"""
run_elmfire_batch.py — Executa elmfire para cada cenário e gera labels (time_of_arrival)

Fluxo por cenário:
  1. Determina UTM zone do ponto de ignição
  2. Baixa GeoTIFFs crus (valores físicos) do LANDFIRE ImageServer em UTM
  3. Gera rasters de clima com valores constantes aleatórios (per paper)
  4. Monta elmfire.data com os paths e parâmetros corretos
  5. Executa elmfire via mpirun
  6. Extrai time_of_arrival (label) do output
  7. Salva par (X=semantic_npy, y=toa) como .pt

Uso:
    python run_elmfire_batch.py \
        --scenarios ../scenarios.csv \
        --semantic-dir ../data/semantic \
        --output-dir ../data/labels \
        --elmfire-bin ../elmfire/repo/build/linux/bin/elmfire \
        --workers 4
"""
import argparse
import csv
import io
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from pyproj import Transformer
import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

LFPS_BASE  = "https://lfps.usgs.gov/arcgis/rest/services"
GRID_SIZE  = 450
CELL_SIZE  = 30       # metros
SIM_HOURS  = 24       # horas de simulação (86400 s)
ELMFIRE_VER = "2025.1002"


def utm_epsg(lon: float, lat: float) -> int:
    """Retorna EPSG do UTM zone correto para o ponto."""
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def bbox_in_utm(lat: float, lon: float, epsg: int) -> tuple:
    """BBox 450×450×30m centrada no ponto em coordenadas UTM."""
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = t.transform(lon, lat)
    half = (GRID_SIZE * CELL_SIZE) / 2
    return (cx - half, cy - half, cx + half, cy + half)


def to_5070(xmin, ymin, xmax, ymax, epsg_src):
    """Converte bbox de qualquer CRS para EPSG:5070 (nativo do LANDFIRE)."""
    t = Transformer.from_crs(f"EPSG:{epsg_src}", "EPSG:5070", always_xy=True)
    x0, y0 = t.transform(xmin, ymin)
    x1, y1 = t.transform(xmax, ymax)
    return (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))


def from_4326_to_5070(lon, lat):
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    return t.transform(lon, lat)


def fetch_raw_tile(folder: str, service: str, bbox_5070: tuple,
                   out_path: Path, out_epsg: int = 5070,
                   size: int = GRID_SIZE, retries: int = 3) -> bool:
    """
    Baixa tile com VALORES BRUTOS do ImageServer (não a renderização colormap)
    e regrava um GeoTIFF limpo com nodata=-9999 explícito.

    Chave: renderingRule={"rasterFunction":"None"} desativa o colormap padrão.
    Sem isso, exportImage devolve a imagem simbolizada e os códigos de
    combustível/valores físicos ficam corrompidos. Além disso, o LANDFIRE usa
    -9999 como NoData mas o TIFF nem sempre carrega a tag — setamos aqui para
    o elmfire tratar corretamente (senão -9999 entraria como combustível).

    bbox_5070: bounding box em EPSG:5070 (nativo LANDFIRE).
    out_epsg:  CRS do raster de saída (igual ao CRS do elmfire.data).
    """
    url = f"{LFPS_BASE}/{folder}/{service}/ImageServer/exportImage"
    xmin, ymin, xmax, ymax = bbox_5070
    params = {
        "bbox":                 f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR":               5070,
        "size":                 f"{size},{size}",
        "imageSR":              out_epsg,   # reprojetar para o CRS de saída
        "format":               "tiff",
        "pixelType":            "S16",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation":        "RSP_NearestNeighbor",
        "renderingRule":        '{"rasterFunction":"None"}',  # valores brutos
        "f":                    "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            href = r.json().get("href")
            if not href:
                raise ValueError(f"Sem href: {r.text[:200]}")
            tif_r = requests.get(href, timeout=120)
            tif_r.raise_for_status()
            # Decodifica, marca nodata (-9999) e regrava GeoTIFF limpo
            with rasterio.open(io.BytesIO(tif_r.content)) as src:
                data = src.read(1)
                profile = src.profile.copy()
            data = np.where(data <= -9990, -9999, data).astype(data.dtype)
            profile.update(driver="GTiff", nodata=-9999)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                log.warning(f"fetch_raw_tile {service}: {e}")
                return False


def write_constant_raster(value: float, template_tif: Path,
                           out_path: Path, dtype="float32") -> None:
    """Cria raster constante com mesma grade/CRS que o template."""
    with rasterio.open(template_tif) as src:
        profile = src.profile.copy()
        profile.update(dtype=dtype, count=1, nodata=-9999.0)
        data = np.full((src.height, src.width), value, dtype=np.float32)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


def make_elmfire_data(inputs_dir: Path, outputs_dir: Path, scratch_dir: Path,
                      epsg: int, xll: float, yll: float,
                      x_ign: float, y_ign: float,
                      lh_moist: float, lw_moist: float,
                      sim_tstop: float) -> str:
    return f"""&INPUTS
FUELS_AND_TOPOGRAPHY_DIRECTORY = '{inputs_dir}'
ASP_FILENAME   = 'asp'
CBD_FILENAME   = 'cbd'
CBH_FILENAME   = 'cbh'
CC_FILENAME    = 'cc'
CH_FILENAME    = 'ch'
DEM_FILENAME   = 'dem'
FBFM_FILENAME  = 'fbfm13'
SLP_FILENAME   = 'slp'
ADJ_FILENAME   = 'adj'
PHI_FILENAME   = 'phi'
DT_METEOROLOGY = 3600.0
WEATHER_DIRECTORY = '{inputs_dir}'
WS_FILENAME  = 'ws'
WD_FILENAME  = 'wd'
M1_FILENAME  = 'm1'
M10_FILENAME = 'm10'
M100_FILENAME= 'm100'
LH_MOISTURE_CONTENT = {lh_moist:.1f}
LW_MOISTURE_CONTENT = {lw_moist:.1f}
/

&OUTPUTS
OUTPUTS_DIRECTORY    = '{outputs_dir}'
DTDUMP               = 3600.
DUMP_FLIN            = .FALSE.
DUMP_SPREAD_RATE     = .FALSE.
DUMP_TIME_OF_ARRIVAL = .TRUE.
CONVERT_TO_GEOTIFF   = .FALSE.
/

&COMPUTATIONAL_DOMAIN
A_SRS = 'EPSG:{epsg}'
COMPUTATIONAL_DOMAIN_CELLSIZE  = {CELL_SIZE}
COMPUTATIONAL_DOMAIN_XLLCORNER = {xll:.2f}
COMPUTATIONAL_DOMAIN_YLLCORNER = {yll:.2f}
/

&TIME_CONTROL
SIMULATION_DT    = 30.0
SIMULATION_TSTOP = {sim_tstop:.1f}
/

&SIMULATOR
NUM_IGNITIONS = 1
X_IGN(1)      = {x_ign:.2f}
Y_IGN(1)      = {y_ign:.2f}
T_IGN(1)      = 0.0
WX_BILINEAR_INTERPOLATION = .TRUE.
WSMFEFF_LOW_MULT = 0.011364
/

&MISCELLANEOUS
PATH_TO_GDAL = '/usr/bin'
SCRATCH      = '{scratch_dir}'
/
"""


LANDFIRE_SERVICES = {
    "dem":    ("Landfire_Topo",   "LF2020_Elev_CONUS"),
    "asp":    ("Landfire_Topo",   "LF2020_Asp_CONUS"),
    "slp":    ("Landfire_Topo",   "LF2020_SlpP_CONUS"),
    "fbfm13": ("Landfire_LF2016", "LF2016_FBFM13_CONUS"),  # Anderson 13 (casa com o semantic .npy)
    "cc":     ("Landfire_LF2016", "LF2016_CC_CONUS"),
    "ch":     ("Landfire_LF2016", "LF2016_CH_CONUS"),
    "cbh":    ("Landfire_LF2016", "LF2016_CBH_CONUS"),
    "cbd":    ("Landfire_LF2016", "LF2016_CBD_CONUS"),
}


def run_scenario(scenario: dict, semantic_dir: Path,
                 output_dir: Path, elmfire_bin: Path) -> tuple[str, bool, str]:
    sid   = scenario["id"]
    split = scenario["split"]
    lat   = float(scenario["lat_inicio"])
    lon   = float(scenario["lon_inicio"])

    pt_path = output_dir / split / f"{sid}.pt"
    if pt_path.exists():
        return sid, True, "já existe"

    npy_path = semantic_dir / split / f"{sid}.npy"
    if not npy_path.exists():
        return sid, False, f"semantic .npy não encontrado"

    epsg = utm_epsg(lon, lat)
    xmin_utm, ymin_utm, xmax_utm, ymax_utm = bbox_in_utm(lat, lon, epsg)

    # Converter bbox UTM → EPSG:5070 para consultar LANDFIRE
    bbox_5070 = to_5070(xmin_utm, ymin_utm, xmax_utm, ymax_utm, epsg)

    # Parâmetros de clima aleatórios, calibrados para condições de incêndio
    # ATIVO (large-scale wildfire). Mantém variabilidade entre cenários (uns
    # mais severos que outros) para a CNN aprender um gradiente de carga, mas
    # com piso de vento e teto de umidade que garantem espalhamento real —
    # mapas de carga densos para o particionamento k-d tree (Entrega 3).
    rng = random.Random(hash(sid) % (2**32))
    ws  = rng.uniform(10, 30)    # wind speed mph  (piso 10 → fogo se propaga)
    wd  = rng.uniform(0, 360)    # wind direction degrees
    m1  = rng.uniform(2, 6)      # 1-hr moisture %  (combustível morto seco)
    m10 = rng.uniform(3, 8)      # 10-hr moisture %
    m100= rng.uniform(5, 12)     # 100-hr moisture %
    lh  = rng.uniform(30, 90)    # live herb moisture %
    lw  = rng.uniform(60, 150)   # live woody moisture %

    with tempfile.TemporaryDirectory(prefix=f"elmfire_{sid}_") as tmpdir:
        tmpdir = Path(tmpdir)
        inputs_dir  = tmpdir / "inputs"
        outputs_dir = tmpdir / "outputs"
        scratch_dir = tmpdir / "scratch"
        for d in (inputs_dir, outputs_dir, scratch_dir):
            d.mkdir()

        # ── Baixar camadas landscape do LANDFIRE em UTM ──────────────────
        for layer, (folder, service) in LANDFIRE_SERVICES.items():
            out_tif = inputs_dir / f"{layer}.tif"
            ok = fetch_raw_tile(folder, service, bbox_5070, out_tif,
                                out_epsg=epsg)   # reprojetar para UTM do cenário
            if not ok:
                return sid, False, f"falha ao baixar {layer}"

        # ── Encontrar ponto de ignição queimável ─────────────────────────
        # FBFM13 não-queimáveis: 91=urban, 92=snow, 93=agri, 98=water, 99=barren
        NON_BURNABLE = {0, 91, 92, 93, 98, 99}
        with rasterio.open(inputs_dir / "fbfm13.tif") as src:
            fbfm = src.read(1)
            transform_utm = src.transform

        cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
        if fbfm[cx, cy] in NON_BURNABLE:
            # Buscar célula queimável mais próxima ao centro (BFS simples)
            from collections import deque
            visited = set()
            queue = deque([(cx, cy)])
            found = None
            while queue and found is None:
                r, c = queue.popleft()
                if (r, c) in visited: continue
                visited.add((r, c))
                if fbfm[r, c] not in NON_BURNABLE and fbfm[r, c] > 0:
                    found = (r, c)
                    break
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr,nc) not in visited:
                        queue.append((nr, nc))
            if found:
                r_ign, c_ign = found
                log.info(f"  {sid}: ignição movida de ({cx},{cy}) fbfm={fbfm[cx,cy]} → ({r_ign},{c_ign}) fbfm={fbfm[r_ign,c_ign]}")
            else:
                return sid, False, "nenhuma célula queimável encontrada no grid"
        else:
            r_ign, c_ign = cx, cy

        # Converter posição de pixel para coordenadas UTM reais
        import rasterio.transform as _rtransform
        x_ign_utm, y_ign_utm = _rtransform.xy(transform_utm, r_ign, c_ign)

        # ── Template para rasters de clima ───────────────────────────────
        template = inputs_dir / "dem.tif"
        write_constant_raster(ws,   template, inputs_dir / "ws.tif")
        write_constant_raster(wd,   template, inputs_dir / "wd.tif")
        write_constant_raster(m1,   template, inputs_dir / "m1.tif")
        write_constant_raster(m10,  template, inputs_dir / "m10.tif")
        write_constant_raster(m100, template, inputs_dir / "m100.tif")
        write_constant_raster(1.0,  template, inputs_dir / "adj.tif")
        write_constant_raster(1.0,  template, inputs_dir / "phi.tif")

        # ── elmfire.data ─────────────────────────────────────────────────
        config = make_elmfire_data(
            inputs_dir, outputs_dir, scratch_dir,
            epsg=epsg,
            xll=xmin_utm, yll=ymin_utm,
            x_ign=x_ign_utm, y_ign=y_ign_utm,
            lh_moist=lh, lw_moist=lw,
            sim_tstop=SIM_HOURS * 3600,
        )
        data_path = inputs_dir / "elmfire.data"
        data_path.write_text(config)

        # ── Executar elmfire ──────────────────────────────────────────────
        result = subprocess.run(
            ["mpirun", "-n", "1", str(elmfire_bin.resolve()), str(data_path)],
            capture_output=True, text=True, cwd=tmpdir, timeout=3600,
        )
        if result.returncode != 0:
            snippet = (result.stderr or result.stdout)[-400:]
            return sid, False, f"elmfire rc={result.returncode}: {snippet}"

        # ── Extrair time_of_arrival ──────────────────────────────────────
        # elmfire com CONVERT_TO_GEOTIFF=.FALSE. gera .bil; converter para .tif
        bil_files = sorted(outputs_dir.glob("time_of_arrival*.bil"))
        toa_tif   = outputs_dir / "time_of_arrival.tif"

        if bil_files:
            subprocess.run(
                ["gdal_translate", "-a_srs", f"EPSG:{epsg}",
                 "-co", "COMPRESS=DEFLATE",
                 str(bil_files[-1]), str(toa_tif)],
                capture_output=True, check=False,
            )

        toa_files = [toa_tif] if toa_tif.exists() else sorted(outputs_dir.glob("time_of_arrival*.tif"))
        if not toa_files:
            return sid, False, f"time_of_arrival não encontrado (outputs: {list(outputs_dir.iterdir())})"

        with rasterio.open(toa_files[-1]) as src:
            toa = src.read(1).astype(np.float32)
            nd  = src.nodata
            if nd is not None:
                toa[toa == nd] = np.nan

        # ── Resize para 450×450 se necessário ────────────────────────────
        if toa.shape != (GRID_SIZE, GRID_SIZE):
            import rasterio.warp as _warp
            toa_resized = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            _warp.reproject(
                source=toa,
                destination=toa_resized,
                resampling=rasterio.warp.Resampling.bilinear,
            )
            toa = toa_resized

        # ── Salvar (X, y) como .pt ───────────────────────────────────────
        X = np.load(npy_path)               # (450, 450, 14)
        X = np.transpose(X, (2, 0, 1))      # → (14, 450, 450)
        y = np.nan_to_num(toa, nan=0.0)     # (450, 450)

        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "X":           torch.tensor(X, dtype=torch.float32),
            "y":           torch.tensor(y, dtype=torch.float32),
            "scenario_id": sid,
        }, pt_path)

    return sid, True, f"OK → {pt_path.name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios",    default="../scenarios.csv")
    parser.add_argument("--semantic-dir", default="../data/semantic")
    parser.add_argument("--output-dir",   default="../data/labels")
    parser.add_argument("--elmfire-bin",
        default="../elmfire/repo/build/linux/bin/elmfire")
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--scenario-id",  default=None)
    args = parser.parse_args()

    elmfire_bin = Path(args.elmfire_bin)
    if not elmfire_bin.exists():
        log.error(f"elmfire não encontrado: {elmfire_bin}")
        sys.exit(1)

    with open(args.scenarios) as f:
        scenarios = list(csv.DictReader(f))
    if args.scenario_id:
        scenarios = [s for s in scenarios if s["id"] == args.scenario_id]

    semantic_dir = Path(args.semantic_dir)
    output_dir   = Path(args.output_dir)
    log.info(f"Rodando elmfire para {len(scenarios)} cenários | workers={args.workers}")

    ok = fail = 0
    failed = []

    if args.workers == 1:
        for sc in scenarios:
            sid, success, msg = run_scenario(sc, semantic_dir, output_dir, elmfire_bin)
            if success:
                ok += 1
                log.info(f"[OK]   {sid}: {msg}")
            else:
                fail += 1
                failed.append((sid, msg))
                log.error(f"[FAIL] {sid}: {msg}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_scenario, sc, semantic_dir, output_dir, elmfire_bin): sc
                for sc in scenarios
            }
            for fut in as_completed(futures):
                sid, success, msg = fut.result()
                if success:
                    ok += 1
                    log.info(f"[OK]   {sid}: {msg}")
                else:
                    fail += 1
                    failed.append((sid, msg))
                    log.error(f"[FAIL] {sid}: {msg}")

    print(f"\n{'='*50}")
    print(f"OK: {ok}  Falhas: {fail}")
    if failed:
        for sid, msg in failed:
            print(f"  FAIL {sid}: {msg}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
