"""
preprocess.py — Pipeline de preprocessing: gera arrays semânticos 450×450×14 por cenário

Estratégia: para cada cenário, consulta diretamente os ImageServers do LANDFIRE
e baixa apenas o tile 450×450 necessário (13.5km × 13.5km). Não precisa baixar
o CONUS inteiro.

Os valores são armazenados BRUTOS (unidades físicas), não normalizados:
elevação em m, slope em %, aspect em graus, fbfm13 em código (1–13 + 91–99
não-queimável), clima em unidade nativa. A normalização por canal para a CNN
é feita em dataset.py. O simulador C consome os valores brutos diretamente.

14 camadas (Kwon et al. 2022):
  Landscape (8) — LANDFIRE ImageServers (valores brutos via renderingRule None):
    0: elevation   → Landfire_Topo/LF2020_Elev_CONUS      (m)
    1: aspect      → Landfire_Topo/LF2020_Asp_CONUS        (graus 0–360)
    2: slope       → Landfire_Topo/LF2020_SlpP_CONUS       (percent)
    3: fbfm13      → Landfire_LF2016/LF2016_FBFM13_CONUS   (código Anderson 1–13; 91–99 não-queimável)
    4: cc          → Landfire_LF2016/LF2016_CC_CONUS       (%)
    5: ch          → Landfire_LF2016/LF2016_CH_CONUS
    6: cbh         → Landfire_LF2016/LF2016_CBH_CONUS
    7: cbd         → Landfire_LF2016/LF2016_CBD_CONUS
  Weather (6) — constantes aleatórias por cenário (per paper), unidades nativas:
    8:  temperature  (50–120 °F)
    9:  humidity     (5–95 %)
    10: cloud_cover  (0–100 %)
    11: precipitation(0–0.5 in/h)
    12: wind_direction(0–360°)
    13: wind_speed   (0–30 mph)

Uso:
    python preprocess.py --scenarios ../scenarios.csv --output-dir ../data/semantic
    python preprocess.py --scenarios ../scenarios.csv --output-dir ../data/semantic --workers 4
    python preprocess.py --scenarios ../scenarios.csv --output-dir ../data/semantic --scenario-id S001
"""
import argparse
import csv
import io
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import rasterio
import rasterio.warp
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fbfm40_crosswalk import fbfm40_to_anderson

# Modo de combustível: "fbfm13" (direto, default) ou "fbfm40" (baixa S&B e traduz)
FUEL_MODE = "fbfm13"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LFPS_BASE  = "https://lfps.usgs.gov/arcgis/rest/services"
CELL_SIZE  = 30        # metros por célula
GRID_SIZE  = 450       # células por lado
DTYPE      = np.float32

# (nome, pasta, serviço, interpolação)
# FBFM13 = categórico (Anderson 13) → NearestNeighbor para preservar os códigos.
# Demais camadas são contínuas → BilinearInterpolation.
NEAREST  = "RSP_NearestNeighbor"
BILINEAR = "RSP_BilinearInterpolation"
LANDSCAPE_SERVICES = [
    ("elevation", "Landfire_Topo",  "LF2020_Elev_CONUS",  BILINEAR),
    ("aspect",    "Landfire_Topo",  "LF2020_Asp_CONUS",   BILINEAR),
    ("slope",     "Landfire_Topo",  "LF2020_SlpP_CONUS",  BILINEAR),
    ("fbfm13",    "Landfire_LF2016","LF2016_FBFM13_CONUS", NEAREST),
    ("cc",        "Landfire_LF2016","LF2016_CC_CONUS",    BILINEAR),
    ("ch",        "Landfire_LF2016","LF2016_CH_CONUS",    BILINEAR),
    ("cbh",       "Landfire_LF2016","LF2016_CBH_CONUS",   BILINEAR),
    ("cbd",       "Landfire_LF2016","LF2016_CBD_CONUS",   BILINEAR),
]

WEATHER_PARAMS = [
    ("temperature",   50.0, 120.0),
    ("humidity",       5.0,  95.0),
    ("cloud_cover",    0.0, 100.0),
    ("precipitation",  0.0,   0.5),
    ("wind_direction", 0.0, 360.0),
    ("wind_speed",     0.0,  30.0),
]

LAYER_NAMES = [s[0] for s in LANDSCAPE_SERVICES] + [w[0] for w in WEATHER_PARAMS]


def meters_to_degrees(meters: float, lat: float) -> tuple[float, float]:
    lat_deg = meters / 111_320
    lon_deg = meters / (111_320 * np.cos(np.radians(lat)))
    return lon_deg, lat_deg


def get_bbox_4326(lat: float, lon: float) -> tuple[float, float, float, float]:
    half_m = (GRID_SIZE * CELL_SIZE) / 2
    dlon, dlat = meters_to_degrees(half_m, lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def bbox_to_5070(bbox_4326: tuple) -> tuple[float, float, float, float]:
    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xmin, ymin, xmax, ymax = bbox_4326
    x0, y0 = t.transform(xmin, ymin)
    x1, y1 = t.transform(xmax, ymax)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def fetch_tile(folder: str, service: str, bbox_5070: tuple,
               interpolation: str = BILINEAR, retries: int = 3) -> np.ndarray | None:
    """
    Busca um tile 450×450 do ImageServer LANDFIRE e retorna array float32
    com os VALORES BRUTOS do raster (não a renderização de colormap).

    Chave: renderingRule={"rasterFunction":"None"} desativa o colormap padrão
    do ImageServer e devolve os valores físicos da fonte (ex.: códigos FBFM13,
    slope em %, elevação em m). Sem isso, exportImage devolve a imagem
    simbolizada e os dados ficam corrompidos.
    """
    url = f"{LFPS_BASE}/{folder}/{service}/ImageServer/exportImage"
    xmin, ymin, xmax, ymax = bbox_5070
    params = {
        "bbox":                 f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR":               5070,
        "size":                 f"{GRID_SIZE},{GRID_SIZE}",
        "imageSR":              5070,          # Albers equal-area → pixels de 30 m
        "format":               "tiff",
        "pixelType":            "S16",         # serviços LANDFIRE são S16
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation":        interpolation,
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

            with rasterio.open(io.BytesIO(tif_r.content)) as src:
                data = src.read(1).astype(DTYPE)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                # LANDFIRE usa -9999 como NoData, mas rasterio nem sempre o detecta
                # (src.nodata=None). Trata o sentinela explicitamente.
                data[data <= -9990] = np.nan

            return data

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"fetch_tile {service} falhou: {e}")


def clean_raw(arr: np.ndarray) -> np.ndarray:
    """Mantém os valores físicos brutos; só substitui nodata/NaN por 0."""
    out = np.nan_to_num(arr, nan=0.0).astype(DTYPE)
    return out


def make_weather_layers(scenario_id: str) -> list[np.ndarray]:
    """Constantes aleatórias por cenário, em UNIDADES NATIVAS (per paper).
    A normalização para a CNN é feita por canal em dataset.py, não aqui."""
    rng = random.Random(hash(scenario_id) % (2**32))
    layers = []
    for _, lo, hi in WEATHER_PARAMS:
        val = rng.uniform(lo, hi)
        layers.append(np.full((GRID_SIZE, GRID_SIZE), val, dtype=DTYPE))
    return layers


def process_scenario(scenario: dict, output_dir: Path) -> tuple[str, bool, str]:
    sid   = scenario["id"]
    lat   = float(scenario["lat_inicio"])
    lon   = float(scenario["lon_inicio"])
    split = scenario["split"]

    out_path = output_dir / split / f"{sid}.npy"
    if out_path.exists():
        return sid, True, "já existe"

    bbox_4326 = get_bbox_4326(lat, lon)
    bbox_5070 = bbox_to_5070(bbox_4326)

    layers = []
    for layer_name, folder, service, interp in LANDSCAPE_SERVICES:
        try:
            # Opção FBFM40: baixa o combustível detalhado (Scott & Burgan, 40
            # modelos) e traduz para Anderson 13 via crosswalk, que é o que o
            # simulador entende. Mantém a camada compatível, com fonte mais rica.
            if layer_name == "fbfm13" and FUEL_MODE == "fbfm40":
                arr = fetch_tile(folder, "LF2016_FBFM40_CONUS", bbox_5070,
                                 interpolation=interp)
                arr = fbfm40_to_anderson(clean_raw(arr))
                layers.append(arr)
            else:
                arr = fetch_tile(folder, service, bbox_5070, interpolation=interp)
                layers.append(clean_raw(arr))
        except Exception as e:
            return sid, False, f"erro em {layer_name}: {e}"

    layers.extend(make_weather_layers(sid))
    assert len(layers) == 14

    semantic = np.stack(layers, axis=-1)   # (450, 450, 14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, semantic)
    return sid, True, f"OK → {out_path.name}"


def main():
    parser = argparse.ArgumentParser(description="Gera arrays semânticos 450×450×14")
    parser.add_argument("--scenarios",   default="../scenarios.csv")
    parser.add_argument("--output-dir",  default="../data/semantic")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--fuel", default="fbfm13", choices=["fbfm13", "fbfm40"],
                        help="fbfm13 = Anderson direto; fbfm40 = baixa Scott&Burgan "
                             "e traduz p/ Anderson via crosswalk")
    args = parser.parse_args()

    global FUEL_MODE
    FUEL_MODE = args.fuel
    output_dir = Path(args.output_dir)
    with open(args.scenarios) as f:
        scenarios = list(csv.DictReader(f))

    if args.scenario_id:
        scenarios = [s for s in scenarios if s["id"] == args.scenario_id]

    log.info(f"Processando {len(scenarios)} cenários | workers={args.workers}")
    ok = fail = skip = 0
    failed = []

    if args.workers == 1:
        for sc in scenarios:
            sid, success, msg = process_scenario(sc, output_dir)
            if success:
                ok += ("já existe" not in msg)
                skip += ("já existe" in msg)
                log.info(f"[OK]   {sid}: {msg}")
            else:
                fail += 1
                failed.append((sid, msg))
                log.error(f"[FAIL] {sid}: {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_scenario, sc, output_dir): sc for sc in scenarios}
            for fut in as_completed(futures):
                sid, success, msg = fut.result()
                if success:
                    ok += ("já existe" not in msg)
                    skip += ("já existe" in msg)
                    log.info(f"[OK]   {sid}: {msg}")
                else:
                    fail += 1
                    failed.append((sid, msg))
                    log.error(f"[FAIL] {sid}: {msg}")

    print(f"\n{'='*50}")
    print(f"OK: {ok}  Pulados: {skip}  Falhas: {fail}")
    if failed:
        for sid, msg in failed:
            print(f"  FAIL {sid}: {msg}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
