import argparse
import io
import random
from pathlib import Path

import numpy as np
import requests
import rasterio
from pyproj import Transformer

GRID = 450
CELL = 30                       # metros/célula
DTYPE = np.float32

ELEV_URL = ("https://elevation3d.arcgis.com/arcgis/rest/services/"
            "WorldElevation3D/Terrain3D/ImageServer/exportImage")
LULC_URL = ("https://ic.imagery1.arcgis.com/arcgis/rest/services/"
            "Sentinel2_10m_LandCover/ImageServer/exportImage")

# Locais reais propensos a incêndio no Brasil (centro do tile)
BR_SITES = [
    ("BR_PANTANAL",    -16.700, -56.800),   # Pantanal (MT/MS)
    ("BR_CERRADO_GO",  -13.800, -47.700),   # Cerrado (Chapada dos Veadeiros)
    ("BR_AMAZONIA_PA", -3.900,  -54.300),   # Arco do desmatamento (PA)
    ("BR_CERRADO_TO",  -10.200, -48.300),   # Cerrado (TO)
]

# Classes Esri Sentinel-2 LULC → FBFM13 (Anderson). 1-13 queimável; 91-99 não.
#  1 Água, 2 Árvores, 4 Veg. alagada, 5 Agricultura, 7 Construído, 8 Solo exposto,
#  9 Neve, 10 Nuvem, 11 Pastagem/campo
LULC_TO_FBFM = {1: 98, 2: 8, 4: 98, 5: 1, 7: 91, 8: 99, 9: 92, 10: 99, 11: 3}
# Copa aproximada por classe (cc %, ch m*10, cbh m*10, cbd kg/m3*100)
LULC_TO_CANOPY = {
    2:  (60.0, 200.0, 30.0, 12.0),   # árvores
    11: (10.0,  20.0,  2.0,  3.0),   # campo
    5:  (5.0,   10.0,  1.0,  2.0),   # agricultura
}


def bbox_3857(lat, lon):
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    cx, cy = t.transform(lon, lat)
    half = GRID * CELL / 2
    return (cx - half, cy - half, cx + half, cy + half)


def fetch(url, bbox, interpolation, retries=3):
    xmin, ymin, xmax, ymax = bbox
    # Os ImageServers globais Esri (Terrain3D, Sentinel2 LandCover) já devolvem
    # valores BRUTOS por padrão e REJEITAM renderingRule=None ("client function").
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": 3857,
        "size": f"{GRID},{GRID}", "imageSR": 3857, "format": "tiff",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": interpolation,
        "f": "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60); r.raise_for_status()
            href = r.json().get("href")
            if not href:
                raise ValueError(f"sem href: {r.text[:150]}")
            tif = requests.get(href, timeout=120); tif.raise_for_status()
            with rasterio.open(io.BytesIO(tif.content)) as s:
                return s.read(1).astype(np.float64)
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"fetch falhou: {e}")


def slope_aspect(elev):
    """Slope (%) e aspect (graus 0-360) a partir do DEM (espaçamento 30 m)."""
    dzdy, dzdx = np.gradient(elev, CELL, CELL)
    slope_pct = 100.0 * np.sqrt(dzdx**2 + dzdy**2)
    aspect = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360.0) % 360.0
    return slope_pct.astype(DTYPE), aspect.astype(DTYPE)


def weather_layers(sid):
    rng = random.Random(hash(sid) % (2**32))
    ranges = [(50, 120), (5, 95), (0, 100), (0, 0.5), (0, 360), (0, 30)]
    return [np.full((GRID, GRID), rng.uniform(lo, hi), DTYPE) for lo, hi in ranges]


def build_scenario(sid, lat, lon):
    bbox = bbox_3857(lat, lon)
    elev = fetch(ELEV_URL, bbox, "RSP_BilinearInterpolation")
    elev = np.nan_to_num(np.where(elev <= -9990, np.nan, elev), nan=0.0)
    slope, aspect = slope_aspect(elev)
    lulc = fetch(LULC_URL, bbox, "RSP_NearestNeighbor")

    fbfm = np.zeros((GRID, GRID), DTYPE)
    cc = np.zeros((GRID, GRID), DTYPE); ch = np.zeros((GRID, GRID), DTYPE)
    cbh = np.zeros((GRID, GRID), DTYPE); cbd = np.zeros((GRID, GRID), DTYPE)
    for cls, fuel in LULC_TO_FBFM.items():
        m = (lulc == cls)
        fbfm[m] = fuel
        if cls in LULC_TO_CANOPY:
            cc[m], ch[m], cbh[m], cbd[m] = LULC_TO_CANOPY[cls]
    fbfm[fbfm == 0] = 99   # classes não mapeadas → não-queimável

    layers = [elev.astype(DTYPE), aspect, slope, fbfm, cc, ch, cbh, cbd]
    layers += weather_layers(sid)
    return np.stack(layers, axis=-1)   # (450,450,14)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/semantic_brazil")
    ap.add_argument("--site", default=None, help="só um site (ex.: BR_PANTANAL)")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    sites = [s for s in BR_SITES if args.site is None or s[0] == args.site]
    for sid, lat, lon in sites:
        try:
            sem = build_scenario(sid, lat, lon)
            np.save(out / f"{sid}.npy", sem)
            fb = sem[..., 3]
            burn = ((fb >= 1) & (fb <= 13)).mean() * 100
            print(f"[OK] {sid}: elev {sem[...,0].min():.0f}-{sem[...,0].max():.0f}m | "
                  f"slope max {sem[...,2].max():.0f}% | queimável {burn:.0f}% | "
                  f"fbfm {sorted(set(np.unique(fb).astype(int)))[:10]}")
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")


if __name__ == "__main__":
    main()
