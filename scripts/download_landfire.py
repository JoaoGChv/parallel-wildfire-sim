import argparse
import csv
import time
from pathlib import Path

import requests
from pyproj import Transformer

BASE_URL = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016"

# Mapa layer_key → nome do ImageServer CONUS
LAYERS = {
    "elevation": "LF2016_Elev_220_CONUS",
    "aspect":    "LF2016_Asp_220_CONUS",
    "slope":     "LF2016_SlpP_220_CONUS",
    "fbfm40":    "LF2016_FBFM40_220_CONUS",
    "cc":        "LF2016_CC_220_CONUS",
    "ch":        "LF2016_CH_220_CONUS",
    "cbh":       "LF2016_CBH_220_CONUS",
    "cbd":       "LF2016_CBD_220_CONUS",
}

# EPSG:5070 é o CRS nativo dos ImageServers LANDFIRE CONUS
SRC_CRS  = "EPSG:5070"
DST_CRS  = "EPSG:4326"


def to_5070(lon: float, lat: float) -> tuple[float, float]:
    t = Transformer.from_crs("EPSG:4326", SRC_CRS, always_xy=True)
    return t.transform(lon, lat)


def bbox_4326_to_5070(bbox_4326: tuple) -> tuple:
    xmin, ymin, xmax, ymax = bbox_4326
    x0, y0 = to_5070(xmin, ymin)
    x1, y1 = to_5070(xmax, ymax)
    return (min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1))


def discover_service_name(layer_key: str) -> str | None:
    """Descobre o nome correto do ImageServer para o layer (versão pode variar)."""
    folder_url = f"{BASE_URL}?f=json"
    r = requests.get(folder_url, timeout=15)
    services = r.json().get("services", [])
    candidates = [s["name"].split("/")[-1] for s in services
                  if s["type"] == "ImageServer"]

    # Prefixos a tentar
    layer_map = {
        "elevation": ["Elev", "DEM"],
        "aspect":    ["Asp"],
        "slope":     ["SlpP", "Slp"],
        "fbfm40":    ["FBFM40"],
        "cc":        ["CC"],
        "ch":        ["CH"],
        "cbh":       ["CBH"],
        "cbd":       ["CBD"],
    }
    prefixes = layer_map.get(layer_key, [])
    for c in candidates:
        for p in prefixes:
            if f"_{p}_" in c and "CONUS" in c and "2016" in c:
                return c
    return None


def export_image(service_name: str, bbox_5070: tuple,
                 size_px: int = 4500) -> bytes | None:
    """Exporta raster do ImageServer como GeoTIFF."""
    xmin, ymin, xmax, ymax = bbox_5070
    url = f"{BASE_URL}/{service_name}/ImageServer/exportImage"
    params = {
        "bbox":                  f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR":                5070,
        "size":                  f"{size_px},{size_px}",
        "imageSR":               4326,
        "format":                "tiff",
        "pixelType":             "UNKNOWN",
        "noDataInterpretation":  "esriNoDataMatchAny",
        "interpolation":         "RSP_BilinearInterpolation",
        "f":                     "json",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "href" not in data:
        print(f"    Sem href: {data}")
        return None

    tif_url = data["href"]
    tif_r = requests.get(tif_url, timeout=120)
    tif_r.raise_for_status()
    return tif_r.content


def download_layer(layer_key: str, bbox_4326: tuple,
                   output_dir: Path, size_px: int = 4500) -> bool:
    dest = output_dir / layer_key
    dest.mkdir(parents=True, exist_ok=True)

    out_tif = dest / f"{layer_key}_lf2016.tif"
    if out_tif.exists() and out_tif.stat().st_size > 10_000:
        print(f"  [{layer_key}] já existe ({out_tif.stat().st_size/1024/1024:.1f} MB), pulando.")
        return True

    # Descobrir nome correto do serviço
    service_name = LAYERS.get(layer_key)
    if not service_name:
        service_name = discover_service_name(layer_key)
    if not service_name:
        print(f"  [{layer_key}] serviço não encontrado")
        return False

    bbox_5070 = bbox_4326_to_5070(bbox_4326)
    print(f"  [{layer_key}] {service_name}")

    for attempt in range(3):
        try:
            content = export_image(service_name, bbox_5070, size_px)
            if content and len(content) > 1000:
                with open(out_tif, "wb") as f:
                    f.write(content)
                print(f"    Salvo: {out_tif.name} ({len(content)/1024/1024:.1f} MB)")
                return True
            else:
                print(f"    Tentativa {attempt+1}: resposta vazia, tentando serviço alternativo")
                # Tentar descobrir nome automaticamente
                service_name = discover_service_name(layer_key)
                if not service_name:
                    break
        except Exception as e:
            print(f"    Tentativa {attempt+1} erro: {e}")
            time.sleep(5)

    return False


def bbox_from_scenarios(csv_path: str, buffer: float = 0.5) -> tuple:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    lons = [float(r["lon_inicio"]) for r in rows]
    lats = [float(r["lat_inicio"]) for r in rows]
    return (min(lons)-buffer, min(lats)-buffer,
            max(lons)+buffer, max(lats)+buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox",      help="xmin,ymin,xmax,ymax (WGS84)")
    parser.add_argument("--scenarios", help="scenarios.csv")
    parser.add_argument("--layers",    nargs="+", default=list(LAYERS.keys()))
    parser.add_argument("--output",    default="../data/landfire")
    parser.add_argument("--size",      type=int, default=4500,
                        help="Tamanho do raster em pixels (default 4500)")
    args = parser.parse_args()

    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
    elif args.scenarios:
        bbox = bbox_from_scenarios(args.scenarios)
        print(f"BBox dos {args.scenarios}: {tuple(round(x,3) for x in bbox)}")
    else:
        parser.error("Informe --bbox ou --scenarios")

    output_dir = Path(args.output)
    print(f"\nDownload de {len(args.layers)} camadas LANDFIRE 2016 → {output_dir}\n")

    results = {}
    for key in args.layers:
        results[key] = download_layer(key, bbox, output_dir, args.size)
        time.sleep(1)

    ok = sum(results.values())
    print(f"\nConcluído: {ok}/{len(args.layers)} camadas OK")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"Falhas: {failed}")


if __name__ == "__main__":
    main()
