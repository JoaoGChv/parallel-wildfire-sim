import argparse
import json
import time
from pathlib import Path
import requests

# FeatureService do NIFC — Historic Perimeters 2016
NIFC_2016_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "Historic_Geomac_Perimeters_2016/FeatureServer/0"
)

# Alternativa: combined dataset com filtro por ano
NIFC_COMBINED_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "Interagency_Fire_Perimeter_History_All_Years/FeatureServer/0"
)

PAGE_SIZE = 2000


def fetch_all(service_url: str, where: str = "1=1") -> list[dict]:
    query_url = service_url + "/query"
    all_features = []
    offset = 0
    while True:
        params = {
            "where":             where,
            "outFields":         "*",
            "outSR":             4326,
            "f":                 "geojson",
            "resultOffset":      offset,
            "resultRecordCount": PAGE_SIZE,
            "returnGeometry":    True,
        }
        r = requests.get(query_url, params=params, timeout=120)
        r.raise_for_status()
        features = r.json().get("features", [])
        all_features.extend(features)
        print(f"  {len(all_features)} features baixadas...", end="\r", flush=True)
        if len(features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.3)
    print()
    return all_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../data/nifc/perimeters_2016")
    parser.add_argument("--use-combined", action="store_true",
                        help="Usar dataset combinado e filtrar por ano 2016")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "nifc_perimeters_2016.geojson"

    if out_file.exists():
        print(f"Já existe: {out_file}")
        return

    if args.use_combined:
        print("Baixando dataset NIFC combinado (filtro FIRE_YEAR=2016)...")
        features = fetch_all(NIFC_COMBINED_URL, where="FIRE_YEAR=2016")
    else:
        print("Baixando dataset NIFC 2016 específico...")
        features = fetch_all(NIFC_2016_URL)

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_file, "w") as f:
        json.dump(geojson, f)

    print(f"Salvo: {out_file}  ({len(features)} perímetros)")


if __name__ == "__main__":
    main()
