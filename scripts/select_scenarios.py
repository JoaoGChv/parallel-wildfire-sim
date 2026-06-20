import argparse
import json
import random
import csv
from pathlib import Path

CONUS_BBOX = (-125.0, 24.5, -66.5, 49.5)

# NWCG fire size class definitions (acres)
CLASS_E_MIN =   300.0
CLASS_E_MAX =   999.9
CLASS_F_MIN = 1_000.0
CLASS_F_MAX = 4_999.9
CLASS_G_MIN = 5_000.0

LARGE_SCALE_MIN = CLASS_E_MIN   # >300 acres = large-scale (paper definition)

TRAIN_COUNT = 51
TEST_COUNT  = 17
RANDOM_SEED = 42


def load_perimeters(geojson_path: str) -> list[dict]:
    with open(geojson_path) as f:
        return json.load(f)["features"]


def get_fire_area_acres(feature: dict) -> float:
    props = feature.get("properties", {})
    for key in ("gisacres", "GIS_ACRES", "GISACRES", "gis_acres", "ACRES", "acres",
                "AREAACRES", "Shape_Area", "shape__Area"):
        val = props.get(key)
        if val is not None:
            try:
                area = float(val)
                # shape__Area está em m² — converter para acres
                if key in ("Shape_Area", "shape__Area") and area > 1e5:
                    area = area / 4046.86
                return area
            except (ValueError, TypeError):
                pass
    return 0.0


def get_fire_center(feature: dict) -> tuple[float, float] | None:
    geom = feature.get("geometry", {})
    if not geom:
        return None
    geom_type = geom.get("type", "")
    coords = geom.get("coordinates", [])
    try:
        ring = coords[0] if geom_type == "Polygon" else coords[0][0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        return (sum(lats) / len(lats), sum(lons) / len(lons))
    except (IndexError, TypeError):
        return None


def is_in_conus(lat: float, lon: float) -> bool:
    xmin, ymin, xmax, ymax = CONUS_BBOX
    return xmin <= lon <= xmax and ymin <= lat <= ymax


def get_fire_name(feature: dict) -> str:
    props = feature.get("properties", {})
    for key in ("incidentname", "FIRE_NAME", "firename", "FIRENAME", "INCIDENT_NAME", "Name", "name"):
        val = props.get(key)
        if val and str(val).strip():
            return str(val).strip().replace(" ", "_").upper()
    return f"FIRE_{abs(hash(str(feature)))%100000}"


def get_ignition_date(feature: dict) -> str:
    props = feature.get("properties", {})
    # perimeterdatetime é timestamp Unix em ms
    for key in ("perimeterdatetime", "datecurrent"):
        val = props.get(key)
        if val:
            try:
                import datetime
                ts = int(val) / 1000
                return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            except Exception:
                pass
    for key in ("ALARM_DATE", "alarm_date", "STARTDATE", "FIRE_START", "ignitiondate"):
        val = props.get(key)
        if val:
            return str(val)[:10]
    return "2016-07-01"


def fire_size_class(acres: float) -> str:
    if acres < 0.25:      return "A"
    if acres < 10:        return "B"
    if acres < 100:       return "C"
    if acres < 300:       return "D"
    if acres < 1_000:     return "E"
    if acres < 5_000:     return "F"
    return "G"


def filter_large_scale(features: list[dict]) -> dict[str, list]:
    """
    Filtra incêndios large-scale (>300 acres) no CONUS.
    Usa apenas perímetro final (latest=Y) para evitar duplicatas.
    Retorna separados por classe F vs restante.
    """
    class_f  = []
    others   = []
    skipped  = {"too_small": 0, "outside_conus": 0, "no_centroid": 0, "not_latest": 0}

    for feat in features:
        # Usar apenas o perímetro final de cada incêndio
        if feat.get("properties", {}).get("latest") != "Y":
            skipped["not_latest"] += 1
            continue
        center = get_fire_center(feat)
        if center is None:
            skipped["no_centroid"] += 1
            continue

        lat, lon = center
        if not is_in_conus(lat, lon):
            skipped["outside_conus"] += 1
            continue

        acres = get_fire_area_acres(feat)
        if acres < LARGE_SCALE_MIN:
            skipped["too_small"] += 1
            continue

        sc = fire_size_class(acres)
        if sc == "F":
            class_f.append(feat)
        else:
            others.append(feat)

    total_valid = len(class_f) + len(others)
    print(f"  Total features NIFC:     {len(features)}")
    print(f"  Perímetros intermediários (latest=N): {skipped['not_latest']}")
    print(f"  Fora do CONUS:           {skipped['outside_conus']}")
    print(f"  Menores de 300 acres:    {skipped['too_small']}")
    print(f"  Sem centroide:           {skipped['no_centroid']}")
    print(f"  Large-scale válidos:     {total_valid}  (paper cita 341)")
    print(f"    Classe F (1k-5k acres): {len(class_f)}  (paper usa 17 como test)")
    print(f"    Outros (E/G):          {len(others)}")
    return {"class_f": class_f, "others": others}


def select_scenarios(filtered: dict) -> tuple[list, list]:
    """
    Seleciona 17 class F para test e 51 aleatórios para train.
    Replica o método do paper (random selection, seed fixo para reprodutibilidade).
    """
    rng = random.Random(RANDOM_SEED)

    class_f = filtered["class_f"]
    others  = filtered["others"]

    # Test: 17 class F (se tiver mais que 17, amostra aleatória)
    if len(class_f) >= TEST_COUNT:
        test = rng.sample(class_f, TEST_COUNT)
    else:
        print(f"  Aviso: apenas {len(class_f)} fires classe F (esperado 17)")
        test = class_f

    # Train: 51 aleatórios do restante (excluindo os selecionados para test)
    test_ids = {id(f) for f in test}
    pool = [f for f in (class_f + others) if id(f) not in test_ids]

    if len(pool) >= TRAIN_COUNT:
        train = rng.sample(pool, TRAIN_COUNT)
    else:
        print(f"  Aviso: apenas {len(pool)} fires disponíveis para treino (esperado 51)")
        train = pool

    return train, test


def build_row(feat: dict, sid: str, split: str) -> dict:
    center = get_fire_center(feat)
    lat, lon = center if center else (0.0, 0.0)
    acres = get_fire_area_acres(feat)
    return {
        "id":            sid,
        "nome_incendio": get_fire_name(feat),
        "lat_inicio":    round(lat, 6),
        "lon_inicio":    round(lon, 6),
        "data_inicio":   get_ignition_date(feat),
        "area_acres":    round(acres, 1),
        "size_class":    fire_size_class(acres),
        "split":         split,
    }


def save_csv(train: list, test: list, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, feat in enumerate(train):
        rows.append(build_row(feat, f"S{i+1:03d}", "train"))
    for i, feat in enumerate(test):
        rows.append(build_row(feat, f"S{TRAIN_COUNT+i+1:03d}", "test"))

    fieldnames = ["id", "nome_incendio", "lat_inicio", "lon_inicio",
                  "data_inicio", "area_acres", "size_class", "split"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nscenarios.csv salvo em: {output_path}")
    print(f"  Treino: {len(train)} cenários")
    print(f"  Teste:  {len(test)} cenários  (todos classe F)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perimeters", required=True)
    parser.add_argument("--output", default="../scenarios.csv")
    args = parser.parse_args()

    print("Carregando perímetros NIFC 2016...")
    features = load_perimeters(args.perimeters)

    print("\nFiltrando large-scale fires (>300 acres, CONUS)...")
    filtered = filter_large_scale(features)

    print("\nSelecionando 51 train + 17 test...")
    train, test = select_scenarios(filtered)

    save_csv(train, test, args.output)


if __name__ == "__main__":
    main()
