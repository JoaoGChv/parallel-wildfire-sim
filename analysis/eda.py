"""
Uso:
    python analysis/eda.py --semantic-dir data/semantic --out analysis/out
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ordem das camadas conforme preprocess.py (LANDSCAPE_SERVICES + WEATHER_PARAMS)
LAYER_NAMES = [
    "elevation", "aspect", "slope", "fbfm13", "cc", "ch", "cbh", "cbd",
    "temperature", "humidity", "cloud_cover", "precipitation",
    "wind_direction", "wind_speed",
]

# Faixas físicas esperadas (LANDFIRE / FARSITE) p/ valores BRUTOS — flagam anomalia.
# (min, max, descrição).
EXPECTED = {
    "elevation":      (-150, 4500,  "metros"),
    "aspect":         (-1, 360,     "graus 0-360 (-1 = plano)"),
    "slope":          (0, 400,      "percent (serviço LANDFIRE 0-400)"),
    "fbfm13":         (0, 99,       "código Anderson 1-13 + 91-99 não-queimável"),
    "cc":             (0, 100,      "cobertura de copa %"),
    "ch":             (0, 600,      "altura de copa (m*10)"),
    "cbh":            (0, 200,      "base de copa (m*10)"),
    "cbd":            (0, 60,       "densidade de copa (kg/m3*100)"),
    "temperature":    (40, 130,     "weather °F (constante/cenário)"),
    "humidity":       (0, 100,      "weather %"),
    "cloud_cover":    (0, 100,      "weather %"),
    "precipitation":  (0, 1,        "weather in/h"),
    "wind_direction": (0, 360,      "weather graus"),
    "wind_speed":     (0, 40,       "weather mph"),
}

# Índices que o simulador C consome (rothermel.h)
SIM_INDICES = {"IDX_ELEVATION": 0, "IDX_ASPECT": 1, "IDX_SLOPE": 2, "IDX_FBFM13": 3}


def collect(files):
    """Agrega estatísticas por camada ao longo de todos os cenários."""
    n = len(LAYER_NAMES)
    per = {i: {"min": [], "max": [], "mean": [], "std": [], "pzero": [],
               "n_unique": [], "spatial_var": []} for i in range(n)}
    any_nan = False
    for f in files:
        a = np.load(f)
        if np.isnan(a).any():
            any_nan = True
        for i in range(n):
            L = a[..., i]
            per[i]["min"].append(float(L.min()))
            per[i]["max"].append(float(L.max()))
            per[i]["mean"].append(float(L.mean()))
            per[i]["std"].append(float(L.std()))
            per[i]["pzero"].append(float((L == 0).mean() * 100))
            per[i]["n_unique"].append(int(np.unique(L).size))
            per[i]["spatial_var"].append(float(L.var()))
    return per, any_nan


def classify(name, gmin, gmax, median_unique, median_std):
    """Retorna (normalizado?, constante?, flags[])."""
    flags = []
    normalized = gmax <= 1.0001
    constant = median_unique <= 1 or median_std < 1e-9
    lo, hi, desc = EXPECTED[name]
    if hi is not None and gmax > hi + 1e-6:
        flags.append(f"max={gmax:.1f} > esperado {hi} ({desc})")
    if lo is not None and gmin < lo - 1e-6:
        flags.append(f"min={gmin:.1f} < esperado {lo}")
    return normalized, constant, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-dir", default="data/semantic")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--montage-ids", nargs="+", default=["S001", "S052"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = sorted(glob.glob(f"{args.semantic_dir}/train/*.npy"))
    test = sorted(glob.glob(f"{args.semantic_dir}/test/*.npy"))
    files = train + test
    per, any_nan = collect(files)

    # ── layer_stats.csv ────────────────────────────────────────────────────
    rows = []
    report_layers = []
    for i, name in enumerate(LAYER_NAMES):
        gmin = min(per[i]["min"])
        gmax = max(per[i]["max"])
        med_mean = float(np.median(per[i]["mean"]))
        med_std = float(np.median(per[i]["std"]))
        med_pzero = float(np.median(per[i]["pzero"]))
        med_unique = float(np.median(per[i]["n_unique"]))
        normalized, constant, flags = classify(name, gmin, gmax, med_unique, med_std)
        rows.append({
            "idx": i, "name": name, "min": gmin, "max": gmax,
            "median_mean": round(med_mean, 4), "median_std": round(med_std, 4),
            "median_pct_zero": round(med_pzero, 2), "median_n_unique": med_unique,
            "normalized": normalized, "constant": constant,
            "flags": " | ".join(flags),
        })
        report_layers.append((i, name, gmin, gmax, med_pzero, normalized, constant, flags))

    import csv
    with open(out / "layer_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── eda_report.md ──────────────────────────────────────────────────────
    lines = []
    lines.append("# EDA — Dataset semântico (450×450×14)\n")
    lines.append(f"- Cenários: **{len(files)}** ({len(train)} treino + {len(test)} teste)")
    lines.append(f"- NaN em algum arquivo: **{any_nan}**")
    lines.append("- O '14' refere-se a **14 CAMADAS por cenário**, não a 14 cenários.\n")

    lines.append("## Tabela de camadas\n")
    lines.append("| L | nome | min | max | %zero(med) | normalizada | constante | flags |")
    lines.append("|---|------|-----|-----|-----------|-------------|-----------|-------|")
    for i, name, gmin, gmax, pz, norm, const, flags in report_layers:
        fl = "; ".join(flags) if flags else "—"
        lines.append(f"| {i} | {name} | {gmin:.2f} | {gmax:.2f} | {pz:.1f}% | "
                     f"{'sim' if norm else 'NÃO'} | {'sim' if const else 'não'} | {fl} |")

    lines.append("\n## Achados\n")
    # Inconsistência de normalização
    normalized_layers = [r[1] for r in report_layers if r[5]]
    raw_layers = [r[1] for r in report_layers if not r[5] and not r[6]]
    const_layers = [r[1] for r in report_layers if r[6]]
    lines.append(f"- **Normalizadas [0,1]:** {', '.join(normalized_layers) or '—'}")
    lines.append(f"- **Cruas (faixa física):** {', '.join(raw_layers) or '—'}")
    lines.append(f"- **Constantes (1 valor / imagem):** {', '.join(const_layers) or '—'} "
                 f"→ explicam as 'imagens vazias/sem relevo' (weather = constante por cenário, conforme paper).")

    flagged = [(r[1], r[7]) for r in report_layers if r[7]]
    if flagged:
        lines.append("\n### ⚠️ Anomalias físicas (camadas suspeitas de erro de ordenação/escala)")
        for name, flags in flagged:
            lines.append(f"- **{name}**: " + "; ".join(flags))

    lines.append("\n### Impacto no simulador C (rothermel.h)")
    for k, v in SIM_INDICES.items():
        nm = LAYER_NAMES[v]
        r = report_layers[v]
        suspect = " ⚠️ SUSPEITA" if r[7] else ""
        lines.append(f"- `{k}={v}` → consome camada **{nm}** (min={r[2]:.1f} max={r[3]:.1f}){suspect}")
    lines.append("\n> Se slope/fbfm40 estiverem na escala errada, a física do Rothermel "
                 "e o `get_fuel()` (clamp 1–13) ficam incorretos → **benchmark inválido até corrigir**.")

    (out / "eda_report.md").write_text("\n".join(lines) + "\n")

    # ── histogramas globais (amostra) ──────────────────────────────────────
    sample = np.load(files[0])
    fig, axes = plt.subplots(2, 7, figsize=(22, 6))
    for i, ax in enumerate(axes.flat):
        ax.hist(sample[..., i].ravel(), bins=50, color="#1565C0")
        ax.set_title(f"L{i} {LAYER_NAMES[i]}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Histogramas por camada — {Path(files[0]).stem}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "hist_layers.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── montagens ──────────────────────────────────────────────────────────
    for sid in args.montage_ids:
        match = [f for f in files if Path(f).stem == sid]
        if not match:
            continue
        a = np.load(match[0])
        fig, axes = plt.subplots(2, 7, figsize=(22, 6.5))
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(a[..., i], cmap="viridis")
            ax.set_title(f"L{i} {LAYER_NAMES[i]}", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"14 camadas semânticas — {sid}", fontsize=14)
        plt.tight_layout()
        plt.savefig(out / f"montage_{sid}.png", dpi=110, bbox_inches="tight")
        plt.close()

    print(f"EDA concluída. Relatório: {out/'eda_report.md'}")
    print(f"Anomalias flagadas em: {[r[1] for r in report_layers if r[7]]}")


if __name__ == "__main__":
    main()
