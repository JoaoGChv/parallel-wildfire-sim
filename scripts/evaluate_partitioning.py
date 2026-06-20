import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
from predict_load import predict_load_map  # noqa: E402

GRID = 450
IDX_FBFM13 = 3
PART_COUNTS = [2, 4, 8, 16, 32]


# ── k-d tree espelhando kdtree/kdtree.c ─────────────────────────────
def _region_load(P, r0, r1, c0, c1):
    """Soma da carga no retângulo [r0,r1)×[c0,c1) via prefix-sum P (R+1×C+1)."""
    return P[r1, c1] - P[r0, c1] - P[r1, c0] + P[r0, c0]


def _split(P, r0, r1, c0, c1, axis, target_half):
    cum = 0.0
    if axis == 0:
        for r in range(r0, r1 - 1):
            cum += _region_load(P, r, r + 1, c0, c1)
            if cum >= target_half:
                return r + 1
        return (r0 + r1) // 2
    else:
        for c in range(c0, c1 - 1):
            cum += _region_load(P, r0, r1, c, c + 1)
            if cum >= target_half:
                return c + 1
        return (c0 + c1) // 2


def kdtree_partition(load_map, n_parts):
    """Retorna lista de retângulos (r0,r1,c0,c1) — folhas da k-d tree."""
    P = np.zeros((GRID + 1, GRID + 1), dtype=np.float64)
    P[1:, 1:] = np.cumsum(np.cumsum(load_map.astype(np.float64), axis=0), axis=1)
    leaves = []

    def build(n, r0, r1, c0, c1, depth):
        if n <= 1 or r1 - r0 <= 1 or c1 - c0 <= 1:
            leaves.append((r0, r1, c0, c1))
            return
        axis = depth % 2
        if axis == 0 and (r1 - r0) < 2: axis = 1
        if axis == 1 and (c1 - c0) < 2: axis = 0
        total = _region_load(P, r0, r1, c0, c1)
        s = _split(P, r0, r1, c0, c1, axis, total / 2.0)
        half = n // 2
        if axis == 0:
            build(half, r0, s, c0, c1, depth + 1)
            build(n - half, s, r1, c0, c1, depth + 1)
        else:
            build(half, r0, r1, c0, s, depth + 1)
            build(n - half, r0, r1, s, c1, depth + 1)

    build(n_parts, 0, GRID, 0, GRID, 0)
    return leaves


def relative_time(leaves, gt_load):
    """max(carga_real_partição) / carga_real_total (Eq.5)."""
    Pg = np.zeros((GRID + 1, GRID + 1), dtype=np.float64)
    Pg[1:, 1:] = np.cumsum(np.cumsum(gt_load.astype(np.float64), axis=0), axis=1)
    loads = [_region_load(Pg, r0, r1, c0, c1) for (r0, r1, c0, c1) in leaves]
    total = sum(loads)
    if total <= 0:
        return None
    return max(loads) / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(ROOT / "scenarios.csv"))
    ap.add_argument("--semantic-dir", default=str(ROOT / "data" / "semantic"))
    ap.add_argument("--labels-dir", default=str(ROOT / "data" / "labels"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=str(ROOT / "results/models/cnn_size21_best.pt"))
    ap.add_argument("--norm",  default=str(ROOT / "results/models/cnn_size21_norm.pt"))
    ap.add_argument("--atomic-size", type=int, default=21)
    ap.add_argument("--target", default="toa", choices=["toa", "burn"],
                    help="toa=carga=TOA predito; burn=carga=prob. de queimar")
    ap.add_argument("--load", default="toa", choices=["toa", "burned"],
                    help="carga REAL p/ medir: toa=valor TOA; burned=binário queimado")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "out"))
    args = ap.parse_args()

    with open(args.scenarios) as f:
        sids = [r["id"] for r in _csv.DictReader(f) if r["split"] == args.split]

    strategies = ["uniform", "points", "proposed", "oracle"]
    # rel[strategy][n_parts] = lista de tempos relativos por cenário
    rel = {s: {n: [] for n in PART_COUNTS} for s in strategies}
    used = 0

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for sid in sids:
        npy = Path(args.semantic_dir) / args.split / f"{sid}.npy"
        pt  = Path(args.labels_dir)   / args.split / f"{sid}.pt"
        if not npy.exists() or not pt.exists():
            continue
        toa = torch.load(pt, weights_only=False)["y"].numpy()
        if (toa > 1).sum() == 0:
            continue   # cenário degenerado (não queimou) — sem carga para balancear
        used += 1

        # carga REAL p/ medir o balanceamento
        gt_load = (toa > 0).astype(np.float32) if args.load == "burned" else toa

        sem = np.load(npy)
        fbfm = sem[..., IDX_FBFM13]
        burnable = ((fbfm >= 1) & (fbfm <= 13)).astype(np.float32)

        pred = predict_load_map(str(npy), args.model, args.norm,
                                args.atomic_size, dev, target=args.target)
        if args.target == "toa":
            pred = np.clip(pred, 0, 86400)   # remove outliers da regressão

        maps = {
            "uniform":  np.ones((GRID, GRID), dtype=np.float32),
            "points":   burnable,
            "proposed": pred,
            "oracle":   gt_load,             # teto: particiona na carga real
        }
        for s in strategies:
            for n in PART_COUNTS:
                rt = relative_time(kdtree_partition(maps[s], n), gt_load)
                if rt is not None:
                    rel[s][n].append(rt)

    # ── Resultado agregado ────────────────────────────────────────────────
    print(f"\nCenários usados: {used}/{len(sids)} ({args.split}) "
          f"| target={args.target} load={args.load}")
    print(f"{'n_parts':>8} | {'uniform':>9} {'points':>9} {'proposta':>9} {'oráculo':>9} | ganho%")
    summary = {s: [] for s in strategies}
    for n in PART_COUNTS:
        vals = {s: (np.mean(rel[s][n]) if rel[s][n] else float('nan')) for s in strategies}
        for s in strategies:
            summary[s].append(vals[s])
        gain = 100 * (min(vals["uniform"], vals["points"]) - vals["proposed"]) \
               / min(vals["uniform"], vals["points"])
        print(f"{n:8d} | {vals['uniform']:9.3f} {vals['points']:9.3f} "
              f"{vals['proposed']:9.3f} {vals['oracle']:9.3f} | {gain:5.1f}%")

    # ── Figura 12 ─────────────────────────────────────────────────────────
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    styles = {"uniform": ("Uniform-sized LB", "o", "#1565C0"),
              "points":  ("Points-based LB", "s", "#E08000"),
              "proposed":("Proposta (CNN+k-d tree)", "^", "#2E7D32"),
              "oracle":  ("Oráculo (carga real)", "x", "#888888")}
    for s in strategies:
        lbl, mk, col = styles[s]
        plt.plot(PART_COUNTS, summary[s], marker=mk, color=col, label=lbl, linewidth=2)
    plt.xscale("log", base=2); plt.xticks(PART_COUNTS, PART_COUNTS)
    plt.xlabel("Número de partições"); plt.ylabel("Tempo de simulação relativo")
    plt.title(f"Métrica do paper (Fig.12) — {used} cenários de {args.split}\n"
              "menor = melhor balanceamento")
    plt.grid(alpha=0.3); plt.legend()
    fig_path = out / "fig12_relative_time.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nFigura salva: {fig_path}")

    import json
    with open(out / "partitioning_metric.json", "w") as f:
        json.dump({s: {str(n): summary[s][i] for i, n in enumerate(PART_COUNTS)}
                   for s in strategies}, f, indent=2)


if __name__ == "__main__":
    main()
