import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SEQ  = ROOT / "simulator" / "simulator"
DEFAULT_OMP  = ROOT / "simulator" / "openmp"   / "simulator_omp"
DEFAULT_CUDA = ROOT / "simulator" / "cuda"     / "fire_cuda"


def npy_to_csv(npy_path: Path, out_csv: Path) -> None:
    arr = np.load(npy_path)
    rows, cols, layers = arr.shape
    with open(out_csv, "w", newline="") as f:
        f.write(",".join(f"layer_{i}" for i in range(layers)) + "\n")
        for r in range(rows):
            for c in range(cols):
                f.write(",".join(f"{v:.6f}" for v in arr[r, c]) + "\n")


def run_sim(cmd: list, timeout: int = 300) -> float | None:
    """Executa simulador e extrai tempo médio em segundos."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if "médio" in line.lower() or "average" in line.lower():
                for tok in line.split():
                    try:
                        return float(tok)
                    except ValueError:
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return None


def benchmark_scenario(sid: str, split: str, semantic_dir: Path,
                        seq_bin: Path, omp_bin: Path, cuda_bin: Path,
                        omp_threads: list, runs: int) -> dict:
    npy = semantic_dir / split / f"{sid}.npy"
    if not npy.exists():
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_csv  = tmp / "map.csv"
        out_csv = tmp / "toa.csv"
        npy_to_csv(npy, in_csv)

        common = [
            "--input",        str(in_csv),
            "--output",       str(out_csv),
            "--ignition-row", "225",
            "--ignition-col", "225",
            "--wind-speed",   "5.0",
            "--wind-dir",     "270",
            "--runs",         str(runs),
        ]

        result = {"scenario_id": sid, "split": split}

        # Sequencial
        if seq_bin.exists():
            t = run_sim([str(seq_bin)] + common)
            result["seq_s"]  = t
            result["seq_ms"] = t * 1000 if t else None

        # OpenMP (vários thread counts)
        for nt in omp_threads:
            if omp_bin.exists():
                t = run_sim([str(omp_bin)] + common + ["--threads", str(nt)])
                result[f"omp_{nt}t_s"]  = t
                result[f"omp_{nt}t_ms"] = t * 1000 if t else None

        # CUDA
        if cuda_bin.exists():
            t = run_sim([str(cuda_bin)] + common, timeout=120)
            result["cuda_s"]  = t
            result["cuda_ms"] = t * 1000 if t else None

    return result


def compute_speedups(results: list, omp_threads: list) -> dict:
    speedups = {nt: [] for nt in omp_threads}
    speedups["cuda"] = []

    for r in results:
        seq = r.get("seq_s")
        if not seq or seq == 0:
            continue
        for nt in omp_threads:
            omp = r.get(f"omp_{nt}t_s")
            if omp and omp > 0:
                speedups[nt].append(seq / omp)
        cuda = r.get("cuda_s")
        if cuda and cuda > 0:
            speedups["cuda"].append(seq / cuda)

    return {k: (sum(v)/len(v) if v else None) for k, v in speedups.items()}


def plot_speedup(results: list, omp_threads: list, out_path: Path) -> None:
    speedups = compute_speedups(results, omp_threads)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Speedup: Sequencial vs. OpenMP vs. CUDA\n"
                 "HPC — Simulação de Incêndios Florestais (Entrega 2)", fontsize=13)

    # ── Gráfico 1: Speedup médio por configuração ──────────────────────────
    labels, vals, colors = [], [], []
    labels.append("Seq\n(baseline)")
    vals.append(1.0)
    colors.append("#555555")

    for nt in omp_threads:
        sp = speedups.get(nt)
        if sp:
            labels.append(f"OMP\n{nt}t")
            vals.append(sp)
            colors.append("#1565C0")

    cuda_sp = speedups.get("cuda")
    if cuda_sp:
        labels.append("CUDA\nRTX4090")
        vals.append(cuda_sp)
        colors.append("#B71C1C")

    bars = axes[0].bar(labels, vals, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.1f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Speedup vs. Sequencial")
    axes[0].set_title("Speedup médio (68 cenários)")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0, max(vals) * 1.2 if vals else 2)
    axes[0].grid(axis="y", alpha=0.3)

    # ── Gráfico 2: Tempo médio por configuração ────────────────────────────
    labels2, times2, colors2 = [], [], []
    for r in results:
        if r.get("seq_ms"):
            avg_seq = sum(r2.get("seq_ms", 0) or 0 for r2 in results) / len(results)
            break

    labels2.append("Sequencial")
    times2.append(sum(r.get("seq_ms", 0) or 0 for r in results) / max(len(results), 1))
    colors2.append("#555555")

    for nt in omp_threads:
        t_list = [r.get(f"omp_{nt}t_ms") for r in results if r.get(f"omp_{nt}t_ms")]
        if t_list:
            labels2.append(f"OMP {nt} threads")
            times2.append(sum(t_list) / len(t_list))
            colors2.append("#1565C0")

    t_cuda = [r.get("cuda_ms") for r in results if r.get("cuda_ms")]
    if t_cuda:
        labels2.append("CUDA (RTX 4090)")
        times2.append(sum(t_cuda) / len(t_cuda))
        colors2.append("#B71C1C")

    bars2 = axes[1].bar(labels2, times2, color=colors2, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars2, times2):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}ms", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("Tempo médio (ms)")
    axes[1].set_title("Tempo de execução médio por cenário")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios",    default=str(ROOT / "scenarios.csv"))
    p.add_argument("--semantic-dir", default=str(ROOT / "data" / "semantic"))
    p.add_argument("--seq",          default=str(DEFAULT_SEQ))
    p.add_argument("--omp",          default=str(DEFAULT_OMP))
    p.add_argument("--cuda",         default=str(DEFAULT_CUDA))
    p.add_argument("--omp-threads",  nargs="+", type=int, default=[1,2,4,8,16])
    p.add_argument("--runs",         type=int, default=3)
    p.add_argument("--n-scenarios",  type=int, default=10,
                   help="Número de cenários a usar (default: 10, mais rápido)")
    p.add_argument("--all-scenarios",action="store_true",
                   help="Usar todos os 68 cenários")
    p.add_argument("--output-dir",   default=str(ROOT / "results"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.scenarios) as f:
        scenarios = list(csv.DictReader(f))

    if not args.all_scenarios:
        scenarios = scenarios[:args.n_scenarios]

    print(f"Benchmark: {len(scenarios)} cenários × {args.runs} runs")
    print(f"Seq:  {args.seq}")
    print(f"OMP:  {args.omp}")
    print(f"CUDA: {args.cuda}\n")

    results = []
    for i, sc in enumerate(scenarios, 1):
        sid   = sc["id"]
        split = sc["split"]
        print(f"[{i:2d}/{len(scenarios)}] {sid} ({sc['nome_incendio']})...")
        r = benchmark_scenario(
            sid, split, Path(args.semantic_dir),
            Path(args.seq), Path(args.omp), Path(args.cuda),
            args.omp_threads, args.runs,
        )
        if r:
            results.append(r)
            parts = [f"seq={r.get('seq_ms','?'):.1f}ms" if r.get('seq_ms') else "seq=?"]
            for nt in args.omp_threads:
                t = r.get(f"omp_{nt}t_ms")
                if t: parts.append(f"omp{nt}t={t:.1f}ms")
            if r.get("cuda_ms"):
                parts.append(f"cuda={r['cuda_ms']:.1f}ms")
            print(f"  {' | '.join(parts)}")

    # Salvar CSV
    if results:
        csv_path = out_dir / "benchmark_results.csv"
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResultados: {csv_path}")

        # Speedup summary
        speedups = compute_speedups(results, args.omp_threads)
        print("\nSpeedup médio vs. sequencial:")
        for nt in args.omp_threads:
            sp = speedups.get(nt)
            if sp: print(f"  OMP {nt:2d} threads: {sp:.2f}×")
        if speedups.get("cuda"):
            print(f"  CUDA:           {speedups['cuda']:.2f}×")

        # Gráfico
        plot_speedup(results, args.omp_threads, out_dir / "speedup_chart.png")

        # JSON
        with open(out_dir / "benchmark_summary.json", "w") as f:
            json.dump({"speedups": {str(k): v for k,v in speedups.items()},
                       "n_scenarios": len(results)}, f, indent=2)


if __name__ == "__main__":
    main()
