import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def npy_to_csv(npy_path: Path, out_csv: Path) -> None:
    arr = np.load(npy_path)   # (450, 450, 14)
    rows, cols, layers = arr.shape
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"layer_{i}" for i in range(layers)])
        for r in range(rows):
            for c in range(cols):
                writer.writerow(arr[r, c].tolist())


def run_simulator(sim_bin: Path, input_csv: Path, output_csv: Path,
                  runs: int = 3) -> float:
    cmd = [
        str(sim_bin),
        "--input",        str(input_csv),
        "--output",       str(output_csv),
        "--ignition-row", "225",
        "--ignition-col", "225",
        "--wind-speed",   "5.0",
        "--wind-dir",     "270",
        "--runs",         str(runs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-300:])
    for line in result.stdout.splitlines():
        if "Tempo médio" in line or "average" in line.lower():
            for tok in line.split():
                try:
                    return float(tok)
                except ValueError:
                    pass
    raise RuntimeError(f"Tempo não encontrado:\n{result.stdout[-300:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios",    default="../scenarios.csv")
    parser.add_argument("--semantic-dir", default="../data/semantic")
    parser.add_argument("--simulator",    default="../simulator/simulator")
    parser.add_argument("--output",       default="../results/sequential_times.csv")
    parser.add_argument("--runs",         type=int, default=3)
    args = parser.parse_args()

    sim_bin = Path(args.simulator)
    if not sim_bin.exists():
        print(f"ERRO: {sim_bin} não encontrado. Execute: cd simulator && make")
        sys.exit(1)

    with open(args.scenarios) as f:
        scenarios = list(csv.DictReader(f))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    failed  = []

    print(f"Benchmarking {len(scenarios)} cenários ({args.runs} runs cada)...\n")
    for i, sc in enumerate(scenarios, 1):
        sid   = sc["id"]
        split = sc["split"]
        npy   = Path(args.semantic_dir) / split / f"{sid}.npy"

        if not npy.exists():
            print(f"[{i:2d}/{len(scenarios)}] {sid} — PULADO (sem .npy)")
            failed.append((sid, "sem .npy"))
            continue

        print(f"[{i:2d}/{len(scenarios)}] {sid} ({sc['nome_incendio']})...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            in_csv  = Path(tmp) / f"{sid}.csv"
            out_csv = Path(tmp) / f"{sid}_toa.csv"
            try:
                npy_to_csv(npy, in_csv)
                avg_s = run_simulator(sim_bin, in_csv, out_csv, args.runs)
                print(f"{avg_s:.4f}s")
                results.append({
                    "scenario_id":   sid,
                    "nome_incendio": sc["nome_incendio"],
                    "split":         split,
                    "area_acres":    sc["area_acres"],
                    "time_seconds":  avg_s,
                    "runs":          args.runs,
                })
            except Exception as e:
                print(f"FALHA: {e}")
                failed.append((sid, str(e)[:80]))

    if results:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        times = [r["time_seconds"] for r in results]
        print(f"\n{'='*50}")
        print(f"Resultados salvos: {out_path}")
        print(f"Cenários OK: {len(results)}/{len(scenarios)}")
        print(f"Tempo médio: {sum(times)/len(times):.4f}s")
        print(f"Tempo mín:   {min(times):.4f}s  ({min(results, key=lambda r: r['time_seconds'])['nome_incendio']})")
        print(f"Tempo máx:   {max(times):.4f}s  ({max(results, key=lambda r: r['time_seconds'])['nome_incendio']})")
    if failed:
        print(f"\nFalhas ({len(failed)}): {[s for s,_ in failed]}")


if __name__ == "__main__":
    main()
