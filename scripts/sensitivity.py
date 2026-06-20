import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "simulator" / "simulator"
GRID = 450
ROS_SCALE = 0.01          # mesma escala realista do dashboard (~horas)
ANISO = 2.5
T_FIXED = 2 * 3600.0      # mede a área alcançada em 2 h


def make_csv(npy_path):
    a = np.load(npy_path)
    tmp = Path(tempfile.gettempdir()) / f"sens_{Path(npy_path).stem}.csv"
    if not tmp.exists():
        with open(tmp, "w") as f:
            f.write(",".join(f"layer_{i}" for i in range(a.shape[2])) + "\n")
            for r in range(GRID):
                for c in range(GRID):
                    f.write(",".join(f"{v:.5f}" for v in a[r, c]) + "\n")
    return str(tmp)


def run(csv_path, ws, wd, m1, aniso=ANISO):
    out = Path(tempfile.gettempdir()) / "sens_toa.csv"
    subprocess.run([str(SIM), "--input", csv_path, "--output", str(out),
                    "--ignition-row", "225", "--ignition-col", "225",
                    "--wind-speed", str(ws), "--wind-dir", str(wd),
                    "--moisture-1h", str(m1), "--ros-scale", str(ROS_SCALE),
                    "--wind-aniso", str(aniso), "--runs", "1"],
                   capture_output=True, timeout=60)
    toa = np.full((GRID, GRID), np.nan, np.float32)
    if out.exists():
        for line in open(out).readlines()[1:]:
            r, c, t = line.split(","); toa[int(r), int(c)] = float(t)
    return toa


def reached_at(toa, T):
    """% da grade alcançada pelo fogo até o tempo T."""
    return 100.0 * (np.nan_to_num(toa, nan=1e30) <= T).mean()


def downwind_shift(toa, T, wd):
    """Deslocamento do centroide do fogo a favor do vento (em células), no tempo T.
    0 ≈ isotrópico/sem vento; positivo grande = fogo empurrado downwind.
    (métrica robusta: a frente é one-sided downwind mas two-sided perpendicular,
    então razão de std confunde — o deslocamento do centroide não.)"""
    m = (~np.isnan(toa)) & (toa <= T)
    if m.sum() < 10:
        return 0.0
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    dy = (yy[m] - 225.0).mean(); dx = (xx[m] - 225.0).mean()
    th = np.radians(wd)
    ux, uy = np.sin(th), -np.cos(th)          # 0°=N(up), 90°=L(+col)
    return float(dx * ux + dy * uy)           # projeção do centroide no vetor do vento


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S002")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=str(ROOT / "analysis" / "out"))
    args = ap.parse_args()

    npy = ROOT / "data" / "semantic" / args.split / f"{args.scenario}.npy"
    csv_path = make_csv(str(npy))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    winds = [0, 3, 6, 9, 12, 15, 18]
    moist = [0.03, 0.06, 0.09, 0.12, 0.18, 0.25]
    results = {"T_fixed_h": T_FIXED / 3600}

    # ── varredura de vento (umidade fixa) ───────────────────────────────────
    m_fix = 0.08
    burn_vs_wind = [reached_at(run(csv_path, ws, 90, m_fix), T_FIXED) for ws in winds]
    # ── varredura de umidade (vento fixo) ───────────────────────────────────
    w_fix = 10
    burn_vs_moist = [reached_at(run(csv_path, w_fix, 90, m), T_FIXED) for m in moist]
    # ── heatmap vento × umidade ─────────────────────────────────────────────
    heat = np.array([[reached_at(run(csv_path, ws, 90, m), T_FIXED)
                      for ws in winds] for m in moist])
    # ── anisotropia vs vento (deslocamento downwind do centroide) ───────────
    elong_vs_wind = [downwind_shift(run(csv_path, ws, 90, m_fix), T_FIXED, 90)
                     for ws in winds]

    results.update({"winds": winds, "burn_vs_wind": burn_vs_wind,
                    "moist": moist, "burn_vs_moist": burn_vs_moist,
                    "downwind_shift_vs_wind": elong_vs_wind})
    json.dump(results, open(out / "sensitivity.json", "w"), indent=2)

    # ── Figura 1: linhas + heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(winds, burn_vs_wind, "o-", color="#1565C0", lw=2)
    ax[0].set_xlabel("Velocidade do vento (m/s)")
    ax[0].set_ylabel(f"Área alcançada em {T_FIXED/3600:.0f} h (%)")
    ax[0].set_title("Sensibilidade ao vento"); ax[0].grid(alpha=0.3)

    ax[1].plot(moist, burn_vs_moist, "s-", color="#B71C1C", lw=2)
    ax[1].set_xlabel("Umidade do combustível 1h")
    ax[1].set_ylabel(f"Área alcançada em {T_FIXED/3600:.0f} h (%)")
    ax[1].set_title("Sensibilidade à umidade (mais seco → mais)"); ax[1].grid(alpha=0.3)

    im = ax[2].imshow(heat, origin="lower", aspect="auto", cmap="inferno",
                      extent=[winds[0], winds[-1], moist[0], moist[-1]])
    ax[2].set_xlabel("Vento (m/s)"); ax[2].set_ylabel("Umidade 1h")
    ax[2].set_title(f"Área alcançada em {T_FIXED/3600:.0f} h (%)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(f"Sensibilidade da propagação — cenário {args.scenario}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "sensitivity_wind_moisture.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── Figura 2: deslocamento downwind ─────────────────────────────────────
    plt.figure(figsize=(6, 4.5))
    plt.plot(winds, [v * 30 / 1000 for v in elong_vs_wind], "^-", color="#2E7D32", lw=2)
    plt.axhline(0.0, color="gray", ls="--", lw=0.8, label="sem viés (isotrópico)")
    plt.xlabel("Velocidade do vento (m/s)")
    plt.ylabel("Deslocamento do fogo a favor do vento (km)")
    plt.title(f"Anisotropia: empurrão downwind vs vento — {args.scenario}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / "sensitivity_anisotropy.png", dpi=140, bbox_inches="tight")
    plt.close()

    print(f"Cenário {args.scenario} | área@{T_FIXED/3600:.0f}h")
    print("vento(m/s):", dict(zip(winds, [round(v, 1) for v in burn_vs_wind])))
    print("umidade   :", dict(zip(moist, [round(v, 1) for v in burn_vs_moist])))
    print("desloc.downwind(células):", dict(zip(winds, [round(v, 1) for v in elong_vs_wind])))
    print(f"Figuras em {out}/sensitivity_*.png")


if __name__ == "__main__":
    main()
