"""
online_rebalancing.py — Entrega 3: rebalanceamento de carga ONLINE durante a simulação

Contribuição do trabalho. Diagnosticamos que prever a carga ESTATICAMENTE (do dado
semântico) não funciona — a propagação é global/anisotrópica (ver
evaluate_partitioning.py: proposta estática < uniform; oráculo >> todos). A carga,
porém, é conhecida DURANTE a simulação (a frente de fogo atual). Então balanceamos
online.

Modelo (dirigido pelo campo TOA real dos labels elmfire):
  - O tempo é discretizado em N_STEPS. No passo k, as células ATIVAS (trabalho
    computacional) são a frente do fogo: TOA ∈ [t_k, t_{k+1}).
  - Com a partição corrente P (N nós), o tempo do passo = max_p (#ativas na partição p)
    — o nó mais lento dita o passo. Tempo total (makespan) = Σ_k max_p load_{k,p}.

Estratégias:
  - static    : particiona 1× por área (uniform) em t=0 e nunca reparticiona.
  - reactive  : reparticiona quando o desbalanceamento medido passa do limiar de 10%
                (Eqs 2-4 do paper), usando a frente ATUAL. (baseline do paper)
  - predictive: NOSSA contribuição — reparticiona usando a frente PREVISTA dos próximos
                W passos (dilatação local da frente ∩ combustível, ponderada por ROS).
                Como antecipa o movimento, a partição vale por mais passos → MENOS
                reparticionamentos para o mesmo (ou melhor) balanço.
  - oracle    : reparticiona TODO passo na carga real (teto de balanço, nº máx. de reparts).

Métricas: tempo total relativo (vs 1 nó) e nº de reparticionamentos por estratégia.

Uso:
    python scripts/online_rebalancing.py --split test --n-parts 8 --out analysis/out
"""
import argparse
import csv as _csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_propagation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "evaluate_partitioning.py")
_ev = importlib.util.module_from_spec(_spec)
exec(compile(open(ROOT / "scripts" / "evaluate_partitioning.py").read()
             .replace('if __name__ == "__main__":', 'if False:'), "ev", "exec"), _ev.__dict__)
kdtree_partition = _ev.kdtree_partition

GRID = 450
IDX_FBFM13 = 3
IDX_WDIR = 12            # direção do vento (graus), camada semântica
THRESH_PCT = 0.10        # limiar de desbalanceamento (Eq.3): 10% da média


def strip_partition(load_map, n_parts, axis):
    """Particiona em N FAIXAS de largura total ao longo de `axis` (0=linhas→faixas
    horizontais; 1=colunas→faixas verticais), balanceadas pela carga. Faixas
    perpendiculares ao avanço da frente fazem a frente ser dividida entre TODOS os
    nós a cada instante → balanço instantâneo mantido enquanto a frente avança."""
    prof = load_map.sum(axis=1 - axis).astype(np.float64)   # perfil de carga no eixo
    total = prof.sum()
    leaves = []
    if total <= 0:                                          # fallback: faixas iguais
        edges = np.linspace(0, GRID, n_parts + 1).astype(int)
    else:
        cs = np.cumsum(prof)
        edges = [0]
        for p in range(1, n_parts):
            target = total * p / n_parts
            edges.append(int(np.searchsorted(cs, target)))
        edges.append(GRID)
        edges = sorted(set(edges))
        while len(edges) < n_parts + 1:                     # garante n_parts faixas
            edges.append(GRID)
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            b = min(a + 1, GRID)
        if axis == 0:
            leaves.append((a, b, 0, GRID))
        else:
            leaves.append((0, GRID, a, b))
    return leaves


def part_loads(leaves, active):
    """Carga (nº de células ativas) por partição."""
    P = np.zeros((GRID + 1, GRID + 1), dtype=np.float64)
    P[1:, 1:] = np.cumsum(np.cumsum(active.astype(np.float64), 0), 1)
    return np.array([P[r1, c1] - P[r0, c1] - P[r1, c0] + P[r0, c0]
                     for (r0, r1, c0, c1) in leaves])


def imbalanced(loads):
    """Eq.2-4: T = max-min ≥ 10% da média ⇒ desbalanceado."""
    if loads.sum() <= 0:
        return False
    return (loads.max() - loads.min()) >= THRESH_PCT * loads.mean()


def simulate(toa, burnable, n_parts, strategy, n_steps=40,
             repart_cost_frac=0.003, lag=1, cooldown=2):
    """Simulação discreta com custo de reparticionamento e lag de detecção.

    - repart_cost_frac: custo de migrar estado entre nós a cada reparticionamento,
      como fração de GRID² (em "células processadas"). Migração é cara em sistemas
      distribuídos reais → poucos reparts é melhor.
    - lag: passos de atraso até o reparticionamento REATIVO surtir efeito (detecção
      + migração). O PREDITIVO age proativamente (lag 0) pois antecipa.

    Retorna (tempo_total, n_reparts, trabalho_total).
    """
    tmax = toa[toa > 0].max()
    edges = np.linspace(0, tmax, n_steps + 1)
    fronts = [((toa >= edges[k]) & (toa < edges[k + 1])).astype(np.float32)
              for k in range(n_steps)]
    repart_cost = repart_cost_frac * GRID * GRID
    yy, xx = np.mgrid[0:GRID, 0:GRID]

    def reachable(active):
        """Região de queima futura prevista = flood-fill do combustível a partir da
        frente atual (só estado atual + mapa de combustível, sem ver o futuro real)."""
        seed = active > 0
        if seed.sum() == 0:
            return burnable
        return binary_propagation(seed, mask=burnable > 0).astype(np.float32)

    def advance_axis(active):
        """Eixo de avanço da frente (centroide relativo à ignição no centro).
        Faixas perpendiculares ao avanço dividem a frente entre todos os nós."""
        m = active > 0
        if m.sum() == 0:
            return 0
        dr = yy[m].mean() - 225.0
        dc = xx[m].mean() - 225.0
        # avanço dominante em coluna → faixas horizontais (axis=0, longas em coluna)
        return 0 if abs(dc) >= abs(dr) else 1

    leaves = kdtree_partition(np.ones((GRID, GRID), np.float32), n_parts)
    cum = np.zeros(len(leaves))
    pending = None            # (novas_folhas, passo_de_aplicação) p/ o lag reativo
    last_repart = -10**9
    n_reparts = 0
    total_time = 0.0
    total_work = 0.0

    for k in range(n_steps):
        if pending is not None and k >= pending[1]:
            leaves = pending[0]; cum = np.zeros(len(leaves)); pending = None

        active = fronts[k]
        loads = part_loads(leaves, active)
        total_time += loads.max()          # barreira: nó mais lento dita o passo
        total_work += active.sum()
        cum += loads
        can_repart = (k - last_repart) >= cooldown     # evita thrashing

        if strategy == "oracle":
            if active.sum() > 0:
                leaves = kdtree_partition(active, n_parts)
                cum = np.zeros(len(leaves))
                total_time += repart_cost; n_reparts += 1; last_repart = k
        elif strategy == "reactive":
            # reage ao desbalanceamento já detectado → particiona na frente ATUAL
            # (míope) e aplica com LAG (detecção + migração)
            if pending is None and can_repart and imbalanced(cum):
                pending = (kdtree_partition(active, n_parts), k + 1 + lag)
                total_time += repart_cost; n_reparts += 1; last_repart = k
        elif strategy == "predictive":
            # PROATIVO: particiona em FAIXAS perpendiculares ao avanço, ponderadas
            # pela região de queima futura prevista. A frente passa a ser dividida
            # entre todos os nós a cada instante → balanço se mantém por muitos
            # passos (poucos reparts) e aplica sem lag.
            if can_repart and imbalanced(cum):
                leaves = strip_partition(reachable(active), n_parts, advance_axis(active))
                cum = np.zeros(len(leaves))
                total_time += repart_cost; n_reparts += 1; last_repart = k
        # static: nunca reparticiona

    return total_time, n_reparts, total_work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(ROOT / "scenarios.csv"))
    ap.add_argument("--labels-dir", default=str(ROOT / "data" / "labels"))
    ap.add_argument("--semantic-dir", default=str(ROOT / "data" / "semantic"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-parts", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--repart-cost", type=float, default=0.003,
                    help="custo de migração por reparticionamento (fração de GRID²); "
                         "0 = mede só qualidade de balanço")
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "analysis" / "out"))
    args = ap.parse_args()

    with open(args.scenarios) as f:
        sids = [r["id"] for r in _csv.DictReader(f) if r["split"] == args.split]

    strategies = ["static", "reactive", "predictive", "oracle"]
    agg = {s: {"reltime": [], "reparts": []} for s in strategies}

    for sid in sids:
        pt  = Path(args.labels_dir)   / args.split / f"{sid}.pt"
        npy = Path(args.semantic_dir) / args.split / f"{sid}.npy"
        if not pt.exists() or not npy.exists():
            continue
        toa = torch.load(pt, weights_only=False)["y"].numpy()
        if (toa > 1).sum() == 0:
            continue
        fbfm = np.load(npy)[..., IDX_FBFM13]
        burnable = ((fbfm >= 1) & (fbfm <= 13)).astype(np.float32)

        for s in strategies:
            tt, nr, work = simulate(toa, burnable, args.n_parts, s, args.n_steps,
                                    repart_cost_frac=args.repart_cost,
                                    cooldown=args.cooldown)
            # tempo relativo vs 1 nó ideal (= trabalho/N): makespan / (work)
            # makespan já é Σ max; standalone 1-nó = Σ active = work (1 passo por vez)
            agg[s]["reltime"].append(tt / max(work, 1))
            agg[s]["reparts"].append(nr)

    print(f"\nRebalanceamento online — {args.split}, N={args.n_parts} partições, "
          f"{args.n_steps} passos")
    print(f"{'estratégia':>12} | {'tempo rel.':>10} | {'#reparts':>9}")
    summary = {}
    for s in strategies:
        rt = float(np.mean(agg[s]["reltime"]))
        nr = float(np.mean(agg[s]["reparts"]))
        summary[s] = {"reltime": rt, "reparts": nr}
        print(f"{s:>12} | {rt:10.3f} | {nr:9.1f}")

    # ganho do preditivo vs reativo
    r, p = summary["reactive"], summary["predictive"]
    print(f"\nPreditivo vs Reativo: tempo {100*(r['reltime']-p['reltime'])/r['reltime']:+.1f}% | "
          f"reparticionamentos {100*(r['reparts']-p['reparts'])/max(r['reparts'],1):+.1f}%")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "online_rebalancing.json", "w") as f:
        json.dump({"n_parts": args.n_parts, "summary": summary}, f, indent=2)

    # gráfico: tempo relativo e nº de reparticionamentos
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cols = {"static": "#999999", "reactive": "#1565C0",
            "predictive": "#2E7D32", "oracle": "#B71C1C"}
    names = {"static": "Estático", "reactive": "Reativo (paper)",
             "predictive": "Preditivo (proposta)", "oracle": "Oráculo"}
    xs = [names[s] for s in strategies]
    ax[0].bar(xs, [summary[s]["reltime"] for s in strategies],
              color=[cols[s] for s in strategies])
    ax[0].set_ylabel("Tempo de simulação relativo"); ax[0].set_title("Makespan (menor = melhor)")
    ax[0].tick_params(axis='x', rotation=20)
    ax[1].bar(xs, [summary[s]["reparts"] for s in strategies],
              color=[cols[s] for s in strategies])
    ax[1].set_ylabel("Nº de reparticionamentos"); ax[1].set_title("Reparticionamentos (menor = menos overhead)")
    ax[1].tick_params(axis='x', rotation=20)
    fig.suptitle(f"Rebalanceamento online — {args.split}, N={args.n_parts}")
    plt.tight_layout()
    plt.savefig(out / "online_rebalancing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSalvo: {out/'online_rebalancing.png'} e .json")


if __name__ == "__main__":
    main()
