"""
Rodar (no servidor):
    streamlit run app/fire_dashboard.py --server.port 8501
Acessar via navegador (port-forward): http://localhost:8501
"""
import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SIM_BIN = ROOT / "simulator" / "simulator"
SEMANTIC = ROOT / "data" / "semantic"
SCENARIOS = ROOT / "scenarios.csv"
GRID = 450
IDX_FBFM13 = 3
IDX_ELEV = 0

st.set_page_config(page_title="Simulador de Incêndio", layout="wide")


@st.cache_data(show_spinner=False)
def list_scenarios():
    rows = []
    for r in csv.DictReader(open(SCENARIOS)):
        npy = SEMANTIC / r["split"] / f"{r['id']}.npy"
        if npy.exists():
            rows.append((r["id"], r["nome_incendio"], str(npy)))
    return rows


@st.cache_data(show_spinner=False)
def load_semantic(npy_path):
    return np.load(npy_path)


@st.cache_data(show_spinner=False)
def scenario_csv(npy_path):
    """Converte o .npy para o CSV que o simulador C lê (cacheado por cenário)."""
    a = np.load(npy_path)
    rows, cols, layers = a.shape
    tmp = Path(tempfile.gettempdir()) / f"firedash_{Path(npy_path).stem}.csv"
    if not tmp.exists():
        with open(tmp, "w") as f:
            f.write(",".join(f"layer_{i}" for i in range(layers)) + "\n")
            for r in range(rows):
                for c in range(cols):
                    f.write(",".join(f"{v:.5f}" for v in a[r, c]) + "\n")
    return str(tmp)


@st.cache_data(show_spinner=False)
def run_simulation(npy_path, ws, wd, m1, ign_r, ign_c, aniso):
    """Roda o simulador C e devolve o campo TOA (450×450, NaN = não queimou)."""
    in_csv = scenario_csv(npy_path)
    out_csv = Path(tempfile.gettempdir()) / f"firedash_toa_{ign_r}_{ign_c}.csv"
    cmd = [str(SIM_BIN), "--input", in_csv, "--output", str(out_csv),
           "--ignition-row", str(ign_r), "--ignition-col", str(ign_c),
           "--wind-speed", str(ws), "--wind-dir", str(wd),
           "--moisture-1h", str(m1), "--ros-scale", "0.01",
           "--wind-aniso", str(aniso), "--runs", "1"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    toa = np.full((GRID, GRID), np.nan, dtype=np.float32)
    if out_csv.exists():
        with open(out_csv) as f:
            next(f)
            for line in f:
                r, c, t = line.split(",")
                toa[int(r), int(c)] = float(t)
    return toa, res.stdout


# ── Sidebar: controles ────────────────────────────────────────────────────────
st.sidebar.title("🔥 Controles")
scen = list_scenarios()
labels = [f"{sid} — {nome}" for sid, nome, _ in scen]
idx = st.sidebar.selectbox("Cenário", range(len(scen)), format_func=lambda i: labels[i])
sid, nome, npy_path = scen[idx]

st.sidebar.subheader("Vento")
ws = st.sidebar.slider("Velocidade (m/s)", 0.0, 20.0, 12.0, 0.5)
wd = st.sidebar.slider("Direção (graus, p/ onde sopra: 0=N, 90=L, 180=S, 270=O)",
                       0, 359, 90, 5)
aniso = st.sidebar.slider("Intensidade do vento (anisotropia)", 0.0, 4.0, 2.5, 0.1,
                          help="Maior = vento domina mais a direção da propagação")

st.sidebar.subheader("Combustível")
m1 = st.sidebar.slider("Umidade do combustível 1h", 0.02, 0.30, 0.05, 0.01,
                       help="Menor = mais seco = espalha mais")

st.sidebar.subheader("Ignição")
ign_r = st.sidebar.slider("Linha", 0, GRID - 1, 225, 5)
ign_c = st.sidebar.slider("Coluna", 0, GRID - 1, 225, 5)

# ── Dados do cenário ──────────────────────────────────────────────────────────
sem = load_semantic(npy_path)
fbfm = sem[..., IDX_FBFM13]
burnable = (fbfm >= 1) & (fbfm <= 13)
ign_burnable = bool(burnable[ign_r, ign_c])

st.title(f"Simulador interativo de incêndio — {sid}")
st.caption(f"{nome} · simulação real (Rothermel + frente de onda, em C) · "
           f"grade {GRID}×{GRID} (30 m/célula)")
if not ign_burnable:
    st.warning(f"A célula de ignição ({ign_r},{ign_c}) é **não-queimável** "
               f"(código {int(fbfm[ign_r,ign_c])}) — o fogo não vai se espalhar. "
               f"Escolha uma célula em área de combustível (mapa à esquerda).")

# ── Rodar simulação ───────────────────────────────────────────────────────────
toa, _ = run_simulation(npy_path, ws, wd, m1, ign_r, ign_c, aniso)
burned_mask = ~np.isnan(toa)
burned_pct = 100.0 * burned_mask.mean()
tmax = np.nanmax(toa) if burned_mask.any() else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Área queimada", f"{burned_pct:.1f}%")
c2.metric("Tempo máx. de chegada", f"{tmax/3600:.1f} h")
c3.metric("Células atingidas", f"{int(burned_mask.sum()):,}")

# ── Animação temporal ─────────────────────────────────────────────────────────
st.subheader("Propagação no tempo")
if burned_mask.any():
    t_h = st.slider("Tempo decorrido (horas)", 0.0, float(tmax / 3600),
                    float(tmax / 3600), 0.25)
    t_s = t_h * 3600
else:
    t_s = 0.0

col_map, col_fire = st.columns(2)

with col_map:
    st.markdown("**Combustível (verde) e não-queimável (cinza)**")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.where(burnable, fbfm, np.nan), cmap="YlGn", vmin=1, vmax=13)
    ax.imshow(np.where(~burnable, 1, np.nan), cmap=ListedColormap(["#777777"]))
    ax.plot(ign_c, ign_r, "r*", markersize=16, markeredgecolor="white")
    ax.set_title("Mapa de combustível + ignição (★)"); ax.axis("off")
    st.pyplot(fig); plt.close(fig)

with col_fire:
    st.markdown("**Fogo até o instante selecionado**")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.where(burnable, 0.3, 0.0), cmap="Greens", vmin=0, vmax=1)
    reached = burned_mask & (toa <= t_s)
    overlay = np.where(reached, toa / max(tmax, 1), np.nan)
    im = ax.imshow(overlay, cmap="inferno", vmin=0, vmax=1)
    # frente atual (chegou nos últimos ~5% do tempo)
    front = reached & (toa >= t_s - 0.05 * tmax)
    ax.imshow(np.where(front, 1, np.nan), cmap=ListedColormap(["#00e5ff"]))
    ax.plot(ign_c, ign_r, "w*", markersize=12)
    ax.set_title(f"t = {t_s/3600:.1f} h  ·  frente em ciano"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="chegada (norm.)")
    st.pyplot(fig); plt.close(fig)

st.info("Mexa nos sliders (vento, umidade, ignição) e observe a propagação mudar. "
        "Direção do vento controla a anisotropia; umidade menor (mais seco) espalha mais.")
