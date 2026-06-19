"""
visualize.py — Gera visualizações dos dados e resultados da Entrega 1

Gera (em results/visualizations/):
  1. semantic_{sid}.png    — as 14 camadas semânticas de um cenário
  2. elmfire_toa_{sid}.png — mapa de time-of-arrival gerado pelo elmfire
  3. simulator_toa_{sid}.png — propagação do simulador C (Rothermel+Dijkstra)
  4. sweep_mse.png         — gráfico do sweep de atomic sizes

Uso:
    python visualize.py --scenario S001 --all-layers
    python visualize.py --scenario S001 --sweep
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

BASE    = Path(__file__).parent.parent
VIZ_DIR = BASE / "results" / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

LAYER_NAMES = [
    "Elevation (DEM)", "Aspect", "Slope",
    "Fuel Model (FBFM40)", "Canopy Cover", "Canopy Height",
    "Canopy Base Height", "Canopy Bulk Density",
    "Temperature", "Humidity", "Cloud Cover",
    "Precipitation", "Wind Direction", "Wind Speed",
]


def load_scenario_info(sid: str) -> dict:
    with open(BASE / "scenarios.csv") as f:
        for row in csv.DictReader(f):
            if row["id"] == sid:
                return row
    raise ValueError(f"Cenário {sid} não encontrado")


# ── 1. Mapa das 14 camadas semânticas ─────────────────────────────────────────

def plot_semantic(sid: str) -> Path:
    sc    = load_scenario_info(sid)
    split = sc["split"]
    npy   = BASE / "data" / "semantic" / split / f"{sid}.npy"
    arr   = np.load(npy)   # (450, 450, 14)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(
        f"{sid} — {sc['nome_incendio']}  |  {sc['lat_inicio']}°N {sc['lon_inicio']}°W  |  {sc['area_acres']} acres",
        fontsize=13, y=1.01
    )

    cmaps = ["terrain","hsv","YlOrRd","tab20b","Greens","YlGn","Blues","PuBu",
             "RdYlBu_r","Blues","Greys","PuBu","twilight","YlOrRd"]

    for i, ax in enumerate(axes.flat):
        if i < 14:
            layer = arr[:, :, i]
            im = ax.imshow(layer, cmap=cmaps[i], origin="upper", vmin=0, vmax=1)
            ax.set_title(LAYER_NAMES[i], fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    plt.tight_layout()
    out = VIZ_DIR / f"semantic_{sid}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")
    return out


# ── 2. Mapa de time-of-arrival do elmfire ─────────────────────────────────────

def plot_elmfire_toa(sid: str) -> Path | None:
    sc    = load_scenario_info(sid)
    split = sc["split"]
    pt    = BASE / "data" / "labels" / split / f"{sid}.pt"
    if not pt.exists():
        print(f"Label não encontrado: {pt}")
        return None

    data = torch.load(pt, weights_only=True)
    y    = data["y"].numpy()   # (450, 450) em segundos

    burned = y > 0
    if not burned.any():
        print(f"{sid}: nenhuma célula queimada no label")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{sid} — {sc['nome_incendio']}  |  Labels elmfire (time of arrival)",
        fontsize=12
    )

    # TOA em horas
    y_h = np.where(burned, y / 3600, np.nan)
    im0 = axes[0].imshow(y_h, cmap="hot_r", origin="upper")
    axes[0].set_title(f"Time of Arrival (horas)\n{burned.sum()} células queimadas ({100*burned.mean():.2f}%)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="horas desde ignição")

    # Mapa binário queimado/não queimado
    axes[1].imshow(burned.astype(float), cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[1].set_title(f"Área queimada\n(simulação 24h, ignição no centro)")
    # Marcar ignição
    axes[1].plot(225, 225, "b*", markersize=12, label="Ignição")
    axes[1].legend(fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()
    out = VIZ_DIR / f"elmfire_toa_{sid}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")
    return out


# ── 3. Propagação de fogo (Python — Rothermel + Dijkstra) ─────────────────────

def rothermel_ros_py(fbfm_norm: float, slope_pct: float, wind_ms: float) -> float:
    """ROS simplificado em m/s para visualização."""
    # Mapear fbfm normalizado → fuel model 1-13
    if fbfm_norm < 0.001:
        return 0.0
    fm = max(1, min(13, int(fbfm_norm * 12) + 1))
    # Parâmetros simplificados por fuel model (spread index proxy)
    base_ros = [0, 0.02, 0.05, 0.15, 0.12, 0.08, 0.10, 0.08,
                0.03, 0.04, 0.06, 0.05, 0.10, 0.20][fm]
    phi_w = 0.4 * (wind_ms / 10.0)
    phi_s = 0.1 * (slope_pct / 100.0)
    return base_ros * (1 + phi_w + phi_s)


def simulate_python(arr: np.ndarray, non_burn_mask: np.ndarray,
                    ign_r: int, ign_c: int,
                    wind_ms: float, wind_dir_deg: float) -> np.ndarray:
    """Dijkstra + Rothermel em Python sobre o array 450×450×14."""
    import heapq

    GRID = 450
    DR = [-1,-1,-1, 0, 0, 1, 1, 1]
    DC = [-1, 0, 1,-1, 1,-1, 0, 1]
    DIST = [42.426, 30.0, 42.426, 30.0, 30.0, 42.426, 30.0, 42.426]
    ANG  = [315., 0., 45., 270., 90., 225., 180., 135.]

    toa = np.full((GRID, GRID), np.inf)
    toa[ign_r, ign_c] = 0.0
    heap = [(0.0, ign_r, ign_c)]

    while heap:
        t, r, c = heapq.heappop(heap)
        if t > toa[r, c] + 1e-6:
            continue
        fbfm = arr[r, c, 3]   # idx 3 = fbfm40
        slope = arr[r, c, 2] * 200.0   # idx 2 = slope
        ros_max = rothermel_ros_py(fbfm, slope, wind_ms)
        if ros_max < 1e-6:
            continue
        for d in range(8):
            nr, nc = r + DR[d], c + DC[d]
            if nr < 0 or nr >= GRID or nc < 0 or nc >= GRID:
                continue
            if non_burn_mask[nr, nc]:
                continue
            c_eff = 0.5 + 0.5 * np.cos(np.radians(ANG[d] - wind_dir_deg))
            ros_d = max(ros_max * c_eff, 0.001)
            new_toa = t + DIST[d] / ros_d
            if new_toa < toa[nr, nc]:
                toa[nr, nc] = new_toa
                heapq.heappush(heap, (new_toa, nr, nc))

    return np.where(np.isinf(toa), np.nan, toa)


def plot_simulator(sid: str, wind_speed: float = 8.0, wind_dir: float = 270.0) -> Path | None:
    sc    = load_scenario_info(sid)
    split = sc["split"]
    npy   = BASE / "data" / "semantic" / split / f"{sid}.npy"
    sim   = BASE / "simulator" / "simulator"

    arr = np.load(npy)   # (450, 450, 14)

    # Baixar FBFM40 cru (valores reais 91-188) para identificar células não-queimáveis
    # Células com FBFM40 = 91 (urban), 92 (snow), 93 (agri), 98 (water), 99 (barren)
    # No array normalizado, usamos o índice 3 (fbfm40). Precisamos desnormalizar.
    # Como não temos os extremos originais, usamos heurística:
    # valores normalizados < 0.07 ou correspondentes a não-queimáveis são mascarados
    # Estratégia: baixar tile cru do LANDFIRE para este cenário
    try:
        from pyproj import Transformer
        import requests, io, rasterio

        lat, lon = float(sc["lat_inicio"]), float(sc["lon_inicio"])
        t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        cx, cy = t.transform(lon, lat)
        half = (450 * 30) / 2
        bbox_5070 = (cx-half, cy-half, cx+half, cy+half)
        xmin, ymin, xmax, ymax = bbox_5070

        url = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2016/LF2016_FBFM40_CONUS/ImageServer/exportImage"
        r = requests.get(url, params={
            "bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": 5070,
            "size": "450,450", "imageSR": 5070,
            "format": "tiff", "pixelType": "UNKNOWN",
            "noDataInterpretation": "esriNoDataMatchAny", "f": "json",
        }, timeout=60)
        href = r.json().get("href")
        tif_r = requests.get(href, timeout=120)
        with rasterio.open(io.BytesIO(tif_r.content)) as src:
            fbfm40_raw = src.read(1)
        NON_BURNABLE = {0, 91, 92, 93, 98, 99}
        non_burn_mask = np.isin(fbfm40_raw, list(NON_BURNABLE))
        print(f"  Células não-queimáveis: {non_burn_mask.sum()} ({100*non_burn_mask.mean():.1f}%)")
    except Exception as e:
        print(f"  Aviso: não obteve FBFM40 cru ({e}), usando máscara do normalizado")
        # Fallback: células com valor normalizado fbfm40 < 0.05 (códigos ~91-93)
        fbfm40_norm = arr[:, :, 3]
        non_burn_mask = fbfm40_norm < 0.05

    # Encontrar ignição queimável mais próxima do centro
    from collections import deque
    cx, cy = 225, 225
    visited2, queue2, ign_r, ign_c = set(), deque([(cx, cy)]), cx, cy
    while queue2:
        r2, c2 = queue2.popleft()
        if (r2, c2) in visited2: continue
        visited2.add((r2, c2))
        if not non_burn_mask[r2, c2]:
            ign_r, ign_c = r2, c2
            break
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r2+dr, c2+dc
            if 0 <= nr < 450 and 0 <= nc < 450 and (nr,nc) not in visited2:
                queue2.append((nr, nc))
    print(f"  Ignição: ({ign_r},{ign_c}), burnable={not non_burn_mask[ign_r,ign_c]}")

    import time as _time
    t0 = _time.time()
    toa = simulate_python(arr, non_burn_mask, ign_r, ign_c, wind_speed, wind_dir)
    exec_time = _time.time() - t0
    print(f"  Simulação Python: {exec_time:.2f}s")

    burned = ~np.isnan(toa)
    toa_h  = np.where(burned, toa / 3600, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{sid} — {sc['nome_incendio']}  |  Simulador C (Rothermel + Dijkstra)\n"
        f"Vento: {wind_speed} m/s @ {wind_dir}°  |  Tempo execução: {exec_time*1000:.1f} ms",
        fontsize=11
    )

    # TOA em horas
    im0 = axes[0].imshow(toa_h, cmap="inferno_r", origin="upper")
    axes[0].set_title(f"Time of Arrival (horas)\n{burned.sum()} células ({100*burned.mean():.2f}%)")
    axes[0].plot(225, 225, "w*", markersize=12)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="horas")

    # Isócronas (frentes de fogo por hora)
    axes[1].imshow(arr[:, :, 0], cmap="terrain", origin="upper", alpha=0.6)
    levels = [h * 3600 for h in range(1, 25) if h * 3600 <= np.nanmax(toa)]
    if levels:
        cs = axes[1].contour(toa, levels=levels, colors="red", linewidths=0.8, alpha=0.8)
        axes[1].clabel(cs, fmt=lambda v: f"{v/3600:.0f}h", fontsize=7, inline=True)
    axes[1].plot(225, 225, "b*", markersize=12, label="Ignição")
    axes[1].set_title("Isócronas horárias\nsobre DEM (terreno)")
    axes[1].legend(fontsize=9)
    axes[1].axis("off")

    # Fuel model overlay
    fuel = arr[:, :, 3]
    axes[2].imshow(fuel, cmap="tab20", origin="upper", alpha=0.7)
    axes[2].contourf(burned.astype(float), levels=[0.5, 1.5], colors=["red"], alpha=0.4)
    axes[2].plot(225, 225, "w*", markersize=12, label="Ignição")
    axes[2].set_title("Fuel Model + área queimada\n(vermelho = área atingida)")
    axes[2].legend(fontsize=9)
    axes[2].axis("off")

    plt.tight_layout()
    out = VIZ_DIR / f"simulator_{sid}_ws{wind_speed:.0f}_wd{wind_dir:.0f}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")
    return out


# ── 4. Gráfico do sweep de atomic sizes ───────────────────────────────────────

def plot_sweep() -> Path:
    import json
    sweep_path = BASE / "results" / "sweep_results.json"
    meta_dir   = BASE / "results" / "models"

    sizes, best_mse, burned_rmse_h = [], [], []
    for s in [9, 12, 15, 18, 21]:
        meta_path = meta_dir / f"cnn_size{s}_meta.json"
        if not meta_path.exists():
            continue
        d = json.load(open(meta_path))
        sizes.append(s)
        best_mse.append(d["best_test_mse"])
        br = d.get("burned_cell_rmse")
        burned_rmse_h.append(br * 24 if br else None)   # normalizado → horas

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CNN Sweep — MSE por Atomic Data Size\n(Kwon et al. 2022 baseline)", fontsize=13)

    # Best test MSE (overall)
    ax1.plot(sizes, best_mse, "o-", color="steelblue", linewidth=2, markersize=8)
    ax1.set_xlabel("Atomic Size (pixels por lado)")
    ax1.set_ylabel("Best Test MSE (normalizado)")
    ax1.set_title("Overall Test MSE\n(dominado por células não-queimadas = 0)")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax1.grid(True, alpha=0.3)
    for s, v in zip(sizes, best_mse):
        ax1.annotate(f"{v:.4f}", (s, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    # Burned cell RMSE em horas
    burned_valid = [(s, v) for s, v in zip(sizes, burned_rmse_h) if v is not None]
    if burned_valid:
        sx, sy = zip(*burned_valid)
        ax2.plot(sx, sy, "o-", color="firebrick", linewidth=2, markersize=8)
        ax2.set_xlabel("Atomic Size (pixels por lado)")
        ax2.set_ylabel("RMSE nas células queimadas (horas)")
        ax2.set_title("RMSE nas células queimadas\n(métrica real de qualidade da predição)")
        ax2.set_xticks([s for s in sizes if s in sx])
        ax2.set_xticklabels([f"{s}×{s}" for s in sizes if s in sx])
        ax2.grid(True, alpha=0.3)
        for s, v in burned_valid:
            ax2.annotate(f"{v:.2f}h", (s, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    out = VIZ_DIR / "sweep_mse.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",   default="S001")
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--elmfire",    action="store_true")
    p.add_argument("--simulator",  action="store_true")
    p.add_argument("--sweep",      action="store_true")
    p.add_argument("--all",        action="store_true")
    p.add_argument("--wind-speed", type=float, default=8.0)
    p.add_argument("--wind-dir",   type=float, default=270.0)
    args = p.parse_args()

    generated = []
    if args.all or args.all_layers:
        generated.append(plot_semantic(args.scenario))
    if args.all or args.elmfire:
        generated.append(plot_elmfire_toa(args.scenario))
    if args.all or args.simulator:
        generated.append(plot_simulator(args.scenario, args.wind_speed, args.wind_dir))
    if args.all or args.sweep:
        generated.append(plot_sweep())

    if not any([args.all, args.all_layers, args.elmfire, args.simulator, args.sweep]):
        # Default: gerar tudo
        generated.append(plot_semantic(args.scenario))
        generated.append(plot_elmfire_toa(args.scenario))
        generated.append(plot_simulator(args.scenario, args.wind_speed, args.wind_dir))
        generated.append(plot_sweep())

    print(f"\nVisualizações salvas em: {VIZ_DIR}")
    for f in generated:
        if f:
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
