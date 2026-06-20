import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))
from cnn import LoadPredictorCNN

GRID = 450
MAX_TOA_S = 86400.0   # mesma normalização de y usada em dataset.py


@torch.no_grad()
def predict_load_map(npy_path, model_path, norm_path,
                     atomic_size=21, device="cuda", row_batch=8, target="toa"):
    a = np.load(npy_path)                                   # (450,450,14)
    X = torch.tensor(np.transpose(a, (2, 0, 1)), dtype=torch.float32)  # (14,450,450)

    norm = torch.load(norm_path, map_location="cpu")
    mean, std = norm["mean"].float(), norm["std"].float()   # (1,14,1,1)
    X = (X.unsqueeze(0) - mean) / std                       # (1,14,450,450)

    half = atomic_size // 2
    Xp = F.pad(X, (half, half, half, half), mode="reflect")[0]  # (14,450+2h,450+2h)
    Xp = Xp.to(device)

    model = LoadPredictorCNN(atomic_size=atomic_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    load = np.zeros((GRID, GRID), dtype=np.float32)
    for r0 in range(0, GRID, row_batch):
        r1 = min(r0 + row_batch, GRID)
        # janelas de todas as colunas para as linhas [r0,r1): (nrows*GRID, 14, A, A)
        block = Xp[:, r0:r1 + 2 * half, :]                  # (14, (r1-r0)+2h, W)
        wins = block.unfold(1, atomic_size, 1).unfold(2, atomic_size, 1)
        # (14, r1-r0, GRID, A, A) → (rows*GRID, 14, A, A)
        nrows = r1 - r0
        wins = wins.permute(1, 2, 0, 3, 4).reshape(nrows * GRID, 14, atomic_size, atomic_size)
        out = model(wins.contiguous())
        if target == "burn":
            pred = torch.sigmoid(out)                        # prob. de queimar [0,1]
        else:
            pred = (out.clamp_min(0.0) * MAX_TOA_S)          # TOA em segundos ≥0
        load[r0:r1] = pred.reshape(nrows, GRID).cpu().numpy()
    return load


def save_csv(load, path):
    rows, cols = load.shape
    with open(path, "w") as f:
        f.write("row,col,load\n")
        for r in range(rows):
            for c in range(cols):
                f.write(f"{r},{c},{load[r, c]:.4f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True)
    p.add_argument("--model", default="results/models/cnn_size21_best.pt")
    p.add_argument("--norm",  default="results/models/cnn_size21_norm.pt")
    p.add_argument("--atomic-size", type=int, default=21)
    p.add_argument("--target", default="toa", choices=["toa", "burn"])
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    load = predict_load_map(args.npy, args.model, args.norm,
                            args.atomic_size, args.device, target=args.target)
    save_csv(load, args.out)
    nz = (load > 1.0).mean() * 100
    print(f"Mapa de carga predito: {args.out}")
    print(f"  load min/max/mean = {load.min():.1f}/{load.max():.1f}/{load.mean():.1f} s")
    print(f"  células com carga>1s: {nz:.1f}%")


if __name__ == "__main__":
    main()
