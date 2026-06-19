"""
train.py — Treina LoadPredictorCNN e registra MSE por atomic_size

Uso:
    python train.py --scenarios ../scenarios.csv --labels-dir ../data/labels \
                    --results-dir ../results --atomic-size 21 --epochs 50
    python train.py ... --sweep-sizes   # treina todos os 5 sizes e compara
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))
from cnn import LoadPredictorCNN
from dataset import build_loaders

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, target="toa"):
    model.eval()
    total, n = 0.0, 0
    # toa: MSE em células queimadas | burn: recall/precision @ prob 0.5
    burned_sq_sum, burned_n = 0.0, 0
    tp = fp = fn = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(X)
        total += criterion(pred, y).item(); n += 1
        if target == "burn":
            phat = (pred > 0)            # logit>0 ⇒ prob>0.5
            yb   = (y > 0.5)
            tp += (phat & yb).sum().item()
            fp += (phat & ~yb).sum().item()
            fn += (~phat & yb).sum().item()
        else:
            mask = y > 0
            if mask.any():
                burned_sq_sum += ((pred[mask] - y[mask]) ** 2).sum().item()
                burned_n      += mask.sum().item()
    if target == "burn":
        recall = tp / max(tp + fn, 1)
        prec   = tp / max(tp + fp, 1)
        secondary = (recall, prec)
    else:
        secondary = burned_sq_sum / max(burned_n, 1)
    return total / max(n, 1), secondary


def train_one(
    scenarios_csv, labels_dir, results_dir,
    atomic_size=21, epochs=50, batch_size=512,
    lr=1e-3, weight_decay=1e-4, patience=10,
    num_workers=4, device="cuda", target="toa",
):
    tag = f"cnn_size{atomic_size}" + ("_burn" if target == "burn" else "")
    log.info(f"\n{'='*50}\n{tag} (target={target})\n{'='*50}")

    train_loader, test_loader = build_loaders(
        scenarios_csv, labels_dir,
        atomic_size=atomic_size, batch_size=batch_size,
        num_workers=num_workers, target=target,
    )
    if len(train_loader.dataset) == 0:
        log.error("Dataset vazio — verifique os arquivos .pt")
        return None

    model     = LoadPredictorCNN(atomic_size=atomic_size).to(device)
    if target == "burn":
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=train_loader.dataset.pos_weight.to(device))
    else:
        criterion = nn.MSELoss()
    opt       = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(opt, patience=5, factor=0.5)

    log.info(f"Parâmetros: {model.count_parameters():,}")
    log.info(f"Train: {len(train_loader.dataset):,} amostras | "
             f"Test: {len(test_loader.dataset):,} amostras")

    best_mse   = float("inf")
    no_improve = 0
    history    = {"train": [], "test": []}

    models_dir = Path(results_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f"{tag}_best.pt"

    # Persistir estatísticas de normalização por canal (do treino). Necessárias
    # para normalizar entradas na inferência (forward CUDA / rebalanceamento E3).
    norm_path = models_dir / f"{tag}_norm.pt"
    torch.save({"mean": train_loader.dataset.norm_mean,
                "std":  train_loader.dataset.norm_std}, norm_path)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, opt, criterion, device)
        te, sec = eval_epoch(model, test_loader, criterion, device, target)
        scheduler.step(te)
        elapsed = time.time() - t0

        history["train"].append(tr)
        history["test"].append(te)
        lr_now = opt.param_groups[0]["lr"]

        if target == "burn":
            rec, prec = sec
            secmsg = f"recall={rec:.3f} prec={prec:.3f}"
        else:
            secmsg = f"burned_mse={sec:.6f}"
        log.info(f"  Época {epoch:3d}/{epochs} | train={tr:.6f} test={te:.6f} "
                 f"{secmsg} lr={lr_now:.1e} {elapsed:.1f}s")

        if te < best_mse:
            best_mse   = te
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stop na época {epoch}")
                break

    # Checkpoint final
    torch.save(model.state_dict(), models_dir / f"{tag}_final.pt")

    _, sec = eval_epoch(model, test_loader, criterion, device, target)
    meta = {
        "atomic_size":       atomic_size,
        "target":            target,
        "epochs_trained":    epoch,
        "best_test_loss":    best_mse,
        "last_test_loss":    history["test"][-1],
        "n_params":          model.count_parameters(),
        "train_samples":     len(train_loader.dataset),
        "test_samples":      len(test_loader.dataset),
        "history":           history,
    }
    if target == "burn":
        meta["recall"], meta["precision"] = sec
    else:
        meta["burned_cell_mse"]  = sec
        meta["burned_cell_rmse"] = sec ** 0.5
    with open(models_dir / f"{tag}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"  Best test loss: {best_mse:.4f}")
    return meta


def sweep(args):
    sizes = [9, 12, 15, 18, 21]
    table = {}
    for size in sizes:
        meta = train_one(
            args.scenarios, args.labels_dir, args.results_dir,
            atomic_size=size, epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, num_workers=args.workers, device=args.device,
            target=args.target,
        )
        if meta:
            table[size] = meta["best_test_loss"]

    print("\n" + "="*40)
    print(f"Sweep — best test loss por atomic_size (target={args.target}):")
    for sz, v in sorted(table.items()):
        print(f"  {sz:2d}×{sz:2d}: {v:.4f}")

    out = Path(args.results_dir) / "sweep_results.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Salvo em: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios",    default="../scenarios.csv")
    p.add_argument("--labels-dir",   default="../data/labels")
    p.add_argument("--results-dir",  default="../results")
    p.add_argument("--atomic-size",  type=int, default=21, choices=[9,12,15,18,21])
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=1024)
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sweep-sizes",  action="store_true")
    p.add_argument("--target",       default="toa", choices=["toa", "burn"],
                   help="toa = regressão do TOA; burn = classificação queima/não (prob)")
    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Device: {args.device}")

    if args.sweep_sizes:
        sweep(args)
    else:
        train_one(
            args.scenarios, args.labels_dir, args.results_dir,
            atomic_size=args.atomic_size, epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience,
            num_workers=args.workers, device=args.device, target=args.target,
        )


if __name__ == "__main__":
    main()
