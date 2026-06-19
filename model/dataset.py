"""
dataset.py — Dataset PyTorch otimizado para LoadPredictorCNN

Pré-carrega todos os 68 tensores em RAM (~826 MB) para eliminar I/O
durante o treino. Cada amostra é uma janela atomic_size×atomic_size
centrada num pixel, com label = TOA normalizado em [0,1].

Labels normalizados: y / 86400.0  (TOA máximo = 24h de simulação)
"""
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)
GRID_SIZE  = 450
MAX_TOA_S  = 86400.0   # 24h em segundos


class WildfireDataset(Dataset):
    """
    Carrega todos os .pt em RAM de uma vez e serve janelas via indexação.
    ~826 MB total (68 cenários × 14 camadas × 450 × 450 float32).
    """
    def __init__(
        self,
        scenarios_csv: str,
        labels_dir: str,
        split: str = "train",
        atomic_size: int = 21,
        norm_mean: torch.Tensor = None,
        norm_std: torch.Tensor = None,
        target: str = "toa",
    ):
        assert split in ("train", "test")
        assert target in ("toa", "burn")
        self.half        = atomic_size // 2
        self.atomic_size = atomic_size
        self.target      = target
        labels_dir       = Path(labels_dir)

        with open(scenarios_csv) as f:
            scenarios = [r for r in csv.DictReader(f) if r["split"] == split]

        # Pré-carregar tudo em memória
        X_list, Y_list = [], []
        loaded = 0
        for sc in scenarios:
            pt = labels_dir / split / f"{sc['id']}.pt"
            if not pt.exists():
                log.warning(f"Faltando: {pt}")
                continue
            data = torch.load(pt, weights_only=True)
            X_list.append(data["X"])          # (14, 450, 450)
            if target == "burn":
                Y_list.append((data["y"] > 0).float())   # 1 = queimou, 0 = não
            else:
                Y_list.append(data["y"] / MAX_TOA_S)     # TOA normalizado [0,1]
            loaded += 1

        if loaded == 0:
            raise RuntimeError(f"Nenhum .pt encontrado em {labels_dir}/{split}")

        # Empilhar: (N_scenarios, 14, 450, 450) e (N_scenarios, 450, 450)
        self.X = torch.stack(X_list)   # float32
        self.Y = torch.stack(Y_list)   # float32

        # Normalização por canal (z-score). X é armazenado em valores físicos
        # BRUTOS com escalas muito diferentes (elevação ~4000, slope 0-400,
        # fbfm13 1-99, clima em unidade nativa). Sem normalizar, a 1ª conv vê
        # canais desbalanceados. Estatísticas SEMPRE do treino (passadas ao
        # teste) para não vazar informação.
        if norm_mean is None or norm_std is None:
            self.norm_mean = self.X.mean(dim=(0, 2, 3), keepdim=True)   # (1,14,1,1)
            self.norm_std  = self.X.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
        else:
            self.norm_mean, self.norm_std = norm_mean, norm_std
        self.X = (self.X - self.norm_mean) / self.norm_std

        # Padding reflexivo para bordas (aplica uma vez, não por sample)
        h = self.half
        self.X_pad = F.pad(self.X, (h, h, h, h), mode="reflect")
        # shape: (N, 14, 450+2h, 450+2h)

        # pos_weight p/ BCE no alvo binário (classes muito desbalanceadas:
        # a maioria das células não queima). = #neg / #pos no treino.
        if target == "burn":
            pos = self.Y.sum().item()
            neg = self.Y.numel() - pos
            self.pos_weight = torch.tensor(neg / max(pos, 1.0))
            self.pos_frac   = pos / self.Y.numel()
        else:
            self.pos_weight = None
            self.pos_frac   = None

        self.n_scenarios = loaded
        self.n_pixels    = GRID_SIZE * GRID_SIZE
        extra = f" | queimadas={self.pos_frac*100:.2f}% pos_weight={self.pos_weight:.0f}" \
                if target == "burn" else ""
        log.info(f"[{split}] {loaded} cenários | "
                 f"{loaded * self.n_pixels:,} amostras | "
                 f"RAM: X={self.X.nbytes/1e6:.0f}MB Y={self.Y.nbytes/1e6:.0f}MB{extra}")

    def __len__(self) -> int:
        return self.n_scenarios * self.n_pixels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sc  = idx // self.n_pixels
        pix = idx  % self.n_pixels
        r   = pix  // GRID_SIZE
        c   = pix  %  GRID_SIZE

        window = self.X_pad[sc, :, r:r + self.atomic_size, c:c + self.atomic_size]
        label  = self.Y[sc, r, c]
        return window, label


def build_loaders(
    scenarios_csv: str,
    labels_dir: str,
    atomic_size: int = 21,
    batch_size: int = 4096,
    num_workers: int = 0,     # 0 = main process (dados já em RAM)
    target: str = "toa",
) -> tuple[DataLoader, DataLoader]:

    train_ds = WildfireDataset(scenarios_csv, labels_dir, "train", atomic_size,
                               target=target)
    # Teste usa as estatísticas de normalização do TREINO (sem data leakage)
    test_ds  = WildfireDataset(scenarios_csv, labels_dir, "test",  atomic_size,
                               norm_mean=train_ds.norm_mean,
                               norm_std=train_ds.norm_std, target=target)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader
