"""
cnn.py — LoadPredictorCNN: predição de carga computacional por célula

Arquitetura (Kwon et al. 2022, Figure 5):
  Input:  (batch, 14, W, W)   W ∈ {9,12,15,18,21}
  3× Conv2d(→BatchNorm→ReLU)
  Flatten → FC(256,ReLU) → FC(1)
  Output: escalar — tempo de chegada do fogo (s)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class LoadPredictorCNN(nn.Module):
    def __init__(self, atomic_size: int = 21, in_channels: int = 14):
        super().__init__()
        assert atomic_size in (9, 12, 15, 18, 21)
        feat_size = atomic_size - 6   # 3 conv sem padding: W - 3×2
        assert feat_size > 0

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_size * feat_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.atomic_size = atomic_size

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for s in (9, 12, 15, 18, 21):
        m = LoadPredictorCNN(atomic_size=s)
        x = torch.randn(4, 14, s, s)
        o = m(x)
        print(f"size={s:2d} | params={m.count_parameters():,} | output={o.shape}")
