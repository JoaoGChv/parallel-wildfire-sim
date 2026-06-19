# Dados (não versionados)

Este diretório contém os dados semânticos e os rótulos de carga. Ele **não é
versionado** (está no `.gitignore`) por ser grande (~3,5 GB) e por ser
**regenerável** a partir dos scripts.

## Estrutura esperada
```
data/
├── semantic/{train,test}/Sxxx.npy     # 14 camadas 450×450 por cenário (LANDFIRE)
├── labels/{train,test}/Sxxx.pt        # rótulos de carga (TOA) via ELMFIRE
└── semantic_brazil/BR_*.npy           # cenários do Brasil (fontes globais)
```

## Como regenerar
```bash
# Dados semânticos dos EUA (68 cenários — requer rede / LANDFIRE)
python scripts/preprocess.py --scenarios scenarios.csv --output-dir data/semantic

# Rótulos de carga (requer o binário do ELMFIRE compilado)
python scripts/run_elmfire_batch.py \
    --semantic-dir data/semantic --output-dir data/labels \
    --elmfire-bin <caminho>/elmfire

# Cenários do Brasil (fontes globais: Esri Terrain3D + Sentinel-2 Land Cover)
python scripts/preprocess_brazil.py --out-dir data/semantic_brazil
```

Opcional: `--fuel fbfm40` no `preprocess.py` usa o combustível Scott & Burgan
(40 modelos) traduzido para Anderson 13 via `scripts/fbfm40_crosswalk.py`.
