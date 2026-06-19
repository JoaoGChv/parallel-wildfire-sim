# Balanceamento de Carga Guiado por Aprendizado para Simulação de Incêndios Florestais

Reprodução, otimização e **análise crítica** do framework de computação distribuída
de **Kwon, Yun & Kim** (*Sensors*, 2022), que usa uma CNN para prever a carga
computacional de uma simulação de incêndio e uma *k-d tree* para particioná-la de
forma balanceada entre nós de computação.

> **Síntese honesta.** O *método* de balanceamento por carga tem teto altíssimo
> (um oráculo de carga real é **35–57% melhor** que o particionamento uniforme).
> O difícil — e nosso principal achado — é **prever a carga**: a propagação do
> fogo é global, anisotrópica e dependente do caminho, então a predição estática a
> partir do dado semântico **não funciona**. Delimitamos *quando* a abordagem paga.

## Principais resultados

- **Correção de dados:** a aquisição original devolvia a imagem renderizada
  (*colormap*) em vez do raster bruto, destruindo os códigos de combustível.
  Corrigido e revalidado (0 anomalias nas 14 camadas).
- **Otimização paralela:** **CUDA 11×** (relaxação iterativa em *shared memory*) e
  **OpenMP 7×** (escrita *lock-free*, sem `critical` global) sobre o *baseline*
  sequencial.
- **Achado central:** a predição estática de carga pela CNN fica *pior* que o
  uniforme; o oráculo mostra o teto do método. Caracterizamos por que a predição
  estática falha.
- **Rebalanceamento online:** o reativo atinge qualidade-oráculo com ¼ dos
  reparticionamentos; o ganho líquido depende do custo de migração × duração da
  simulação.
- **Extras:** simulador interativo (Streamlit), análise de sensibilidade a
  clima/vento, generalização do *pipeline* ao **Brasil** e suporte a FBFM40.

Detalhes completos em [`RESULTADOS.md`](RESULTADOS.md) e no artigo em
[`artigo/`](artigo/).

## Estrutura do repositório

```
scripts/          Pré-processamento e avaliação (preprocess, elmfire, FBFM40,
                  evaluate_partitioning, online_rebalancing, sensitivity, benchmark)
model/            CNN preditora de carga (cnn, dataset, train, predict_load)
simulator/        Simulador de Rothermel em C
  ├── (raiz)        versão sequencial
  ├── openmp/       versão OpenMP
  └── cuda/         versão CUDA
kdtree/           Particionador k-d tree balanceado por carga (C)
analysis/         EDA, figuras e relatório de análise
app/              Simulador interativo (dashboard Streamlit)
artigo/           Artigo no padrão SBC (LaTeX, pronto para Overleaf)
data/             Dados semânticos e rótulos (NÃO versionado — ver data/README.md)
results/          Modelos treinados e saídas (NÃO versionado)
RESULTADOS.md     Resultados consolidados
```

## Como executar

```bash
pip install -r requirements.txt

# 1. Dados (ver data/README.md — requer rede; regenera os 68 cenários)
python scripts/preprocess.py --scenarios scenarios.csv --output-dir data/semantic
python scripts/run_elmfire_batch.py --semantic-dir data/semantic --output-dir data/labels

# 2. Treinar a CNN
python model/train.py --sweep-sizes

# 3. Compilar e rodar os simuladores
make -C simulator && make -C simulator/openmp && make -C simulator/cuda && make -C kdtree

# 4. Avaliação (métrica do paper, rebalanceamento online, sensibilidade)
python scripts/evaluate_partitioning.py --split test
python scripts/online_rebalancing.py --split test --n-parts 8

# 5. Simulador interativo
streamlit run app/fire_dashboard.py --server.port 8501
```

## Dependências de sistema
- Compilador C com OpenMP (`gcc`), `nvcc` (CUDA) para a versão GPU.
- [ELMFIRE](https://elmfire.io/) (externo) para gerar os rótulos de carga.

## Referência
Kwon, J.-W.; Yun, S.-J.; Kim, W.-T. *A Semantic Data-Based Distributed Computing
Framework to Accelerate Digital Twin Services for Large-Scale Disasters.*
**Sensors** 2022, 22(18), 6749. <https://doi.org/10.3390/s22186749>

> Projeto de Computação de Alto Desempenho — Instituto de Informática, UFG.
