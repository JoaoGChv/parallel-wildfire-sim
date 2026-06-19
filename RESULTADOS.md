# Resultados — Projeto HPC: Simulação de Incêndios com Balanceamento de Carga

Reprodução e melhoria de Kwon, Yun, Kim (*Sensors* 2022, 22, 6749): framework de
computação distribuída que usa CNN para prever carga computacional e k-d tree para
particionar a simulação de incêndio de forma balanceada.

> **Síntese honesta:** o *método* de balanceamento por carga tem teto altíssimo
> (oráculo 35–57% melhor que particionamento uniforme). O difícil — e o nosso
> principal achado — é **prever a carga**: a propagação do fogo é global,
> anisotrópica e dependente do caminho, então predição estática a partir do dado
> semântico não funciona. Delimitamos **quando** a abordagem paga e quando não.

---

## 1. Correção da fundação de dados (pré-requisito)

A EDA ([analysis/eda.py](analysis/eda.py), relatório em
[analysis/out/eda_report.md](analysis/out/eda_report.md)) revelou que o dataset
estava **corrompido na origem** e foi regenerado. Três bugs corrigidos no
[scripts/preprocess.py](scripts/preprocess.py) e [scripts/run_elmfire_batch.py](scripts/run_elmfire_batch.py):

1. **Colormap em vez de raster bruto** — `exportImage` sem `renderingRule` devolvia
   a imagem simbolizada; os códigos de combustível FBFM estavam destruídos
   (viravam rampa contínua 0–100). Corrigido com `renderingRule={"rasterFunction":"None"}`.
2. **FBFM40 vs simulador** — o `rothermel.c` implementa os 13 modelos de Anderson;
   trocamos para `LF2016_FBFM13_CONUS` (códigos 1–13 + 91–99 não-queimável).
3. **Nodata −9999** não tratado em 6 cenários.

Esclarecimento para a banca: **"14" = 14 camadas por cenário, não 14 cenários**
(temos 68 = 51 treino + 17 teste, igual ao paper). As "imagens vazias" são as 6
camadas de clima (constantes por cenário, como o paper define) + áreas sem floresta.

Validação pós-correção: **0 anomalias** nas 14 camadas. Labels do elmfire
regenerados com clima calibrado para fogo ativo (mediana 1.25%, máx 29% de área
queimada).

---

## 2. Entrega 2 — Paralelização (HPC)

### 2.1. Otimização dos simuladores
Correção do `get_fuel` (códigos 91–99 = não-queimável) nos três; os três concordam
em ~93.3% de área queimada (corretude validada). Otimizações:

| Simulador | Antes | Depois | Ganho | Mudança |
|-----------|-------|--------|-------|---------|
| **CUDA** ([fire_propagation.cu](simulator/cuda/fire_propagation.cu)) | 3.17 s (416 lançamentos) | **0.28 s** (37) | **11×** | relaxação iterativa interna na shared memory; correção do sentinela INF |
| **OpenMP** ([simulator_omp.c](simulator/openmp/simulator_omp.c)) | 1.15 s | **0.157 s** (8 threads) | **7×** | remoção do `critical` global (escrita lock-free por linhas) |

Observação honesta: ambos seguem acima do Dijkstra sequencial (40 ms) porque o grid
450×450 é pequeno e o sequencial já é quase ótimo — por isso a avaliação foca na
**métrica de balanceamento do paper**, não em speedup bruto de uma sim de 40 ms.

### 2.2. Métrica do paper — tempo de simulação relativo (Figura 12)
[scripts/evaluate_partitioning.py](scripts/evaluate_partitioning.py) — compara
estratégias de particionamento via k-d tree, medindo `max(carga_partição)/carga_total`
(Eq.5). Figura em [analysis/out/fig12_relative_time.png](analysis/out/fig12_relative_time.png).

| nº partições | uniform | points-based | **proposta (CNN estática)** | **oráculo (carga real)** |
|---|---|---|---|---|
| 2 | 0.855 | 0.852 | 0.917 | **0.545** |
| 4 | 0.659 | 0.706 | 0.771 | **0.281** |
| 8 | 0.566 | 0.601 | 0.667 | **0.165** |
| 16 | 0.512 | 0.540 | 0.605 | **0.103** |
| 32 | 0.422 | 0.447 | 0.504 | **0.070** |

**Leitura:** o **oráculo** (particionar conhecendo a carga real) é 35–57% melhor que
o uniform → o método tem teto enorme. Mas a **proposta com CNN estática é pior que o
uniform** — a predição estática não captura onde o fogo realmente vai.

---

## 3. Achado central: por que a predição estática falha

A CNN ([model/cnn.py](model/cnn.py)) foi treinada de duas formas, ambas falhando como
mapa de carga para particionar:
- **Regressão do TOA** (time-of-arrival): MSE de teste ~0.027, mas piora vs uniform.
- **Classificação queima/não** (binária, com `pos_weight`): recall 0.08, overfit
  (perda de teste sobe).
- **Proxy radial** (distância à ignição): correlação queima×proximidade só **0.213**.

**Causa raiz:** quais células queimam (e quando) depende do **caminho global** do
fogo desde a ignição — vento, conectividade do combustível, barreiras. Uma janela
local 21×21 de dado **estático** não tem essa informação. (Verificado: a k-d tree
está correta — em carga sintética concentrada atinge o balanço ideal exato.)

---

## 4. Entrega 3 — Rebalanceamento online

Como a carga é conhecida **durante** a simulação (a frente de fogo atual),
implementamos rebalanceamento online ([scripts/online_rebalancing.py](scripts/online_rebalancing.py)):
modelo discreto dirigido pelo campo TOA real; o tempo de cada passo = nó mais lento
(barreira de sincronização). Estratégias: estático, reativo (paper, Eqs 2–4),
preditivo (proposta) e oráculo.

### 4.1. Qualidade de balanço (sem custo de migração) — N=8
| estratégia | tempo relativo | nº reparts |
|---|---|---|
| estático | 0.168 | 0 |
| **reativo (paper)** | **0.151** | 9.4 |
| preditivo (proposta) | 0.164 | 9.4 |
| oráculo | 0.144 | 37 |

O **rebalanceamento reativo melhora o balanço** (~10% vs estático) e se aproxima do
oráculo usando **¼ dos reparticionamentos** do oráculo. (Fig.
[analysis/out_online_nocost/online_rebalancing.png](analysis/out_online_nocost/online_rebalancing.png))

### 4.2. Com custo de migração realista (0.003·GRID²/repart) — N=8
| estratégia | tempo relativo | nº reparts |
|---|---|---|
| **estático** | **0.168** | 0 |
| reativo | 0.180 | 9.4 |
| preditivo | 0.192 | 9.4 |
| oráculo | 0.256 | 37 |

Quando se cobra a migração de estado entre nós, **no nosso regime (fogos curtos e
espalhados) o estático ganha** — o custo de reparticionar supera o ganho de balanço.

### 4.3. Resultado honesto sobre o preditivo
A contribuição preditiva **não superou o reativo** de forma robusta. A razão é
geométrica: o fogo se espalha de modo irregular/radial, então nenhuma partição fixa
(blocos ou faixas) mantém o balanço acumulado enquanto a frente varre o domínio.
Não forçamos um resultado positivo por *parameter-fishing*.

---

## 5. Conclusões

1. **Balanceamento por carga funciona** quando a carga é conhecida (oráculo: −57%).
2. **Predição estática de carga não funciona** para incêndios — propagação global e
   dependente do caminho. (achado/contribuição negativa bem caracterizada)
3. **Rebalanceamento online reativo** recupera quase todo o ganho do oráculo em
   *balanço*, mas o ganho líquido depende do **trade-off custo-de-migração × duração
   da simulação**: paga em simulações grandes/longas/concentradas (regime do paper),
   não em fogos curtos/espalhados (nosso regime).
4. **Onde melhoramos vs o paper:** pipeline de dados corrigido e validado; CUDA 11× e
   OpenMP 7× mais rápidos; e a **delimitação empírica honesta** de quando a abordagem
   de balanceamento paga — incluindo a demonstração de que predição estática é
   insuficiente, motivando o rebalanceamento online.

## 6. Extras

### 6.1. Simulador interativo (visualização dinâmica)
[app/fire_dashboard.py](app/fire_dashboard.py) — dashboard Streamlit que roda a
simulação **real** em C e deixa variar vento (vel./direção/anisotropia), umidade e
ignição, com animação temporal da chegada do fogo. (MuJoCo/Gazebo/Isaac descartados:
são engines de robótica, não de fogo celular 2D.) Validado: vento leste → fogo p/
direita; oeste → esquerda. Rodar: `streamlit run app/fire_dashboard.py --server.port 8501`.

### 6.2. Análise de sensibilidade a clima/vento
[scripts/sensitivity.py](scripts/sensitivity.py) — varredura com a simulação real
(figuras em [analysis/out/sensitivity_wind_moisture.png](analysis/out/sensitivity_wind_moisture.png)
e [sensitivity_anisotropy.png](analysis/out/sensitivity_anisotropy.png)):
- **Vento:** sem vento o fogo quase não se propaga (0% em 2 h); a 18 m/s alcança ~46%.
- **Umidade:** efeito de extinção claro — acima de ~0.15–0.18 de umidade 1h o fogo não
  se sustenta (0%); seco (0.03) alcança ~41%.
- **Direção:** o vento empurra o centroide do fogo ~4 km a favor (anisotropia).

Confirma que o modelo de Rothermel implementado responde fisicamente a vento e
umidade — sustentando o uso da simulação como ground-truth de carga.

### 6.3. Dataset do Brasil (generalização além dos EUA)
[scripts/preprocess_brazil.py](scripts/preprocess_brazil.py) — o paper só usou
incêndios dos EUA (LANDFIRE). Geramos cenários para 4 locais reais propensos a
incêndio no Brasil (Pantanal, Cerrado GO/TO, arco da Amazônia) usando **fontes
globais**: elevação do **Esri WorldElevation3D/Terrain3D** (slope/aspect derivados
do gradiente) e uso do solo do **Esri Sentinel-2 10m Land Cover**, mapeado para
combustível FBFM13 (árvores→modelo 8, pastagem→3, agricultura→1; água/construído/
solo→não-queimável). Clima por constantes aleatórias (como o paper).

Resultado em [analysis/out/brazil_demo.png](analysis/out/brazil_demo.png): o pipeline
roda fim-a-fim e a **simulação real propaga fogo sobre o terreno brasileiro**,
respondendo ao vento, com o combustível e as barreiras (água/área construída) reais
de cada bioma. Demonstra que a metodologia generaliza além do dataset original.

### 6.4. Suporte a FBFM40 (Scott & Burgan) via crosswalk
[scripts/fbfm40_crosswalk.py](scripts/fbfm40_crosswalk.py) + opção `--fuel fbfm40` no
[preprocess.py](scripts/preprocess.py). O simulador implementa os 13 modelos de
Anderson; o crosswalk traduz os 40 modelos de Scott & Burgan (códigos 101–204:
grama/grama-arbusto/arbusto/timber/slash) para o Anderson equivalente em
comportamento (ex.: 101→1, 143→6, 183→8, 201→11; 91–99 preservados). Validado no
S001: `--fuel fbfm40` produz códigos válidos (1–13 + não-queimável), compatíveis com
o simulador, a partir da fonte de combustível mais detalhada.

## 7. Limitações e trabalho futuro
- Regime de avaliação favorável ao paper (sim. longas/grandes) para evidenciar o
  ganho líquido do rebalanceamento.
- Preditor de carga melhor: contexto global / trajetória da frente (anisotropia),
  não janela local estática.
- Dataset do Brasil: copa/combustível mais fiéis (ex.: MapBiomas) e perímetros
  reais de ignição (INPE/Programa Queimadas), no lugar das aproximações atuais.
