# EDA — Dataset semântico (450×450×14)

- Cenários: **68** (51 treino + 17 teste)
- NaN em algum arquivo: **False**
- O '14' refere-se a **14 CAMADAS por cenário**, não a 14 cenários.

## Tabela de camadas

| L | nome | min | max | %zero(med) | normalizada | constante | flags |
|---|------|-----|-----|-----------|-------------|-----------|-------|
| 0 | elevation | -1.00 | 3973.00 | 0.0% | NÃO | não | — |
| 1 | aspect | -1.00 | 359.00 | 0.1% | NÃO | não | — |
| 2 | slope | 0.00 | 397.00 | 0.2% | NÃO | não | — |
| 3 | fbfm13 | 0.00 | 99.00 | 0.0% | NÃO | não | — |
| 4 | cc | 0.00 | 95.00 | 46.0% | NÃO | não | — |
| 5 | ch | 0.00 | 509.00 | 45.4% | NÃO | não | — |
| 6 | cbh | 0.00 | 100.00 | 48.9% | NÃO | não | — |
| 7 | cbd | 0.00 | 45.00 | 48.3% | NÃO | não | — |
| 8 | temperature | 52.50 | 119.30 | 0.0% | NÃO | sim | — |
| 9 | humidity | 5.60 | 93.26 | 0.0% | NÃO | sim | — |
| 10 | cloud_cover | 4.17 | 99.72 | 0.0% | NÃO | sim | — |
| 11 | precipitation | 0.00 | 0.50 | 0.0% | sim | sim | — |
| 12 | wind_direction | 0.65 | 359.98 | 0.0% | NÃO | sim | — |
| 13 | wind_speed | 0.25 | 29.89 | 0.0% | NÃO | sim | — |

## Achados

- **Normalizadas [0,1]:** precipitation
- **Cruas (faixa física):** elevation, aspect, slope, fbfm13, cc, ch, cbh, cbd
- **Constantes (1 valor / imagem):** temperature, humidity, cloud_cover, precipitation, wind_direction, wind_speed → explicam as 'imagens vazias/sem relevo' (weather = constante por cenário, conforme paper).

### Impacto no simulador C (rothermel.h)
- `IDX_ELEVATION=0` → consome camada **elevation** (min=-1.0 max=3973.0)
- `IDX_ASPECT=1` → consome camada **aspect** (min=-1.0 max=359.0)
- `IDX_SLOPE=2` → consome camada **slope** (min=0.0 max=397.0)
- `IDX_FBFM13=3` → consome camada **fbfm13** (min=0.0 max=99.0)

> Se slope/fbfm40 estiverem na escala errada, a física do Rothermel e o `get_fuel()` (clamp 1–13) ficam incorretos → **benchmark inválido até corrigir**.
