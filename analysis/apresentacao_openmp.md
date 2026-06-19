# OpenMP — trechos de código para a apresentação

Paralelização do simulador de incêndio ([simulator/openmp/simulator_omp.c](../simulator/openmp/simulator_omp.c)).
Estratégia: **decomposição de domínio** — o grid 450×450 é dividido em faixas
horizontais (uma por thread); cada thread roda Dijkstra na sua faixa; a propagação
entre faixas acontece por **halo exchange** iterativo até convergência.

---

## 1. Região paralela: uma faixa por thread + convergência global

```c
while (global_updated && iteration < MAX_ITER) {
    global_updated = 0;
    iteration++;

    #pragma omp parallel reduction(|:global_updated)
    {
        int tid = omp_get_thread_num();
        int r0  = tid * rows_per_part;                 // início da faixa da thread
        int r1  = (r0 + rows_per_part < G_MAP.rows) ? r0 + rows_per_part : G_MAP.rows;
        if (r0 < G_MAP.rows)
            global_updated |= dijkstra_partition(r0, r1);   // Dijkstra local
    }
}
```

- `#pragma omp parallel` cria as threads; cada uma processa **só a sua faixa** `[r0,r1)`.
- `reduction(|:global_updated)` combina (OR) o "houve atualização?" de todas as
  threads numa flag global, sem corrida — o laço externo repete enquanto **alguma**
  thread ainda propaga fogo (halo exchange até convergir).

---

## 2. A OTIMIZAÇÃO principal: escrita lock-free (sem `critical` global)

**Antes (lento):** toda relaxação de vizinho passava por um `#pragma omp critical`
global → serializava as threads e destruía o paralelismo (28× mais lento que o seq).

**Depois (7× mais rápido):** cada thread escreve **apenas nas suas próprias linhas**
→ sem disputa, sem lock. As linhas das vizinhas entram só como **fonte** (leitura).

```c
static int dijkstra_partition(int row_start, int row_end) {
    PQ* pq = pq_new(64 * 1024);
    int updated = 0;

    // Seed: inclui as linhas-fantasma (ghost) das faixas vizinhas como FONTES.
    // É assim que o fogo "entra" de uma faixa na outra sem ninguém escrever
    // na linha da outra thread.
    int gs = (row_start - 1 < 0) ? 0 : row_start - 1;
    int ge = (row_end + 1 > G_MAP.rows) ? G_MAP.rows : row_end + 1;
    for (int r = gs; r < ge; r++)
        for (int c = 0; c < G_MAP.cols; c++)
            if (G_MAP.toa[r][c] < FLT_MAX) pq_push(pq, G_MAP.toa[r][c], r, c);

    while (!pq_empty(pq)) {
        PQNode nd = pq_pop(pq);
        float t = nd.time; int r = nd.row, c = nd.col;
        if (t > G_MAP.toa[r][c] + 1e-6f) continue;
        ...
        for (int d = 0; d < 8; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            ...
            // Escreve SOMENTE nas próprias linhas → lock-free (exclusivo da thread).
            if (nr < row_start || nr >= row_end) continue;

            float new_toa = t + DIST[d] / ros_d;
            if (new_toa < G_MAP.toa[nr][nc]) {
                G_MAP.toa[nr][nc] = new_toa;   // sem #pragma omp critical!
                updated = 1;
                pq_push(pq, new_toa, nr, nc);
            }
        }
    }
    pq_free(pq); return updated;
}
```

**Ideia-chave para a banca:** o gargalo de paralelismo não era o cálculo, era a
**sincronização**. Trocar um lock global por um particionamento onde cada thread só
escreve no que é seu (e lê as bordas das vizinhas) eliminou a contenção.

---

## 3. Resultado

| Threads | Tempo | vs sequencial |
|---|---|---|
| baseline (critical global) | 1.15 s | 28× mais lento |
| **8 threads (lock-free)** | **0.157 s** | **7× mais rápido que o baseline** |

Compilação: `gcc -O3 -march=native -fopenmp -o simulator_omp simulator_omp.c -lm`
(ver [simulator/openmp/Makefile](../simulator/openmp/Makefile)).

> Observação honesta: o Dijkstra **sequencial** roda em ~40 ms (grid pequeno, 450×450),
> então nem OpenMP nem CUDA o batem em wall-clock — por isso a avaliação do projeto
> usa a **métrica de balanceamento do paper** (Figura 12), não speedup bruto.
