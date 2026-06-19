/*
 * simulator_omp.c — Simulador de propagação de incêndio com OpenMP
 *
 * Estratégia de paralelismo:
 *   O grid 450×450 é dividido em N_THREADS partições horizontais.
 *   Cada thread OpenMP roda Dijkstra independente em sua partição.
 *   Halo exchange: células de fronteira são trocadas entre iterações
 *   até convergência global (nenhuma atualização de TOA pendente).
 *
 * Compilar: make  (ver Makefile)
 * Uso:
 *   ./simulator_omp --input map.csv --output toa.csv \
 *                   --ignition-row 225 --ignition-col 225 \
 *                   --threads 8 --runs 3
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

/* ── Reutilizar código base da Entrega 1 ──────────────────────────────────── */
#define MAX_ROWS    450
#define MAX_COLS    450
#define N_LAYERS    14
#define CELL_SIZE_M 30.0f
#define IDX_ELEVATION 0
#define IDX_ASPECT    1
#define IDX_SLOPE     2
#define IDX_FBFM13    3

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { float time; int row; int col; } PQNode;
typedef struct { PQNode* data; int size; int cap; } PQ;

typedef struct { float layers[N_LAYERS]; int burned; float time_burned; } Cell;

typedef struct {
    int   model_id;
    float w0_1h, delta, mx, heat, sv_1h;
} FM;

static const FM FUEL_MODELS[] = {
    { 1, 0.166f,0.305f,0.12f,18622.f,11483.f},
    { 2, 0.898f,0.305f,0.15f,18622.f,11483.f},
    { 3, 0.675f,0.762f,0.25f,18622.f,11483.f},
    { 4, 2.246f,1.829f,0.20f,18622.f, 6562.f},
    { 5, 0.225f,0.610f,0.20f,18622.f, 6562.f},
    { 6, 0.785f,0.762f,0.25f,18622.f, 6562.f},
    { 7, 0.561f,0.762f,0.40f,18622.f, 6562.f},
    { 8, 0.336f,0.061f,0.30f,18622.f,11483.f},
    { 9, 0.336f,0.061f,0.25f,18622.f,13123.f},
    {10, 0.561f,0.305f,0.25f,18622.f, 8202.f},
    {11, 0.785f,0.305f,0.15f,18622.f, 6562.f},
    {12, 3.370f,0.762f,0.20f,18622.f, 6562.f},
    {13, 7.415f,0.914f,0.25f,18622.f, 6562.f},
};
#define N_FM 13

static int get_fuel(const Cell* c) {
    /* FBFM13: 1-13 queimável; 0 e 91-99 (urbano/água/rocha) não-queimável */
    int id = (int)(c->layers[IDX_FBFM13] + 0.5f);
    if (id < 1 || id > 13) return 0;
    return id;
}

static float rothermel_ros(int fid, float slope_pct, float wind_ms) {
    if (fid <= 0 || fid > N_FM) return 0.f;
    const FM* fm = &FUEL_MODELS[fid-1];
    float sv = fm->sv_1h;
    float bd = fm->w0_1h / (fm->delta > 0 ? fm->delta : 1.f);
    float beta = bd / 32.f;
    float bop  = 3.348f * powf(sv, -0.8189f);
    if (sv <= 0 || bop <= 0) return 0.f;

    float A  = 133.f * powf(sv, -0.7913f);
    float gm = powf(sv, 1.5f) / (495.f + 0.594f * powf(sv, 1.5f));
    float fb = powf(beta/bop, A) * expf(A * (1.f - beta/bop));
    float ir = gm * fb * fm->w0_1h * fm->heat * 0.417439f;
    float xi = expf((0.792f + 0.681f*sqrtf(sv))*(beta+0.1f))/(192.f+0.2595f*sv);

    float wf = 0.f;
    float wf_ftmin = wind_ms * 196.85f;
    float C = 7.47f*expf(-0.133f*powf(sv,0.55f));
    float B = 0.02526f*powf(sv,0.54f);
    float E = 0.715f*expf(-3.59e-4f*sv);
    if (bop > 0) wf = C * powf(wf_ftmin+1.f, B) * powf(beta/bop, -E);

    float tan_phi = slope_pct / 100.f;
    float ws = (bop > 0) ? 5.275f * powf(beta/bop,-0.3f) * tan_phi * tan_phi : 0.f;

    float Q = 250.f + 1116.f * 0.05f;  /* m1h=5% default */
    float rho_b = bd;
    if (rho_b * Q <= 0) return 0.f;
    float R = (ir * xi * (1.f + wf + ws)) / (rho_b * Q);
    return (R < 0) ? 0.f : R * 0.00508f;
}

/* ── Priority Queue (thread-local) ──────────────────────────────────────── */
static PQ* pq_new(int cap) {
    PQ* pq = malloc(sizeof(PQ));
    pq->data = malloc(cap * sizeof(PQNode));
    pq->size = 0; pq->cap = cap;
    return pq;
}
static void pq_free(PQ* pq) { free(pq->data); free(pq); }
static void pq_swap(PQ* pq, int a, int b) {
    PQNode t = pq->data[a]; pq->data[a] = pq->data[b]; pq->data[b] = t;
}
static void pq_sift_up(PQ* pq, int i) {
    while (i > 0) {
        int p = (i-1)/2;
        if (pq->data[p].time <= pq->data[i].time) break;
        pq_swap(pq,p,i); i = p;
    }
}
static void pq_sift_dn(PQ* pq, int i) {
    int n = pq->size;
    while (1) {
        int s=i, l=2*i+1, r=2*i+2;
        if (l<n && pq->data[l].time<pq->data[s].time) s=l;
        if (r<n && pq->data[r].time<pq->data[s].time) s=r;
        if (s==i) break;
        pq_swap(pq,i,s); i=s;
    }
}
static void pq_push(PQ* pq, float t, int r, int c) {
    if (pq->size >= pq->cap) {
        pq->cap *= 2;
        pq->data = realloc(pq->data, pq->cap * sizeof(PQNode));
    }
    pq->data[pq->size] = (PQNode){t, r, c};
    pq_sift_up(pq, pq->size++);
}
static PQNode pq_pop(PQ* pq) {
    PQNode top = pq->data[0]; pq->size--;
    if (pq->size > 0) { pq->data[0] = pq->data[pq->size]; pq_sift_dn(pq, 0); }
    return top;
}
static int pq_empty(const PQ* pq) { return pq->size == 0; }

/* ── Vizinhança ──────────────────────────────────────────────────────────── */
static const int   DR[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
static const int   DC[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
static const float DIST[8]= {42.426f,30.f,42.426f,30.f,30.f,42.426f,30.f,42.426f};
static const float ANG[8] = {315.f,0.f,45.f,270.f,90.f,225.f,180.f,135.f};

/* ── Dados globais compartilhados entre threads ───────────────────────────── */
typedef struct {
    Cell  cells[MAX_ROWS][MAX_COLS];
    float toa[MAX_ROWS][MAX_COLS];  /* lido/escrito com omp atomic */
    int   rows, cols;
} Map;

static Map G_MAP;
static float G_WIND_MS  = 5.f;
static float G_WIND_DIR = 270.f;

/*
 * dijkstra_partition — roda Dijkstra apenas nas linhas [row_start, row_end).
 * Lê/escreve G_MAP.toa com CAS-style (só atualiza se melhora).
 * Retorna 1 se algum TOA foi atualizado nesta iteração.
 */
static int dijkstra_partition(int row_start, int row_end) {
    PQ* pq = pq_new(64 * 1024);
    int updated = 0;

    /* Seed: células finitas em [row_start-1, row_end+1) — inclui as linhas-fantasma
       (ghost) das partições vizinhas como FONTES de propagação. É assim que o fogo
       entra de uma partição na outra, sem nenhuma thread escrever na linha da outra. */
    int gs = (row_start - 1 < 0) ? 0 : row_start - 1;
    int ge = (row_end + 1 > G_MAP.rows) ? G_MAP.rows : row_end + 1;
    for (int r = gs; r < ge; r++)
        for (int c = 0; c < G_MAP.cols; c++)
            if (G_MAP.toa[r][c] < FLT_MAX) pq_push(pq, G_MAP.toa[r][c], r, c);

    while (!pq_empty(pq)) {
        PQNode nd = pq_pop(pq);
        float t = nd.time; int r = nd.row, c = nd.col;

        if (t > G_MAP.toa[r][c] + 1e-6f) continue;

        int fid = get_fuel(&G_MAP.cells[r][c]);
        if (fid <= 0) continue;

        float slope = G_MAP.cells[r][c].layers[IDX_SLOPE];
        float ros   = rothermel_ros(fid, slope, G_WIND_MS);
        if (ros < 1e-6f) continue;

        for (int d = 0; d < 8; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            if (nr < 0 || nr >= G_MAP.rows || nc < 0 || nc >= G_MAP.cols) continue;

            /* Escreve SOMENTE nas próprias linhas [row_start,row_end) → sem lock,
               pois cada linha pertence a uma única thread. As linhas-fantasma são
               apenas fontes (leitura). Remove o gargalo do critical global. */
            if (nr < row_start || nr >= row_end) continue;

            float c_eff = 0.5f + 0.5f * cosf((ANG[d] - G_WIND_DIR) * (float)M_PI / 180.f);
            float ros_d = fmaxf(ros * c_eff, 0.001f);
            float new_toa = t + DIST[d] / ros_d;

            if (new_toa < G_MAP.toa[nr][nc]) {
                G_MAP.toa[nr][nc] = new_toa;
                updated = 1;
                pq_push(pq, new_toa, nr, nc);
            }
        }
    }

    pq_free(pq);
    return updated;
}

static int load_csv(const char* path) {
    FILE* f = fopen(path, "r"); if (!f) return -1;
    char line[8192]; int cell_idx = 0;
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    while (fgets(line, sizeof(line), f) && cell_idx < MAX_ROWS*MAX_COLS) {
        int row = cell_idx / MAX_COLS, col = cell_idx % MAX_COLS, layer = 0;
        char* tok = strtok(line, ",");
        while (tok && layer < N_LAYERS) {
            G_MAP.cells[row][col].layers[layer] = atof(tok);
            tok = strtok(NULL, ","); layer++;
        }
        G_MAP.cells[row][col].burned = 0;
        G_MAP.toa[row][col] = FLT_MAX;
        cell_idx++;
    }
    G_MAP.rows = (cell_idx > 0) ? MAX_ROWS : 0; G_MAP.cols = (cell_idx > 0) ? MAX_COLS : 0;
    fclose(f); return 0;
}

static void save_toa(const char* path) {
    FILE* f = fopen(path, "w"); if (!f) return;
    fprintf(f, "row,col,time_of_arrival_s\n");
    for (int r = 0; r < G_MAP.rows; r++)
        for (int c = 0; c < G_MAP.cols; c++)
            if (G_MAP.toa[r][c] < FLT_MAX)
                fprintf(f, "%d,%d,%.2f\n", r, c, G_MAP.toa[r][c]);
    fclose(f);
}

/* ── Simulação OpenMP ─────────────────────────────────────────────────────── */
static void simulate_omp(int ign_r, int ign_c, int nthreads) {
    /* Resetar TOA */
    for (int r = 0; r < G_MAP.rows; r++)
        for (int c = 0; c < G_MAP.cols; c++)
            G_MAP.toa[r][c] = FLT_MAX;

    G_MAP.toa[ign_r][ign_c] = 0.f;
    G_MAP.cells[ign_r][ign_c].burned = 1;

    /* Dividir grid em nthreads partições horizontais */
    int rows_per_part = (G_MAP.rows + nthreads - 1) / nthreads;

    int global_updated = 1;
    int iteration = 0;
    const int MAX_ITER = 200;

    omp_set_num_threads(nthreads);

    while (global_updated && iteration < MAX_ITER) {
        global_updated = 0;
        iteration++;

        #pragma omp parallel reduction(|:global_updated)
        {
            int tid = omp_get_thread_num();
            int r0  = tid * rows_per_part;
            int r1  = (r0 + rows_per_part < G_MAP.rows) ? r0 + rows_per_part : G_MAP.rows;
            if (r0 < G_MAP.rows)
                global_updated |= dijkstra_partition(r0, r1);
        }
    }

    printf("  Iterações de halo exchange: %d\n", iteration);
}

static void usage(const char* p) {
    fprintf(stderr, "Uso: %s --input <csv> --output <csv> "
            "[--ignition-row N] [--ignition-col N] "
            "[--wind-speed <ms>] [--wind-dir <deg>] "
            "[--threads N] [--runs N]\n", p);
}

int main(int argc, char* argv[]) {
    const char *in = NULL, *out = NULL;
    int ign_r = 225, ign_c = 225, runs = 3, nthreads = 4;
    float ws = 5.f, wd = 180.f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--input")        && i+1<argc) in       = argv[++i];
        else if (!strcmp(argv[i],"--output")  && i+1<argc) out      = argv[++i];
        else if (!strcmp(argv[i],"--ignition-row") && i+1<argc) ign_r = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ignition-col") && i+1<argc) ign_c = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--wind-speed")   && i+1<argc) ws    = atof(argv[++i]);
        else if (!strcmp(argv[i],"--wind-dir")     && i+1<argc) wd    = atof(argv[++i]);
        else if (!strcmp(argv[i],"--threads")      && i+1<argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--runs")         && i+1<argc) runs  = atoi(argv[++i]);
    }
    if (!in || !out) { usage(argv[0]); return 1; }

    memset(&G_MAP, 0, sizeof(G_MAP));
    printf("Carregando: %s\n", in);
    if (load_csv(in) < 0) { fprintf(stderr, "Erro ao abrir %s\n", in); return 1; }
    printf("Mapa: %d×%d | Threads: %d\n", G_MAP.rows, G_MAP.cols, nthreads);

    G_WIND_MS  = ws;
    G_WIND_DIR = wd;

    double total = 0.0;
    for (int run = 0; run < runs; run++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        simulate_omp(ign_r, ign_c, nthreads);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
        total += el;
        printf("  Run %d: %.4f s\n", run+1, el);
    }

    double avg = total / runs;
    printf("Tempo médio (OMP threads=%d): %.4f s\n", nthreads, avg);

    int burned = 0;
    for (int r = 0; r < G_MAP.rows; r++)
        for (int c = 0; c < G_MAP.cols; c++)
            if (G_MAP.toa[r][c] < FLT_MAX) burned++;
    printf("Células atingidas: %d / %d (%.1f%%)\n",
           burned, G_MAP.rows*G_MAP.cols, 100.0*burned/(G_MAP.rows*G_MAP.cols));

    save_toa(out);
    printf("TOA salvo: %s\n", out);
    return 0;
}
