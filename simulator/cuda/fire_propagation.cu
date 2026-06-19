/*
 * fire_propagation.cu — Kernel CUDA de propagação de incêndio
 *
 * Algoritmo: Parallel Wavefront Dijkstra (Bellman-Ford relaxation em GPU)
 *   - Cada thread processa uma célula do grid 450×450
 *   - Iterações até convergência: nenhum TOA muda mais de EPSILON
 *   - Shared memory usada para troca de valores na borda de cada bloco CUDA
 *
 * Compilar: nvcc -O3 -arch=sm_89 fire_propagation.cu -o fire_cuda -lm
 *           (sm_89 = RTX 4090; ajustar conforme GPU)
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_ROWS    450
#define MAX_COLS    450
#define N_LAYERS    14
#define CELL_SIZE_M 30.0f

#define IDX_ELEVATION 0
#define IDX_ASPECT    1
#define IDX_SLOPE     2
#define IDX_FBFM13    3

/* ── Parâmetros de fuel model (em device constant memory) ──────────────── */
struct FuelModel {
    float w0_1h, delta, mx, heat, sv_1h;
};

__constant__ FuelModel d_fm[14] = {
    {0.166f,0.305f,0.12f,18622.f,11483.f},
    {0.898f,0.305f,0.15f,18622.f,11483.f},
    {0.675f,0.762f,0.25f,18622.f,11483.f},
    {2.246f,1.829f,0.20f,18622.f, 6562.f},
    {0.225f,0.610f,0.20f,18622.f, 6562.f},
    {0.785f,0.762f,0.25f,18622.f, 6562.f},
    {0.561f,0.762f,0.40f,18622.f, 6562.f},
    {0.336f,0.061f,0.30f,18622.f,11483.f},
    {0.336f,0.061f,0.25f,18622.f,13123.f},
    {0.561f,0.305f,0.25f,18622.f, 8202.f},
    {0.785f,0.305f,0.15f,18622.f, 6562.f},
    {3.370f,0.762f,0.20f,18622.f, 6562.f},
    {7.415f,0.914f,0.25f,18622.f, 6562.f},
};

__constant__ int   d_DR[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
__constant__ int   d_DC[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
__constant__ float d_DIST[8] = {42.426f,30.f,42.426f,30.f,30.f,42.426f,30.f,42.426f};
__constant__ float d_ANG[8]  = {315.f,0.f,45.f,270.f,90.f,225.f,180.f,135.f};

/* ── Rothermel ROS no device ──────────────────────────────────────────────── */
__device__ float d_rothermel_ros(int fid, float slope_pct, float wind_ms) {
    if (fid <= 0 || fid > 13) return 0.f;
    const FuelModel* fm = &d_fm[fid-1];
    float sv = fm->sv_1h;
    float bd = fm->w0_1h / (fm->delta > 0 ? fm->delta : 1.f);
    float beta = bd / 32.f;
    float bop  = 3.348f * powf(sv, -0.8189f);
    if (sv <= 0 || bop <= 0) return 0.f;

    float A  = 133.f * powf(sv, -0.7913f);
    float gm = powf(sv, 1.5f) / (495.f + 0.594f * powf(sv, 1.5f));
    float fb = powf(beta/bop, A) * expf(A*(1.f-beta/bop));
    float ir = gm * fb * fm->w0_1h * fm->heat * 0.417439f;
    float xi = expf((0.792f+0.681f*sqrtf(sv))*(beta+0.1f))/(192.f+0.2595f*sv);

    float wf_ftmin = wind_ms * 196.85f;
    float C = 7.47f*expf(-0.133f*powf(sv,0.55f));
    float B = 0.02526f*powf(sv,0.54f);
    float E = 0.715f*expf(-3.59e-4f*sv);
    float wf = C * powf(wf_ftmin+1.f, B) * powf(beta/bop, -E);

    float tan_phi = slope_pct / 100.f;
    float ws = 5.275f * powf(beta/bop,-0.3f) * tan_phi * tan_phi;

    float Q = 250.f + 1116.f * 0.05f;
    float rho_b = bd;
    if (rho_b * Q <= 0) return 0.f;
    float R = (ir * xi * (1.f + wf + ws)) / (rho_b * Q);
    return fmaxf(R * 0.00508f, 0.f);
}

/* ────────────────────────────────────────────────────────────────────────────
 * Kernel principal: Bellman-Ford relaxation paralela
 *
 * Cada thread lê o TOA de suas células vizinhas e atualiza o TOA local
 * se encontrar um caminho mais curto. Usa shared memory para os valores
 * de borda do bloco (halo cells = 1 célula ao redor do tile).
 *
 * Grid: (COLS/TILE_W, ROWS/TILE_H) blocos
 * Block: (TILE_W, TILE_H) threads
 *
 * Shared memory: (TILE_W+2) × (TILE_H+2) floats (halo incluído)
 * ──────────────────────────────────────────────────────────────────────────*/
#define TILE_W 16
#define TILE_H 16
#define SM_W (TILE_W + 2)
#define SM_H (TILE_H + 2)

/* Sentinela de "não queimado". NÃO usar FLT_MAX via memset 0x7f
   (0x7f7f7f7f ≈ 3.39e38 < FLT_MAX → contagem de queimados ficava 100% falso). */
#define INF_TOA   1e30f
#define BURN_THRESH 1e29f   /* TOA < BURN_THRESH ⇒ célula queimou */

/* Número de relaxações INTERNAS (em shared memory) por lançamento de kernel.
   Cada iteração propaga o fogo 1 célula; com INNER_ITERS ~ tamanho do tile, um
   único lançamento propaga o wavefront por todo o bloco usando o halo como
   condição de contorno. Reduz os lançamentos globais (e syncs com o host) de
   centenas para ~dezenas. */
#define INNER_ITERS 24

__global__ void fire_relax_kernel(
    const float* __restrict__ d_layers,   /* N_LAYERS × ROWS × COLS */
    float*       d_toa,                    /* ROWS × COLS  (in/out) */
    int*         d_updated,               /* flag global de convergência */
    int          rows, int cols,
    float        wind_ms, float wind_dir_deg
) {
    int gx = blockIdx.x * TILE_W + threadIdx.x;
    int gy = blockIdx.y * TILE_H + threadIdx.y;
    int tx = threadIdx.x + 1;   /* posição na shared mem (com halo) */
    int ty = threadIdx.y + 1;
    bool inside = (gx < cols && gy < rows);

    __shared__ float sm_toa[SM_H][SM_W];

    /* ── Carregar tile + halo na shared memory ─────────────────────────── */
    float my_toa = inside ? d_toa[gy * cols + gx] : INF_TOA;
    sm_toa[ty][tx] = my_toa;

    if (threadIdx.x == 0)
        sm_toa[ty][0]      = (gx > 0)        ? d_toa[gy*cols + (gx-1)]     : INF_TOA;
    if (threadIdx.x == TILE_W-1)
        sm_toa[ty][SM_W-1] = (gx+1 < cols)   ? d_toa[gy*cols + (gx+1)]     : INF_TOA;
    if (threadIdx.y == 0)
        sm_toa[0][tx]      = (gy > 0)        ? d_toa[(gy-1)*cols + gx]     : INF_TOA;
    if (threadIdx.y == TILE_H-1)
        sm_toa[SM_H-1][tx] = (gy+1 < rows)   ? d_toa[(gy+1)*cols + gx]     : INF_TOA;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sm_toa[0][0]           = (gx>0&&gy>0)                   ? d_toa[(gy-1)*cols+(gx-1)]         : INF_TOA;
        sm_toa[0][SM_W-1]      = (gx+TILE_W<cols&&gy>0)         ? d_toa[(gy-1)*cols+(gx+TILE_W)]    : INF_TOA;
        sm_toa[SM_H-1][0]      = (gx>0&&gy+TILE_H<rows)         ? d_toa[(gy+TILE_H)*cols+(gx-1)]    : INF_TOA;
        sm_toa[SM_H-1][SM_W-1] = (gx+TILE_W<cols&&gy+TILE_H<rows)? d_toa[(gy+TILE_H)*cols+(gx+TILE_W)]: INF_TOA;
    }
    __syncthreads();

    /* ── ROS da célula (calculado uma vez) + ROS por direção ──────────────── */
    bool burnable = false;
    float ros_d[8];
    if (inside) {
        int base = gy * cols + gx;
        int fid = (int)(d_layers[IDX_FBFM13 * rows * cols + base] + 0.5f);
        if (fid >= 1 && fid <= 13) {                 /* 0 e 91-99 não-queimável */
            float slope = d_layers[IDX_SLOPE * rows * cols + base];
            float ros_max = d_rothermel_ros(fid, slope, wind_ms);
            if (ros_max >= 1e-6f) {
                burnable = true;
                for (int d = 0; d < 8; d++) {
                    float c_eff = 0.5f + 0.5f * cosf((d_ANG[d]-wind_dir_deg)*M_PI/180.f);
                    ros_d[d] = fmaxf(ros_max * c_eff, 0.001f);
                }
            }
        }
    }

    /* ── Relaxação iterativa dentro do bloco (halo fixo como contorno) ────── */
    for (int it = 0; it < INNER_ITERS; it++) {
        __syncthreads();                 /* sm_toa estável para leitura */
        float best = my_toa;
        if (burnable) {
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                float nbr = sm_toa[ty + d_DR[d]][tx + d_DC[d]];
                if (nbr < INF_TOA)
                    best = fminf(best, nbr + d_DIST[d] / ros_d[d]);
            }
        }
        __syncthreads();                 /* todos leram antes de escrever */
        my_toa = best;
        sm_toa[ty][tx] = best;
    }

    /* ── Escrever de volta no global (atomicMin via CAS) se melhorou ──────── */
    if (inside && burnable) {
        int base = gy * cols + gx;
        float gval = d_toa[base];
        if (my_toa < gval - 1e-3f) {
            int* addr = (int*)&d_toa[base];
            int old_int = __float_as_int(gval), assumed;
            do {
                assumed = old_int;
                if (__int_as_float(assumed) <= my_toa) break;
                old_int = atomicCAS(addr, assumed, __float_as_int(my_toa));
            } while (assumed != old_int);
            atomicOr(d_updated, 1);
        }
    }
}

/* ── Host: leitura CSV (formato da Entrega 1) ──────────────────────────── */
static int ROWS, COLS;
static float H_LAYERS[N_LAYERS][MAX_ROWS][MAX_COLS];  /* layers[layer][row][col] */
static float H_TOA[MAX_ROWS][MAX_COLS];

static int load_csv(const char* path) {
    FILE* f = fopen(path, "r"); if (!f) return -1;
    char line[8192]; int cell_idx = 0;
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    while (fgets(line, sizeof(line), f) && cell_idx < MAX_ROWS*MAX_COLS) {
        int row = cell_idx / MAX_COLS, col = cell_idx % MAX_COLS, layer = 0;
        char* tok = strtok(line, ",");
        while (tok && layer < N_LAYERS) {
            H_LAYERS[layer][row][col] = atof(tok);
            tok = strtok(NULL, ","); layer++;
        }
        H_TOA[row][col] = INF_TOA;
        cell_idx++;
    }
    ROWS = (cell_idx > 0) ? MAX_ROWS : 0; COLS = (cell_idx > 0) ? MAX_COLS : 0;
    fclose(f); return 0;
}

static void save_toa(const char* path) {
    FILE* f = fopen(path, "w"); if (!f) return;
    fprintf(f, "row,col,time_of_arrival_s\n");
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            if (H_TOA[r][c] < INF_TOA)
                fprintf(f, "%d,%d,%.2f\n", r, c, H_TOA[r][c]);
    fclose(f);
}

/* ── Main ──────────────────────────────────────────────────────────────── */
static void usage(const char* p) {
    fprintf(stderr, "Uso: %s --input <csv> --output <csv> "
            "[--ignition-row N] [--ignition-col N] "
            "[--wind-speed <ms>] [--wind-dir <deg>] "
            "[--runs N] [--max-iter N]\n", p);
}

int main(int argc, char* argv[]) {
    const char *in = NULL, *out = NULL;
    int ign_r = 225, ign_c = 225, runs = 3, max_iter = 500;
    float ws = 5.f, wd = 270.f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"--input")        && i+1<argc) in      = argv[++i];
        else if (!strcmp(argv[i],"--output")       && i+1<argc) out     = argv[++i];
        else if (!strcmp(argv[i],"--ignition-row") && i+1<argc) ign_r   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ignition-col") && i+1<argc) ign_c   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--wind-speed")   && i+1<argc) ws      = atof(argv[++i]);
        else if (!strcmp(argv[i],"--wind-dir")     && i+1<argc) wd      = atof(argv[++i]);
        else if (!strcmp(argv[i],"--runs")         && i+1<argc) runs    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-iter")     && i+1<argc) max_iter = atoi(argv[++i]);
    }
    if (!in || !out) { usage(argv[0]); return 1; }

    /* GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    /* Carregar dados */
    memset(H_LAYERS, 0, sizeof(H_LAYERS));
    memset(H_TOA, 0, sizeof(H_TOA));
    printf("Carregando: %s\n", in);
    if (load_csv(in) < 0) { fprintf(stderr, "Erro ao abrir %s\n", in); return 1; }
    printf("Mapa: %d×%d\n", ROWS, COLS);

    /* Alocar device */
    int N = ROWS * COLS;
    float *d_layers, *d_toa; int *d_updated;
    cudaMalloc(&d_layers,  N_LAYERS * N * sizeof(float));
    cudaMalloc(&d_toa,     N * sizeof(float));
    cudaMalloc(&d_updated, sizeof(int));

    /* Copiar layers (formato planar: layer×row×col) */
    for (int l = 0; l < N_LAYERS; l++)
        cudaMemcpy(d_layers + l*N, H_LAYERS[l], N*sizeof(float), cudaMemcpyHostToDevice);

    /* Configurar grid */
    dim3 block(TILE_W, TILE_H);
    dim3 grid((COLS + TILE_W - 1)/TILE_W, (ROWS + TILE_H - 1)/TILE_H);

    double total = 0.0;

    /* Buffer de init: TOA = INF_TOA em todas as células (sentinela correto) */
    float* h_toa_init = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_toa_init[i] = INF_TOA;

    for (int run = 0; run < runs; run++) {
        /* Reset TOA */
        cudaMemcpy(d_toa, h_toa_init, N * sizeof(float), cudaMemcpyHostToDevice);
        float zero = 0.f;
        cudaMemcpy(d_toa + ign_r*COLS + ign_c, &zero, sizeof(float), cudaMemcpyHostToDevice);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int iter = 0;
        int h_updated = 1;
        while (h_updated && iter < max_iter) {
            h_updated = 0;
            cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice);

            fire_relax_kernel<<<grid, block>>>(
                d_layers, d_toa, d_updated,
                ROWS, COLS, ws, wd
            );
            cudaDeviceSynchronize();
            cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);
            iter++;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
        total += el;
        printf("  Run %d: %.4f s (%d iters)\n", run+1, el, iter);
    }

    double avg = total / runs;
    printf("Tempo médio (CUDA): %.4f s\n", avg);

    /* Copiar resultado de volta */
    float* flat_toa = (float*)malloc(N * sizeof(float));
    cudaMemcpy(flat_toa, d_toa, N*sizeof(float), cudaMemcpyDeviceToHost);
    int burned = 0;
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) {
            H_TOA[r][c] = flat_toa[r*COLS + c];
            if (H_TOA[r][c] < INF_TOA) burned++;
        }
    free(flat_toa);
    printf("Células atingidas: %d / %d (%.1f%%)\n", burned, N, 100.0*burned/N);

    save_toa(out);
    printf("TOA salvo: %s\n", out);

    cudaFree(d_layers); cudaFree(d_toa); cudaFree(d_updated);
    return 0;
}
