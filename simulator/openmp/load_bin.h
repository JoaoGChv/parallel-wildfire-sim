#ifndef LOAD_BIN_H
#define LOAD_BIN_H
/*
 * Formato binário para comunicação Python → C:
 *   4B int32: rows
 *   4B int32: cols
 *   4B int32: layers
 *   rows×cols×layers × 4B float32: dados em ordem [row][col][layer]
 *
 * Python gera com:
 *   arr = np.load("S001.npy").astype(np.float32)  # (450,450,14)
 *   header = np.array([450, 450, 14], dtype=np.int32)
 *   header.tofile(f); arr.tofile(f)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 450
#define MAX_COLS 450
#define N_LAYERS 14

typedef struct {
    float layers[N_LAYERS];
    int burned;
    float time_burned;
} Cell;

typedef struct {
    Cell  cells[MAX_ROWS][MAX_COLS];
    float toa[MAX_ROWS][MAX_COLS];
    int   rows, cols;
} Map;

static inline int map_load_bin(const char* path, Map* m) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Erro ao abrir: %s\n", path); return -1; }

    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, f) != 3) { fclose(f); return -1; }
    int rows = hdr[0], cols = hdr[1], layers = hdr[2];
    if (rows > MAX_ROWS || cols > MAX_COLS || layers != N_LAYERS) {
        fprintf(stderr, "Dimensões inválidas: %d×%d×%d\n", rows, cols, layers);
        fclose(f); return -1;
    }

    /* Ler todos os floats de uma vez e distribuir */
    float* buf = (float*)malloc((size_t)rows * cols * layers * sizeof(float));
    if (!buf) { fclose(f); return -1; }
    size_t n = (size_t)rows * cols * layers;
    if (fread(buf, sizeof(float), n, f) != n) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);

    m->rows = rows; m->cols = cols;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            for (int l = 0; l < layers; l++)
                m->cells[r][c].layers[l] = buf[(r*cols + c)*layers + l];
            m->cells[r][c].burned = 0;
            m->cells[r][c].time_burned = 3.4e38f;
            m->toa[r][c] = 3.4e38f;
        }
    free(buf);
    return 0;
}

static inline void map_save_toa(const char* path, const Map* m) {
    FILE* f = fopen(path, "w"); if (!f) return;
    fprintf(f, "row,col,time_of_arrival_s\n");
    for (int r = 0; r < m->rows; r++)
        for (int c = 0; c < m->cols; c++)
            if (m->toa[r][c] < 3.4e38f)
                fprintf(f, "%d,%d,%.2f\n", r, c, m->toa[r][c]);
    fclose(f);
}

#endif /* LOAD_BIN_H */
