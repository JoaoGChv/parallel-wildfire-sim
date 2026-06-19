/*
 * main_kdtree.c — Demonstração do particionamento k-d tree
 *
 * Lê o mapa de carga computacional (predição CNN ou TOA do elmfire)
 * e particiona em N partições balanceadas via k-d tree.
 *
 * Uso:
 *   ./kdtree_demo --load-map <csv_toa> --n-parts 4 --output partition_map.csv
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kdtree.h"

#define ROWS 450
#define COLS 450

static float g_load[ROWS * COLS];

/* Lê TOA CSV e usa como proxy de carga (células queimadas = carga > 0) */
static void load_toa_as_load(const char* path) {
    for (int i = 0; i < ROWS * COLS; i++) g_load[i] = 0.1f; /* default: carga mínima */
    FILE* f = fopen(path, "r"); if (!f) return;
    char line[256]; fgets(line, sizeof(line), f); /* header */
    while (fgets(line, sizeof(line), f)) {
        int r, c; float t;
        if (sscanf(line, "%d,%d,%f", &r, &c, &t) == 3)
            g_load[r * COLS + c] = t;   /* TOA como proxy de carga */
    }
    fclose(f);
}

static void save_partition_map(const char* path, Partition* parts, int n) {
    /* Cria CSV: row,col,partition_id */
    int* pmap = calloc(ROWS * COLS, sizeof(int));
    for (int i = 0; i < ROWS * COLS; i++) pmap[i] = -1;

    for (int p = 0; p < n; p++) {
        for (int r = parts[p].row_min; r < parts[p].row_max; r++)
            for (int c = parts[p].col_min; c < parts[p].col_max; c++)
                pmap[r * COLS + c] = parts[p].id;
    }

    FILE* f = fopen(path, "w"); if (!f) { free(pmap); return; }
    fprintf(f, "row,col,partition_id\n");
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            fprintf(f, "%d,%d,%d\n", r, c, pmap[r*COLS+c]);
    fclose(f);
    free(pmap);
}

int main(int argc, char* argv[]) {
    const char* load_map_path = NULL;
    const char* out_path      = "partition_map.csv";
    int n_parts = 4;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--load-map") && i+1<argc) load_map_path = argv[++i];
        else if (!strcmp(argv[i],"--n-parts") && i+1<argc) n_parts = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--output")  && i+1<argc) out_path = argv[++i];
    }

    /* Aceitar n_parts como potência de 2 entre 2 e 32 */
    if (n_parts < 2) n_parts = 2;
    if (n_parts > MAX_PARTITIONS) n_parts = MAX_PARTITIONS;

    if (load_map_path) {
        printf("Carregando mapa de carga: %s\n", load_map_path);
        load_toa_as_load(load_map_path);
    } else {
        printf("Mapa de carga uniforme (sem --load-map)\n");
        for (int i = 0; i < ROWS * COLS; i++) g_load[i] = 1.f;
    }

    printf("Construindo k-d tree com %d partições...\n", n_parts);
    int next_id = 0;
    KDNode* tree = kdtree_build(g_load, ROWS, COLS, n_parts,
                                0, ROWS, 0, COLS, 0, &next_id);

    /* Exibir árvore */
    printf("\nEstrutura da k-d tree:\n");
    kdtree_print(tree, 0);

    /* Coletar folhas */
    Partition parts[MAX_PARTITIONS];
    int count = 0;
    kdtree_get_leaves(tree, parts, &count);

    /* Estatísticas de balanceamento */
    printf("\nEstatísticas de balanceamento (%d partições):\n", count);
    float total = 0.f, min_load = 1e30f, max_load = 0.f;
    for (int i = 0; i < count; i++) {
        total    += parts[i].load;
        if (parts[i].load < min_load) min_load = parts[i].load;
        if (parts[i].load > max_load) max_load = parts[i].load;
    }
    float avg = total / count;
    printf("  Carga média:    %.1f\n", avg);
    printf("  Carga mínima:   %.1f (%.1f%% da média)\n", min_load, 100*min_load/avg);
    printf("  Carga máxima:   %.1f (%.1f%% da média)\n", max_load, 100*max_load/avg);
    printf("  Imbalance ratio: %.3f\n", max_load / (min_load > 0 ? min_load : 1.f));

    /* Salvar mapa de partições */
    printf("\nSalvando mapa de partições: %s\n", out_path);
    save_partition_map(out_path, parts, count);

    kdtree_free(tree);
    return 0;
}
