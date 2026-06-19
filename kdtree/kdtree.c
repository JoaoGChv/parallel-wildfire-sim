/*
 * kdtree.c — Implementação do k-d tree de particionamento
 *
 * Referência: Kwon et al. (2022) — "the partitioning algorithm followed the k-d tree,
 * The numbers of partitions and computing nodes were squared numbers of 2"
 * → Partições: 2, 4, 8, 16, 32 (potências de 2)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "kdtree.h"

/* Soma da carga em uma região retangular */
float region_load(const float* load_map, int cols,
                  int r0, int r1, int c0, int c1) {
    float total = 0.f;
    for (int r = r0; r < r1; r++)
        for (int c = c0; c < c1; c++)
            total += load_map[r * cols + c];
    return total;
}

/*
 * Encontra a posição de corte que melhor equilibra a carga.
 * axis=0: corte horizontal (divide em row_min..split e split..row_max)
 * axis=1: corte vertical   (divide em col_min..split e split..col_max)
 */
static int find_balanced_split(const float* load_map, int cols,
                                int r0, int r1, int c0, int c1,
                                int axis, float target_half) {
    float cumulative = 0.f;
    if (axis == 0) {
        for (int r = r0; r < r1 - 1; r++) {
            cumulative += region_load(load_map, cols, r, r+1, c0, c1);
            if (cumulative >= target_half) return r + 1;
        }
        return (r0 + r1) / 2;
    } else {
        for (int c = c0; c < c1 - 1; c++) {
            cumulative += region_load(load_map, cols, r0, r1, c, c+1);
            if (cumulative >= target_half) return c + 1;
        }
        return (c0 + c1) / 2;
    }
}

/*
 * Constrói a k-d tree recursivamente.
 * n_partitions: potência de 2 (ex: 4 → 2 níveis de divisão)
 * depth: nível atual (usado para alternar eixo de corte)
 */
KDNode* kdtree_build(const float* load_map, int rows, int cols,
                     int n_partitions, int r0, int r1,
                     int c0, int c1, int depth, int* next_id) {
    KDNode* node = calloc(1, sizeof(KDNode));

    float total_load = region_load(load_map, cols, r0, r1, c0, c1);

    if (n_partitions <= 1 || r1 - r0 <= 1 || c1 - c0 <= 1) {
        /* Folha */
        node->is_leaf      = 1;
        node->part.row_min = r0;
        node->part.row_max = r1;
        node->part.col_min = c0;
        node->part.col_max = c1;
        node->part.load    = total_load;
        node->part.id      = (*next_id)++;
        return node;
    }

    /* Escolher eixo: alternar entre linhas e colunas por profundidade */
    int axis = depth % 2;

    /* Se a região for muito estreita num eixo, forçar o outro */
    if (axis == 0 && (r1 - r0) < 2) axis = 1;
    if (axis == 1 && (c1 - c0) < 2) axis = 0;

    int split = find_balanced_split(load_map, cols, r0, r1, c0, c1,
                                    axis, total_load / 2.f);

    node->is_leaf     = 0;
    node->split_axis  = axis;
    node->split_pos   = split;

    int half = n_partitions / 2;
    if (axis == 0) {
        node->left  = kdtree_build(load_map, rows, cols, half,
                                   r0, split, c0, c1, depth+1, next_id);
        node->right = kdtree_build(load_map, rows, cols, n_partitions - half,
                                   split, r1, c0, c1, depth+1, next_id);
    } else {
        node->left  = kdtree_build(load_map, rows, cols, half,
                                   r0, r1, c0, split, depth+1, next_id);
        node->right = kdtree_build(load_map, rows, cols, n_partitions - half,
                                   r0, r1, split, c1, depth+1, next_id);
    }

    return node;
}

void kdtree_get_leaves(KDNode* node, Partition* parts, int* count) {
    if (!node) return;
    if (node->is_leaf) {
        parts[(*count)++] = node->part;
        return;
    }
    kdtree_get_leaves(node->left,  parts, count);
    kdtree_get_leaves(node->right, parts, count);
}

void kdtree_free(KDNode* node) {
    if (!node) return;
    kdtree_free(node->left);
    kdtree_free(node->right);
    free(node);
}

void kdtree_print(KDNode* node, int depth) {
    if (!node) return;
    char indent[64] = {0};
    for (int i = 0; i < depth*2 && i < 62; i++) indent[i] = ' ';
    if (node->is_leaf) {
        printf("%sLeaf #%d  rows[%d,%d) cols[%d,%d)  load=%.1f\n",
               indent, node->part.id,
               node->part.row_min, node->part.row_max,
               node->part.col_min, node->part.col_max,
               node->part.load);
    } else {
        printf("%sSplit axis=%s pos=%d\n",
               indent, node->split_axis==0?"row":"col", node->split_pos);
        kdtree_print(node->left,  depth+1);
        kdtree_print(node->right, depth+1);
    }
}
