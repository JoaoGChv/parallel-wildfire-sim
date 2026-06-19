#ifndef KDTREE_H
#define KDTREE_H
/*
 * kdtree.h — Particionamento k-d tree balanceado para o grid 450×450
 *
 * A k-d tree divide o grid recursivamente alternando cortes horizontais
 * e verticais. O critério de balanceamento usa a carga computacional
 * prevista pela CNN (ou simplesmente o número de células queimáveis).
 *
 * Cada folha da árvore é uma partição que será atribuída a um nó/thread.
 */

#define MAX_PARTITIONS 32

typedef struct {
    int row_min, row_max;   /* [row_min, row_max) */
    int col_min, col_max;   /* [col_min, col_max) */
    float load;             /* carga computacional prevista (CNN) */
    int   id;
} Partition;

typedef struct KDNode {
    int is_leaf;
    Partition part;         /* válido se is_leaf */
    int split_axis;         /* 0=horizontal(row), 1=vertical(col) */
    int split_pos;
    struct KDNode* left;
    struct KDNode* right;
} KDNode;

/* API */
KDNode* kdtree_build(const float* load_map, int rows, int cols,
                     int n_partitions, int row_min, int row_max,
                     int col_min, int col_max, int depth, int* next_id);
void    kdtree_get_leaves(KDNode* node, Partition* parts, int* count);
void    kdtree_free(KDNode* node);
void    kdtree_print(KDNode* node, int depth);
float   region_load(const float* load_map, int cols,
                    int r0, int r1, int c0, int c1);

#endif
