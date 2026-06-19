#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H

typedef struct { float time; int row; int col; } PQNode;
typedef struct { PQNode* data; int size; int capacity; } PriorityQueue;

PriorityQueue* pq_create(int capacity);
void           pq_destroy(PriorityQueue* pq);
void           pq_push(PriorityQueue* pq, float time, int row, int col);
PQNode         pq_pop(PriorityQueue* pq);
int            pq_empty(const PriorityQueue* pq);

#endif
