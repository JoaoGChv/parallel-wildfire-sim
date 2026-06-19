#include <stdlib.h>
#include <stdio.h>
#include "priority_queue.h"

PriorityQueue* pq_create(int capacity) {
    PriorityQueue* pq = malloc(sizeof(PriorityQueue));
    pq->data = malloc(capacity * sizeof(PQNode));
    pq->size = 0; pq->capacity = capacity;
    return pq;
}
void pq_destroy(PriorityQueue* pq) { free(pq->data); free(pq); }

static void swap(PQNode* a, PQNode* b) { PQNode t=*a; *a=*b; *b=t; }
static void sift_up(PriorityQueue* pq, int i) {
    while (i>0) { int p=(i-1)/2; if(pq->data[p].time<=pq->data[i].time) break; swap(&pq->data[p],&pq->data[i]); i=p; }
}
static void sift_down(PriorityQueue* pq, int i) {
    int n=pq->size;
    while(1){ int s=i,l=2*i+1,r=2*i+2;
        if(l<n&&pq->data[l].time<pq->data[s].time) s=l;
        if(r<n&&pq->data[r].time<pq->data[s].time) s=r;
        if(s==i) break; swap(&pq->data[i],&pq->data[s]); i=s; }
}
void pq_push(PriorityQueue* pq, float time, int row, int col) {
    if(pq->size>=pq->capacity){ pq->capacity*=2; pq->data=realloc(pq->data,pq->capacity*sizeof(PQNode)); }
    pq->data[pq->size]=(PQNode){time,row,col}; sift_up(pq,pq->size++);
}
PQNode pq_pop(PriorityQueue* pq) {
    PQNode top=pq->data[0]; pq->size--;
    if(pq->size>0){ pq->data[0]=pq->data[pq->size]; sift_down(pq,0); }
    return top;
}
int pq_empty(const PriorityQueue* pq) { return pq->size==0; }
