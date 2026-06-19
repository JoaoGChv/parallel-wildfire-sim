/*
 * simulator.c — Propagação sequencial de incêndio (Rothermel + Dijkstra/Huygens)
 *
 * Input:  CSV com colunas layer_0..layer_13 (dados semânticos normalizados)
 * Output: CSV com row,col,time_of_arrival_s + tempo de execução medido
 *
 * Uso:
 *   ./simulator --input map.csv --output toa.csv [--ignition-row 225] [--ignition-col 225]
 *               [--wind-speed 5.0] [--wind-dir 270] [--runs 3]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "rothermel.h"
#include "priority_queue.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const int   DR[8] = {-1,-1,-1, 0, 0, 1, 1, 1};
static const int   DC[8] = {-1, 0, 1,-1, 1,-1, 0, 1};
static const float DIST[8]= {42.426f,30.f,42.426f,30.f,30.f,42.426f,30.f,42.426f};
static const float ANG[8] = {315.f,0.f,45.f,270.f,90.f,225.f,180.f,135.f};

typedef struct { Cell cells[MAX_ROWS][MAX_COLS]; float toa[MAX_ROWS][MAX_COLS]; int rows,cols; } Map;

static int load_csv(const char* path, Map* m) {
    FILE* f=fopen(path,"r"); if(!f){fprintf(stderr,"Não abre: %s\n",path);return -1;}
    char line[8192]; int cell_idx=0;
    if(!fgets(line,sizeof(line),f)){fclose(f);return -1;} /* header */
    while(fgets(line,sizeof(line),f)&&cell_idx<MAX_ROWS*MAX_COLS) {
        int row=cell_idx/MAX_COLS, col=cell_idx%MAX_COLS, layer=0; char* tok=strtok(line,",");
        while(tok&&layer<N_LAYERS){
            m->cells[row][col].layers[layer]=atof(tok);
            tok=strtok(NULL,","); layer++;
        }
        m->cells[row][col].burned=0; m->cells[row][col].time_burned=FLT_MAX;
        m->toa[row][col]=FLT_MAX; cell_idx++;
    }
    m->rows=(cell_idx>0)?MAX_ROWS:0; m->cols=(cell_idx>0)?MAX_COLS:0; fclose(f); return 0;
}

static void save_toa(const char* path, const Map* m) {
    FILE* f=fopen(path,"w"); if(!f){fprintf(stderr,"Não salva: %s\n",path);return;}
    fprintf(f,"row,col,time_of_arrival_s\n");
    for(int r=0;r<m->rows;r++) for(int c=0;c<m->cols;c++)
        if(m->toa[r][c]<FLT_MAX) fprintf(f,"%d,%d,%.2f\n",r,c,m->toa[r][c]);
    fclose(f);
}

static int get_fuel(const Cell* cell) {
    /* FBFM13 (Anderson): 1-13 = modelos queimáveis; 0 e 91-99
       (urbano, neve, agrícola, água, rocha) = não-queimável. */
    int id=(int)(cell->layers[IDX_FBFM13]+0.5f);
    if (id<1 || id>13) return 0;   /* inclui 91-99 → não-queimável */
    return id;
}

static float G_WIND_ANISO=0.5f;   /* anisotropia direcional do vento (--wind-aniso) */
static float dir_ros(float ros_max, float wind_dir, float spread_angle) {
    float c=G_WIND_ANISO;
    float d=(spread_angle-wind_dir)*(float)M_PI/180.f;
    float ros=ros_max*(1.f+c*cosf(d))/(1.f+c);
    return (ros<0.01f)?0.01f:ros;
}

static void simulate(Map* m, int ir, int ic, float ws_ms, float wd_deg, float m1h, float m10h, float m100h, float ros_scale) {
    for(int r=0;r<m->rows;r++) for(int c=0;c<m->cols;c++) m->toa[r][c]=FLT_MAX;
    PriorityQueue* pq=pq_create(MAX_ROWS*MAX_COLS);
    m->toa[ir][ic]=0; m->cells[ir][ic].burned=1; pq_push(pq,0,ir,ic);
    Weather wx={ws_ms,wd_deg,m1h,m10h,m100h,0.6f};
    while(!pq_empty(pq)){
        PQNode nd=pq_pop(pq); float t=nd.time; int r=nd.row,c=nd.col;
        if(t>m->toa[r][c]+1e-6f) continue;
        int fuel=get_fuel(&m->cells[r][c]);
        if(fuel<=0) continue;  /* não-queimável */
        float slope_pct=m->cells[r][c].layers[IDX_SLOPE];
        float ros_max=rothermel_ros(fuel,slope_pct,m->cells[r][c].layers[IDX_ASPECT],&wx)*ros_scale;
        if(ros_max<1e-6f) continue;
        for(int d=0;d<8;d++){
            int nr=r+DR[d], nc=c+DC[d];
            if(nr<0||nr>=m->rows||nc<0||nc>=m->cols) continue;
            float ros_d=dir_ros(ros_max,wd_deg,ANG[d]);
            if(ros_d<1e-6f) continue;
            float new_toa=t+DIST[d]/ros_d;
            if(new_toa<m->toa[nr][nc]){
                m->toa[nr][nc]=new_toa; m->cells[nr][nc].burned=1; pq_push(pq,new_toa,nr,nc);
            }
        }
    }
    pq_destroy(pq);
}

static void usage(const char* p){
    fprintf(stderr,"Uso: %s --input <csv> --output <csv> [--ignition-row N] [--ignition-col N]\n"
        "         [--wind-speed <m/s>] [--wind-dir <deg>] [--runs N]\n",p);
}

int main(int argc, char* argv[]) {
    const char *in=NULL,*out=NULL;
    int ir=225,ic=225,runs=3;
    float ws=5.f,wd=180.f,m1=0.05f,m10=0.08f,m100=0.12f,ros_scale=1.0f;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--input")&&i+1<argc)       in=argv[++i];
        else if(!strcmp(argv[i],"--output")&&i+1<argc) out=argv[++i];
        else if(!strcmp(argv[i],"--ignition-row")&&i+1<argc) ir=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ignition-col")&&i+1<argc) ic=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--wind-speed")&&i+1<argc)   ws=atof(argv[++i]);
        else if(!strcmp(argv[i],"--wind-dir")&&i+1<argc)     wd=atof(argv[++i]);
        else if(!strcmp(argv[i],"--moisture-1h")&&i+1<argc)  m1=atof(argv[++i]);
        else if(!strcmp(argv[i],"--moisture-10h")&&i+1<argc) m10=atof(argv[++i]);
        else if(!strcmp(argv[i],"--moisture-100h")&&i+1<argc)m100=atof(argv[++i]);
        else if(!strcmp(argv[i],"--ros-scale")&&i+1<argc)    ros_scale=atof(argv[++i]);
        else if(!strcmp(argv[i],"--wind-aniso")&&i+1<argc)   G_WIND_ANISO=atof(argv[++i]);
        else if(!strcmp(argv[i],"--runs")&&i+1<argc)         runs=atoi(argv[++i]);
    }
    if(!in||!out){usage(argv[0]);return 1;}

    static Map map; memset(&map,0,sizeof(map));
    printf("Carregando: %s\n",in);
    if(load_csv(in,&map)<0) return 1;
    printf("Mapa: %d×%d células\n",map.rows,map.cols);

    double total=0;
    for(int run=0;run<runs;run++){
        struct timespec t0,t1;
        clock_gettime(CLOCK_MONOTONIC,&t0);
        simulate(&map,ir,ic,ws,wd,m1,m10,m100,ros_scale);
        clock_gettime(CLOCK_MONOTONIC,&t1);
        double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
        total+=el; printf("  Run %d: %.4f s\n",run+1,el);
    }
    double avg=total/runs;
    printf("Tempo médio: %.4f s (%d runs)\n",avg,runs);

    int burned=0;
    for(int r=0;r<map.rows;r++) for(int c=0;c<map.cols;c++) if(map.toa[r][c]<FLT_MAX) burned++;
    printf("Células atingidas: %d / %d (%.1f%%)\n",burned,map.rows*map.cols,100.0*burned/(map.rows*map.cols));

    save_toa(out,&map);
    printf("TOA salvo: %s\n",out);
    return 0;
}
