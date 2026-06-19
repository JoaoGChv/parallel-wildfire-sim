#ifndef ROTHERMEL_H
#define ROTHERMEL_H

#define MAX_ROWS 450
#define MAX_COLS 450
#define N_LAYERS 14
#define CELL_SIZE_M 30.0

/* Índices conforme LAYER_ORDER do preprocess.py:
   0=elevation 1=aspect 2=slope 3=fbfm40 4=cc 5=ch 6=cbh 7=cbd */
#define IDX_ELEVATION 0
#define IDX_ASPECT    1
#define IDX_SLOPE     2
#define IDX_FBFM13    3   /* fbfm40 na nossa pipeline */

typedef struct { float layers[N_LAYERS]; int burned; float time_burned; } Cell;
typedef struct { float wind_speed_ms; float wind_dir_deg; float m1h; float m10h; float m100h; float mlh; } Weather;
typedef struct {
    int model_id; float w0_1h,w0_10h,w0_100h,w0_herb;
    float sv_1h,sv_10h,sv_100h,sv_herb; float delta,mx,heat;
} FuelModel;

const FuelModel* get_fuel_model(int model_id);
float rothermel_ros(int fuel_model_id, float slope_pct, float aspect_deg, const Weather* wx);

#endif
