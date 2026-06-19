#include <math.h>
#include <stddef.h>
#include "rothermel.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const FuelModel FM[] = {
    { 1,0.166,0,0,0,        11483,358,98,0,    0.305,0.12,18622},
    { 2,0.898,0.449,0,0.449,11483,358,98,4921, 0.305,0.15,18622},
    { 3,0.675,0,0,0,        11483,358,98,0,    0.762,0.25,18622},
    { 4,2.246,1.123,0.561,0,6562,358,98,0,     1.829,0.20,18622},
    { 5,0.225,0,0,0.449,    6562,358,98,6562,  0.610,0.20,18622},
    { 6,0.785,0.674,0.337,0,6562,358,98,0,     0.762,0.25,18622},
    { 7,0.561,0.449,0.337,0,6562,358,98,0,     0.762,0.40,18622},
    { 8,0.336,0.561,1.123,0,11483,358,98,0,    0.061,0.30,18622},
    { 9,0.336,0.112,0.112,0,13123,358,98,0,    0.061,0.25,18622},
    {10,0.561,1.123,1.123,0,8202,358,98,0,     0.305,0.25,18622},
    {11,0.785,1.123,1.571,0,6562,358,98,0,     0.305,0.15,18622},
    {12,3.370,2.809,2.246,0,6562,358,98,0,     0.762,0.20,18622},
    {13,7.415,3.370,2.246,0,6562,358,98,0,     0.914,0.25,18622},
};
#define NFM (int)(sizeof(FM)/sizeof(FM[0]))

const FuelModel* get_fuel_model(int id) {
    for(int i=0;i<NFM;i++) if(FM[i].model_id==id) return &FM[i];
    return &FM[0];
}

static float moisture_damping(float m, float mx) {
    if(mx<=0||m>=mx) return 0;
    float r=m/mx; return 1-2.59f*r+5.11f*r*r-3.52f*r*r*r;
}

float rothermel_ros(int fuel_model_id, float slope_pct, float aspect_deg, const Weather* wx) {
    (void)aspect_deg;
    const FuelModel* fm = get_fuel_model(fuel_model_id);
    float sv=fm->sv_1h, bd=(fm->w0_1h+fm->w0_10h+fm->w0_100h+fm->w0_herb)/(fm->delta>0?fm->delta:1);
    float beta=bd/32.0f, bop=3.348f*powf(sv,-0.8189f);
    if(sv<=0||bop<=0||fm->delta<=0) return 0;
    float A=133.0f*powf(sv,-0.7913f);
    float gmax=powf(sv,1.5f)/(495+0.594f*powf(sv,1.5f));
    float fb=powf(beta/bop,A)*expf(A*(1-beta/bop));
    float ir=gmax*fb*(fm->w0_1h*fm->heat)*moisture_damping(wx->m1h,fm->mx)*0.417439f;
    float xi=expf((0.792f+0.681f*sqrtf(sv))*(beta+0.1f))/(192+0.2595f*sv);
    float wf=0,ws=0;
    float wf_ftmin=wx->wind_speed_ms*196.85f;
    float C=7.47f*expf(-0.133f*powf(sv,0.55f)), B=0.02526f*powf(sv,0.54f), E=0.715f*expf(-3.59e-4f*sv);
    if(bop>0) wf=C*powf(wf_ftmin+1,B)*powf(beta/bop,-E);
    float tan_phi=slope_pct/100.0f;
    if(bop>0) ws=5.275f*powf(beta/bop,-0.3f)*tan_phi*tan_phi;
    float rho_b=bd, Q=250+1116*wx->m1h;
    if(rho_b*Q<=0) return 0;
    float R=(ir*xi*(1+wf+ws))/(rho_b*Q);
    return (R<0)?0:R*0.00508f;
}
