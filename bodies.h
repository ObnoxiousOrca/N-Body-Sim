
#ifndef BODIES_H
#define BODIES_H

#define ASTEROID_COUNT 10000
#define SATELLITE_COUNT 13
#define PLANET_COUNT 8

#define N (1 + PLANET_COUNT + SATELLITE_COUNT + ASTEROID_COUNT)

typedef struct Bodies {
    double *x;
    double *y;
    double *vx;
    double *vy;
    double *mass;
    double *ax;
    double *ay;
} Bodies;

#endif
