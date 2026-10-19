
#include "init.h"

#include <math.h>
#include <stdlib.h>

#define G 6.67430e-11
#define M_SUN 1.98847e30
#define AU 1.495978707e11
#define PI 3.14159265358979323846

enum {
    SUN = 0,
    MERCURY, VENUS, EARTH, MARS,
    JUPITER, SATURN, URANUS, NEPTUNE
};

typedef struct {
    double mass;
    double a;       /* Semi-major axis in metres */
    double phase;   /* Initial orbital angle */
} PlanetData;

typedef struct {
    int parent;
    double mass;
    double radius;  /* Approximate orbital radius around parent */
    double phase;
} SatelliteData;

static const PlanetData planets[PLANET_COUNT] = {
    {3.3011e23,  0.3871 * AU, 0.4},  /* Mercury */
    {4.8675e24,  0.7233 * AU, 1.2},  /* Venus */
    {5.9722e24,  1.0000 * AU, 2.1},  /* Earth */
    {6.4171e23,  1.5237 * AU, 3.0},  /* Mars */
    {1.8982e27,  5.2028 * AU, 4.1},  /* Jupiter */
    {5.6834e26,  9.5388 * AU, 5.0},  /* Saturn */
    {8.6810e25, 19.1914 * AU, 5.7},  /* Uranus */
    {1.02413e26, 30.0611 * AU, 0.8}  /* Neptune */
};

/* Moon masses and approximate orbital radii.
   Parent indices refer to the planet enum above. */
static const SatelliteData satellites[SATELLITE_COUNT] = {
    {EARTH,   7.342e22, 3.844e8,  0.3}, /* Moon */
    {MARS,    1.066e16, 9.376e6,  1.1}, /* Phobos */
    {MARS,    1.476e15, 2.346e7,  2.0}, /* Deimos */

    {JUPITER, 8.932e22, 4.217e8,  0.5}, /* Io */
    {JUPITER, 4.800e22, 6.711e8,  1.4}, /* Europa */
    {JUPITER, 1.482e23, 1.0704e9, 2.4}, /* Ganymede */
    {JUPITER, 1.076e23, 1.8827e9, 3.5}, /* Callisto */

    {SATURN,  1.345e23, 1.22187e9, 0.7}, /* Titan */
    {SATURN,  1.080e20, 2.380e8,   1.8}, /* Enceladus */
    {SATURN,  2.306e21, 5.270e8,   2.9}, /* Rhea */

    {URANUS,  3.527e21, 4.360e8,   0.9}, /* Titania */
    {URANUS,  3.014e21, 5.840e8,   2.2}, /* Oberon */

    {NEPTUNE, 2.140e22, 3.5476e8,  1.6}  /* Triton */
};

/* Uniform random number in (0, 1). */
static double random_unit(void)
{
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

/* Exponential distribution with the given mean. */
static double exponential_mass(double mean_mass)
{
    return -mean_mass * log(1.0 - random_unit());
}

void initialize_bodies(Bodies *B)
{
    int i;
    int index;

    /* Sun */
    B->x[SUN] = 0.0;
    B->y[SUN] = 0.0;
    B->vx[SUN] = 0.0;
    B->vy[SUN] = 0.0;
    B->mass[SUN] = M_SUN;
    B->ax[SUN] = 0.0;
    B->ay[SUN] = 0.0;

    /* Planets */
    for (i = 0; i < PLANET_COUNT; i++) {
        int p = i + 1;

        double r = planets[i].a;
        double theta = planets[i].phase;
        double speed = sqrt(G * M_SUN / r);

        B->x[p] = r * cos(theta);
        B->y[p] = r * sin(theta);

        /* Velocity perpendicular to the radius */
        B->vx[p] = -speed * sin(theta);
        B->vy[p] =  speed * cos(theta);

        B->mass[p] = planets[i].mass;
        B->ax[p] = 0.0;
        B->ay[p] = 0.0;
    }

    /* Primary satellites */
    index = 1 + PLANET_COUNT;

    for (i = 0; i < SATELLITE_COUNT; i++) {
        int parent = satellites[i].parent;

        double r = satellites[i].radius;
        double theta = satellites[i].phase;

        double parent_mass = B->mass[parent];

        /* Approximate orbital speed around the parent */
        double speed = sqrt(G * parent_mass / r);

        double rel_x = r * cos(theta);
        double rel_y = r * sin(theta);

        double rel_vx = -speed * sin(theta);
        double rel_vy =  speed * cos(theta);

        /* Add the parent planet's heliocentric state */
        B->x[index] = B->x[parent] + rel_x;
        B->y[index] = B->y[parent] + rel_y;

        B->vx[index] = B->vx[parent] + rel_vx;
        B->vy[index] = B->vy[parent] + rel_vy;

        B->mass[index] = satellites[i].mass;
        B->ax[index] = 0.0;
        B->ay[index] = 0.0;

        index++;
    }

    /* Asteroid belt: approximately 2.1 to 3.3 AU */
    for (i = 0; i < ASTEROID_COUNT; i++) {
        double r = (2.1 + 1.2 * random_unit()) * AU;
        double theta = 2.0 * PI * random_unit();

        /* Random mass drawn from an exponential distribution.
           Mean asteroid mass = 1e12 kg. */
        double mass = exponential_mass(1.0e12);

        /* Nearly circular orbit around the Sun */
        double speed = sqrt(G * M_SUN / r);

        B->x[index] = r * cos(theta);
        B->y[index] = r * sin(theta);

        B->vx[index] = -speed * sin(theta);
        B->vy[index] =  speed * cos(theta);

        B->mass[index] = mass;
        B->ax[index] = 0.0;
        B->ay[index] = 0.0;

        index++;
    }

    /*
     * Give the Sun the velocity needed to make the total
     * initial linear momentum approximately zero.
     */
    {
        double px = 0.0;
        double py = 0.0;

        for (i = 1; i < N; i++) {
            px += B->mass[i] * B->vx[i];
            py += B->mass[i] * B->vy[i];
        }

        B->vx[SUN] = -px / B->mass[SUN];
        B->vy[SUN] = -py / B->mass[SUN];
    }
}
