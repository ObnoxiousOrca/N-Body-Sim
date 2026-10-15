
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include "bodies.h"

#ifdef __cplusplus
extern "C" {
#endif

#define G 6.67430e-11
#define M_SUN 1.98847e30
#define AU 1.495978707e11
#define PI 3.14159265358979323846
void initialize_bodies(Bodies *B);

#ifdef __cplusplus
}
#endif


#endif
