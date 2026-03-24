#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include <stdbool.h>

#define EPSILON 1e-10
#define MAX_ITERATIONS 100

bool mollweide_forward(double lon, double lat, double radius, double lon_center,
                       double *x, double *y);
bool mollweide_inverse(double x, double y, double radius, double lon_center,
                       double *lon, double *lat);

#endif
