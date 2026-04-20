#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include <stdbool.h>

#define EPSILON 1e-10
#define MAX_ITERATIONS 100

bool mollweide_forward(double lon, double lat, double radius, double lon_center,
                       double *x, double *y);
bool mollweide_inverse(double x, double y, double radius, double lon_center,
                       double *lon, double *lat);
bool equal_earth_forward(double lon, double lat, double radius, double lon_center,
                         double *x, double *y);
bool equal_earth_inverse(double x, double y, double radius, double lon_center,
                         double *lon, double *lat);
bool mbtfpq_forward(double lon, double lat, double radius, double lon_center,
                    double *x, double *y);
bool mbtfpq_inverse(double x, double y, double radius, double lon_center,
                    double *lon, double *lat);
bool hammer_forward(double lon, double lat, double radius, double lon_center,
                    double *x, double *y);
bool hammer_inverse(double x, double y, double radius, double lon_center,
                    double *lon, double *lat);
bool laea_forward(double lon, double lat, double radius,
                  double lon_center, double lat_center,
                  double *x, double *y);
bool laea_inverse(double x, double y, double radius,
                  double lon_center, double lat_center,
                  double *lon, double *lat);
#endif
