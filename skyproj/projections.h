#ifndef _PROJECTIONS_H
#define _PROJECTIONS_H

#include <stdbool.h>

#define EPSILON 1e-10
#define MAX_ITERATIONS 100

/**
 * Precomputed Albers Equal-Area Conic constants
 */
typedef struct {
    double lon_center;
    double lat1;
    double lat2;
    double n;
    double n_inv;
    double C;
    double rho0;
    bool n_negative;
} albers_params_t;

typedef struct {
    double lon_0;       /* central meridian (geographic), subtracted first */
    double lamp;        /* lon_p */
    double sphip;       /* sin(lat_p) */
    double cphip;       /* cos(lat_p) */
} oblique_mollweide_params_t;

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
bool gnomonic_forward(double lon, double lat, double radius,
                      double lon_center, double lat_center,
                      double *x, double *y);
bool gnomonic_inverse(double x, double y, double radius,
                      double lon_center, double lat_center,
                      double *lon, double *lat);
bool albers_init(albers_params_t *params,
                 double lon_center,
                 double lat1, double lat2);
bool albers_forward(const albers_params_t *params,
                    double lon, double lat, double radius,
                    double *x, double *y);
bool albers_inverse(const albers_params_t *params,
                    double x, double y, double radius,
                    double *lon, double *lat);
bool oblique_mollweide_init(oblique_mollweide_params_t *params,
                            double lon_p, double lat_p,
                            double lon_0);
bool oblique_mollweide_forward(const oblique_mollweide_params_t *params,
                               double lon, double lat, double radius,
                               double *x, double *y);
bool oblique_mollweide_inverse(const oblique_mollweide_params_t *params,
                               double x, double y, double radius,
                               double *lon, double *lat);
#endif
