#include <math.h>
#include <stdbool.h>
#include "skyproj.h"

/**
 * Compute the central angle between two points on a sphere
 * using the Vincenty formula (stable for all distances)
 */
static double central_angle(double lat0, double lon0, double lat1, double lon1) {
    double sin_lat0 = sin(lat0);
    double cos_lat0 = cos(lat0);
    double sin_lat1 = sin(lat1);
    double cos_lat1 = cos(lat1);
    double dlon = lon1 - lon0;
    double cos_dlon = cos(dlon);
    double sin_dlon = sin(dlon);

    double a = cos_lat1 * sin_dlon;
    double b = cos_lat0 * sin_lat1 - sin_lat0 * cos_lat1 * cos_dlon;
    double c = sin_lat0 * sin_lat1 + cos_lat0 * cos_lat1 * cos_dlon;

    return atan2(sqrt(a * a + b * b), c);
}

/**
 * Compute a point along the great circle from (lat0,lon0) to (lat1,lon1)
 * at fractional distance f (0 = start, 1 = end)
 *
 * Uses the spherical interpolation (slerp) formula:
 *   P(f) = sin((1-f)*d)/sin(d) * P0 + sin(f*d)/sin(d) * P1
 *
 * where d is the central angle between P0 and P1
 */
static void great_circle_point(double lat0, double lon0,
                               double lat1, double lon1,
                               double d, double sin_d,
                               double f,
                               double *lat_out, double *lon_out) {
    /* Handle zero or near-zero distance */
    if (sin_d < 1e-15) {
        *lat_out = lat0;
        *lon_out = lon0;
        return;
    }

    double a = sin((1.0 - f) * d) / sin_d;
    double b = sin(f * d) / sin_d;

    /* Convert to Cartesian, interpolate, convert back */
    double cos_lat0 = cos(lat0);
    double cos_lat1 = cos(lat1);

    double x = a * cos_lat0 * cos(lon0) + b * cos_lat1 * cos(lon1);
    double y = a * cos_lat0 * sin(lon0) + b * cos_lat1 * sin(lon1);
    double z = a * sin(lat0) + b * sin(lat1);

    *lat_out = atan2(z, sqrt(x * x + y * y));
    *lon_out = atan2(y, x);
}

/**
 * Generate points along a geodesic (great circle) between two points
 *
 * @param lon0          Start longitude in degrees
 * @param lat0          Start latitude in degrees
 * @param lon1          End longitude in degrees
 * @param lat1          End latitude in degrees
 * @param radius        Sphere radius
 * @param npts          Number of points to generate
 * @param include_start Include the start point (0 or 1)
 * @param include_end   Include the end point (0 or 1)
 * @param degrees       If true, output in degrees; if false, radians
 * @param lonlat_data   Output array of size npts*2: [lon0,lat0,lon1,lat1,...]
 * @return true if successful
 */
bool geod_interp_sp(double lon0, double lat0,
                    double lon1, double lat1,
                    double radius,
                    int npts, int include_start, int include_end,
                    int degrees,
                    double *lonlat_data) {
    if (lonlat_data == NULL || npts <= 0 || radius <= 0) {
        return false;
    }

    (void)radius;

    double lon0_r = lon0 * SP_D2R;
    double lat0_r = lat0 * SP_D2R;
    double lon1_r = lon1 * SP_D2R;
    double lat1_r = lat1 * SP_D2R;

    double d = central_angle(lat0_r, lon0_r, lat1_r, lon1_r);
    double sin_d = sin(d);

    int npts_stepsize = npts;
    int offset = 0;

    if (!include_start && include_end) {
        offset = 1;
    } else if (include_start && include_end) {
        npts_stepsize -= 1;
    } else if (!include_start && !include_end) {
        npts_stepsize += 1;
        offset = 1;
    }

    if (npts_stepsize <= 0) npts_stepsize = 1;

    double stepsize = 1.0 / (double)npts_stepsize;
    double conv = degrees ? SP_R2D : 1.0;

    for (int i = 0; i < npts; i++) {
        double f = (i + offset) * stepsize;
        double lat_out, lon_out;

        great_circle_point(lat0_r, lon0_r, lat1_r, lon1_r,
                           d, sin_d, f,
                           &lat_out, &lon_out);

        lonlat_data[i * 2]     = lon_out * conv;
        lonlat_data[i * 2 + 1] = lat_out * conv;
    }

    return true;
}

/**
 * Solve the geodesic direct problem on a sphere.
 *
 * Given a starting point, azimuth, and distance, find the endpoint.
 *
 * @param lon0      Start longitude in radians
 * @param lat0      Start latitude in radians
 * @param azimuth   Azimuth (bearing) in radians, clockwise from north
 * @param distance  Distance along the sphere surface
 * @param radius    Sphere radius
 * @param degrees   If true, output in degrees; if false, radians
 * @param lon_out   Output longitude in radians
 * @param lat_out   Output latitude in radians
 * @return true if successful
 */
bool geod_direct_sp(double lon0, double lat0,
                    double azimuth, double distance,
                    double radius,
                    int degrees,
                    double *lon_out, double *lat_out) {
    if (lon_out == NULL || lat_out == NULL || radius <= 0) {
        return false;
    }

    double lon0_r = lon0 * SP_D2R;
    double lat0_r = lat0 * SP_D2R;
    double az_r = azimuth * SP_D2R;

    double d = distance / radius;

    double sin_d = sin(d);
    double cos_d = cos(d);
    double sin_lat0 = sin(lat0_r);
    double cos_lat0 = cos(lat0_r);
    double sin_az = sin(az_r);
    double cos_az = cos(az_r);

    double sin_lat1 = sin_lat0 * cos_d + cos_lat0 * sin_d * cos_az;
    sin_lat1 = fmax(-1.0, fmin(1.0, sin_lat1));

    double lat1 = asin(sin_lat1);

    double y = sin_az * sin_d * cos_lat0;
    double x = cos_d - sin_lat0 * sin_lat1;
    double lon1 = lon0_r + atan2(y, x);

    lon1 = fmod(lon1, 2.0 * SP_PI);
    if (lon1 > SP_PI) lon1 -= 2.0 * SP_PI;
    if (lon1 < -SP_PI) lon1 += 2.0 * SP_PI;

    double conv = degrees ? SP_R2D : 1.0;
    *lon_out = lon1 * conv;
    *lat_out = lat1 * conv;

    return true;
}
