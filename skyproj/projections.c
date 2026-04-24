/*
  This code adapts projections from the OSGeo PROJ library with assistance from
  Claude Opus 4.6.

  The PROJ license is very permissive and I am replicating it here.

  ------

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/

#include <math.h>
#include <stdbool.h>

#include "skyproj.h"
#include "projections.h"

/**
 * Normalize angle to [-π, π] range (inclusive of -π, π)
 *
 * @param angle Input angle in radians
 * @return Normalized angle in [-π, π]
 */

/**
 * Normalize angle to [-π, π] using floor-based approach
 */
static double normalize_angle(double angle) {
    if (!isfinite(angle)) {
        return 0.0;
    }

    // Bring into approximate range
    angle = fmod(angle, 2.0 * SP_PI);

    // Ensure (-π, π] with correct boundary behavior
    if (angle > SP_PI) {
        angle -= 2.0 * SP_PI;
    } else if (angle < -SP_PI) {
        angle += 2.0 * SP_PI;
    }

    return angle;
}

/**
 * Adjust longitude to be within ±180° of central meridian
 * This matches PROJ library behavior
 */
static double adjust_lon(double lon) {
    // Normalize to [-π, π]
    if (!isfinite(lon)) {
        return 0.0;
    }

    lon = fmod(lon, 2.0 * SP_PI);

    if (lon > SP_PI) {
        lon -= 2.0 * SP_PI;
    } else if (lon < -SP_PI) {
        lon += 2.0 * SP_PI;
    }

    return lon;
}

/**
 * Calculate delta longitude the way PROJ does it
 * This maintains consistent behavior across the ±180° boundary
 */
static double delta_longitude(double lon, double lon_center) {
    // First, normalize both longitudes to [-π, π]
    lon = adjust_lon(lon);
    lon_center = adjust_lon(lon_center);

    // Compute difference
    double delta = lon - lon_center;

    // Adjust to ensure delta is in [-π, π]
    // This is the key: we normalize the DIFFERENCE, not compute shortest path
    while (delta > SP_PI) {
        delta -= 2.0 * SP_PI;
    }
    while (delta < -SP_PI) {
        delta += 2.0 * SP_PI;
    }

    return delta;
}

/* Plate Carree */

bool platecarree_forward(double lon, double lat, double radius, double lon_center, double *x,
                         double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, lon_center);

    *x = radius * delta_lon;
    *y = radius * lat;

    return true;
}

bool platecarree_inverse(double x, double y, double radius, double lon_center, double *lon,
                         double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    *lat = y / radius;

    if (*lat > SP_PI / 2.0) *lat = SP_PI / 2.0;
    if (*lat < -SP_PI / 2.0) *lat = -SP_PI / 2.0;

    double delta_lon = x / radius;
    if (delta_lon > SP_PI) delta_lon = SP_PI;
    if (delta_lon < -SP_PI) delta_lon = -SP_PI;

    *lon = normalize_angle(lon_center + delta_lon);

    return true;
}

/**
 * Improved initial guess for theta using a rational approximation
 * This significantly reduces iterations needed
 */
static double theta_initial_guess(double lat) {
    // For small latitudes, theta ≈ lat works well
    if (fabs(lat) < 0.1) {
        return lat;
    }

    // Better approximation for larger latitudes
    // Based on series expansion of the inverse function
    double sin_lat = sin(lat);
    double lat2 = lat * lat;

    // Rational approximation that's accurate to ~1e-6
    return lat * (1.0 - 0.16211 * lat2) / (1.0 + 0.05082 * lat2);
}

/**
 * Forward Mollweide projection
 * Converts latitude/longitude to x/y coordinates
 *
 * @param lon Longitude in radians [-π, π]
 * @param lat Latitude in radians [-π/2, π/2]
 * @param radius Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians (longitude at center of map)
 * @param x Output x coordinate
 * @param y Output y coordinate
 * @return true if successful, false otherwise
 */
bool mollweide_forward(double lon, double lat, double radius, double lon_center, double *x,
                       double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    // Calculate delta longitude from central meridian
    double delta_lon = delta_longitude(lon, lon_center);

    // Handle poles specially
    if (fabs(lat - SP_PI / 2) < EPSILON) {
        *x = 0.0;
        *y = sqrt(2.0) * radius;
        return true;
    }
    if (fabs(lat + SP_PI / 2) < EPSILON) {
        *x = 0.0;
        *y = -sqrt(2.0) * radius;
        return true;
    }

    // Solve for auxiliary angle theta using Newton-Raphson iteration
    // Equation: 2*theta + sin(2*theta) = pi*sin(lat)
    double theta = theta_initial_guess(lat);
    double target = SP_PI * sin(lat);

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double sin_2theta = sin(2.0 * theta);
        double cos_2theta = cos(2.0 * theta);

        double f = 2.0 * theta + sin_2theta - target;
        double df = 2.0 + 2.0 * cos_2theta;
        double d2f = -4.0 * sin_2theta;

        if (fabs(df) < EPSILON) break;

        // Halley's method: x_n+1 = x_n - 2*f*f' / (2*f'^2 - f*f'')
        double delta = 2.0 * f * df / (2.0 * df * df - f * d2f);
        theta -= delta;

        if (fabs(delta) < EPSILON) break;
    }

    // Calculate x and y coordinates
    // x = (2*sqrt(2)/π) * R * delta_lon * cos(theta)
    // y = sqrt(2) * R * sin(theta)
    *x = (2.0 * sqrt(2.0) / SP_PI) * radius * delta_lon * cos(theta);
    *y = sqrt(2.0) * radius * sin(theta);

    return true;
}

/**
 * Inverse Mollweide projection
 * Converts x/y coordinates back to latitude/longitude
 *
 * @param x Input x coordinate
 * @param y Input y coordinate
 * @param radius Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians (longitude at center of map)
 * @param lon Output longitude in radians [-π, π]
 * @param lat Output latitude in radians [-π/2, π/2]
 * @return true if successful, false if point is outside valid region
 */
/**
 * Optimized inverse with better numerical stability
 */
bool mollweide_inverse(double x, double y, double radius, double lon_center, double *lon,
                       double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    // Check for invalid inputs
    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    // Normalize coordinates by radius
    double x_scaled = x / radius;
    double y_scaled = y / radius;

    // Quick bounds check
    double abs_y = fabs(y_scaled);
    double sqrt2 = sqrt(2.0);

    if (abs_y > sqrt2 + EPSILON) {
        return false;
    }

    // More precise ellipse check
    double x_norm = x_scaled / (2.0 * sqrt2);
    double y_norm = y_scaled / sqrt2;
    double dist_sq = x_norm * x_norm + y_norm * y_norm;

    if (dist_sq > 1.0 + EPSILON) {
        return false;
    }

    // Clamp y to valid range to prevent asin domain errors
    if (y_scaled > sqrt2) y_scaled = sqrt2;
    if (y_scaled < -sqrt2) y_scaled = -sqrt2;

    // Calculate theta from y
    double theta = asin(y_scaled / sqrt2);
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);

    // Calculate latitude first (always works)
    double sin_2theta = 2.0 * sin_theta * cos_theta;
    double sin_lat = (2.0 * theta + sin_2theta) / SP_PI;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    // Handle poles and near-poles for longitude
    // Use a slightly larger threshold for safety
    if (fabs(cos_theta) < EPSILON * 10.0) {
        *lon = lon_center;
        return true;
    }

    // Calculate longitude with bounds checking
    double delta_lon = SP_PI * x_scaled / (2.0 * sqrt2 * cos_theta);

    // Clamp to valid range
    if (delta_lon > SP_PI) delta_lon = SP_PI;
    if (delta_lon < -SP_PI) delta_lon = -SP_PI;

    *lon = normalize_angle(lon_center + delta_lon);

    return true;
}

// Equal Earth projection parameters
// From: Šavrič, B., Patterson, T., & Jenny, B. (2019)
// International Journal of Geographical Information Science, 33(3), 454-465
#define EE_A1 1.340264
#define EE_A2 -0.081106
#define EE_A3 0.000893
#define EE_A4 0.003796
#define EE_M (sqrt(3.0) / 2.0)

/**
 * Forward Equal Earth projection
 * Converts latitude/longitude to x/y coordinates
 *
 * @param lon Longitude in radians [-π, π]
 * @param lat Latitude in radians [-π/2, π/2]
 * @param radius Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param x Output x coordinate
 * @param y Output y coordinate
 * @return true if successful, false otherwise
 */
bool equal_earth_forward(double lon, double lat, double radius, double lon_center, double *x,
                         double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double lambda = delta_longitude(lon, lon_center);

    double sin_theta = EE_M * sin(lat);
    sin_theta = fmax(-1.0, fmin(1.0, sin_theta));
    double theta = asin(sin_theta);

    double theta2 = theta * theta;
    double theta6 = theta2 * theta2 * theta2;
    double theta8 = theta6 * theta2;

    double cos_theta = cos(theta);

    /* dy/dtheta = A1 + 3*A2*theta^2 + 7*A3*theta^6 + 9*A4*theta^8 */
    double dy_dtheta =
        EE_A1 + 3.0 * EE_A2 * theta2 + 7.0 * EE_A3 * theta6 + 9.0 * EE_A4 * theta8;

    /* x = (2*sqrt(3) * R * lambda * cos(theta)) / (3 * dy_dtheta) */
    *x = (4.0 * EE_M * radius * lambda * cos_theta) / (3.0 * dy_dtheta);

    /* y = R * (A1*theta + A2*theta^3 + A3*theta^7 + A4*theta^9) */
    *y = radius * (EE_A1 * theta + EE_A2 * theta * theta2 + EE_A3 * theta * theta6 +
                   EE_A4 * theta * theta8);

    return true;
}

/**
 * Inverse Equal Earth projection
 * Converts x/y coordinates back to latitude/longitude
 *
 * @param x Input x coordinate
 * @param y Input y coordinate
 * @param radius Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lon Output longitude in radians
 * @param lat Output latitude in radians
 * @return true if successful, false if point is outside valid region
 */
bool equal_earth_inverse(double x, double y, double radius, double lon_center, double *lon,
                         double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    double y_scaled = y / radius;

    /* Initial guess */
    double theta = y_scaled / EE_A1;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double theta2 = theta * theta;
        double theta6 = theta2 * theta2 * theta2;
        double theta8 = theta6 * theta2;

        double f = EE_A1 * theta + EE_A2 * theta * theta2 + EE_A3 * theta * theta6 +
                   EE_A4 * theta * theta8 - y_scaled;

        double df = EE_A1 + 3.0 * EE_A2 * theta2 + 7.0 * EE_A3 * theta6 + 9.0 * EE_A4 * theta8;

        if (fabs(df) < EPSILON) break;

        double delta = f / df;
        theta -= delta;

        if (fabs(delta) < EPSILON) break;
    }

    /* Latitude: sin(phi) = 2*sin(theta)/sqrt(3) = sin(theta)/M */
    double sin_phi = sin(theta) / EE_M;
    sin_phi = fmax(-1.0, fmin(1.0, sin_phi));
    *lat = asin(sin_phi);

    /* Longitude */
    double cos_theta = cos(theta);
    if (fabs(cos_theta) < EPSILON) {
        *lon = lon_center;
        return true;
    }

    double theta2 = theta * theta;
    double theta6 = theta2 * theta2 * theta2;
    double theta8 = theta6 * theta2;

    double dy_dtheta =
        EE_A1 + 3.0 * EE_A2 * theta2 + 7.0 * EE_A3 * theta6 + 9.0 * EE_A4 * theta8;

    double lambda = 3.0 * x * dy_dtheta / (4.0 * EE_M * radius * cos_theta);

    *lon = normalize_angle(lon_center + lambda);

    return true;
}

/*
 * McBryde-Thomas Flat Polar Quartic Projection
 *
 * Pseudocylindrical equal-area projection.
 *
 * Auxiliary angle theta solved from:
 *   sin(theta/2) + sin(theta) = C_p * sin(lat)
 *
 * where C_p = 1 + sqrt(2)/2
 *
 * Forward:
 *   x = C_x * R * delta_lon * (1 + 2*cos(theta)/sqrt(3))
 *   y = C_y * R * sin(theta/2)
 *
 * Constants from PROJ (PJ_mbtfpq.c):
 *   C_x = 0.4052847345693511
 *   C_y = 1.1463476799243318
 *   C_p = 3.4141356237309505  (= 2 * (1 + sqrt(2)/2))
 */

#define MBTFPQ_C 1.70710678118654752440
#define MBTFPQ_RC 0.58578643762690495119
#define MBTFPQ_FYC 1.87475828462269495505
#define MBTFPQ_RYC 0.53340209679417701685
#define MBTFPQ_FXC 0.31245971410378249250
#define MBTFPQ_RXC 3.20041258076506210122

/**
 * Forward McBryde-Thomas Flat Polar Quartic projection
 *
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-π/2, π/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false otherwise
 */
bool mbtfpq_forward(double lon, double lat, double radius, double lon_center, double *x,
                    double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, lon_center);
    double sin_lat = sin(lat);
    double target = MBTFPQ_C * sin_lat;

    double theta = lat;

    double sin_h, cos_h;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double half_t = 0.5 * theta;
        sin_h = sin(half_t);
        cos_h = cos(half_t);

        double sin_t = 2.0 * sin_h * cos_h;
        double cos_t = 2.0 * cos_h * cos_h - 1.0;

        double dtheta = (sin_h + sin_t - target) / (0.5 * cos_h + cos_t);
        theta -= dtheta;
        if (fabs(dtheta) < EPSILON) break;
    }

    /* Recompute for final theta */
    double half_t = 0.5 * theta;
    sin_h = sin(half_t);
    cos_h = cos(half_t);

    /* cos(theta)/cos(theta/2) = 2*cos_h - 1/cos_h */
    double x_shape = 1.0 + 2.0 * (2.0 * cos_h - 1.0 / cos_h);

    *x = radius * MBTFPQ_FXC * delta_lon * x_shape;
    *y = radius * MBTFPQ_FYC * sin_h;

    return true;
}

/**
 * Inverse McBryde-Thomas Flat Polar Quartic projection
 *
 * @param x          Input x coordinate
 * @param y          Input y coordinate
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lon        Output longitude in radians
 * @param lat        Output latitude in radians
 * @return true if successful, false if point is outside valid region
 */
bool mbtfpq_inverse(double x, double y, double radius, double lon_center, double *lon,
                    double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    double sin_half_theta = y * MBTFPQ_RYC / radius;
    double theta;
    double t;

    if (fabs(sin_half_theta) > 1.0) {
        if (fabs(sin_half_theta) > 1.000001) {
            return false;
        }
        if (sin_half_theta < 0.0) {
            t = -1.0;
            theta = -SP_PI;
        } else {
            t = 1.0;
            theta = SP_PI;
        }
    } else {
        t = sin_half_theta;
        theta = 2.0 * asin(sin_half_theta);
    }

    /* Longitude */
    double cos_half = cos(0.5 * theta);
    double delta_lon;

    if (cos_half == 0.0) {
        delta_lon = 0.0;
    } else {
        double cos_theta = cos(theta);
        double shape = 1.0 + 2.0 * cos_theta / cos_half;
        delta_lon = x * MBTFPQ_RXC / (radius * shape);
    }

    *lon = normalize_angle(lon_center + delta_lon);

    /* Latitude */
    double sin_lat = MBTFPQ_RC * (t + sin(theta));
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    return true;
}

/*
 * Hammer-Aitoff Projection
 *
 * Equal-area projection derived from the azimuthal equal-area projection.
 *
 * Forward:
 *   Let w = sqrt(1 + cos(lat)*cos(lon/2))   (the "Aitoff weight")
 *   x = 2*sqrt(2) * R * cos(lat)*sin(lon/2) / w
 *   y = sqrt(2) * R * sin(lat) / w
 *
 * Inverse:
 *   z = sqrt(1 - (x/(4R))^2 - (y/(2R))^2)
 *   lon = 2 * atan2(z*x, 2*(2*z^2 - 1)) + lon_center
 *   lat = asin(z*y / R)          (with appropriate scaling)
 *
 * From PROJ PJ_hammer.c with W=0.5, M=1 (standard Hammer).
 */

/**
 * Forward Hammer-Aitoff projection
 *
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-pi/2, pi/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false otherwise
 */
bool hammer_forward(double lon, double lat, double radius, double lon_center, double *x,
                    double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, lon_center);

    double half_lon = 0.5 * delta_lon;
    double cos_lat = cos(lat);
    double sin_lat = sin(lat);
    double cos_half_lon = cos(half_lon);
    double sin_half_lon = sin(half_lon);

    double d = cos_lat * cos_half_lon;

    /* 1 + cos(lat)*cos(lon/2) must be positive for a valid point */
    double sum = 1.0 + d;
    if (sum < EPSILON) {
        /* Point is at the anti-meridian edge; clamp */
        *x = 0.0;
        *y = (sin_lat >= 0.0) ? sqrt(2.0) * radius : -sqrt(2.0) * radius;
        return true;
    }

    double w = sqrt(sum);

    /* x = 2*sqrt(2) * R * cos(lat)*sin(lon/2) / w */
    /* y = sqrt(2) * R * sin(lat) / w */
    static const double SQRT2 = 1.41421356237309504880;
    static const double TWO_SQRT2 = 2.82842712474619009760;

    *x = radius * TWO_SQRT2 * cos_lat * sin_half_lon / w;
    *y = radius * SQRT2 * sin_lat / w;

    return true;
}

/**
 * Inverse Hammer-Aitoff projection
 *
 * @param x          Input x coordinate
 * @param y          Input y coordinate
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lon        Output longitude in radians
 * @param lat        Output latitude in radians
 * @return true if successful, false if point is outside valid region
 */
bool hammer_inverse(double x, double y, double radius, double lon_center, double *lon,
                    double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    /* Normalize */
    double x_scaled = x / radius;
    double y_scaled = y / radius;

    double x_az = x_scaled * 0.5; /* undo the x doubling */
    double y_az = y_scaled;

    double rho_sq = x_az * x_az + y_az * y_az;

    /* Boundary check: rho must be <= 2 for valid azimuthal equal-area */
    if (rho_sq > 4.0 + EPSILON) {
        return false;
    }
    if (rho_sq > 4.0) rho_sq = 4.0;

    double rho = sqrt(rho_sq);

    /* Handle origin */
    if (rho < EPSILON) {
        *lat = 0.0;
        *lon = lon_center;
        return true;
    }

    /* c = 2*asin(rho/2) */
    double half_rho = rho * 0.5;
    if (half_rho > 1.0) half_rho = 1.0;
    double c = 2.0 * asin(half_rho);
    double sin_c = sin(c);
    double cos_c = cos(c);

    /* lat = asin(y_az * sin_c / rho) */
    double sin_lat = y_az * sin_c / rho;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    /* half_lon = atan2(x_az * sin_c, rho * cos_c) */
    double half_lon = atan2(x_az * sin_c, rho * cos_c);

    /* lon = 2 * half_lon + lon_center */
    *lon = normalize_angle(lon_center + 2.0 * half_lon);

    return true;
}

/*
 * Lambert Azimuthal Equal-Area Projection
 *
 * Forward:
 *   k = sqrt(2 / (1 + sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(dlon)))
 *   x = R * k * cos(lat) * sin(dlon)
 *   y = R * k * (cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(dlon))
 *
 * Inverse:
 *   rho = sqrt(x^2 + y^2)
 *   c = 2 * asin(rho / (2*R))
 *   lat = asin(cos(c)*sin(lat0) + y*sin(c)*cos(lat0)/rho)
 *   lon = lon0 + atan2(x*sin(c), rho*cos(lat0)*cos(c) - y*sin(lat0)*sin(c))
 */

/**
 * Forward Lambert Azimuthal Equal-Area projection
 *
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-pi/2, pi/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lat_center Central latitude in radians
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false otherwise
 */
bool laea_forward(double lon, double lat, double radius, double lon_center, double lat_center,
                  double *x, double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, lon_center);

    double sin_lat = sin(lat);
    double cos_lat = cos(lat);
    double sin_lat0 = sin(lat_center);
    double cos_lat0 = cos(lat_center);
    double cos_dlon = cos(delta_lon);
    double sin_dlon = sin(delta_lon);

    /* d = 1 + sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(dlon) */
    double d = 1.0 + sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon;

    /* d <= 0 means the point is on or beyond the opposite side of the globe */
    if (d < EPSILON) {
        /* Antipodal point — project to boundary */
        /* Place at maximum distance in the appropriate direction */
        double rho_max = 2.0 * radius;
        double bearing =
            atan2(cos_lat * sin_dlon, cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon);
        *x = rho_max * sin(bearing);
        *y = rho_max * cos(bearing);
        return true;
    }

    double k = sqrt(2.0 / d);

    *x = radius * k * cos_lat * sin_dlon;
    *y = radius * k * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon);

    return true;
}

/**
 * Inverse Lambert Azimuthal Equal-Area projection
 *
 * @param x          Input x coordinate
 * @param y          Input y coordinate
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lat_center Central latitude in radians
 * @param lon        Output longitude in radians
 * @param lat        Output latitude in radians
 * @return true if successful, false if point is outside valid region
 */
bool laea_inverse(double x, double y, double radius, double lon_center, double lat_center,
                  double *lon, double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    double x_scaled = x / radius;
    double y_scaled = y / radius;

    double rho_sq = x_scaled * x_scaled + y_scaled * y_scaled;

    /* Maximum rho for the full sphere is 2 (diameter of the disk) */
    if (rho_sq > 4.0 + EPSILON) {
        return false;
    }
    if (rho_sq > 4.0) rho_sq = 4.0;

    /* Handle origin */
    if (rho_sq < EPSILON * EPSILON) {
        *lat = lat_center;
        *lon = lon_center;
        return true;
    }

    double rho = sqrt(rho_sq);

    /* c = 2 * asin(rho / 2) */
    double half_rho = rho * 0.5;
    if (half_rho > 1.0) half_rho = 1.0;
    double c = 2.0 * asin(half_rho);
    double sin_c = sin(c);
    double cos_c = cos(c);

    double sin_lat0 = sin(lat_center);
    double cos_lat0 = cos(lat_center);

    /* lat = asin(cos_c * sin_lat0 + y_scaled * sin_c * cos_lat0 / rho) */
    double sin_lat = cos_c * sin_lat0 + y_scaled * sin_c * cos_lat0 / rho;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    /* lon = lon0 + atan2(x*sin_c, rho*cos_lat0*cos_c - y*sin_lat0*sin_c) */
    double numer = x_scaled * sin_c;
    double denom = rho * cos_lat0 * cos_c - y_scaled * sin_lat0 * sin_c;

    *lon = normalize_angle(lon_center + atan2(numer, denom));

    return true;
}

/*
 * Gnomonic Projection
 *
 * A perspective projection from the center of the sphere onto a tangent plane.
 * Only the hemisphere facing the tangent point can be projected (points
 * at or beyond 90° from center are invalid).
 *
 * Forward:
 *   cos_c = sin(lat0)*sin(lat) + cos(lat0)*cos(lat)*cos(dlon)
 *   x = R * cos(lat)*sin(dlon) / cos_c
 *   y = R * (cos(lat0)*sin(lat) - sin(lat0)*cos(lat)*cos(dlon)) / cos_c
 *
 * Inverse:
 *   rho = sqrt(x^2 + y^2)
 *   c = atan(rho / R)
 *   lat = asin(cos(c)*sin(lat0) + y*sin(c)*cos(lat0)/(rho))
 *   lon = lon0 + atan2(x*sin(c), rho*cos(lat0)*cos(c) - y*sin(lat0)*sin(c))
 */

/**
 * Forward Gnomonic projection
 *
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-pi/2, pi/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lat_center Central latitude in radians
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false if point is on back hemisphere
 */
bool gnomonic_forward(double lon, double lat, double radius, double lon_center,
                      double lat_center, double *x, double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, lon_center);

    double sin_lat = sin(lat);
    double cos_lat = cos(lat);
    double sin_lat0 = sin(lat_center);
    double cos_lat0 = cos(lat_center);
    double cos_dlon = cos(delta_lon);
    double sin_dlon = sin(delta_lon);

    /* cos_c = cos of angular distance from center */
    double cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon;

    /* Point must be on the near hemisphere (cos_c > 0) */
    if (cos_c <= EPSILON) {
        return false;
    }

    double rcp = radius / cos_c;

    *x = rcp * cos_lat * sin_dlon;
    *y = rcp * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon);

    return true;
}

/**
 * Inverse Gnomonic projection
 *
 * @param x          Input x coordinate
 * @param y          Input y coordinate
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param lat_center Central latitude in radians
 * @param lon        Output longitude in radians
 * @param lat        Output latitude in radians
 * @return true if successful, false otherwise
 */
bool gnomonic_inverse(double x, double y, double radius, double lon_center, double lat_center,
                      double *lon, double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    double x_scaled = x / radius;
    double y_scaled = y / radius;

    double rho_sq = x_scaled * x_scaled + y_scaled * y_scaled;
    double rho = sqrt(rho_sq);

    /* Handle origin */
    if (rho < EPSILON) {
        *lat = lat_center;
        *lon = lon_center;
        return true;
    }

    /* c = atan(rho) — angular distance from center */
    double c = atan(rho);
    double sin_c = sin(c);
    double cos_c = cos(c);

    double sin_lat0 = sin(lat_center);
    double cos_lat0 = cos(lat_center);

    /* lat = asin(cos_c * sin_lat0 + y_scaled * sin_c * cos_lat0 / rho) */
    double sin_lat = cos_c * sin_lat0 + y_scaled * sin_c * cos_lat0 / rho;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    /* lon = lon0 + atan2(x*sin_c, rho*cos_lat0*cos_c - y*sin_lat0*sin_c) */
    double numer = x_scaled * sin_c;
    double denom = rho * cos_lat0 * cos_c - y_scaled * sin_lat0 * sin_c;

    *lon = normalize_angle(lon_center + atan2(numer, denom));

    return true;
}

/*
 * Albers Equal-Area Conic Projection
 *
 * Constants (computed from standard parallels lat1, lat2):
 *   n = (sin(lat1) + sin(lat2)) / 2
 *   C = cos^2(lat1) + 2*n*sin(lat1)
 *   rho0 = sqrt(C) / n    (y-origin at equator)
 *
 * Forward:
 *   theta = n * delta_lon
 *   rho = sqrt(C - 2*n*sin(lat)) / n
 *   x = R * rho * sin(theta)
 *   y = R * (rho0 - rho * cos(theta))
 *
 * Inverse:
 *   rho = sqrt(x^2 + (rho0*R - y)^2) * sign(n)
 *   theta = atan2(x, rho0*R - y)
 *   lat = asin((C - (rho*n/R)^2) / (2*n))
 *   lon = lon0 + theta / n
 */

/**
 * Initialize Albers constants from projection parameters.
 *
 * @param params     Output constants structure
 * @param lon_center Central meridian in radians
 * @param lat1       First standard parallel in radians
 * @param lat2       Second standard parallel in radians
 * @return true if valid, false if degenerate (e.g. n ≈ 0)
 */
bool albers_init(albers_params_t *params, double lon_center, double lat1, double lat2) {
    if (params == NULL) {
        return false;
    }

    params->lon_center = lon_center;
    params->lat1 = lat1;
    params->lat2 = lat2;

    double sin_lat1 = sin(lat1);
    double sin_lat2 = sin(lat2);
    double cos_lat1 = cos(lat1);

    params->n = 0.5 * (sin_lat1 + sin_lat2);

    if (fabs(params->n) < EPSILON) {
        return false;
    }

    params->n_inv = 1.0 / params->n;
    params->C = cos_lat1 * cos_lat1 + 2.0 * params->n * sin_lat1;

    /* rho0 with y-origin at equator (lat=0, sin(lat)=0) */
    params->rho0 = sqrt(params->C) * params->n_inv;

    params->n_negative = (params->n < 0.0);

    return true;
}

/**
 * Forward Albers Equal-Area Conic projection
 *
 * @param params     Precomputed constants from albers_init
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-pi/2, pi/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false otherwise
 */
bool albers_forward(const albers_params_t *params, double lon, double lat, double radius,
                    double *x, double *y) {
    if (params == NULL || x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, params->lon_center);

    double theta = params->n * delta_lon;
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);

    double rho_sq = params->C - 2.0 * params->n * sin(lat);
    if (rho_sq < 0.0) rho_sq = 0.0;
    double rho = sqrt(rho_sq) * params->n_inv;

    *x = radius * rho * sin_theta;
    *y = radius * (params->rho0 - rho * cos_theta);

    return true;
}

/**
 * Inverse Albers Equal-Area Conic projection
 *
 * @param params     Precomputed constants from albers_init
 * @param x          Input x coordinate
 * @param y          Input y coordinate
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon        Output longitude in radians
 * @param lat        Output latitude in radians
 * @return true if successful, false if point is outside valid region
 */
bool albers_inverse(const albers_params_t *params, double x, double y, double radius,
                    double *lon, double *lat) {
    if (params == NULL || lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    double x_scaled = x / radius;
    double y_adj = params->rho0 - y / radius;

    double rho = sqrt(x_scaled * x_scaled + y_adj * y_adj);

    double theta;
    if (params->n_negative) {
        rho = -rho;
        theta = atan2(-x_scaled, -y_adj);
    } else {
        theta = atan2(x_scaled, y_adj);
    }

    double rho_n = rho * params->n;
    double sin_lat = (params->C - rho_n * rho_n) * 0.5 * params->n_inv;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    double delta_lon = theta * params->n_inv;
    if (delta_lon > SP_PI) delta_lon = SP_PI;
    if (delta_lon < -SP_PI) delta_lon = -SP_PI;

    *lon = normalize_angle(params->lon_center + delta_lon);

    return true;
}

/*
 * Oblique Mollweide Projection
 *
 * This applies a coordinate rotation to move the projection pole
 * to (lon_p, lat_p), then applies the standard Mollweide projection.
 *
 * The rotation transforms geographic (lon, lat) to oblique (olon, olat)
 * where the new "north pole" is at the specified (lon_p, lat_p).
 *
 * Forward:
 *   1. Rotate (lon, lat) -> (olon, olat) with pole at (lon_p, lat_p)
 *   2. Apply standard Mollweide to (olon, olat)
 *
 * Inverse:
 *   1. Apply inverse Mollweide to get (olon, olat)
 *   2. Rotate (olon, olat) -> (lon, lat) back to geographic
 */

/**
 * Initialize oblique Mollweide parameters
 *
 * @param params     Output parameter structure
 * @param lon_p      Longitude of the oblique pole in radians
 * @param lat_p      Latitude of the oblique pole in radians
 * @param lon_0 Central meridian in radians (in the oblique system)
 * @return true if successful
 */
bool oblique_mollweide_init(oblique_mollweide_params_t *params, double lon_p, double lat_p,
                            double lon_0) {
    if (params == NULL) {
        return false;
    }

    params->lon_0 = lon_0;
    params->lamp = lon_p;
    params->sphip = sin(lat_p);
    params->cphip = cos(lat_p);

    return true;
}

/**
 * PROJ o_forward rotation: geographic -> oblique
 * Input lon must already have lon_0 subtracted.
 */
static void geographic_to_oblique(const oblique_mollweide_params_t *params, double lon,
                                  double lat, double *olon, double *olat) {
    double coslam = cos(lon);
    double sinphi = sin(lat);
    double cosphi = cos(lat);

    /* Snyder (5-8b) */
    *olon =
        atan2(cosphi * sin(lon), params->sphip * cosphi * coslam + params->cphip * sinphi) +
        params->lamp;

    /* Normalize */
    *olon = fmod(*olon, 2.0 * SP_PI);
    if (*olon > SP_PI) *olon -= 2.0 * SP_PI;
    if (*olon <= -SP_PI) *olon += 2.0 * SP_PI;

    /* Snyder (5-7) */
    double sin_olat = params->sphip * sinphi - params->cphip * cosphi * coslam;
    sin_olat = fmax(-1.0, fmin(1.0, sin_olat));
    *olat = asin(sin_olat);
}

/**
 * PROJ o_inverse rotation: oblique -> geographic
 * Returns lon WITHOUT lon_0 added back.
 */
static void oblique_to_geographic(const oblique_mollweide_params_t *params, double olon,
                                  double olat, double *lon, double *lat) {
    /* Subtract lamp first */
    olon -= params->lamp;

    double coslam = cos(olon);
    double sinphi = sin(olat);
    double cosphi = cos(olat);

    /* Snyder (5-9) */
    double sin_lat = params->sphip * sinphi + params->cphip * cosphi * coslam;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    /* Snyder (5-10b) */
    *lon = atan2(cosphi * sin(olon), params->sphip * cosphi * coslam - params->cphip * sinphi);
}

/**
 * Forward oblique Mollweide projection
 *
 * @param params     Precomputed oblique parameters
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-pi/2, pi/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false otherwise
 */
bool oblique_mollweide_forward(const oblique_mollweide_params_t *params, double lon,
                               double lat, double radius, double *x, double *y) {
    if (params == NULL || x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    /* Subtract lon_0 (same as PROJ pipeline does before o_forward) */
    double lon_adj = lon - params->lon_0;

    /* Rotate to oblique */
    double olon, olat;
    geographic_to_oblique(params, lon_adj, lat, &olon, &olat);

    /* Standard Mollweide with lon_center=0 */
    return mollweide_forward(olon, olat, radius, 0.0, x, y);
}

/**
 * Inverse oblique Mollweide projection
 *
 * @param params     Precomputed oblique parameters
 * @param x          Input x coordinate
 * @param y          Input y coordinate
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon        Output longitude in radians
 * @param lat        Output latitude in radians
 * @return true if successful, false if point is outside valid region
 */
bool oblique_mollweide_inverse(const oblique_mollweide_params_t *params, double x, double y,
                               double radius, double *lon, double *lat) {
    if (params == NULL || lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    /* Inverse Mollweide with lon_center=0 */
    double olon, olat;
    if (!mollweide_inverse(x, y, radius, 0.0, &olon, &olat)) {
        return false;
    }

    /* Rotate back to geographic */
    double geo_lon, geo_lat;
    oblique_to_geographic(params, olon, olat, &geo_lon, &geo_lat);

    /* Add lon_0 back and normalize */
    *lon = normalize_angle(geo_lon + params->lon_0);
    *lat = geo_lat;

    return true;
}
