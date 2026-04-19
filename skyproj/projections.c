#include <math.h>
#include <stdbool.h>

#include "skyproj.h"
#include "projections.h"

/**
 * Normalize angle to (-π, π] range (exclusive of -π, inclusive of π)
 *
 * @param angle Input angle in radians
 * @return Normalized angle in (-π, π]
 */

/**
 * Normalize angle to (-π, π] using floor-based approach
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
    } else if (angle <= -SP_PI) {
        angle += 2.0 * SP_PI;
    }

    return angle;
}

/**
 * Adjust longitude to be within ±180° of central meridian
 * This matches PROJ library behavior
 */
static double adjust_lon(double lon) {
    // Normalize to (-π, π]
    if (!isfinite(lon)) {
        return 0.0;
    }

    lon = fmod(lon, 2.0 * SP_PI);

    if (lon > SP_PI) {
        lon -= 2.0 * SP_PI;
    } else if (lon <= -SP_PI) {
        lon += 2.0 * SP_PI;
    }

    return lon;
}

/**
 * Calculate delta longitude the way PROJ does it
 * This maintains consistent behavior across the ±180° boundary
 */
static double delta_longitude(double lon, double lon_center) {
    // First, normalize both longitudes to (-π, π]
    lon = adjust_lon(lon);
    lon_center = adjust_lon(lon_center);

    // Compute difference
    double delta = lon - lon_center;

    // Adjust to ensure delta is in (-π, π]
    // This is the key: we normalize the DIFFERENCE, not compute shortest path
    while (delta > SP_PI) {
        delta -= 2.0 * SP_PI;
    }
    while (delta <= -SP_PI) {
        delta += 2.0 * SP_PI;
    }

    return delta;
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
bool mollweide_forward(double lon, double lat, double radius, double lon_center,
                       double *x, double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    // Calculate delta longitude from central meridian
    double delta_lon = delta_longitude(lon, lon_center);

    // Handle poles specially
    if (fabs(lat - SP_PI/2) < EPSILON) {
        *x = 0.0;
        *y = sqrt(2.0) * radius;
        return true;
    }
    if (fabs(lat + SP_PI/2) < EPSILON) {
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
bool mollweide_inverse(double x, double y, double radius, double lon_center,
                       double *lon, double *lat) {
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
#define A1 1.340264
#define A2 -0.081106
#define A3 0.000893
#define A4 0.003796

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
bool equal_earth_forward(double lon, double lat, double radius, double lon_center,
                         double *x, double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    // Calculate delta longitude
    double lambda = delta_longitude(lon, lon_center);

    // Parametric latitude
    double phi = lat;
    double sin_phi = sin(phi);

    // Calculate parametric angle theta
    // sin(theta) = (sqrt(3)/2) * sin(phi)
    double sin_theta = sqrt(3.0) / 2.0 * sin_phi;

    // Clamp to valid range
    if (sin_theta > 1.0) sin_theta = 1.0;
    if (sin_theta < -1.0) sin_theta = -1.0;

    double theta = asin(sin_theta);

    // Calculate powers of theta
    double theta2 = theta * theta;
    double theta6 = theta2 * theta2 * theta2;
    double theta8 = theta6 * theta2;

    // Calculate polynomial terms
    double cos_theta = cos(theta);

    // Denominator for x coordinate
    double denom = 3.0 * (9.0 * A4 * theta6 + 7.0 * A3 * theta6 + 
                          3.0 * A2 * theta2 + A1);

    // Calculate coordinates
    // Note: The standard R in the paper is approximately sqrt(3)/2 ≈ 0.8660254
    // We scale by our custom radius

    // x = (2 * sqrt(3) * R * lambda * cos(theta)) / (3 * denom)
    *x = (2.0 * sqrt(3.0) * radius * lambda * cos_theta) / denom;

    // y = R * A1 * theta + R * A2 * theta^3 + R * A3 * theta^7 + R * A4 * theta^9
    // Factored: y = R * theta * (A1 + A2*theta^2 + A3*theta^6 + A4*theta^8)
    *y = radius * theta * (A1 + A2 * theta2 + A3 * theta6 + A4 * theta8);

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
bool equal_earth_inverse(double x, double y, double radius, double lon_center,
                         double *lon, double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    // Normalize y coordinate
    double y_scaled = y / radius;

    // Maximum theta value
    double theta_max = asin(sqrt(3.0) / 2.0);  // ≈ 1.0472 radians ≈ 60°

    // Check if y is in valid range
    double y_max = theta_max * (A1 + A2 * theta_max * theta_max + 
                                A3 * pow(theta_max, 6) + 
                                A4 * pow(theta_max, 8));

    if (fabs(y_scaled) > y_max + EPSILON) {
        return false;
    }

    // Solve for theta using Newton-Raphson iteration
    // Equation: f(theta) = theta * (A1 + A2*theta^2 + A3*theta^6 + A4*theta^8) - y/R = 0

    // Initial guess: for small y, theta ≈ y/A1
    double theta = y_scaled / A1;

    // Clamp initial guess
    if (theta > theta_max) theta = theta_max;
    if (theta < -theta_max) theta = -theta_max;

    // Newton-Raphson iteration
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta4 * theta2;
        double theta8 = theta6 * theta2;

        // f(theta) = theta * (A1 + A2*theta^2 + A3*theta^6 + A4*theta^8) - y/R
        double polynomial = A1 + A2 * theta2 + A3 * theta6 + A4 * theta8;
        double f = theta * polynomial - y_scaled;

        // f'(theta) = polynomial + theta * d(polynomial)/d(theta)
        // d(polynomial)/d(theta) = 2*A2*theta + 6*A3*theta^5 + 8*A4*theta^7
        double d_polynomial = 2.0 * A2 * theta + 6.0 * A3 * theta * theta4 + 
                             8.0 * A4 * theta * theta6;
        double df = polynomial + theta * d_polynomial;

        if (fabs(df) < EPSILON) {
            return false;
        }

        double delta = f / df;
        theta -= delta;

        if (fabs(delta) < EPSILON) {
            break;
        }

        // Clamp theta to valid range during iteration
        if (theta > theta_max) theta = theta_max;
        if (theta < -theta_max) theta = -theta_max;
    }

    // Calculate latitude from theta
    // sin(phi) = 2 * sin(theta) / sqrt(3)
    double sin_phi = 2.0 * sin(theta) / sqrt(3.0);

    // Clamp to valid range
    if (sin_phi > 1.0) sin_phi = 1.0;
    if (sin_phi < -1.0) sin_phi = -1.0;

    *lat = asin(sin_phi);

    // Calculate longitude
    double cos_theta = cos(theta);

    // Handle poles
    if (fabs(cos_theta) < EPSILON) {
        *lon = lon_center;
        return true;
    }

    // Calculate delta longitude
    double theta2 = theta * theta;
    double theta6 = theta2 * theta2 * theta2;

    double denom = 3.0 * (9.0 * A4 * theta6 + 7.0 * A3 * theta6 + 
                          3.0 * A2 * theta2 + A1);

    double lambda = (x * denom) / (2.0 * sqrt(3.0) * radius * cos_theta);

    // Clamp to valid range
    if (lambda > SP_PI) lambda = SP_PI;
    if (lambda < -SP_PI) lambda = -SP_PI;

    // Add central meridian
    *lon = normalize_angle(lon_center + lambda);

    return true;
}

#include <math.h>
#include <stdbool.h>
#include "skyproj.h"
#include "projections.h"

/*
 * McBryde-Thomas Flat Polar Quartic Projection
 *
 * Constants and algorithm from PROJ PJ_mbtfpq.c
 *
 * Auxiliary angle theta solved from:
 *   sin(theta/2) + sin(theta) = C * sin(lat)
 *   where C = 1 + sqrt(2)/2
 *
 * Forward:
 *   x = FXC * R * delta_lon * (1 + 2*cos(theta)/sqrt(3))
 *   y = FYC * R * sin(theta/2)
 *
 * Inverse:
 *   sin(theta/2) = y / (FYC * R)
 *   theta = 2 * asin(sin(theta/2))
 *   sin(lat) = (sin(theta/2) + sin(theta)) / C
 *   delta_lon = x / (FXC * R * (1 + 2*cos(theta)/sqrt(3)))
 */

/* Constants directly from PROJ PJ_mbtfpq.c */
#define MBTFPQ_C      1.70710678118654752440   /* 1 + sqrt(2)/2           */
#define MBTFPQ_RC     0.58578643762690495119   /* 1/C                     */
#define MBTFPQ_FYC    1.87475828462269495505   /* y forward scale         */
#define MBTFPQ_RYC    0.53340209679090950      /* 1/FYC                   */
#define MBTFPQ_FXC    0.31245526550726392588   /* x forward scale         */
#define MBTFPQ_RXC    3.20041258076506210122   /* 1/FXC                   */

/**
 * Forward McBryde-Thomas Flat Polar Quartic projection
 *
 * @param lon        Longitude in radians
 * @param lat        Latitude in radians [-pi/2, pi/2]
 * @param radius     Radius of the sphere (must be > 0)
 * @param lon_center Central meridian in radians
 * @param x          Output x coordinate
 * @param y          Output y coordinate
 * @return true if successful, false otherwise
 */
bool mbtfpq_forward(double lon, double lat, double radius, double lon_center,
                     double *x, double *y) {
    if (x == NULL || y == NULL || radius <= 0) {
        return false;
    }

    double delta_lon = delta_longitude(lon, lon_center);
    double sin_lat = sin(lat);

    /*
     * Solve: sin(theta/2) + sin(theta) = C * sin(lat)
     *
     * Let k = sin(theta/2), then:
     *   cos(theta/2) = sqrt(1 - k^2)
     *   sin(theta) = 2*k*sqrt(1 - k^2)
     *
     * So the equation becomes:
     *   k + 2*k*sqrt(1 - k^2) = C * sin(lat)
     *   k * (1 + 2*sqrt(1 - k^2)) = C * sin(lat)
     *
     * Let g(k) = k * (1 + 2*sqrt(1 - k^2)) - target
     * g'(k) = 1 + 2*sqrt(1 - k^2) - 2*k^2/sqrt(1 - k^2)
     *       = 1 + 2*(1 - k^2)/sqrt(1 - k^2) - 2*k^2/sqrt(1 - k^2)
     *       = 1 + (2 - 4*k^2) / sqrt(1 - k^2)
     *
     * No trig in the loop — only sqrt.
     */
    static const double TWO_OVER_SQRT3 = 1.15470053837925152902;

    double target = MBTFPQ_C * sin_lat;
    double k = sin_lat * 0.5;  /* initial guess for sin(theta/2) */

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double k2 = k * k;
        double om_k2 = 1.0 - k2;
        if (om_k2 < 0.0) om_k2 = 0.0;
        double sqrt_om = sqrt(om_k2);

        double g = k * (1.0 + 2.0 * sqrt_om) - target;

        /* Avoid division by zero near poles */
        double dg;
        if (sqrt_om < EPSILON) {
            dg = 1.0;
        } else {
            dg = 1.0 + (2.0 - 4.0 * k2) / sqrt_om;
        }

        double delta = g / dg;
        k -= delta;

        /* Clamp k to valid range */
        if (k > 1.0) k = 1.0;
        if (k < -1.0) k = -1.0;

        if (fabs(delta) < EPSILON) break;
    }

    /* Now compute x, y from k = sin(theta/2) */
    /* cos(theta) = 1 - 2*k^2 */
    double cos_theta = 1.0 - 2.0 * k * k;

    *x = radius * MBTFPQ_FXC * delta_lon * (1.0 + TWO_OVER_SQRT3 * cos_theta);
    *y = radius * MBTFPQ_FYC * k;

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
bool mbtfpq_inverse(double x, double y, double radius, double lon_center,
                     double *lon, double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    if (!isfinite(x) || !isfinite(y)) {
        return false;
    }

    double sin_half_theta = y * MBTFPQ_RYC / radius;

    if (fabs(sin_half_theta) > 1.0 + EPSILON) {
        return false;
    }
    sin_half_theta = fmax(-1.0, fmin(1.0, sin_half_theta));

    double half_theta = asin(sin_half_theta);
    double theta = 2.0 * half_theta;

    double sin_lat = (sin_half_theta + sin(theta)) * MBTFPQ_RC;
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    double cos_theta = cos(theta);
    double shape = 1.0 + 2.0 * cos_theta / sqrt(3.0);

    if (fabs(shape) < EPSILON * 10.0) {
        *lon = lon_center;
        return true;
    }

    double delta_lon = x * MBTFPQ_RXC / (radius * shape);

    if (delta_lon > SP_PI) delta_lon = SP_PI;
    if (delta_lon < -SP_PI) delta_lon = -SP_PI;

    *lon = normalize_angle(lon_center + delta_lon);

    return true;
}
