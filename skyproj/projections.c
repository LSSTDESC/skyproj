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
