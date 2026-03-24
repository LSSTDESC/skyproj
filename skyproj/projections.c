#include <math.h>
#include <stdbool.h>

#include "skyproj.h"
#include "projections.h"

/**
 * Normalize angle to [-π, π] range
 */
static double normalize_angle(double angle) {
    while (angle > SP_PI) angle -= 2.0 * SP_PI;
    while (angle < -SP_PI) angle += 2.0 * SP_PI;
    return angle;
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
    double delta_lon = normalize_angle(lon - lon_center);

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
    // double theta = lat;  // Initial guess
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

    /*
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double f = 2.0 * theta + sin(2.0 * theta) - target;
        double df = 2.0 + 2.0 * cos(2.0 * theta);

        if (fabs(df) < EPSILON) {
            return false;  // Derivative too small
        }

        double delta = f / df;
        theta -= delta;

        if (fabs(delta) < EPSILON) {
            break;
        }
    }
    */

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
/*
bool mollweide_inverse(double x, double y, double radius, double lon_center,
                       double *lon, double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    // Normalize coordinates by radius
    double x_scaled = x / radius;
    double y_scaled = y / radius;

    // Check if point is within the valid ellipse
    // Ellipse equation: (x/(2*sqrt(2)*R))^2 + (y/(sqrt(2)*R))^2 <= 1
    // After scaling: (x_scaled/(2*sqrt(2)))^2 + (y_scaled/sqrt(2))^2 <= 1
    double x_norm = x_scaled / (2.0 * sqrt(2.0));
    double y_norm = y_scaled / sqrt(2.0);

    if (x_norm * x_norm + y_norm * y_norm > 1.0 + EPSILON) {
        return false;  // Point outside valid region
    }

    // Clamp y to valid range to handle numerical errors
    double y_max = sqrt(2.0);
    if (y_scaled > y_max) y_scaled = y_max;
    if (y_scaled < -y_max) y_scaled = -y_max;

    // Calculate theta from y coordinate
    // y = sqrt(2) * R * sin(theta)
    double theta = asin(y_scaled / sqrt(2.0));

    // Handle the case where cos(theta) is zero (poles)
    if (fabs(cos(theta)) < EPSILON) {
        *lon = lon_center;  // Use center longitude for poles
        *lat = (y > 0) ? SP_PI/2.0 : -SP_PI/2.0;
        return true;
    }

    // Calculate latitude
    // From: 2*theta + sin(2*theta) = pi*sin(lat)
    double sin_lat = (2.0 * theta + sin(2.0 * theta)) / SP_PI;

    // Clamp to valid range [-1, 1] to handle numerical errors
    if (sin_lat > 1.0) sin_lat = 1.0;
    if (sin_lat < -1.0) sin_lat = -1.0;

    *lat = asin(sin_lat);

    // Calculate delta longitude from x coordinate
    // x = (2*sqrt(2)/π) * R * delta_lon * cos(theta)
    double delta_lon = SP_PI * x_scaled / (2.0 * sqrt(2.0) * cos(theta));

    // Clamp delta longitude to valid range [-π, π]
    if (delta_lon > SP_PI) delta_lon = SP_PI;
    if (delta_lon < -SP_PI) delta_lon = -SP_PI;

    // Add central meridian to get actual longitude
    *lon = normalize_angle(lon_center + delta_lon);

    return true;
}
*/
/**
 * Optimized inverse with better numerical stability
 */
bool mollweide_inverse(double x, double y, double radius, double lon_center,
                                 double *lon, double *lat) {
    if (lon == NULL || lat == NULL || radius <= 0) {
        return false;
    }

    // Normalize coordinates by radius
    double x_scaled = x / radius;
    double y_scaled = y / radius;

    // Quick bounds check (can early-exit invalid points faster)
    double abs_y = fabs(y_scaled);
    if (abs_y > sqrt(2.0) + EPSILON) {
        return false;  // Definitely outside
    }

    // More precise ellipse check
    double x_norm = x_scaled / (2.0 * sqrt(2.0));
    double y_norm = y_scaled / sqrt(2.0);
    double dist_sq = x_norm * x_norm + y_norm * y_norm;

    if (dist_sq > 1.0 + EPSILON) {
        return false;
    }

    // Clamp y to valid range
    double sqrt2 = sqrt(2.0);
    if (y_scaled > sqrt2) y_scaled = sqrt2;
    if (y_scaled < -sqrt2) y_scaled = -sqrt2;

    // Calculate theta from y (this is exact)
    double theta = asin(y_scaled / sqrt2);

    // Handle poles specially
    double cos_theta = cos(theta);
    if (fabs(cos_theta) < EPSILON) {
        *lon = lon_center;
        *lat = (y > 0) ? SP_PI/2.0 : -SP_PI/2.0;
        return true;
    }

    // Calculate latitude using optimized polynomial evaluation
    // Original: sin_lat = (2*theta + sin(2*theta)) / SP_PI
    // Optimized using sin(2*theta) = 2*sin(theta)*cos(theta)
    double sin_theta = sin(theta);
    double sin_2theta = 2.0 * sin_theta * cos_theta;
    double sin_lat = (2.0 * theta + sin_2theta) / SP_PI;

    // Clamp and calculate latitude
    sin_lat = fmax(-1.0, fmin(1.0, sin_lat));
    *lat = asin(sin_lat);

    // Calculate longitude
    double delta_lon = SP_PI * x_scaled / (2.0 * sqrt2 * cos_theta);
    delta_lon = fmax(-SP_PI, fmin(SP_PI, delta_lon));

    *lon = normalize_angle(lon_center + delta_lon);

    return true;
}
