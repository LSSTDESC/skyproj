#ifndef _GEODESICS_H

#define _GEODESICS_H

bool geod_interp_sp(double lon0, double lat0,
                    double lon1, double lat1,
                    double radius,
                    int npts, int include_start, int include_end,
                    int degrees,
                    double *lonlat_data);
bool geod_direct_sp(double lon0, double lat0,
                    double azimuth, double distance,
                    double radius,
                    int degrees,
                    double *lon_out, double *lat_out);

#endif
