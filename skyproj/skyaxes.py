import functools
import numpy as np

import matplotlib.axes
from pyproj import Geod

from .skycrs import PlateCarreeCRS

__all__ = ["SkyAxes"]


def _add_lonlat(func):
    """Decorator to add lonlat option."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if kwargs.pop('lonlat', True):
            kwargs['transform'] = self.projection
        return func(self, *args, **kwargs)
    return wrapper


class SkyAxes(matplotlib.axes.Axes):
    # docstring inherited
    def __init__(self, *args, **kwargs):
        self.projection = kwargs.pop("sky_crs")

        self.plate_carree = PlateCarreeCRS()

        super().__init__(*args, **kwargs)

    def cla(self):
        """Clear the current axes."""
        result = super().cla()
        self.xaxis.set_visible(False)
        self.yaxis.set_visible(False)

        self.set_frame_on(False)

        # Always equal aspect ratio.
        self.set_aspect('equal')

        return result

    def set_extent(self, extent, lonlat=True):
        """Set the extent of the axes.

        Parameters
        ----------
        extent : `tuple` [`float`]
            Set the extent by [lon0, lon1, lat0, lat1] or [x0, x1, y0, y1].
        lonlat : `bool`, optional
            Extent is specified in lon/lat coordinates?  Otherwise native.
        """
        if not lonlat:
            x0, x1, y0, y1 = extent
        else:
            # Need to do transformation.
            lon0, lon1, lat0, lat1 = extent

            # Check if the longitude range is the full sphere, and special case that.
            if np.isclose(np.abs(lon1 - lon0), 360.0):
                lat_steps = [lat0, lat1]
                if lat0 < 0.0 and lat1 > 0.0:
                    # Make sure we have the equator included
                    lat_steps.append(0.0)
                lon, lat = np.meshgrid(np.linspace(0, 360.0, 360), lat_steps)
                xy = self.projection.transform_points(self.plate_carree, lon.ravel(), lat.ravel())
                # Need to offset this by some small amount to ensure we don't get
                # out-of-bounds transformations.
                eps = 1e-5
                x0 = (1 - eps)*np.min(xy[:, 0])
                x1 = (1 - eps)*np.max(xy[:, 0])
                y0 = (1 - eps)*np.min(xy[:, 1])
                y1 = (1 - eps)*np.max(xy[:, 1])
            else:
                # Make a ring of points and check their extent.
                npt = 100
                lon_pts = np.linspace(lon0, lon1, npt)
                lat_pts = np.linspace(lat0, lat1, npt)
                lon = np.concatenate((lon_pts, lon_pts, np.repeat(lon0, npt), np.repeat(lon1, npt)))
                lat = np.concatenate((np.repeat(lat0, npt), np.repeat(lat1, npt), lat_pts, lat_pts))
                xy = self.projection.transform_points(self.plate_carree, lon, lat)
                # FIXME NOTE NEED TO KNOW LON_0/WRAP OF PROJECTION...
                x0 = np.min(xy[:, 0])
                x1 = np.max(xy[:, 0])
                y0 = np.min(xy[:, 1])
                y1 = np.max(xy[:, 1])

        self.set_xlim([x0, x1])
        self.set_ylim([y0, y1])

    def get_extent(self, lonlat=True):
        """Get the extent of the axes.

        Parameters
        ----------
        lonlat : `bool`, optional
            Return extent in lon/lat coordinates?  Otherwise native.
        """
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()

        if not lonlat:
            extent = (x0, x1, y0, y1)
        else:
            # Make a ring of points and check their extent.
            npt = 100
            x_pts = np.linspace(x0, x1, npt)
            y_pts = np.linspace(y0, y1, npt)
            x = np.concatenate((x_pts, x_pts, np.repeat(x0, npt), np.repeat(x1, npt)))
            y = np.concatenate((np.repeat(y0, npt), np.repeat(y1, npt), y_pts, y_pts))
            lonlat = self.plate_carree.transform_points(self.projection, x, y)

            # We may have nans from out-of-bounds for certain projections (e.g. Mollweide):
            if np.any(np.isnan(lonlat)):
                if np.any(np.isnan(lonlat[0: npt, 0])):
                    # Bottom is at the limit
                    lonlat[0: npt, 1] = -90.0 + 1e-5
                if np.any(np.isnan(lonlat[npt: 2*npt, 0])):
                    # Top is at the limit
                    lonlat[npt: 2*npt, 1] = 90.0 - 1e-5
                if np.any(np.isnan(lonlat[2*npt: 3*npt, 1])):
                    # Right is at the limit
                    lonlat[2*npt: 3*npt, 0] = 180.0 - 1e-5
                if np.any(np.isnan(lonlat[3*npt: 4*npt, 1])):
                    # Left is at the limit
                    lonlat[3*npt: 4*npt, 0] = -180.0 + 1e-5
            else:
                # Check for out-of-bounds by reverse-projecting
                xy = self.projection.transform_points(self.plate_carree, lonlat[:, 0], lonlat[:, 1])
                bad = ((~np.isclose(xy[:, 0], x)) | (~np.isclose(xy[:, 1], y)))
                lonlat[bad, :] = np.nan

            # FIXME CHECK FOR WRAPPING!!!

            lon0 = np.nanmin(lonlat[:, 0])
            lon1 = np.nanmax(lonlat[:, 0])
            lat0 = np.nanmin(lonlat[:, 1])
            lat1 = np.nanmax(lonlat[:, 1])

            if self.xaxis_inverted():
                extent = (lon1, lon0, lat0, lat1)
            else:
                extent = (lon0, lon1, lat0, lat1)

        return extent

    @_add_lonlat
    def plot(self, *args, **kwargs):
        # docstring inherited

        # The transformation code will automatically plot geodesics
        # and split line segements that cross the wrapping boundary.
        result = super().plot(*args, **kwargs)

        return result

    @_add_lonlat
    def scatter(self, *args, **kwargs):
        # docstring inherited
        result = super().scatter(*args, **kwargs)

        return result

    @_add_lonlat
    def pcolormesh(self, X, Y, C, **kwargs):
        # docstring inherited
        C_temp = C.copy()

        if kwargs.get('lonlat', True):
            # Check for wrapping by projecting and looking for jumps.
            proj_xy = self.projection.transform_points(self.plate_carree, X, Y)
            X_proj = proj_xy[..., 0]
            Y_proj = proj_xy[..., 1]

            dist = np.hypot(X_proj[1:, 1:] - X_proj[0: -1, 0: -1],
                            Y_proj[1:, 1:] - Y_proj[0: -1, 0: -1])

            # If we have a jump of 10% of the radius, assume it's bad,
            # except if we are using PlateCarree which doesn't use the radius.
            if self.projection == self.plate_carree:
                max_dist = 90.0
            else:
                max_dist = 0.1*self.projection.radius

            split = (dist > max_dist).nonzero()

            # By marking these jumps as masked then pcolormesh works just fine.
            C_temp.mask[split] = True

        result = super().pcolormesh(X, Y, C_temp, **kwargs)

        return result

    @_add_lonlat
    def fill(self, *args, **kwargs):
        # docstring inherited
        result = super().fill(*args, **kwargs)

        return result

    @_add_lonlat
    def text(self, *args, **kwargs):
        # docstring inherited
        result = super().text(*args, **kwargs)

        return result

    @_add_lonlat
    def circle(self, lon, lat, radius, nsamp=100, fill=False, **kwargs):
        """Draw a geodesic circle centered at given position.

        Parameters
        ----------
        lon : `float`
            Longitude of center of circle (degrees).
        lat : `float`
            Latitude of center of circle (degrees).
        radius : `float`
            Radius of circle (degrees).
        nsamp : `int`, optional
            Number of points to sample.
        fill : `bool`, optional
            Draw filled circle?
        **kwargs : `dict`
            Extra plotting kwargs.
        """
        geod = Geod(a=self.projection.radius)

        # We need the radius in meters
        radius_m = self.projection.radius*np.deg2rad(radius)

        az = np.linspace(360.0, 0.0, nsamp)
        lons, lats, _ = geod.fwd(
            np.full(nsamp, lon, dtype=np.float64),
            np.full(nsamp, lat, dtype=np.float64),
            az,
            np.full(nsamp, radius_m)
        )
        if fill:
            return self.fill(lons, lats, **kwargs)
        else:
            return self.plot(lons, lats, **kwargs)

    @_add_lonlat
    def ellipse(self, lon, lat, a, b, theta, nsamp=100, fill=False, **kwargs):
        """Draw a geodesic ellipse centered at given position.

        Parameters
        ----------
        lon : `float`
            Longitude of center of ellipse (degrees).
        lat : `float`
            Latitude of center of ellipse (degrees).
        a : `float`
            Semi-major axis of ellipse (degrees).
        b : `float`
            Semi-minor axis of ellipse (degrees).
        theta : `float`
            Position angle of ellipse.  Degrees East of North.
        nsamp : `int`, optional
            Number of points to sample.
        fill : `bool`, optional
            Draw filled ellipse?
        **kwargs : `dict`
            Extra plotting kwargs.
        """
        geod = Geod(a=self.projection.radius)

        # We need the radius in meters
        a_m = self.projection.radius * np.deg2rad(a)
        b_m = self.projection.radius * np.deg2rad(b)

        az = np.linspace(360.0, 0.0, nsamp)

        phase_rad = np.deg2rad(az - theta)

        # Position Angle is defined as degrees East from North
        denom = np.sqrt((b_m * np.cos(phase_rad))**2 + (a_m * np.sin(phase_rad))**2)
        dist = a_m * b_m / denom

        lons, lats, _ = geod.fwd(
            np.full(nsamp, lon, dtype=np.float64),
            np.full(nsamp, lat, dtype=np.float64),
            az,
            dist,
        )
        if fill:
            return self.fill(lons, lats, **kwargs)
        else:
            return self.plot(lons, lats, **kwargs)

    @property
    def lon_0(self):
        return self.projection.lon_0

    @property
    def lat_0(self):
        return self.projection.lat_0

    def update_projection(self, crs_new):
        """Update the projection central coordinate.

        Parameters
        ----------
        crs_new : `skyproj.SkyCRS`
        """
        self.projection = crs_new
