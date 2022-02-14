import numpy as np
from packaging import version

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

from .utils import wrap_values

import matplotlib
if version.parse(matplotlib.__version__) >= version.parse('3.5'):
    _celestial_angle_360 = True
else:
    _celestial_angle_360 = False

__all__ = ['WrappedFormatterDMS', 'ExtremeFinderWrapped', 'GridHelperSkyproj']


class WrappedFormatterDMS(angle_helper.FormatterDMS):
    def __init__(self, wrap, longitude_ticks):
        self._wrap = wrap
        if longitude_ticks == 'positive':
            self._longitude_ticks = 1
        elif longitude_ticks == 'symmetric':
            self._longitude_ticks = -1
        else:
            raise ValueError("longitude_ticks must be `positive` or `symmetric`.")
        super().__init__()

    def _wrap_values(self, factor, values):
        """Wrap the values according to the wrap angle.

        Parameters
        ----------
        factor : `float`
            Scaling factor for input values
        values : `list`
            List of values to format

        Returns
        -------
        wrapped_values : `np.ndarray`
            Array of wrapped values, scaled by factor.
        """
        _values = np.atleast_1d(values)/factor
        # _values = (_values + self._wrap) % 360 - self._wrap
        _values = wrap_values(_values, wrap=self._wrap)
        if self._longitude_ticks == 1:
            # Values should all be positive, 0 to 360
            _values %= 360.0
        else:
            # Additional test to set -180 to positive 180
            _values[np.isclose(_values, -180.0)] = 180.0
        return factor*_values

    def __call__(self, direction, factor, values):
        return super().__call__(direction, factor, self._wrap_values(factor, values))


class ExtremeFinderWrapped(ExtremeFinderSimple):
    # docstring inherited

    def __init__(self, nx, ny, wrap_angle):
        """
        Find extremes with configurable wrap angle and correct limits.

        Parameters
        ----------
        nx : `int`
            Number of samples in x direction.
        ny : `int`
            Number of samples in y direction.
        wrap_angle : `float`
            Angle at which the 360-degree cycle should be wrapped.
        """
        self.nx, self.ny = nx, ny
        self._wrap = wrap_angle
        self._eps = 1e-5

    def __call__(self, transform_xy, x1, y1, x2, y2):
        # docstring inherited
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        with np.errstate(invalid='ignore'):
            lon = wrap_values(lon, wrap=self._wrap)

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        lon_min, lon_max, lat_min, lat_max = \
            self._add_pad(lon_min, lon_max, lat_min, lat_max)

        lat_min = np.clip(lat_min, -90.0 + self._eps, 90.0 - self._eps)
        lat_max = np.clip(lat_max, -90.0 + self._eps, 90.0 - self._eps)

        lon_min = np.clip(lon_min, self._wrap - 360. + self._eps, self._wrap - self._eps)
        lon_max = np.clip(lon_max, self._wrap - 360. + self._eps, self._wrap - self._eps)

        return lon_min, lon_max, lat_min, lat_max


class GridHelperSkyproj(GridHelperCurveLinear):
    """GridHelperCurveLinear with tick overlap protection and additional labels.
    """
    def __init__(self, *args, celestial=True, lon_0=0.0, full_sky_top_bottom_lon_0=False, **kwargs):
        self._celestial = celestial
        self._full_sky = False
        self._lon_0 = lon_0
        self._full_sky_top_bottom_lon_0 = full_sky_top_bottom_lon_0
        super().__init__(*args, **kwargs)

    def set_full_sky(self, full_sky):
        """Set the grid helper for full sky mode.

        Parameters
        ----------
        full_sky : `bool`
        """
        self._full_sky = full_sky

    def get_tick_iterator(self, nth_coord, axis_side, minor=False):

        try:
            _grid_info = self._grid_info
        except AttributeError:
            _grid_info = self.grid_info

        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if lon_or_lat == "lon":
            # Need to compute maximum extent in the x direction
            x_min = 1e100
            x_max = -1e100
            for line in _grid_info['lon_lines']:
                x_min = min((x_min, np.min(line[0])))
                x_max = max((x_max, np.max(line[0])))
            delta_x = x_max - x_min
        if not minor:  # major ticks
            tick_locs = _grid_info[lon_or_lat]["tick_locs"][axis_side]
            tick_labels = _grid_info[lon_or_lat]["tick_labels"][axis_side]

            if lon_or_lat == "lat" and self._full_sky and axis_side in ['left', 'right']:
                # We need additional ticks following the curved boundary
                if axis_side == 'left':
                    deltas = np.array([-179.99999, -175.])
                    dx_offset_sign = 1.0
                else:
                    deltas = np.array([179.99999, 175.0])
                    dx_offset_sign = -1.0

                levels = []
                tick_locs = []
                for lat_level in _grid_info["lat"]["levels"]:
                    # Skip any levels near the pole
                    if np.abs(np.abs(lat_level) - 90.0) < 5.0:
                        continue

                    xy = self.grid_finder.transform_xy(self._lon_0 + deltas, [lat_level]*2)
                    dx = xy[0, 1] - xy[0, 0]
                    dy = xy[1, 1] - xy[1, 0]
                    angle = np.rad2deg(np.arctan2(dy, dx))

                    label_dx = 0.0
                    label_dy = 0.0
                    if lat_level < 0.0:
                        # Extra shift for labels in the south.
                        xy2 = self.grid_finder.transform_xy([self._lon_0 + deltas[0]]*2,
                                                            [lat_level - 1.0, lat_level + 1.0])
                        dx2 = xy2[0, 1] - xy2[0, 0]
                        dy2 = xy2[1, 1] - xy2[1, 0]

                        if dx2 != 0:
                            bound_slope = dy2/dx2
                            label_dx = dx_offset_sign*dx/bound_slope/2.
                            label_dy = dx/bound_slope/2.

                    tick_locs.append(((xy[0, 0] + label_dx, xy[1, 0] + label_dy), angle))

                    levels.append(lat_level)

                tick_labels = self.grid_finder.tick_formatter2(axis_side, 1.0, levels)

            elif (lon_or_lat == "lon" and self._full_sky and self._full_sky_top_bottom_lon_0
                  and axis_side in ["top", "bottom"]):
                if axis_side == "top":
                    lat = 89.99999
                else:
                    lat = -89.99999
                xy = self.grid_finder.transform_xy(self._lon_0, lat)

                tick_locs = [(tuple(xy[:, 0]), 0.0)]
                if np.isclose(self._lon_0, 179.9999):
                    lon_0 = 180.0
                else:
                    lon_0 = self._lon_0
                tick_labels = self.grid_finder.tick_formatter1(axis_side, 1.0, [lon_0])

            prev_xy = None
            for ctr, ((xy, a), l) in enumerate(zip(tick_locs, tick_labels)):
                if self._celestial and _celestial_angle_360:
                    angle_normal = 360.0 - a
                else:
                    angle_normal = a

                if ctr > 0 and lon_or_lat == 'lon':
                    # Check if this is too close to the last label.
                    if abs(xy[0] - prev_xy[0])/delta_x < 0.05:
                        continue
                prev_xy = xy
                yield xy, angle_normal, angle_tangent, l
        else:
            for (xy, a), l in zip(
                    _grid_info[lon_or_lat]["tick_locs"][axis_side],
                    _grid_info[lon_or_lat]["tick_labels"][axis_side]):
                if self._celestial and _celestial_angle_360:
                    angle_normal = 360.0 - a
                else:
                    angle_normal = a
                yield xy, angle_normal, angle_tangent, ""
