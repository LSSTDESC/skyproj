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
    """A tick formatter that handles longitude wrapping.

    Parameters
    ----------
    wrap : `float`
        Angle (degrees) which should be wrapped (subtract 360).
    longitude_ticks : `str`
        Type of longitude ticks, either ``positive`` (0 to 360)
        or ``symmetric`` (-180 to 180).
    """
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
    def __init__(self, nx, ny, wrap_angle):
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
    """GridHelperCurveLinear with tick overlap protection.

    Parameters
    ----------
    *args : `list`
        Arguments for ``GridHelperCurveLinear``.
    celestial : `bool`, optional
        Plot is celestial, and angles should be 0 to 360.  Otherwise -180 to 180.
    equatorial_labels : `bool`, optional
        Longitude labels are marked on the equator instead of edges.
    delta_cut : `float`, optional
        Gridline step (degrees) to signify a jump around a wrapped edge.
    **kwargs : `dict`, optional
        Additional kwargs for ``GridHelperCurveLinear``.
    """
    def __init__(self, *args, celestial=True, equatorial_labels=False, delta_cut=80.0, **kwargs):
        self._celestial = celestial
        self._equatorial_labels = equatorial_labels
        self._delta_cut = delta_cut

        super().__init__(*args, **kwargs)

    def get_gridlines(self, which="major", axis="both"):
        # docstring inherited
        grid_lines = []
        if axis in ["both", "x"]:
            for gl in self._grid_info["lon"]["lines"]:
                grid_lines.extend(self._cut_grid_line_jumps(gl))
        if axis in ["both", "y"]:
            for gl in self._grid_info["lat"]["lines"]:
                grid_lines.extend(self._cut_grid_line_jumps(gl))
        return grid_lines

    def _cut_grid_line_jumps(self, gl):
        """Check for jumps and cut gridlines into multiple sections.

        Parameters
        ----------
        gl : `list` [`tuple`]
            Input gridlines.  List of tuples of numpy arrays.

        Returns
        -------
        gl_new : `list` [`tuple`]
            New gridlines.  Jumps have been replaced with `np.nan`
            values to ensure lines are not connected around edges.
        """
        dx = gl[0][0][1:] - gl[0][0][: -1]
        dy = gl[0][1][1:] - gl[0][1][: -1]

        split, = (np.hypot(dx, dy) > self._delta_cut).nonzero()

        if split.size == 0:
            return gl

        gl_new = [(np.insert(gl[0][0], split + 1, np.nan),
                   np.insert(gl[0][1], split + 1, np.nan))]

        return gl_new

    def get_tick_iterator(self, nth_coord, axis_side, minor=False):
        # docstring inherited
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

            if lon_or_lat == "lon" and self._equatorial_labels:
                min_lat = np.min(_grid_info["lat"]["levels"])
                max_lat = np.max(_grid_info["lat"]["levels"])
                if (min_lat < -89.0 and axis_side == "bottom") \
                   or (max_lat > 89.0 and axis_side == "top"):
                    tick_locs = []
                    tick_labels = []

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
