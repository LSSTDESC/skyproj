import numpy as np

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.axis_artist import TickLabels

from .utils import wrap_values


__all__ = ['WrappedFormatterDMS', 'ExtremeFinderWrapped', 'SkyTickLabels']


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


class SkyTickLabels(TickLabels):
    def __init__(self, *, axis_direction="bottom", visible=True, **kwargs):
        super().__init__(axis_direction=axis_direction, **kwargs)

        self._axis_direction = axis_direction
        self._visible = visible

    @property
    def visible(self):
        return self._visible

    def set_from_tick_iterator(self, tick_iter):
        """
        """
        ticklabel_add_angle = dict(left=180, right=0, bottom=0, top=180)[self._axis_direction]

        # ticks_loc_angle = []
        ticklabels_loc_angle_label = []

        for loc, angle_normal, angle_tangent, label in tick_iter:
            angle_label = angle_tangent - 90 + ticklabel_add_angle
            ticklabels_loc_angle_label.append([loc, angle_label, label])

        self.set_locs_angles_labels(ticklabels_loc_angle_label)
