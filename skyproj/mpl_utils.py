import numpy as np

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.axis_artist import TickLabels, LabelBase

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
    def __init__(self, nx, ny, wrap_angle, transform_lonlat_to_xy):
        self.nx, self.ny = nx, ny
        self._wrap = wrap_angle
        self._eps = 1e-5

        # When we set this up we know the maximum possible range in x and y
        # and make sure that we don't go out of this range.
        lon, lat = np.meshgrid(
            wrap_values(np.linspace(-180.0, 180.0, self.nx), wrap=self._wrap),
            np.linspace(-90.0, 90.0, self.ny),
        )
        x, y = transform_lonlat_to_xy(np.ravel(lon), np.ravel(lat))
        self.xmin = np.nanmin(x)
        self.xmax = np.nanmax(x)
        self.ymin = np.nanmin(y)
        self.ymax = np.nanmax(y)

    def __call__(self, transform_xy_to_lonlat, x1, y1, x2, y2):
        # docstring inherited
        x, y = np.meshgrid(
            np.linspace(
                np.clip(x1, self.xmin, self.xmax),
                np.clip(x2, self.xmin, self.xmax),
                self.nx,
            ),
            np.linspace(
                np.clip(y1, self.ymin, self.ymax),
                np.clip(y2, self.ymin, self.ymax),
                self.ny,
            ),
        )
        lon, lat = transform_xy_to_lonlat(np.ravel(x), np.ravel(y))

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
        # self.zorder = 10

    @property
    def visible(self):
        return self._visible

    def set_from_tick_iterator(self, tick_iter):
        """
        """
        ticklabel_add_angle = dict(left=180, right=0, bottom=0, top=180)[self._axis_direction]

        ticklabels_loc_angle_label = []
        self._alignments = []
        self._outers = []

        for loc, angle_normal, angle_tangent, label, alignment, outer in tick_iter:
            angle_label = angle_tangent - 90 + ticklabel_add_angle
            ticklabels_loc_angle_label.append([loc, angle_label, label])
            self._alignments.append(alignment)
            self._outers.append(outer)

        self.set_locs_angles_labels(ticklabels_loc_angle_label)

    def get_texts_widths_heights_descents(self, renderer):
        """
        Return a list of ``(width, height, descent)`` tuples for ticklabels.

        Empty labels are left out.
        """
        whd_list = []
        for (_loc, _angle, label), outer in zip(self._locs_angles_labels, self._outers):
            if not label.strip() or not outer:
                continue
            clean_line, ismath = self._preprocess_math(label)
            whd = renderer.get_text_width_height_descent(
                clean_line, self._fontproperties, ismath=ismath)
            whd_list.append(whd)
        return whd_list

    def draw(self, renderer):
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return

        print("SkyTickLabels: ", self.zorder)

        r, total_width = self._get_ticklabels_offsets(renderer,
                                                      self._axis_direction)

        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad

        ha_default = self.get_ha()
        va_default = self.get_va()

        for ((x, y), a, l), (ha, va) in zip(self._locs_angles_labels, self._alignments):
            if not l.strip():
                continue
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            if ha == "":
                self.set_ha(ha_default)
            else:
                self.set_ha(ha)
            if va == "":
                self.set_va(va_default)
            else:
                self.set_va(va)
            # print("setting text to ", x, y, a, ha, va, ha_default, va_default, l)
            self.set_text(l)
            LabelBase.draw(self, renderer)

        # the value saved will be used to draw axislabel.
        self._axislabel_pad = total_width + pad
        print(self._axis_direction, total_width, pad, self._external_pad, self._axislabel_pad)
