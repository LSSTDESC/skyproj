# License agreement for matplotlib versions 1.3.0 and later
# =========================================================
#
# 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
# ("MDT"), and the Individual or Organization ("Licensee") accessing and
# otherwise using matplotlib software in source or binary form and its
# associated documentation.
#
# 2. Subject to the terms and conditions of this License Agreement, MDT
# hereby grants Licensee a nonexclusive, royalty-free, world-wide license
# to reproduce, analyze, test, perform and/or display publicly, prepare
# derivative works, distribute, and otherwise use matplotlib
# alone or in any derivative version, provided, however, that MDT's
# License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
# 2012- Matplotlib Development Team; All Rights Reserved" are retained in
# matplotlib  alone or in any derivative version prepared by
# Licensee.
#
# 3. In the event Licensee prepares a derivative work that is based on or
# incorporates matplotlib or any part thereof, and wants to
# make the derivative work available to others as provided herein, then
# Licensee hereby agrees to include in any such work a brief summary of
# the changes made to matplotlib .
#
# 4. MDT is making matplotlib available to Licensee on an "AS
# IS" basis.  MDT MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
# IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND
# DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
# FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB
# WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
#
# 5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB
#  FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR
# LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING
# MATPLOTLIB , OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF
# THE POSSIBILITY THEREOF.
#
# 6. This License Agreement will automatically terminate upon a material
# breach of its terms and conditions.
#
# 7. Nothing in this License Agreement shall be deemed to create any
# relationship of agency, partnership, or joint venture between MDT and
# Licensee.  This License Agreement does not grant permission to use MDT
# trademarks or trade name in a trademark sense to endorse or promote
# products or services of Licensee, or any third party.
#
# 8. By copying, installing or otherwise using matplotlib ,
# Licensee agrees to be bound by the terms and conditions of this License
# Agreement.

import numpy as np

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
import mpl_toolkits.axisartist.angle_helper as angle_helper

import matplotlib.text as mtext
from matplotlib.transforms import Affine2D
from matplotlib import _api

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
        return np.round(factor*_values).astype(np.int64)

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
            wrap_values(np.linspace(-180.0, 180.0 - self._eps, self.nx), wrap=self._wrap),
            np.linspace(-90.0 + self._eps, 90.0 - self._eps, self.ny),
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


class SkyTickLabels(mtext.Text):
    def __init__(self, *, axis_direction="bottom", visible=True, **kwargs):
        super().__init__(**kwargs)

        # Initialization vendored from AxisLabel.
        self.set_axis_direction(axis_direction)
        self._axislabel_pad = 0
        self._pad = 5
        self._external_pad = 0

        # Initialization vendored from LabelBase.
        self.locs_and_labels = []
        self._ref_angle = 0
        self._offset_radius = 0.
        self.set_rotation_mode("anchor")
        self._text_follow_ref_angle = True

        # Extra initialization.
        self._visible = visible
        self._padding_computed = False
        self._alignments = []
        self._outers = []
        self.ticklabels_loc_angle_label = []

    # ==================================================
    # The following methods are vendored from AxisLabel.
    def get_pad(self):
        """
        Return the internal pad in points.

        See `.set_pad` for more details.
        """
        return self._pad

    def set_pad(self, pad):
        """
        Set the internal pad in points.

        The actual pad will be the sum of the internal pad and the
        external pad (the latter is set automatically by the `.AxisArtist`).

        Parameters
        ----------
        pad : float
            The internal pad in points.
        """
        self._pad = pad

    def set_default_alignment(self, d):
        """
        Set the default alignment. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        va, ha = _api.check_getitem(self._default_alignments, d=d)
        self.set_va(va)
        self.set_ha(ha)

    def set_default_angle(self, d):
        """
        Set the default angle. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_rotation(_api.check_getitem(self._default_angles, d=d))

    def set_axis_direction(self, d):
        """
        Adjust the text angle and text alignment of axis label
        according to the matplotlib convention.

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_default_alignment(d)
        self.set_default_angle(d)
        self._axis_direction = d

    # The previous methods are vendored from AxisLabel.
    # =================================================

    # =================================================
    # The following methods are vendored from TickLabel.

    _default_alignments = dict(left=("center", "right"),
                               right=("center", "left"),
                               bottom=("baseline", "center"),
                               top=("baseline", "center"))

    _default_angles = dict(left=90,
                           right=-90,
                           bottom=0,
                           top=180)

    def set_locs_angles_labels(self, locs_angles_labels):
        self._locs_angles_labels = locs_angles_labels

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

    def _get_ticklabels_offsets(self, renderer, label_direction):
        """
        Calculate the ticklabel offsets from the tick and their total heights.

        The offset only takes account the offset due to the vertical alignment
        of the ticklabels: if axis direction is bottom and va is 'top', it will
        return 0; if va is 'baseline', it will return (height-descent).
        """
        whd_list = self.get_texts_widths_heights_descents(renderer)

        if not whd_list:
            return 0, 0

        r = 0
        va, ha = self.get_va(), self.get_ha()

        if label_direction == "left":
            pad = max(w for w, h, d in whd_list)
            if ha == "left":
                r = pad
            elif ha == "center":
                r = .5 * pad
        elif label_direction == "right":
            pad = max(w for w, h, d in whd_list)
            if ha == "right":
                r = pad
            elif ha == "center":
                r = .5 * pad
        elif label_direction == "bottom":
            pad = max(h for w, h, d in whd_list)
            if va == "bottom":
                r = pad
            elif va == "center":
                r = .5 * pad
            elif va == "baseline":
                max_ascent = max(h - d for w, h, d in whd_list)
                max_descent = max(d for w, h, d in whd_list)
                r = max_ascent
                pad = max_ascent + max_descent
        elif label_direction == "top":
            pad = max(h for w, h, d in whd_list)
            if va == "top":
                r = pad
            elif va == "center":
                r = .5 * pad
            elif va == "baseline":
                max_ascent = max(h - d for w, h, d in whd_list)
                max_descent = max(d for w, h, d in whd_list)
                r = max_descent
                pad = max_ascent + max_descent

        # r : offset
        # pad : total height of the ticklabels. This will be used to
        # calculate the pad for the axislabel.
        return r, pad

    # The previous methods are vendored from TickLabel.
    # =================================================

    # =================================================
    # The following methods are vendored from LabelBase.

    @property
    def _offset_ref_angle(self):
        return self._ref_angle

    @property
    def _text_ref_angle(self):
        if self._text_follow_ref_angle:
            return self._ref_angle + 90
        else:
            return 0

    # The previous methods are vendored from LabelBase.
    # =================================================

    @property
    def visible(self):
        return self._visible

    def reset_tick_iterator(self):
        """Reset the tick iterators.
        """
        self.ticklabels_loc_angle_label = []
        self._alignments = []
        self._outers = []

    def set_from_tick_iterator(self, tick_iter, reset=True):
        """Set label parameters from a tick iterator.

        Parameters
        ----------
        tick_iter : `tuple` [`tuple` [`float`], `float`, `float`, `str`, `str`, `bool`
            Tuple with x/y position, normal angle, tangent angle,
            label name, alignment string, and whether this is
            an outer label or not.
        reset : `bool`, optional
            Reset the ticks?
        """
        ticklabel_add_angle = dict(left=180, right=0, bottom=0, top=180)[self._axis_direction]

        if reset:
            self.reset_tick_iterator()

        for loc, angle_normal, angle_tangent, label, alignment, outer in tick_iter:
            angle_label = angle_tangent - 90 + ticklabel_add_angle
            self.ticklabels_loc_angle_label.append([loc, angle_label, label])
            self._alignments.append(alignment)
            self._outers.append(outer)

        self.set_locs_angles_labels(self.ticklabels_loc_angle_label)

    def compute_padding(self, renderer):
        """Compute axis padding from labels.

        This will set the _axislabel_pad attribute based on the sizes
        of the tick labels, as rendered by the renderer.

        Parameters
        ----------
        renderer : `matplotlib.backends.Renderer`
            Matplotlib renderer.
        """
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return

        self._padding_computed = True

        r, total_width = self._get_ticklabels_offsets(renderer,
                                                      self._axis_direction)

        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad

        self._axislabel_pad = total_width + pad

    def draw(self, renderer):
        # docstring inherited
        if not self._padding_computed:
            self.compute_padding(renderer)

        if not self.get_visible():
            return

        ha_default = self.get_ha()
        va_default = self.get_va()

        tr = self.get_transform()
        angle_orig = self.get_rotation()

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
            self.set_text(l)

            # The following code is adapted from LabelBase.draw()
            theta = np.deg2rad(self._offset_ref_angle)
            dd = self._offset_radius
            dx, dy = dd * np.cos(theta), dd * np.sin(theta)

            self.set_transform(tr + Affine2D().translate(dx, dy))
            self.set_rotation(self._text_ref_angle + angle_orig)
            super().draw(renderer)

        # Restore original properties
        self.set_transform(tr)
        self.set_rotation(angle_orig)
        self.set_ha(ha_default)
        self.set_va(va_default)

        # Reset this variable.
        self._padding_computed = False
