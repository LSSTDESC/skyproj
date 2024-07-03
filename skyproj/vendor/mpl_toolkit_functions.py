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

# These are mpl_toolking GridHelperCurveLinear functions vendored from matplotlib 3.8.
# These are copied in here because the API changes with every version which breaks
# skyproj. Soon skyproj will be using its own tools that will not be so fragile, but
# until then this should be more stable.

import numpy as np
from itertools import chain

from mpl_toolkits.axisartist.axislines import GridHelperBase, _FixedAxisArtistHelperBase
from matplotlib.transforms import Bbox, Transform
from mpl_toolkits.axisartist.grid_finder import _User2DTransform
from mpl_toolkits.axisartist.axis_artist import AxisArtist
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple, MaxNLocator, FormatterPrettyPrint


class FixedAxisArtistHelper(_FixedAxisArtistHelperBase):
    """
    Helper class for a fixed axis.
    """

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super().__init__(loc=side)

        self.grid_helper = grid_helper
        if nth_coord_ticks is None:
            nth_coord_ticks = self.nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        v1, v2 = axes.get_ylim() if self.nth_coord == 0 else axes.get_xlim()
        if v1 > v2:  # Inverted limits.
            side = {"left": "right", "right": "left",
                    "top": "bottom", "bottom": "top"}[self.side]
        else:
            side = self.side
        g = self.grid_helper
        ti1 = g.get_tick_iterator(self.nth_coord_ticks, side)
        ti2 = g.get_tick_iterator(1-self.nth_coord_ticks, side, minor=True)
        return chain(ti1, ti2), iter([])


def _find_line_box_crossings(xys, bbox):
    """
    Find the points where a polyline crosses a bbox, and the crossing angles.

    Parameters
    ----------
    xys : (N, 2) array
        The polyline coordinates.
    bbox : `.Bbox`
        The bounding box.

    Returns
    -------
    list of ((float, float), float)
        Four separate lists of crossings, for the left, right, bottom, and top
        sides of the bbox, respectively.  For each list, the entries are the
        ``((x, y), ccw_angle_in_degrees)`` of the crossing, where an angle of 0
        means that the polyline is moving to the right at the crossing point.

        The entries are computed by linearly interpolating at each crossing
        between the nearest points on either side of the bbox edges.
    """
    crossings = []
    dxys = xys[1:] - xys[:-1]
    for sl in [slice(None), slice(None, None, -1)]:
        us, vs = xys.T[sl]  # "this" coord, "other" coord
        dus, dvs = dxys.T[sl]
        umin, vmin = bbox.min[sl]
        umax, vmax = bbox.max[sl]
        for u0, inside in [(umin, us > umin), (umax, us < umax)]:
            crossings.append([])
            idxs, = (inside[:-1] ^ inside[1:]).nonzero()
            for idx in idxs:
                v = vs[idx] + (u0 - us[idx]) * dvs[idx] / dus[idx]
                if not vmin <= v <= vmax:
                    continue
                crossing = (u0, v)[sl]
                theta = np.degrees(np.arctan2(*dxys[idx][::-1]))
                crossings[-1].append((crossing, theta))
    return crossings


class GridFinder:
    """
    Internal helper for `~.grid_helper_curvelinear.GridHelperCurveLinear`, with
    the same constructor parameters; should not be directly instantiated.
    """

    def __init__(self,
                 transform,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        if extreme_finder is None:
            extreme_finder = ExtremeFinderSimple(20, 20)
        if grid_locator1 is None:
            grid_locator1 = MaxNLocator()
        if grid_locator2 is None:
            grid_locator2 = MaxNLocator()
        if tick_formatter1 is None:
            tick_formatter1 = FormatterPrettyPrint()
        if tick_formatter2 is None:
            tick_formatter2 = FormatterPrettyPrint()
        self.extreme_finder = extreme_finder
        self.grid_locator1 = grid_locator1
        self.grid_locator2 = grid_locator2
        self.tick_formatter1 = tick_formatter1
        self.tick_formatter2 = tick_formatter2
        self.set_transform(transform)

    def get_grid_info(self, x1, y1, x2, y2):
        """
        lon_values, lat_values : list of grid values. if integer is given,
                           rough number of grids in each direction.
        """

        extremes = self.extreme_finder(self.inv_transform_xy, x1, y1, x2, y2)

        # min & max rage of lat (or lon) for each grid line will be drawn.
        # i.e., gridline of lon=0 will be drawn from lat_min to lat_max.

        lon_min, lon_max, lat_min, lat_max = extremes
        lon_levs, lon_n, lon_factor = self.grid_locator1(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        lat_levs, lat_n, lat_factor = self.grid_locator2(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)

        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor

        lon_lines, lat_lines = self._get_raw_grid_lines(lon_values,
                                                        lat_values,
                                                        lon_min, lon_max,
                                                        lat_min, lat_max)

        ddx = (x2-x1)*1.e-10
        ddy = (y2-y1)*1.e-10
        bb = Bbox.from_extents(x1-ddx, y1-ddy, x2+ddx, y2+ddy)

        grid_info = {
            "extremes": extremes,
            "lon_lines": lon_lines,
            "lat_lines": lat_lines,
            "lon": self._clip_grid_lines_and_find_ticks(
                lon_lines, lon_values, lon_levs, bb),
            "lat": self._clip_grid_lines_and_find_ticks(
                lat_lines, lat_values, lat_levs, bb),
        }

        tck_labels = grid_info["lon"]["tick_labels"] = {}
        for direction in ["left", "bottom", "right", "top"]:
            levs = grid_info["lon"]["tick_levels"][direction]
            tck_labels[direction] = self.tick_formatter1(
                direction, lon_factor, levs)

        tck_labels = grid_info["lat"]["tick_labels"] = {}
        for direction in ["left", "bottom", "right", "top"]:
            levs = grid_info["lat"]["tick_levels"][direction]
            tck_labels[direction] = self.tick_formatter2(
                direction, lat_factor, levs)

        return grid_info

    def _get_raw_grid_lines(self,
                            lon_values, lat_values,
                            lon_min, lon_max, lat_min, lat_max):

        lons_i = np.linspace(lon_min, lon_max, 100)  # for interpolation
        lats_i = np.linspace(lat_min, lat_max, 100)

        lon_lines = [self.transform_xy(np.full_like(lats_i, lon), lats_i)
                     for lon in lon_values]
        lat_lines = [self.transform_xy(lons_i, np.full_like(lons_i, lat))
                     for lat in lat_values]

        return lon_lines, lat_lines

    def _clip_grid_lines_and_find_ticks(self, lines, values, levs, bb):
        gi = {
            "values": [],
            "levels": [],
            "tick_levels": dict(left=[], bottom=[], right=[], top=[]),
            "tick_locs": dict(left=[], bottom=[], right=[], top=[]),
            "lines": [],
        }

        tck_levels = gi["tick_levels"]
        tck_locs = gi["tick_locs"]
        for (lx, ly), v, lev in zip(lines, values, levs):
            tcks = _find_line_box_crossings(np.column_stack([lx, ly]), bb)
            gi["levels"].append(v)
            gi["lines"].append([(lx, ly)])

            for tck, direction in zip(tcks,
                                      ["left", "right", "bottom", "top"]):
                for t in tck:
                    tck_levels[direction].append(lev)
                    tck_locs[direction].append(t)

        return gi

    def set_transform(self, aux_trans):
        if isinstance(aux_trans, Transform):
            self._aux_transform = aux_trans
        elif len(aux_trans) == 2 and all(map(callable, aux_trans)):
            self._aux_transform = _User2DTransform(*aux_trans)
        else:
            raise TypeError("'aux_trans' must be either a Transform "
                            "instance or a pair of callables")

    def get_transform(self):
        return self._aux_transform

    update_transform = set_transform  # backcompat alias.

    def transform_xy(self, x, y):
        return self._aux_transform.transform(np.column_stack([x, y])).T

    def inv_transform_xy(self, x, y):
        return self._aux_transform.inverted().transform(
            np.column_stack([x, y])).T

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k in ["extreme_finder",
                     "grid_locator1",
                     "grid_locator2",
                     "tick_formatter1",
                     "tick_formatter2"]:
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown update property {k!r}")


class GridHelperCurveLinear(GridHelperBase):
    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        Parameters
        ----------
        aux_trans : `.Transform` or tuple[Callable, Callable]
            The transform from curved coordinates to rectilinear coordinate:
            either a `.Transform` instance (which provides also its inverse),
            or a pair of callables ``(trans, inv_trans)`` that define the
            transform and its inverse.  The callables should have signature::

                x_rect, y_rect = trans(x_curved, y_curved)
                x_curved, y_curved = inv_trans(x_rect, y_rect)

        extreme_finder

        grid_locator1, grid_locator2
            Grid locators for each axis.

        tick_formatter1, tick_formatter2
            Tick formatters for each axis.
        """
        super().__init__()
        self._grid_info = None
        self.grid_finder = GridFinder(aux_trans,
                                      extreme_finder,
                                      grid_locator1,
                                      grid_locator2,
                                      tick_formatter1,
                                      tick_formatter2)

    def update_grid_finder(self, aux_trans=None, **kwargs):
        if aux_trans is not None:
            self.grid_finder.update_transform(aux_trans)
        self.grid_finder.update(**kwargs)
        self._old_limits = None  # Force revalidation.

    def new_fixed_axis(self, loc,
                       nth_coord=None,
                       axis_direction=None,
                       offset=None,
                       axes=None):
        if axes is None:
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        helper = FixedAxisArtistHelper(self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        # Why is clip not set on axisline, unlike in new_floating_axis or in
        # the floating_axig.GridHelperCurveLinear subclass?
        return axisline

    def new_floating_axis(self, nth_coord,
                          value,
                          axes=None,
                          axis_direction="bottom"
                          ):
        """
        if axes is None:
            axes = self.axes
        helper = FloatingAxisArtistHelper(
            self, nth_coord, value, axis_direction)
        axisline = AxisArtist(axes, helper)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        # axisline.major_ticklabels.set_visible(True)
        # axisline.minor_ticklabels.set_visible(False)
        return axisline
        """
        raise NotImplementedError("Removed from vendored code.")

    def _update_grid(self, x1, y1, x2, y2):
        self._grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)

    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []
        if axis in ["both", "x"]:
            for gl in self._grid_info["lon"]["lines"]:
                grid_lines.extend(gl)
        if axis in ["both", "y"]:
            for gl in self._grid_info["lat"]["lines"]:
                grid_lines.extend(gl)
        return grid_lines

    def get_tick_iterator(self, nth_coord, axis_side, minor=False):

        # axisnr = dict(left=0, bottom=1, right=2, top=3)[axis_side]
        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        # angle = [0, 90, 180, 270][axisnr]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if not minor:  # major ticks
            for (xy, a), l in zip(
                    self._grid_info[lon_or_lat]["tick_locs"][axis_side],
                    self._grid_info[lon_or_lat]["tick_labels"][axis_side]):
                angle_normal = a
                yield xy, angle_normal, angle_tangent, l
        else:
            for (xy, a), l in zip(
                    self._grid_info[lon_or_lat]["tick_locs"][axis_side],
                    self._grid_info[lon_or_lat]["tick_labels"][axis_side]):
                angle_normal = a
                yield xy, angle_normal, angle_tangent, ""
            # for xy, a, l in self._grid_info[lon_or_lat]["ticks"][axis_side]:
            #     yield xy, a, ""
