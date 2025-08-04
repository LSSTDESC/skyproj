import functools
import numpy as np
import matplotlib.collections
from matplotlib.transforms import Bbox

from .skycrs import proj, proj_inverse
from .mpl_utils import ExtremeFinderWrapped, WrappedFormatterDMS
import mpl_toolkits.axisartist.angle_helper as angle_helper

__all__ = ["SkyGridHelper", "SkyGridlines"]


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
            cross = []
            idxs, = (inside[:-1] ^ inside[1:]).nonzero()
            for idx in idxs:
                v = vs[idx] + (u0 - us[idx]) * dvs[idx] / dus[idx]
                if not vmin <= v <= vmax:
                    continue
                crossing = (u0, v)[sl]
                theta = np.degrees(np.arctan2(*dxys[idx][::-1]))
                cross.append((crossing, theta, ("", "")))
            crossings.append(cross)
    return crossings


def _find_inner_crossings(line_x, line_y, lon_or_lat, min_x, max_x, min_y, max_y):
    """
    Find projection boundary crossings that are inside the axis bounding box.

    Parameters
    ----------
    line_x : `np.ndarray`
        x coordinates of the line.
    line_y : `np.ndarray`
        y coordinates of the line.
    lon_or_lat : `str`
        Must be ``lon`` or ``lat``.
    min_x : `float`
        Minimum x value of the bounding box.
    max_x : `float`
        Maximum x value of the bounding box.
    min_y : `float`
        Minimum y value of the bounding box.
    max_y : `float`
        Maximum y value of the bounding box.

    Returns
    -------
    crossings : `dict` [`str`, `tuple`]
        A crossing dictionary keyed by side (``left``/``right`` or
        ``top``/``bottom``) with a tuple of position, angle, and
        horizontal/vertical alignment strings.
    """
    crossings = {}
    if lon_or_lat == "lat":
        for side in ["left", "right"]:
            if side == "right":
                line_x = line_x[::-1]
                line_y = line_y[::-1]
            if line_x[0] <= min_x or line_x[0] >= max_x or \
               line_y[0] <= min_y or line_y[0] >= max_y:
                continue
            if line_y[0] < 0.0:
                va = "top"
            elif line_y[0] > 0.0:
                va = "bottom"
            else:
                va = "center"
            crossings[side] = ((line_x[0], line_y[0]), 0.0, ("", va))
    else:
        for side in ["top", "bottom"]:
            if side == "top":
                line_index = -1
                va = "bottom"
            else:
                line_index = 0
                va = "top"

            if line_x[line_index] <= min_x or line_x[line_index] >= max_x or \
               line_y[line_index] <= min_y or line_y[line_index] >= max_y:
                continue

            if line_x[0] < 0.0:
                ha = "left"
            elif line_x[0] > 0.0:
                ha = "right"
            else:
                ha = "center"

            crossings[side] = ((line_x[line_index], line_y[line_index]), 0.0, (ha, va))

    return crossings


class SkyGridHelper:
    """A helper class to compute quantities for a SkyGrid.

    Parameters
    ----------
    projection : `skyproj.SkyCRS`
        Coordinate reference system for this projection.
    wrap : `float`, optional
        Wrapping angle (degrees).
    n_grid_lon_default : `int`, optional
        Number of longitude grid lines for default.
    n_grid_lat_default : `int`, optional
        Number of latitude grid lines for default.
    longitude_ticks : `str`, optional
        If this is ``positive`` then longitudes will be from 0 to 360; if it
        is ``symmetric`` then the longitudes will be from -180 to 180.
    celestial : `bool`, optional
        Is this a celestial plot, inverting the latitude?
    equatorial_labels : `bool`, optional
        Should this map have longitude labels along the equator?
    full_circle : `bool`, optional
        Does this projection have a fully circular boundary?
    min_lon_ticklabel_delta : `float`, optional
        What is the minimal fraction of the total x axis size to allow
        a tick label to be plotted?
    draw_inner_lon_labels : `bool`, optional
        Draw "inner" longitude labels? Only necessary for certain
        projections (e.g. Albers).
    """
    def __init__(
        self,
        projection,
        wrap=0.0,
        n_grid_lon_default=None,
        n_grid_lat_default=None,
        longitude_ticks="positive",
        celestial=True,
        equatorial_labels=False,
        full_circle=False,
        min_lon_ticklabel_delta=0.1,
        draw_inner_lon_labels=False,
    ):
        self._transform_lonlat_to_xy = functools.partial(proj, projection=projection, wrap=wrap)
        self._transform_xy_to_lonlat = functools.partial(proj_inverse, projection=projection)
        self._wrap = wrap
        self._n_grid_lon_default = n_grid_lon_default
        self._n_grid_lat_default = n_grid_lat_default
        self._extreme_finder = ExtremeFinderWrapped(20, 20, wrap, self.transform_xy)
        self._grid_locator_lon = None
        self._grid_locator_lat = None
        self._tick_formatters = {
            "lon": WrappedFormatterDMS(180.0, longitude_ticks),
            "lat": angle_helper.FormatterDMS(),
        }
        self._celestial = celestial
        self._equatorial_labels = equatorial_labels
        self._full_circle = full_circle
        if projection.name == "cyl":
            self._delta_cut = 80.0
        else:
            self._delta_cut = 0.5*projection.radius
        self._min_lon_ticklabel_delta = min_lon_ticklabel_delta
        self._draw_inner_lon_labels = draw_inner_lon_labels

        self._grid_info = None
        self._old_limits = None

    def get_gridlines(self, axis="both"):
        """
        Get the grid lines associated with this grid helper.

        Parameters
        ----------
        axis : `str`, optional
            Get for the ``x`` axis, ``y`` axis, or ``both`` axes.

        Returns
        -------
        grid_lines : `list` of xy tuples.
            The grid lines in projected coordinates.
        """
        if self._grid_info is None:
            raise RuntimeError("Must first call update_lim(axis)")

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

    def update_lim(self, axis):
        """Update grid limits.

        Parameters
        ----------
        axis : `matplotlib.axis.Axis`
            Axis to use for limits to create grid limits.
        """
        x1, x2 = axis.get_xlim()
        y1, y2 = axis.get_ylim()
        if self._old_limits != (x1, x2, y1, y2):
            _n_grid_lon, _n_grid_lat = self._compute_n_grid_from_extent(
                axis.get_extent(),
                n_grid_lon_default=self._n_grid_lon_default,
                n_grid_lat_default=self._n_grid_lat_default,
            )

            if self._wrap == 180.0 and not self._full_circle:
                _include_last_lon = True
            else:
                _include_last_lon = False

            self._grid_locator_lon = angle_helper.LocatorD(_n_grid_lon, include_last=_include_last_lon)
            self._grid_locator_lat = angle_helper.LocatorD(_n_grid_lat, include_last=True)

            self._update_grid(x1, y1, x2, y2)
            self._old_limits = (x1, x2, y1, y2)

    def _update_grid(self, x1, y1, x2, y2):
        """Update the grid based on axis limits.

        Parameters
        ----------
        x1 : `float`
            Axis lower xlim.
        x2 : `float`
            Axis upper xlim.
        y1 : `float`
            Axis lower ylim.
        y2 : `float`
            Axis upper ylim.
        """
        self._grid_info = self._get_grid_info(x1, y1, x2, y2)

    def _get_grid_info(self, x1, y1, x2, y2):
        """Get the grid info structure for this grid.

        Parameters
        ----------
        x1 : `float`
            Axis lower xlim.
        x2 : `float`
            Axis upper xlim.
        y1 : `float`
            Axis lower ylim.
        y2 : `float`
            Axis upper ylim.

        Returns
        -------
        grid_info : `dict`
            Dictionary with ``extremes``, ``bbox``, ``lon``, and ``lat``.
        """
        extremes = self._extreme_finder(self.inv_transform_xy, x1, y1, x2, y2)

        # min & max rage of lat (or lon) for each grid line will be drawn.
        # i.e., gridline of lon=0 will be drawn from lat_min to lat_max.

        lon_min, lon_max, lat_min, lat_max = extremes
        lon_levs, lon_n, lon_factor = self._grid_locator_lon(lon_min, lon_max)
        lon_levs = np.asarray(lon_levs)
        lat_levs, lat_n, lat_factor = self._grid_locator_lat(lat_min, lat_max)
        lat_levs = np.asarray(lat_levs)

        lon_values = lon_levs[:lon_n] / lon_factor
        lat_values = lat_levs[:lat_n] / lat_factor

        lon_lines, lat_lines = self._get_raw_grid_lines(lon_values,
                                                        lat_values,
                                                        lon_min, lon_max,
                                                        lat_min, lat_max)

        bb = Bbox.from_extents(x1, y1, x2, y2).expanded(1 + 2e-10, 1 + 2e-10)

        grid_info = {
            "extremes": extremes,
            "bounding_box": bb,
            # "lon", "lat", filled below.
        }

        use_equatorial_labels = False
        if self._equatorial_labels:
            if np.min(lat_values) < -89.0 and np.max(lat_values) > 89.0:
                use_equatorial_labels = True

        # inverted = False
        if x1 < x2:
            # gi_side_map = {side: side for side in ['left', 'right']}
            min_x = x1
            max_x = x2
        else:
            # gi_side_map = {'left': 'right',
            #                'right': 'left'}
            # inverted = True
            min_x = x2
            max_x = x1

        for lon_or_lat, levs, factor, values, lines in [
                ("lon", lon_levs, lon_factor, lon_values, lon_lines),
                ("lat", lat_levs, lat_factor, lat_values, lat_lines),
        ]:
            grid_info[lon_or_lat] = gi = {
                "lines": [[line] for line in lines],
                "ticks": {"left": [], "right": [], "bottom": [], "top": []},
            }
            for (lx, ly), v, level in zip(lines, values, levs):
                if lon_or_lat == "lon" and use_equatorial_labels:
                    xy = self.transform_xy(level, 0.0)[:, 0]
                    # Make sure we don't try to label at the extreme edges.
                    if xy[0] > min_x and xy[0] < max_x:
                        gi["ticks"]["top"].append(
                            {
                                "level": level,
                                "loc": (xy, 0.0, ("right", "center")),
                                "outer": False,
                            },
                        )
                else:
                    all_crossings = _find_line_box_crossings(np.column_stack([lx, ly]), bb)
                    for side, crossings in zip(
                            ["left", "right", "bottom", "top"], all_crossings):
                        for crossing in crossings:
                            if side in ("left", "right"):
                                crossing_ = (crossing[0], crossing[1], ("", "center"))
                            else:
                                crossing_ = (crossing[0], crossing[1], ("center", ""))
                            gi["ticks"][side].append({"level": level, "loc": crossing_, "outer": True})

                    # Don't mark the inner poles.
                    if lon_or_lat == "lat" and level in [-90.0, 90.0]:
                        continue
                    # Do not draw inner longitudes.
                    if lon_or_lat == "lon" and not self._draw_inner_lon_labels:
                        continue

                    inner_crossings = _find_inner_crossings(lx, ly, lon_or_lat, min_x, max_x, y1, y2)

                    for side in ["left", "right", "bottom", "top"]:
                        if side in inner_crossings:
                            gi["ticks"][side].append(
                                {
                                    "level": level,
                                    "loc": inner_crossings[side],
                                    "outer": False,
                                }
                            )

            for side in gi["ticks"]:
                levs = [tick["level"] for tick in gi["ticks"][side]]
                labels = self._tick_formatters[lon_or_lat](side, factor, levs)
                for tick, label in zip(gi["ticks"][side], labels):
                    tick["label"] = label

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

    def transform_xy(self, lon, lat):
        """Transform from lon/lat to x/y.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitudes.
        lat : `np.ndarray`
            Array of latitudes.

        Returns
        -------
        x : `np.ndarray`
            Array of projected x values.
        y : `np.ndarray`
            Array of projected y values.
        """
        return np.column_stack(self._transform_lonlat_to_xy(lon, lat)).T

    def inv_transform_xy(self, x, y):
        """Transform from x/y to lon/lat.

        Parameters
        ----------
        x : `np.ndarray`
            Array of projected x values.
        y : `np.ndarray`
            Array of projected y values.

        Returns
        -------
        lon : `np.ndarray`
            Array of longitudes.
        lat : `np.ndarray`
            Array of latitudes.
        """
        return np.column_stack(self._transform_xy_to_lonlat(x, y)).T

    def get_tick_iterator(self, lon_or_lat, axis_side):
        """Get the tick iterator for a given axis/side.

        Parameters
        ----------
        lon_or_lat : `str`
            Must be ``lon`` or ``lat``.
        axis_side : `str`
            Must be ``left``, ``right``, ``top``, or ``bottom``.

        Returns
        -------
        tick_iterator : `Iterator`
            Iterator with xy (coordinate tuple), angle_normal (normal angle
            to the axis at the coordinate location), angle_tangent (tangent
            angle), label (label string), alignment (label alignment when
            drawing), and outer (boolean to denote if this is a label
            outside the boundary [True] or inside [False]).
        """
        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        if lon_or_lat == "lon":
            # Need to compute maximum extent in the x direction
            delta_x = np.abs(self._grid_info["bounding_box"].x0 - self._grid_info["bounding_box"].x1)

        tick_locs = [item["loc"] for item in self._grid_info[lon_or_lat]["ticks"][axis_side]]
        tick_labels = [item["label"] for item in self._grid_info[lon_or_lat]["ticks"][axis_side]]
        tick_outers = [item["outer"] for item in self._grid_info[lon_or_lat]["ticks"][axis_side]]

        prev_xy = None
        for ctr, ((xy, a, alignment), label, outer) in enumerate(zip(tick_locs, tick_labels, tick_outers)):
            if self._celestial:
                angle_normal = 360.0 - a
            else:
                angle_normal = a

            if ctr > 0 and lon_or_lat == 'lon':
                # Check if this is too close to the last label.
                if abs(xy[0] - prev_xy[0])/delta_x < self._min_lon_ticklabel_delta:
                    continue
            prev_xy = xy
            yield xy, angle_normal, angle_tangent, label, alignment, outer

    def _compute_n_grid_from_extent(self, extent, n_grid_lat_default=None, n_grid_lon_default=None):
        """Compute the number of grid lines from the extent.

        This will respect values that were set at initialization time.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        n_grid_lat_default : `int`, optional
            Requested number of latitude gridlines; otherwise automatic.
        n_grid_lon_default : `int`, optional
            Requested number of longitude gridlines; otherwise automatic.

        Returns
        -------
        n_grid_lon : `int`
            Number of gridlines in the longitude direction.
        n_grid_lat : `int`
            Number of gridlines in the latitude direction.
        """
        if n_grid_lat_default is None:
            n_grid_lat = 6
        else:
            n_grid_lat = n_grid_lat_default

        if n_grid_lon_default is None:
            latscale = np.cos(np.deg2rad(np.mean(extent[2:])))
            ratio = np.clip(np.abs(extent[1] - extent[0])*latscale/(extent[3] - extent[2]), 1./3., 5./3.)
            n_grid_lon = int(np.ceil(ratio * n_grid_lat))
        else:
            n_grid_lon = n_grid_lon_default

        return n_grid_lon, n_grid_lat

    @property
    def full_circle(self):
        return self._full_circle


class SkyGridlines(matplotlib.collections.LineCollection):
    """A class to describe a set of grid lines on a SkyProj plot.

    Parameters
    ----------
    segments : `list` [array-like]
        List of line segments. See matplotlib.collections.LineCollection.
    grid_helper : `skyproj.SkyGridHelper`, optional
        Helper class for computing grid values. May be initialized empty,
        but must be followed up with `set_grid_helper()` later.
    """
    def __init__(self, segments=[], grid_helper=None, **kwargs):
        super().__init__(segments, **kwargs)

        # Note that this will not work unless you call
        # set_clip_box(axes.bbox) before drawing.

        self.set_clip_on(True)

        self._grid_helper = grid_helper

    def set_grid_helper(self, grid_helper):
        """Set the grid helper.

        Parameters
        ----------
        grid_helper : `skyproj.SkyGridHelper`
        """
        self._grid_helper = grid_helper

    def get_tick_iterator(self, lon_or_lat, axis_side):
        """Get a tick iterator from the grid helper.

        Parameters
        ----------
        lon_or_lat : `str`
            String determining whether to return the longitude or latitude
            tick iterator.
        axis_side : `str`
            Axis side for tick iterator (``left``, ``right``, ``top``, or
            ``bottom``.)

        Returns
        -------
        tick_iterator : `Iterator`
            Iterator with xy (coordinate tuple), angle_normal (normal angle
            to the axis at the coordinate location), angle_tangent (tangent
            angle), label (label string), alignment (label alignment when
            drawing), and outer (boolean to denote if this is a label
            outside the boundary [True] or inside [False]).
        """
        return self._grid_helper.get_tick_iterator(lon_or_lat, axis_side)

    @property
    def full_circle(self):
        return self._grid_helper.full_circle

    def draw(self, renderer):
        # docstring inherited
        gridlines = self._grid_helper.get_gridlines()

        self.set_segments([np.transpose(line) for line in gridlines])

        super().draw(renderer)
