import functools
import numpy as np
import warnings

import matplotlib as mpl
import matplotlib.axes
from pyproj import Geod

from .utils import wrap_values
from .skygrid import SkyGridlines, SkyGridHelper
from .mpl_utils import SkyTickLabels


__all__ = ["SkyAxes"]


GRIDLINES_ZORDER_DEFAULT = 10


def _add_lonlat(func):
    """Decorator to add lonlat and geodesic options."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if kwargs.pop("lonlat", True):
            kwargs["transform"] = self.projection
            geodesics = kwargs.pop("geodesic", True)
            kwargs["transform"].set_plot_geodesics(geodesics)
        return func(self, *args, **kwargs)
    return wrapper


class SkyAxes(matplotlib.axes.Axes):
    # docstring inherited
    def __init__(self, *args, **kwargs):
        self.projection = kwargs.pop("sky_crs", None)

        if self.projection is None:
            raise RuntimeError("Must specify sky_crs for initializing SkyAxes.")

        # We create empty gridlines and _ticklabels_visibility so that
        # the super().__init() has placeholders on first initialization.
        self.gridlines = SkyGridlines([])
        self.gridlines.set_zorder(GRIDLINES_ZORDER_DEFAULT)
        self._ticklabels_visibility = {}
        self._ticklabels = []

        super().__init__(*args, **kwargs)

        # The ``inherit`` name is special and means to use the same
        # color as x/ytick.color.
        if mpl.rcParams["xtick.labelcolor"] == "inherit":
            self._xlabelcolor = mpl.rcParams["xtick.color"]
        else:
            self._xlabelcolor = mpl.rcParams["xtick.labelcolor"]

        if mpl.rcParams["ytick.labelcolor"] == "inherit":
            self._ylabelcolor = mpl.rcParams["ytick.color"]
        else:
            self._ylabelcolor = mpl.rcParams["ytick.labelcolor"]

        self._ticklabels = {
            "left": SkyTickLabels(
                axis_direction="left",
                figure=self.figure,
                transform=self.transData,
                fontsize=mpl.rcParams["ytick.labelsize"],
                pad=mpl.rcParams["ytick.major.pad"],
                color=self._ylabelcolor,
            ),
            "right": SkyTickLabels(
                axis_direction="right",
                figure=self.figure,
                transform=self.transData,
                fontsize=mpl.rcParams["ytick.labelsize"],
                pad=mpl.rcParams["ytick.major.pad"],
                color=self._ylabelcolor,
            ),
            "top": SkyTickLabels(
                axis_direction="top",
                figure=self.figure,
                transform=self.transData,
                fontsize=mpl.rcParams["xtick.labelsize"],
                pad=mpl.rcParams["xtick.major.pad"],
                color=self._xlabelcolor,
            ),
            "bottom": SkyTickLabels(
                axis_direction="bottom",
                figure=self.figure,
                transform=self.transData,
                fontsize=mpl.rcParams["xtick.labelsize"],
                pad=mpl.rcParams["xtick.major.pad"],
                color=self._xlabelcolor,
            ),
        }

        self._ticklabels_visibility = {
            "left": True,
            "right": False,
            "top": True,
            "bottom": True,
        }

        self._xlabelpad = mpl.rcParams["axes.labelpad"]
        self._ylabelpad = mpl.rcParams["axes.labelpad"]

        # This needs to happen to make sure that it's all set correctly.
        self.clear()

    def clear(self):
        """Clear the current axes."""
        result = super().clear()

        # This will turn off all the built-in ticks.
        tick_param_dict = {
            "left": False,
            "right": False,
            "top": False,
            "bottom": False,
            "labelleft": False,
            "labelright": False,
            "labelbottom": False,
            "labeltop": False,
        }
        self.xaxis.set_tick_params(**tick_param_dict)
        self.yaxis.set_tick_params(**tick_param_dict)

        self.set_frame_on(False)

        # Always equal aspect ratio.
        self.set_aspect('equal')

        self._set_artist_props(self.gridlines)

        return result

    def grid(self, visible=False, which="major", axis="both",
             n_grid_lon=None, n_grid_lat=None,
             longitude_ticks="positive", equatorial_labels=False, celestial=True,
             full_circle=False, wrap=0.0, min_lon_ticklabel_delta=0.1,
             draw_inner_lon_labels=False,
             **kwargs):
        # docstring inherited

        self._grid_visible = visible

        if visible:
            # Set up grid finder and grid lines.

            grid_helper = SkyGridHelper(
                self.projection,
                wrap,
                n_grid_lon_default=n_grid_lon,
                n_grid_lat_default=n_grid_lat,
                longitude_ticks=longitude_ticks,
                celestial=celestial,
                equatorial_labels=equatorial_labels,
                full_circle=full_circle,
                min_lon_ticklabel_delta=min_lon_ticklabel_delta,
                draw_inner_lon_labels=draw_inner_lon_labels,
            )
            grid_helper.update_lim(self)

            self.gridlines.set_grid_helper(grid_helper)

        # We don't want the projection here because the gridlines
        # are all in projected coordinates.
        self.gridlines.set(**kwargs)

    def draw(self, renderer):
        # docstring inherited

        # Note that we first need to compute all the lon/lat label locations
        # and sizes to know the correct padding for the axis label.  But
        # we need to defer drawing of the labels until the end to ensure that
        # they end up on top.
        xaxis_pad, yaxis_pad = 0.0, 0.0

        labels_to_draw = []
        if self._grid_visible:
            # We need to update the limits and ensure the gridlines know
            # about the limits for clipping.
            self.gridlines._grid_helper.update_lim(self)
            self.gridlines.set_clip_box(self.bbox)

            for side in ["left", "right", "bottom", "top"]:
                self._ticklabels[side].reset_tick_iterator()

            # We only do labels if we have grid lines.
            for lon_or_lat, side in [("lon", "top"), ("lon", "bottom"), ("lat", "left"), ("lat", "right")]:
                if self._ticklabels_visibility[self._ticklabels[side]._axis_direction]:
                    tick_iter = self.gridlines.get_tick_iterator(lon_or_lat, side)
                    self._ticklabels[side].set_from_tick_iterator(tick_iter)
                    self._ticklabels[side].compute_padding(renderer)
                    labels_to_draw.append(self._ticklabels[side])
                    if side == "top" or side == "bottom":
                        xaxis_pad = max(xaxis_pad, self._ticklabels[side]._axislabel_pad)
                    else:
                        yaxis_pad = max(yaxis_pad, self._ticklabels[side]._axislabel_pad)

            if self.gridlines.full_circle:
                # In the case of full circle, we will draw the left/right
                # longitude labels if we want to draw either left or right.
                if self._ticklabels_visibility["left"] or self._ticklabels_visibility["right"]:
                    for lon_or_lat, side in [("lon", "left"), ("lon", "right")]:
                        tick_iter = self.gridlines.get_tick_iterator(lon_or_lat, side)
                        self._ticklabels[side].set_from_tick_iterator(tick_iter, reset=False)
                        self._ticklabels[side].compute_padding(renderer)
                        labels_to_draw.append(self._ticklabels[side])
                        yaxis_pad = max(yaxis_pad, self._ticklabels[side]._axislabel_pad)

        self.xaxis.labelpad = xaxis_pad/renderer.points_to_pixels(1.0) + self._xlabelpad
        self.yaxis.labelpad = yaxis_pad/renderer.points_to_pixels(1.0) + self._ylabelpad

        if self._grid_visible:
            self.add_artist(self.gridlines)

        super().draw(renderer)

        if self._grid_visible:
            # RA/Dec labels must be drawn on top, after everything else
            # is rendered.
            for label_to_draw in labels_to_draw:
                label_to_draw.draw(renderer)

    def invert_xaxis(self):
        super().invert_xaxis()

        if self.xaxis_inverted():
            self._ticklabels["left"].set_axis_direction("right")
            self._ticklabels["right"].set_axis_direction("left")
        else:
            self._ticklabels["left"].set_axis_direction("left")
            self._ticklabels["right"].set_axis_direction("right")

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
                xy = self.projection.transform_points(lon.ravel(), lat.ravel())
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
                xy = self.projection.transform_points(lon, lat)
                # FIXME NOTE NEED TO KNOW LON_0/WRAP OF PROJECTION...
                x0 = np.min(xy[:, 0])
                x1 = np.max(xy[:, 0])
                y0 = np.min(xy[:, 1])
                y1 = np.max(xy[:, 1])

        self.set_xlim([x0, x1])
        self.set_ylim([y0, y1])

        # FIXME: do automatic inversion here.

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
            # Make a ring of points + center and check their extent.
            npt = 500
            x_pts = np.linspace(x0, x1, npt)
            y_pts = np.linspace(y0, y1, npt)
            # The Lambert Azimuthal Equal Area projection needs the center point.
            if self.projection.name == "laea":
                # Also ensure we have the center point, and 0, 0 if that is
                # contained.
                x_center = (x0 + x1) / 2.
                y_center = (y0 + y1) / 2.
                if (y0 < 0.0 and y1 > 0.0) and \
                   ((x0 < 0.0 and x1 > 0.0) or (x0 > 0.0 and x1 < 0.0)):
                    x_centers = [x_center, 0.0]
                    y_centers = [y_center, 0.0]
                else:
                    x_centers = [x_center]
                    y_centers = [y_center]
            else:
                x_centers = []
                y_centers = []
            x = np.concatenate((x_pts, x_pts, np.repeat(x0, npt), np.repeat(x1, npt), x_centers))
            y = np.concatenate((np.repeat(y0, npt), np.repeat(y1, npt), y_pts, y_pts, y_centers))
            lonlat = self.projection.transform_points(x, y, inverse=True)

            # Check for out-of-bounds by reverse-projecting
            xy = self.projection.transform_points(lonlat[:, 0], lonlat[:, 1])
            bad = ((~np.isclose(xy[:, 0], x)) | (~np.isclose(xy[:, 1], y)))
            lonlat[bad, :] = np.nan

            # We need to wrap values to get the correct range
            wrap = (self.projection.lon_0 + 180.) % 360.
            with warnings.catch_warnings():
                # Some longitude values may be nan, so we filter these expected warnings.
                warnings.simplefilter("ignore")
                lon_wrap = wrap_values(lonlat[:, 0], wrap)

            if np.all(np.isnan(lon_wrap)):
                lon0 = -180.0 + 1e-5
                lon1 = 180.0 - 1e-5
            else:
                lon0 = np.nanmin(lon_wrap)
                lon1 = np.nanmax(lon_wrap)

                if np.isclose(lon0, -180.0, atol=1.0):
                    lon0 = -180.0 + 1e-5
                if np.isclose(lon1, 180.0, atol=1.0):
                    lon1 = 180.0 - 1e-5

            if np.all(np.isnan(lonlat[:, 1])):
                lat0 = -90.0 + 1e-5
                lat1 = 90.0 - 1e-5
            else:
                lat0 = np.nanmin(lonlat[:, 1])
                lat1 = np.nanmax(lonlat[:, 1])

                if np.isclose(lat0, -90.0, atol=1.0):
                    lat0 = -90.0 + 1e-5
                if np.isclose(lat1, 90.0, atol=1.0):
                    lat1 = 90.0 - 1e-5

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
            proj_xy = self.projection.transform_points(X, Y)
            X_proj = proj_xy[..., 0]
            Y_proj = proj_xy[..., 1]

            dist = np.hypot(X_proj[1:, 1:] - X_proj[0: -1, 0: -1],
                            Y_proj[1:, 1:] - Y_proj[0: -1, 0: -1])

            # If we have a jump of 10% of the radius, assume it's bad,
            # except if we are using PlateCarree/cyl which doesn't use the
            # radius.
            if self.projection.name == "cyl":
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

    def annotate(self, *args, **kwargs):
        # docstring inherited
        if kwargs.pop("lonlat", True):
            if "xycoords" not in kwargs:
                kwargs["xycoords"] = self.projection._as_mpl_transform(self)
            if "textcoords" not in kwargs:
                kwargs["textcoords"] = self.projection._as_mpl_transform(self)

        result = super().annotate(*args, **kwargs)

        return result

    def legend(self, *args, loc="upper left", zorder=GRIDLINES_ZORDER_DEFAULT + 1, **kwargs):
        legend = super().legend(*args, loc=loc, **kwargs)
        legend.set_zorder(zorder)

        return legend

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

    def set_xlabel(self, xlabel, fontsize="xx-large", **kwargs):
        """Set the label on the x axis.

        Parameters
        ----------
        xlabel : `str`
            x label string.
        fontsize : `int` or `str`, optional
            Font size for label.
        **kwargs : `dict`
            Additional keyword arguments accepted by ax.set_xlabel().
        """
        self._xlabelpad = kwargs.pop("labelpad", mpl.rcParams["axes.labelpad"])
        return super().set_xlabel(xlabel, labelpad=0, fontsize=fontsize, **kwargs)

    def set_ylabel(self, ylabel, fontsize="xx-large", **kwargs):
        """Set the label on the y axis.

        Parameters
        ----------
        ylabel : `str`
            y label string.
        fontsize : `int` or `str`, optional
            Font size for label.
        **kwargs : `dict`
            Additional keyword arguments accepted by ax.set_ylabel().
        """
        self._ylabelpad = kwargs.pop("labelpad", mpl.rcParams["axes.labelpad"])
        return super().set_ylabel(ylabel, labelpad=0, fontsize=fontsize, **kwargs)

    def tick_params(self, axis="both", **kwargs):
        # docstring inherited
        if len(self._ticklabels) == 0:
            # Nothing to do here since axis is not initialized.
            return

        if axis not in ("x", "y", "both"):
            raise ValueError("axis keyword must be one of ``x``, ``y``, or ``both``.")
        which = kwargs.pop("which", "major")
        if which not in ("major", "both"):
            raise ValueError("which keyword must be one of ``major``, ``minor``, or ``both``.")
        if which == "minor":
            # Nothing to do.
            return
        reset = kwargs.pop("reset", False)

        axis_mapping = {
            "x": ["top", "bottom"],
            "y": ["left", "right"],
        }

        for _axis in ("x", "y"):
            if axis not in (_axis, "both"):
                continue

            labelsize = kwargs.pop("labelsize", None)
            labelcolor = kwargs.pop("labelcolor", None)
            labelfontfamily = kwargs.pop("labelfontfamily", None)
            labelbottom = kwargs.pop("labelbottom", None)
            labeltop = kwargs.pop("labeltop", None)
            labelleft = kwargs.pop("labelleft", None)
            labelright = kwargs.pop("labelright", None)
            pad = kwargs.pop("pad", None)

            for side in axis_mapping[_axis]:
                if labelsize is not None:
                    self._ticklabels[side].set(fontsize=labelsize)
                elif reset:
                    self._ticklabels[side].set(fontsize=mpl.rcParams["ytick.labelsize"])
                if labelcolor is not None:
                    self._ticklabels[side].set(color=labelcolor)
                elif reset:
                    if _axis == "x":
                        self._ticklabels[side].set(color=self._xlabelcolor)
                    else:
                        self._ticklabels[side].set(color=self._ylabelcolor)
                if labelfontfamily is not None:
                    self._ticklabels[side].set(fontfamily=labelfontfamily)
                elif reset:
                    self._ticklabels[side].set(fontfamily=None)
                if pad is not None:
                    self._ticklabels[side].set_pad(pad)
                elif reset:
                    self._ticklabels[side].set_pad(mpl.rcParams[f"{axis}tick.major.pad"])

            if labelbottom is not None:
                self._ticklabels_visibility["bottom"] = labelbottom
            elif reset:
                self._ticklabels_visibility["bottom"] = True
            if labeltop is not None:
                self._ticklabels_visibility["top"] = labeltop
            elif reset:
                self._ticklabels_visibility["top"] = True
            if labelleft is not None:
                self._ticklabels_visibility["left"] = labelleft
            elif reset:
                self._ticklabels_visibility["left"] = True
            if labelright is not None:
                self._ticklabels_visibility["right"] = labelright
            elif reset:
                self._ticklabels_visibility["right"] = False

    def update_projection(self, crs_new):
        """Update the projection central coordinate.

        Parameters
        ----------
        crs_new : `skyproj.SkyCRS`
        """
        self.projection = crs_new

    def minorticks_on(self):
        """This is a no-op; skyproj does not support minor ticks."""
        warnings.warn("Skyproj does not support minor ticks.")
