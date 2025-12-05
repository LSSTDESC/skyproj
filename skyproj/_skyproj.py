import warnings

import matplotlib

import numpy as np
import hpgeom as hpg

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

from .skyaxes import GRIDLINES_ZORDER_DEFAULT
from .skycrs import get_crs, GnomonicCRS, proj, proj_inverse
from .hpx_utils import (
    healpix_pixels_range,
    hspmap_to_xy,
    hpxmap_to_xy,
    healpix_to_xy,
    healpix_bin,
    NoValidPixelsError,
)
from .utils import wrap_values, _get_boundary_poly_xy, get_autoscale_vmin_vmax

from ._docstrings import skyproj_init_parameters, skyproj_kwargs_par
from ._docstrings import (
    add_func_docstr,
    draw_hpxmap_docstr,
    draw_hpxpix_docstr,
    draw_hspmap_docstr,
    draw_hpxbin_docstr,
)


class _Skyproj():
    __doc__ = skyproj_init_parameters(
        "Base class for creating Skyproj objects.",
        include_projection_name=True,
    ) + skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        projection_name='cyl',
        lon_0=0.0,
        gridlines=True,
        celestial=True,
        extent=None,
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=None,
        n_grid_lat=None,
        min_lon_ticklabel_delta=0.1,
        **kwargs,
    ):
        self._redraw_dict = {'hpxmap': None,
                             'hspmap': None,
                             'im': None,
                             'inset_colorbar': None,
                             'inset_colorbar_kwargs': {},
                             'colorbar': None,
                             'colorbar_kwargs': {},
                             'lon_range_home': None,
                             'lat_range_home': None,
                             'vmin': None,
                             'vmax': None,
                             'norm': None,
                             'xsize': None,
                             'kwargs_pcolormesh': None,
                             'nside': None,
                             'nest': None,
                             'rasterized': True}

        if longitude_ticks == 'positive':
            self._longitude_ticks = longitude_ticks
        elif longitude_ticks == 'symmetric':
            self._longitude_ticks = longitude_ticks
        else:
            raise ValueError("longitude_ticks must be 'positive' or 'symmetric'.")

        self._n_grid_lon = n_grid_lon
        self._n_grid_lat = n_grid_lat
        self._min_lon_ticklabel_delta = min_lon_ticklabel_delta

        if ax is None:
            # If we don't have an axis, we need to ask the matplotlib
            # caching system which involves an import of matplotlib.pyplot.
            import matplotlib.pyplot as plt
            ax = plt.gca()

        fig = ax.figure
        # This code does not work with the constrained_layout option
        try:
            # Newer matplotlib
            fig.set_layout_engine('none')
        except AttributeError:
            # Older matplotlib
            fig.set_constrained_layout(False)
        subspec = ax.get_subplotspec()
        fig.delaxes(ax)

        # Map lon_0 to be between -180.0 and 180.0
        lon_0 = wrap_values(lon_0)

        if abs(lon_0) == 180.0:
            # We must move this by epsilon or the code gets confused with 0 == 360
            lon_0 = 179.9999

        kwargs['lon_0'] = lon_0
        crs = get_crs(projection_name, **kwargs)

        if len(rcparams) > 0:
            warnings.warn("rcparams is deprecated as a keyword, and is now ignored. "
                          "Please use skyproj.ax.tick_params() to set tick label parameters.")

        with matplotlib.rc_context(
            {
                "xtick.minor.visible": False,
                "ytick.minor.visible": False
            },
        ):
            self._ax = fig.add_subplot(subspec, projection=crs)

        self._crs_orig = crs
        self._reprojected = False

        self._celestial = celestial
        self._gridlines = gridlines
        self._autorescale = autorescale
        self._galactic = galactic

        self._wrap = (self.lon_0 + 180.) % 360.

        extent_xy = None
        if extent is None:
            extent = self._full_sky_extent_initial
            if self._init_extent_xy:
                # Certain projections (such as laea) require us to know the x/y extent
                proj_boundary_xy = self._compute_proj_boundary_xy()
                pts = np.concatenate((proj_boundary_xy['left'],
                                      proj_boundary_xy['right'],
                                      proj_boundary_xy['top'],
                                      proj_boundary_xy['bottom']))
                extent_xy = [np.min(pts[:, 0]), np.max(pts[:, 0]),
                             np.min(pts[:, 1]), np.max(pts[:, 1])]
        else:
            extent_xy = None

        self._boundary_lines = None

        self._initialize_axes(extent, extent_xy=extent_xy)

        # Set up callbacks on axis zoom.
        self._add_change_axis_callbacks()

        # Set up callback on figure resize.
        self._dc = self.ax.figure.canvas.mpl_connect('draw_event', self._draw_callback)
        self._initial_extent_xy = [0]*4

        # Set up reproject callback.
        self._rpc = self.ax.figure.canvas.mpl_connect('key_press_event', self._keypress_callback)

        self._draw_bounds()

    def proj(self, lon, lat):
        """Apply forward projection to a set of lon/lat positions.

        Convert from lon/lat to x/y.

        Parameters
        ----------
        lon : `float` or `list` or `np.ndarray`
            Array of longitude(s) (degrees).
        lat : `float` or `list` or `np.ndarray`
            Array of latitudes(s) (degrees).

        Returns
        -------
        x : `np.ndarray`
            Array of x values.
        y : `np.ndarray`
            Array of y values.
        """
        return proj(lon, lat, projection=self.crs, pole_clip=self._pole_clip)

    def proj_inverse(self, x, y):
        """Apply inverse projection to a set of points.

        Convert from x/y to lon/lat.

        Parameters
        ----------
        x : `float` or `list` or `np.ndarray`
            Projected x values.
        y : `float` or `list` or `np.ndarray`
            Projected y values.

        Returns
        -------
        lon : `np.ndarray`
            Array of longitudes (degrees).
        lat : `np.ndarray`
            Array of latitudes (degrees).
        """
        return proj_inverse(x, y, self.crs)

    def _initialize_axes(self, extent, extent_xy=None):
        """Initialize the axes with a given extent.

        Note that calling this method will remove all formatting options.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        extent_xy : array-like, optional
            Extent in xy space [x_min, x_max, y_min, y_max].
            Used for full-sky initialization.
        """
        self._set_axes_limits(extent, extent_xy=extent_xy, invert=False)
        self._create_axes(extent)
        self._set_axes_limits(extent, extent_xy=extent_xy, invert=self._celestial)

        self._ax.set_frame_on(False)
        if self._gridlines:
            self._ax.grid(visible=True, linestyle=':', color='k', lw=0.5,
                          n_grid_lon=self._n_grid_lon, n_grid_lat=self._n_grid_lat,
                          longitude_ticks=self._longitude_ticks,
                          equatorial_labels=self._equatorial_labels,
                          celestial=self._celestial,
                          full_circle=self._full_circle,
                          wrap=self._wrap,
                          min_lon_ticklabel_delta=self._min_lon_ticklabel_delta,
                          draw_inner_lon_labels=self._inner_longitude_labels)

        self._extent_xy = self._ax.get_extent(lonlat=False)
        self._changed_x_axis = False
        self._changed_y_axis = False

    def set_extent(self, extent):
        """Set the extent.

        Axes will be properly inverted if Skyproj was initialized with
        ``celestial=True``.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        self._set_axes_limits(extent, invert=self._celestial)
        self._extent_xy = self._ax.get_extent(lonlat=False)

        self._draw_bounds()

    def _draw_bounds(self):
        """Draw the map boundary."""
        # Remove any previous lines
        if self._boundary_lines:
            for line in self._boundary_lines:
                line.remove()
            self._boundary_lines = None

        extent_xy = self._ax.get_extent(lonlat=False)
        bounds_xy = self._compute_proj_boundary_xy()
        bounds_xy_clipped = _get_boundary_poly_xy(bounds_xy, extent_xy, self.proj, self.proj_inverse)

        self._boundary_lines = self._ax.plot(bounds_xy_clipped[:, 0],
                                             bounds_xy_clipped[:, 1],
                                             'k-',
                                             lonlat=False,
                                             clip_on=False,
                                             linewidth=matplotlib.rcParams['axes.linewidth'])

    def get_extent(self):
        """Get the extent in lon/lat coordinates.

        Returns
        -------
        extent : `list`
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        return self._ax.get_extent(lonlat=True)

    def set_autorescale(self, autorescale):
        """Set automatic rescaling after zoom.

        Parameters
        ----------
        autorescale : `bool`
            Automatically rescale after zoom?
        """
        self._autorescale = autorescale

    def _set_axes_limits(self, extent, extent_xy=None, invert=True):
        """Set axis limits from an extent.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        extent_xy : array-like, optional
            Extent in xy space [x_min, x_max, y_min, y_max].
            Used for full-sky initialization in place of extent.
        """
        if len(extent) != 4:
            raise ValueError("Must specify extent as a 4-element array.")
        if extent_xy:
            if len(extent_xy) != 4:
                raise ValueError("Must specify extent_xy as a 4-element array.")

        if extent_xy:
            self._ax.set_extent(extent_xy, lonlat=False)
        else:
            self._ax.set_extent(extent, lonlat=True)

        if invert:
            # This should probably be done inside the axes.
            # Okay, so the act of setting the extent will reset the inversion.
            # We want our set_extent to automatically do the inversion if it
            # is set to be inverted, I think.
            # And can use ax.xaxis_inverted() to check if it already is.
            # FIXME
            self._ax.invert_xaxis()

        return self._ax.get_xlim(), self._ax.get_ylim()

    def _create_axes(self, extent):
        """Create axes.

        Parameters
        ----------
        extent : `list`
            Axis extent [lon_min, lon_max, lat_min, lat_max] (degrees).
        """
        # FIXME simplify.
        fig = self._ax.figure

        self._ax.set_xlabel(self._default_xy_labels[0])
        self._ax.set_ylabel(self._default_xy_labels[1])

        self._ax.format_coord = self._format_coord

        fig.sca(self._ax)

        return fig, self._ax

    def _format_coord(self, x, y):
        """Return a coordinate format string.

        Parameters
        ----------
        x : `float`
            x position in projected coords.
        y : `float`
            y position in projected coords.

        Returns
        -------
        coord_string : `str`
            Formatted string.
        """
        lon, lat = self.proj_inverse(x, y)
        # Check out-of-bounds with reversibility (unless already out-of-bounds)
        if np.isfinite(lon) and np.isfinite(lat):
            xx, yy = self.proj(lon, lat)
            if not np.isclose(x, xx) or not np.isclose(y, yy):
                lon, lat = np.nan, np.nan
        if self._longitude_ticks == 'positive' and np.isfinite(lon):
            lon %= 360.0
        coord_string = 'lon=%.6f, lat=%.6f' % (lon, lat)
        if np.isnan(lon) or np.isnan(lat):
            val = hpg.UNSEEN
        elif self._redraw_dict['hspmap'] is not None:
            val = self._redraw_dict['hspmap'].get_values_pos(lon, lat)
        elif self._redraw_dict['hpxmap'] is not None:
            pix = hpg.angle_to_pixel(self._redraw_dict['nside'],
                                     lon,
                                     lat,
                                     nest=self._redraw_dict['nest'])
            val = self._redraw_dict['hpxmap'][pix]
        else:
            return coord_string

        if np.isclose(val, hpg.UNSEEN):
            coord_string += ', val=UNSEEN'
        else:
            coord_string += ', val=%f' % (val)
        return coord_string

    def _add_change_axis_callbacks(self):
        """Add callbacks to change axis."""
        self._xlc = self._ax.callbacks.connect('xlim_changed', self._change_axis)
        self._ylc = self._ax.callbacks.connect('ylim_changed', self._change_axis)

    def _remove_change_axis_callbacks(self):
        """Remove callbacks to change axis."""
        self._ax.callbacks.disconnect(self._xlc)
        self._ax.callbacks.disconnect(self._ylc)

    def _change_axis(self, ax):
        """Callback for axis change.

        Parameters
        ----------
        ax : `skyproj.SkyAxesSubplot`
        """
        extent_xy = ax.get_extent(lonlat=False)
        if not np.isclose(extent_xy[0], self._extent_xy[0]) \
           or not np.isclose(extent_xy[1], self._extent_xy[1]):
            self._changed_x_axis = True
        if not np.isclose(extent_xy[2], self._extent_xy[2]) or \
           not np.isclose(extent_xy[3], self._extent_xy[3]):
            self._changed_y_axis = True

        if not self._changed_x_axis or not self._changed_y_axis:
            # Nothing to do yet.
            return

        extent = ax.get_extent(lonlat=True)

        gone_home = False
        if np.all(np.isclose(ax.get_extent(lonlat=False), self._initial_extent_xy)):
            gone_home = True

            if self._reprojected:
                self._remove_change_axis_callbacks()
                ax.update_projection(self._crs_orig)
                self._initialize_axes(self._initial_extent_lonlat)
                self._add_change_axis_callbacks()

        # Reset to new extent
        self._changed_x_axis = False
        self._changed_y_axis = False
        self._extent_xy = extent_xy

        self._draw_bounds()

        if gone_home:
            lon_range = self._redraw_dict['lon_range_home']
            lat_range = self._redraw_dict['lat_range_home']
        else:
            lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
            lat_range = [extent[2], extent[3]]

        if self._redraw_dict['hpxmap'] is not None:
            lon_raster, lat_raster, values_raster = hpxmap_to_xy(self._redraw_dict['hpxmap'],
                                                                 lon_range,
                                                                 lat_range,
                                                                 nest=self._redraw_dict['nest'],
                                                                 xsize=self._redraw_dict['xsize'])
        elif self._redraw_dict['hspmap'] is not None:
            lon_raster, lat_raster, values_raster = hspmap_to_xy(self._redraw_dict['hspmap'],
                                                                 lon_range,
                                                                 lat_range,
                                                                 xsize=self._redraw_dict['xsize'])
        else:
            # Nothing to do
            return

        redraw_colorbar = False
        redraw_inset_colorbar = False
        norm = self._redraw_dict["norm"]
        if not isinstance(norm, str):
            vmin = None
            vmax = None
        elif self._autorescale:
            # Recompute scaling
            try:
                vmin, vmax = get_autoscale_vmin_vmax(values_raster.compressed(), None, None)

                self._redraw_dict['vmin'] = vmin
                self._redraw_dict['vmax'] = vmax

                if self._redraw_dict['colorbar']:
                    redraw_colorbar = True
                if self._redraw_dict['inset_colorbar']:
                    redraw_inset_colorbar = True
            except IndexError:
                # We have zoomed to a blank spot, don't rescale
                vmin = self._redraw_dict['vmin']
                vmax = self._redraw_dict['vmax']
        else:
            vmin = self._redraw_dict['vmin']
            vmax = self._redraw_dict['vmax']

        if self._redraw_dict['im'] is not None:
            self._redraw_dict['im'].remove()
            self._redraw_dict['im'] = None

        im = self._ax.pcolormesh(
            lon_raster,
            lat_raster,
            values_raster,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            rasterized=self._redraw_dict['rasterized'],
            **self._redraw_dict['kwargs_pcolormesh'],
        )
        self._redraw_dict['im'] = im
        self._ax._sci(im)

        if redraw_colorbar or redraw_inset_colorbar:
            if isinstance(norm, str):
                if norm == "log":
                    map_norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    map_norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                map_norm = norm

            if redraw_colorbar:
                mappable = ScalarMappable(map_norm, cmap=self._redraw_dict['colorbar'].cmap)
                self._redraw_dict['colorbar'].update_normal(mappable)
            else:
                mappable = ScalarMappable(map_norm, cmap=self._redraw_dict['inset_colorbar'].cmap)
                self._redraw_dict['inset_colorbar'].update_normal(mappable)

    def _draw_callback(self, event):
        # We need to set the initial extent on first draw
        if np.allclose(self._initial_extent_xy, 0):
            self._initial_extent_xy = self._ax.get_extent(lonlat=False)
            self._initial_extent_lonlat = self._ax.get_extent(lonlat=True)

    def _keypress_callback(self, event):
        if event.key == 'R':
            self._remove_change_axis_callbacks()
            extent = self._ax.get_extent(lonlat=True)

            lon_0_new = (extent[0] + extent[1])/2.
            if self.lat_0 is not None:
                lat_0_new = (extent[2] + extent[3])/2.
            else:
                lat_0_new = None

            # Decide if gnomonic or not
            if (extent[1] - extent[0])/2. < 1.0 and (extent[3] - extent[2])/2. < 1.0:
                # Make this a gnomonic projection
                crs_new = GnomonicCRS(lon_0=lon_0_new, lat_0=(extent[2] + extent[3])/2.)
            else:
                crs_new = self._crs_orig.with_new_center(lon_0_new, lat_0=lat_0_new)

            self._ax.update_projection(crs_new)
            self._initialize_axes(extent)
            self._changed_x_axis = True
            self._changed_y_axis = True
            self._change_axis(self._ax)
            self._add_change_axis_callbacks()
            self._ax.figure.canvas.draw()

            self._reprojected = True

    @property
    def ax(self):
        return self._ax

    def compute_extent(self, lon, lat):
        """Compute plotting extent for a set of lon/lat points.

        Uses a simple looping algorithm to find the ideal range so that
        all the points fit within the projected frame, with a small border.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude values.
        lat : `np.ndarray`
            Array of latitude values.

        Returns
        -------
        extent : `list`
            Plotting extent [lon_max, lon_min, lat_min, lat_max]
        """
        lon_wrap = wrap_values(lon, wrap=self._wrap)

        # Compute lat range with cushion
        lat_min0 = np.min(lat)
        lat_max0 = np.max(lat)
        lat_range = lat_max0 - lat_min0
        lat_min = np.clip(lat_min0 - 0.05*lat_range, -90.0, None)
        lat_max = np.clip(lat_max0 + 0.05*lat_range, None, 90.0)

        # Compute an ideally fitting lon range with cushion
        x, y = self.proj(lon, lat)

        lon_min0 = np.min(lon_wrap)
        lon_max0 = np.max(lon_wrap)
        lon_step = (lon_max0 - lon_min0)/20.
        lon_cent = (lon_min0 + lon_max0)/2.

        # Compute lon_min so that it fits all the data.
        enclosed = False
        lon_min = lon_cent - lon_step
        while not enclosed and lon_min > (self.lon_0 - 180.0):
            e_x, e_y = self.proj([lon_min + lon_step/2., lon_min + lon_step/2.], [lat_min, lat_max])
            n_out = np.sum(x < e_x.min())
            if n_out == 0:
                enclosed = True
            else:
                lon_min = np.clip(lon_min - lon_step, self.lon_0 - 180., None)

        # Compute lon_max so that it fits all the data
        enclosed = False
        lon_max = lon_cent + lon_step
        while not enclosed and lon_max < (self.lon_0 + 180.0):
            e_x, e_y = self.proj([lon_max - lon_step/2., lon_max - lon_step/2.], [lat_min, lat_max])
            n_out = np.sum(x > e_x.max())
            if n_out == 0:
                enclosed = True
            else:
                lon_max = np.clip(lon_max + lon_step, None, self.lon_0 + 180.)

        return [lon_max, lon_min, lat_min, lat_max]

    def plot(self, *args, **kwargs):
        warnings.warn(
            "skyproj.plot() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.plot()",
            FutureWarning,
        )
        return self._ax.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        warnings.warn(
            "skyproj.scatter() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.scatter()",
            FutureWarning,
        )
        return self._ax.scatter(*args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        warnings.warn(
            "skyproj.pcolormesh() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.pcolormesh()",
            FutureWarning,
        )
        return self._ax.pcolormesh(*args, **kwargs)

    def fill(self, *args, **kwargs):
        warnings.warn(
            "skyproj.fill() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.fill()",
            FutureWarning,
        )
        return self._ax.fill(*args, **kwargs)

    def circle(self, *args, **kwargs):
        warnings.warn(
            "skyproj.circle() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.circle()",
            FutureWarning,
        )
        return self._ax.circle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        warnings.warn(
            "skyproj.ellipse() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.ellipse()",
            FutureWarning,
        )
        return self._ax.ellipse(*args, **kwargs)

    def legend(self, *args, loc='upper left', zorder=GRIDLINES_ZORDER_DEFAULT + 1, **kwargs):
        """Add legend to the axis with ax.legend(*args, **kwargs, zorder=zorder).

        By default the legend will be placed on top of all other elements. This
        can be adjusted with the zorder parameter.
        """
        warnings.warn(
            "skyproj.legend() has been deprecated and will be removed in v2.5. "
            "Please access via skyproj.ax.legend()",
            FutureWarning,
        )
        legend = self._ax.legend(*args, loc=loc, **kwargs)
        legend.set_zorder(zorder)
        return legend

    def draw_polygon(self, lon, lat, edgecolor='red', linestyle='solid',
                     facecolor=None, **kwargs):
        """Plot a polygon from a list of lon, lat coordinates.

        This routine is a convenience wrapper around plot() and fill(), both
        of which work in geodesic (great circle) coordinates.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude points in polygon.
        lat : `np.ndarray`
            Array of latitude points in polygon.
        edgecolor : `str`, optional
            Color of polygon boundary.  Set to None for no boundary.
        linestyle : `str`, optional
            Line style for boundary.
        facecolor : `str`, optional
            Color of polygon face.  Set to None for no fill color.
        **kwargs : `dict`, optional
            Additional keywords passed to plot.
        """
        if linestyle is not None and edgecolor is not None:
            self._ax.plot(np.append(lon, lon[0]),
                          np.append(lat, lat[0]),
                          color=edgecolor, linestyle=linestyle, **kwargs)
        if facecolor is not None:
            self.ax.fill(lon, lat, color=facecolor, **kwargs)

    def draw_polygon_file(self, filename, reverse=True,
                          edgecolor='red', linestyle='solid', **kwargs):
        """Draw a text file containing lon, lat coordinates of polygon(s).

        Parameters
        ----------
        filename : `str`
            Name of file containing the polygon(s) [lon, lat, poly]
        reverse : `bool`
            Reverse drawing order of points in each polygon.
        edgecolor : `str`
            Color of polygon boundary.
        linestyle : `str`, optional
            Line style for boundary.
        **kwargs : `dict`
            Additional keywords passed to plot.
        """
        try:
            data = np.genfromtxt(filename, names=['lon', 'lat', 'poly'])
        except ValueError:
            from numpy.lib.recfunctions import append_fields
            data = np.genfromtxt(filename, names=['lon', 'lat'])
            data = append_fields(data, 'poly', np.zeros(len(data)))

        for p in np.unique(data['poly']):
            poly = data[data['poly'] == p]
            lon = poly['lon'][::-1] if reverse else poly['lon']
            lat = poly['lat'][::-1] if reverse else poly['lat']
            self.draw_polygon(lon,
                              lat,
                              edgecolor=edgecolor,
                              linestyle=linestyle,
                              **kwargs)
            # Only add the label to the first polygon plotting.
            kwargs.pop('label', None)

    def draw_box(self, lon, lat, edgecolor='red', linestyle='solid',
                 facecolor=None, **kwargs):
        """Plot a box from a list of lon, lat coordinates.

        This will draw a box with sides of constant lon/lat, and does
        not operate in geodesic (great circle) coordinates, unlike
        draw_polygon().

        Parameters
        ----------
        lon : `np.ndarray`
            Array of four longitude points describing box.
        lat : `np.ndarray`
            Array of four latitude points describing box.
        edgecolor : `str`, optional
            Color of box boundary.  Set to None for no boundary.
        linestyle : `str`, optional
            Line style for boundary.
        facecolor : `str`, optional
            Color of box face.  Set to None for no fill color.
        **kwargs : `dict`, optional
            Additional keywords passed to plot.
        """
        if len(lon) != 4 or len(lat) != 4:
            raise ValueError("draw_box requires 4 longitude and latitude points.")

        if linestyle is not None and edgecolor is not None:
            self._ax.plot(np.append(lon, lon[0]),
                          np.append(lat, lat[0]),
                          color=edgecolor, linestyle=linestyle,
                          geodesic=False, **kwargs)
        if facecolor is not None:
            self.ax.fill(lon, lat, color=facecolor, geodesic=False, **kwargs)

    @add_func_docstr(draw_hpxmap_docstr)
    def draw_hpxmap(self, hpxmap, nest=False, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None,
                    norm="linear", **kwargs):

        nside = hpg.npixel_to_nside(hpxmap.size)
        pixels, = np.where(hpxmap != hpg.UNSEEN)

        lon_range_set = lon_range is not None
        lat_range_set = lat_range is not None

        if lon_range is None or lat_range is None:
            if zoom:
                _lon_range, _lat_range = healpix_pixels_range(nside,
                                                              pixels,
                                                              self._wrap,
                                                              nest=nest)
            else:
                extent = self.get_extent()
                _lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
                _lat_range = [extent[2], extent[3]]

            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = hpxmap_to_xy(hpxmap,
                                                             lon_range,
                                                             lat_range,
                                                             nest=nest,
                                                             xsize=xsize)

        if isinstance(norm, str):
            _vmin, _vmax = get_autoscale_vmin_vmax(
                values_raster.compressed(),
                vmin,
                vmax,
            )
        else:
            _vmin = None
            _vmax = None

        if zoom:
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)
        elif lon_range_set and lat_range_set:
            self.set_extent([lon_range[1], lon_range[0], lat_range[0], lat_range[1]])

        im = self._ax.pcolormesh(
            lon_raster,
            lat_raster,
            values_raster,
            norm=norm,
            vmin=_vmin,
            vmax=_vmax,
            rasterized=rasterized,
            **kwargs,
        )
        self._ax._sci(im)

        # Link up callbacks
        self._redraw_dict['hspmap'] = None
        self._redraw_dict['hpxmap'] = hpxmap
        self._redraw_dict['lon_range_home'] = lon_range
        self._redraw_dict['lat_range_home'] = lat_range
        self._redraw_dict['nside'] = nside
        self._redraw_dict['nest'] = nest
        self._redraw_dict['vmin'] = vmin
        self._redraw_dict['vmax'] = vmax
        self._redraw_dict['norm'] = norm
        self._redraw_dict['xsize'] = xsize
        self._redraw_dict['rasterized'] = rasterized
        self._redraw_dict['kwargs_pcolormesh'] = kwargs

        return im, lon_raster, lat_raster, values_raster

    @add_func_docstr(draw_hpxpix_docstr)
    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None,
                    norm="linear", **kwargs):

        lon_range_set = lon_range is not None
        lat_range_set = lat_range is not None

        if lon_range is None or lat_range is None:
            if zoom:
                _lon_range, _lat_range = healpix_pixels_range(nside,
                                                              pixels,
                                                              self._wrap,
                                                              nest=nest)
            else:
                extent = self.get_extent()
                _lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
                _lat_range = [extent[2], extent[3]]

            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = healpix_to_xy(
            nside,
            pixels,
            values,
            nest=nest,
            xsize=xsize,
            lon_range=lon_range,
            lat_range=lat_range
        )

        if isinstance(norm, str):
            _vmin, _vmax = get_autoscale_vmin_vmax(
                values_raster.compressed(),
                vmin,
                vmax,
            )
        else:
            _vmin = None
            _vmax = None

        if zoom:
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)
        elif lon_range_set and lat_range_set:
            self.set_extent([lon_range[1], lon_range[0], lat_range[0], lat_range[1]])

        im = self._ax.pcolormesh(
            lon_raster,
            lat_raster,
            values_raster,
            norm=norm,
            vmin=_vmin,
            vmax=_vmax,
            rasterized=rasterized,
            **kwargs,
        )
        self._ax._sci(im)

        return im, lon_raster, lat_raster, values_raster

    @add_func_docstr(draw_hspmap_docstr)
    def draw_hspmap(self, hspmap, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, valid_mask=False,
                    norm="linear", **kwargs):
        self._hspmap = hspmap
        self._hpxmap = None

        lon_range_set = lon_range is not None
        lat_range_set = lat_range is not None

        if lon_range is None or lat_range is None:
            if zoom:
                # Using the coverage map is much faster even if approximate.
                try:
                    _lon_range, _lat_range = healpix_pixels_range(
                        hspmap.nside_coverage,
                        np.where(hspmap.coverage_mask)[0],
                        self._wrap,
                        nest=True,
                    )
                except NoValidPixelsError:
                    warnings.warn("No valid pixels found; auto-zoom not possible.")
                    zoom = False

            if not zoom:
                extent = self.get_extent()
                _lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
                _lat_range = [extent[2], extent[3]]

            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        if hspmap.is_rec_array and not valid_mask:
            warnings.warn(
                """
                draw_hspmap called with a record array HealSparseMap.  Assuming valid_mask=True.
                To instead visualize component "A" of the record array, draw_hspmap(hspmap["A"])
                """
            )
            valid_mask = True

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = hspmap_to_xy(hspmap,
                                                             lon_range,
                                                             lat_range,
                                                             xsize=xsize,
                                                             valid_mask=valid_mask)

        if isinstance(norm, str):
            _vmin, _vmax = get_autoscale_vmin_vmax(
                values_raster.compressed(),
                vmin,
                vmax,
            )
        else:
            _vmin = None
            _vmax = None

        if zoom:
            # Watch for masked array here...
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)
        elif lon_range_set and lat_range_set:
            self.set_extent([lon_range[1], lon_range[0], lat_range[0], lat_range[1]])

        im = self._ax.pcolormesh(
            lon_raster,
            lat_raster,
            values_raster,
            norm=norm,
            vmin=_vmin,
            vmax=_vmax,
            rasterized=rasterized,
            **kwargs,
        )
        self._ax._sci(im)

        # Link up callbacks
        self._redraw_dict['hspmap'] = hspmap
        self._redraw_dict['hpxmap'] = None
        self._redraw_dict['lon_range_home'] = lon_range
        self._redraw_dict['lat_range_home'] = lat_range
        self._redraw_dict['im'] = im
        self._redraw_dict['vmin'] = vmin
        self._redraw_dict['vmax'] = vmax
        self._redraw_dict['norm'] = norm
        self._redraw_dict['xsize'] = xsize
        self._redraw_dict['rasterized'] = rasterized
        self._redraw_dict['kwargs_pcolormesh'] = kwargs

        return im, lon_raster, lat_raster, values_raster

    @add_func_docstr(draw_hpxbin_docstr)
    def draw_hpxbin(self, lon, lat, C=None, nside=256, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None,
                    norm="linear", **kwargs):
        hpxmap = healpix_bin(lon, lat, C=C, nside=nside, nest=nest)

        im, lon_raster, lat_raster, values_raster = self.draw_hpxmap(
            hpxmap, nest=nest, zoom=zoom, xsize=xsize, vmin=vmin,
            vmax=vmax, rasterized=rasterized, lon_range=lon_range,
            lat_range=lat_range, norm=norm,
            **kwargs)

        return hpxmap, im, lon_raster, lat_raster, values_raster

    def draw_inset_colorbar(self, format=None, label=None, ticks=None, fontsize=11,
                            width="25%", height="5%", loc=7, bbox_to_anchor=(0., -0.04, 1, 1),
                            orientation='horizontal', ax=None, **kwargs):
        """Draw an inset colorbar.

        Parameters
        ----------
        format : `str`, optional
            Format string for tick labels.
        label : `str`, optional
            Label to attach to inset colorbar.
        ticks : `list`, optional
            List of tick values.
        fontsize : `int`, optional
            Font size to use for ticks.
        width : `str`, optional
            Fraction of total axis width for inset colorbar.
        height : `str`, optional
            Fraction of total axis height for inset colorbar.
        loc : `int`, optional
            Matplotlib location code.
        bbox_to_anchor : `tuple`, optional
            Where to put inset colorbar bbox.
        orientation : `str`, optional
            Inset colorbar orientation (``horizontal`` or ``vertical``).
        ax : `SkyAxesSubplot`, optional
            Axis associated with inset colorbar.  If None, use
            skyaxes associated with map.
        **kwargs : `dict`, optional
            Additional kwargs to pass to inset_axes or colorbar.

        Returns
        -------
        colorbar : `matplotlib.colorbar.Colorbar`
        colorbar_axis : `mpl_toolkits.axes_grid1.parasite_axes.AxesHostAxes`
        """
        if ax is None:
            ax = self._ax

        im = ax._gci()

        cax = inset_axes(ax,
                         width=width,
                         height=height,
                         loc=loc,
                         bbox_to_anchor=bbox_to_anchor,
                         bbox_transform=ax.transAxes,
                         **kwargs)
        cmin, cmax = im.get_clim()

        if ticks is None and cmin is not None and cmax is not None:
            cmed = (cmax + cmin)/2.
            delta = (cmax - cmin)/10.
            ticks = np.array([cmin + delta, cmed, cmax - delta])

        tmin = np.min(np.abs(ticks[0]))
        tmax = np.max(np.abs(ticks[1]))

        if format is None:
            if (tmin < 1e-2) or (tmax > 1e3):
                format = '$%.1e$'
            elif (tmin > 0.1) and (tmax < 100):
                format = '$%.1f$'
            elif (tmax > 100):
                format = '$%i$'
            else:
                format = '$%.2g$'

        custom_format = False
        if format == 'custom':
            custom_format = True
            ticks = np.array([cmin, 0.85*cmax])
            format = '$%.0e$'

        cbar = ax.figure.colorbar(
            im,
            ax=ax,
            cax=cax,
            orientation=orientation,
            ticks=ticks,
            format=format,
            **kwargs,
        )
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', labelsize=fontsize)

        if custom_format:
            ticklabels = cax.get_xticklabels()
            for i, lab in enumerate(ticklabels):
                val, exp = ticklabels[i].get_text().split('e')
                ticklabels[i].set_text(r'$%s \times 10^{%i}$'%(val, int(exp)))
            cax.set_xticklabels(ticklabels)

        if label is not None:
            cbar.set_label(label, size=fontsize)
            cax.xaxis.set_label_position('top')

        ax.figure.sca(ax)

        # Save reference to colorbar for zooming
        cbar_kwargs = kwargs
        cbar_kwargs['format'] = format
        cbar_kwargs['label'] = label
        cbar_kwargs['ticks'] = ticks
        cbar_kwargs['fontsize'] = fontsize
        cbar_kwargs['width'] = width
        cbar_kwargs['height'] = height
        cbar_kwargs['loc'] = loc
        cbar_kwargs['bbox_to_anchor'] = bbox_to_anchor
        cbar_kwargs['orientation'] = orientation
        self._redraw_dict['inset_colorbar'] = cbar
        self._redraw_dict['inset_colorbar_kwargs'] = cbar_kwargs

        return cbar, cax

    def draw_colorbar(self, label=None, ticks=None, fontsize=11,
                      fraction=0.15, location='right', pad=0.0,
                      ax=None, **kwargs):
        """Draw a colorbar.

        Parameters
        ----------
        label : `str`, optional
            Label to attach to colorbar.
        ticks : `list`, optional
            List of tick values.
        fontsize : `int`, optional
            Font size to use for ticks.
        fraction : `float`, optional
            Fraction of original axes to use for colorbar.
        location : `str`, optional
            Colorbar location (``right``, ``bottom``, ``left``, ``top``).
        pad : `float`, optional
            Fraction of original axes between colorbar and original axes.
        ax : `SkyAxesSubplot`, optional
            Axis associated with inset colorbar.  If None, use
            skyaxes associated with map.
        **kwargs : `dict`, optional
            Additional kwargs to send to colorbar().

        Returns
        -------
        colorbar : `matplotlib.colorbar.Colorbar`
        """
        if ax is None:
            ax = self._ax

        cbar = ax.figure.colorbar(
            ax._gci(),
            ax=ax,
            location=location,
            ticks=ticks,
            fraction=fraction,
            pad=pad,
            **kwargs,
        )

        if location == 'right' or location == 'left':
            cbar_axis = 'y'
        else:
            cbar_axis = 'x'

        cbar.ax.tick_params(axis=cbar_axis, labelsize=fontsize)

        if label is not None:
            cbar.set_label(label, size=fontsize)

        # Reset the "home" position because axis has been shifted.
        self._initial_extent_xy = self._ax.get_extent(lonlat=False)

        ax.figure.sca(ax)

        # Save reference to colorbar for zooming
        cbar_kwargs = kwargs
        cbar_kwargs['label'] = label
        cbar_kwargs['ticks'] = ticks
        cbar_kwargs['fontsize'] = fontsize
        cbar_kwargs['fraction'] = fraction
        cbar_kwargs['location'] = location
        cbar_kwargs['pad'] = pad
        self._redraw_dict['colorbar'] = cbar
        self._redraw_dict['colorbar_kwargs'] = cbar_kwargs

        return cbar

    def draw_milky_way(self, width=10, linewidth=1.5, color='black', linestyle='-', **kwargs):
        """Draw the Milky Way galaxy.

        Parameters
        ----------
        width : `float`
            Number of degrees north and south to draw dotted lines.
        linewidth : `float`
            Width of line along the plane.
        color : `str`
            Color of Milky Way plane.
        linestyle : `str`
            Style of line.
        **kwargs : `dict`
            Additional kwargs to pass to plot.
        """
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        glon = np.linspace(0, 360, 500)
        glat = np.zeros_like(glon)

        if not self._galactic:
            gc = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
            radec = gc.fk5
            lon = radec.ra.degree
            lat = radec.dec.degree
        else:
            lon = glon
            lat = glat

        self._ax.plot(lon, lat, linewidth=linewidth, color=color, linestyle=linestyle, **kwargs)
        # pop any labels
        kwargs.pop('label', None)
        if width > 0:
            for delta in [+width, -width]:
                if not self._galactic:
                    gc = SkyCoord(l=glon*u.degree, b=(glat + delta)*u.degree, frame='galactic')
                    radec = gc.fk5
                    lon = radec.ra.degree
                    lat = radec.dec.degree
                else:
                    lon = glon
                    lat = glat + delta
                self._ax.plot(lon, lat, linewidth=1.0, color=color,
                              linestyle='--', **kwargs)

    def tissot_indicatrices(self, radius=5.0, num_lon=9, num_lat=5, color='red', alpha=0.5):
        """Draw Tissot indicatrices.

        See https://en.wikipedia.org/wiki/Tissot%27s_indicatrix for details.

        Parameters
        ----------
        radius : `float`
            Radius of each indicatrix circle.
        num_lon : `int`
            Number of indicatrices in the longitude direction.
        num_lat : `int`
            Number of indicatrices in the latitude direction.
        color : `str`, optional
            Color of indicatrices.
        alpha : `float`, optional
            Alpha of indicatrices.
        """
        lons = np.linspace(-175.0, 175.0, num_lon)
        lats = np.linspace(-80.0, 80.0, num_lat)

        for lat in lats:
            # We want to skip alternate indicatrices at high latitudes.
            skip_alternate = False
            if np.abs(lat) >= 75.0:
                skip_alternate = True
            skipped = False
            for lon in lons:
                if skip_alternate and not skipped:
                    skipped = True
                    continue
                _ = self._ax.circle(lon, lat, radius, fill=True, color=color, alpha=alpha)
                skipped = False

    @property
    def lon_0(self):
        return self._ax.lon_0

    @property
    def lat_0(self):
        return self._ax.lat_0

    @property
    def crs(self):
        return self._ax.projection

    @property
    def projection_name(self):
        return self._ax.projection.name

    @property
    def _full_sky_extent_initial(self):
        return [self.lon_0 - 180.0,
                self.lon_0 + 180.0,
                -90.0 + self._pole_clip,
                90.0 - self._pole_clip]

    @property
    def _pole_clip(self):
        # Allow clipping of poles for full-sky projections; this
        # can avoid inversion problems in (e.g.) Mollweide projections.
        return 0.0

    @property
    def _full_circle(self):
        # Is this projection a full circle?
        return False

    @property
    def _equatorial_labels(self):
        # Should the longitude labels be along the equator?
        return False

    @property
    def _radial_labels(self):
        # Are there radial labels?
        return False

    @property
    def _inner_longitude_labels(self):
        # Are there inner longitude labels?
        return False

    @property
    def _init_extent_xy(self):
        # Is the initial extent in x/y space?
        return False

    @property
    def _default_xy_labels(self):
        # Default labels in x, y
        if self._galactic:
            return ("Galactic Longitude", "Galactic Latitude")
        else:
            return ("Right Ascension", "Declination")
