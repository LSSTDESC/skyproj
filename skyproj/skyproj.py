import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

import mpl_toolkits.axisartist as axisartist
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .projections import get_projection, PlateCarree
from .hpx_utils import healpix_pixels_range, hspmap_to_xy, hpxmap_to_xy, healpix_to_xy, healpix_bin
from .mpl_utils import ExtremeFinderWrapped, WrappedFormatterDMS, GridHelperSkyproj
from .utils import wrap_values

__all__ = ['Skyproj', 'McBrydeSkyproj', 'LaeaSkyproj', 'MollweideSkyproj',
           'HammerSkyproj', 'EqualEarthSkyproj']


class Skyproj():
    """Base class for creating Skyproj objects.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`, optional
        Axis object to replace with a skyproj axis
    projection_name : `str`, optional
        Valid proj4/cartosky projection name.
    lon_0 : `float`, optional
        Central longitude of projection.
    gridlines : `bool`, optional
        Draw gridlines?
    celestial : `bool`, optional
        Do celestial plotting (e.g. invert longitude axis).
    extent : iterable, optional
        Default exent of the map, [lon_min, lon_max, lat_min, lat_max].
        Note that lon_min, lon_max can be specified in any order, the
        orientation of the map is set by ``celestial``.
    longitude_ticks : `str`, optional
        Label longitude ticks from 0 to 360 degrees (``positive``) or
        from -180 to 180 degrees (``symmetric``).
    autorescale : `bool`, optional
        Automatically rescale map visualizations on zoom?
    **kwargs : `dict`, optional
        Additional arguments to send to cartosky/proj4 projection initialization.
    """
    pole_clip = 0.0

    def __init__(self, ax=None, projection_name='cyl', lon_0=0, gridlines=True, celestial=True,
                 extent=None, longitude_ticks='positive', autorescale=True, **kwargs):
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
                             'xsize': None,
                             'kwargs_pcolormesh': None,
                             'nside': None,
                             'nest': None}

        if longitude_ticks == 'positive':
            self._longitude_ticks = longitude_ticks
        elif longitude_ticks == 'symmetric':
            self._longitude_ticks = longitude_ticks
        else:
            raise ValueError("longitude_ticks must be 'positive' or 'symmetric'.")

        if ax is None:
            ax = plt.gca()

        fig = ax.figure
        # This code does not work with the constrained_layout option
        fig.set_constrained_layout(False)
        subspec = ax.get_subplotspec()
        fig.delaxes(ax)

        # Map lon_0 to be between -180.0 and 180.0
        lon_0 = wrap_values(lon_0)

        if abs(lon_0) == 180.0:
            # We must move this by epsilon or the code gets confused with 0 == 360
            lon_0 = 179.9999

        kwargs['lon_0'] = lon_0
        self.projection = get_projection(projection_name, **kwargs)
        self.projection_name = projection_name
        self._ax = fig.add_subplot(subspec, projection=self.projection)

        self._aa = None

        self.do_celestial = celestial
        self.do_gridlines = gridlines
        self._autorescale = autorescale

        self._wrap = (lon_0 + 180.) % 360.
        self._lon_0 = self.projection.proj4_params['lon_0']

        if extent is None:
            extent = [lon_0 - 180.0, lon_0 + 180.0, -90.0 + self.pole_clip, 90.0 - self.pole_clip]

        self._initialize_axes(extent)

        # Set up callbacks on axis zoom.
        self._xlc = self._ax.callbacks.connect('xlim_changed', self._change_axis)
        self._ylc = self._ax.callbacks.connect('ylim_changed', self._change_axis)

        # Set up callback on figure resize.
        self._frc = self.ax.figure.canvas.mpl_connect('resize_event', self._change_size)
        self._dc = self.ax.figure.canvas.mpl_connect('draw_event', self._draw_callback)
        self._initial_extent_xy = self._ax.get_extent(lonlat=False)
        self._has_zoomed = False

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
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)
        proj_xy = self.projection.transform_points(PlateCarree(), lon, lat)
        return proj_xy[..., 0], proj_xy[..., 1]

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
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        proj_lonlat = PlateCarree().transform_points(self.projection, x, y)
        return proj_lonlat[..., 0], proj_lonlat[..., 1]

    def _initialize_axes(self, extent):
        """Initialize the axes with a given extent.

        Note that calling this method will remove all formatting options.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        # Reset any axis artist if necessary
        if self._aa is not None:
            self._aa.remove()
            self._aa = None

        self._set_axes_limits(extent, invert=False)
        self._create_axes(extent)
        self._set_axes_limits(extent, invert=self.do_celestial)

        self._ax.set_frame_on(False)
        if self.do_gridlines:
            self._aa.grid(True, linestyle=':', color='k', lw=0.5)

        # Draw the outer edges of the projection.  This needs to be forward-
        # projected and drawn in that space to prevent out-of-bounds clipping.
        # It also needs to be done just inside -180/180 to prevent the transform
        # from resolving to the same line.
        x, y = self.proj(np.linspace(self._lon_0 - 179.9999, self._lon_0 - 179.9999),
                         np.linspace(-90., 90.))
        self._ax.plot(x, y, 'k-', lonlat=False)
        x, y = self.proj(np.linspace(self._lon_0 + 179.9999, self._lon_0 + 179.9999),
                         np.linspace(-90., 90.))
        self._ax.plot(x, y, 'k-', lonlat=False)

        self._extent = self._ax.get_extent(lonlat=True)
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
        self._set_axes_limits(extent, invert=self.do_celestial)
        self._extent = self._ax.get_extent(lonlat=True)

    def get_extent(self):
        """Get the extent in lon/lat coordinates.

        Returns
        -------
        extent : `list`
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        return self._extent

    def set_autorescale(self, autorescale):
        """Set automatic rescaling after zoom.

        Parameters
        ----------
        autorescale : `bool`
            Automatically rescale after zoom?
        """
        self._autorescale = autorescale

    def _set_axes_limits(self, extent, invert=True):
        """Set axis limits from an extent.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        if len(extent) != 4:
            raise ValueError("Must specify extent as a 4-element array.")

        self._ax.set_extent(extent, lonlat=True)

        if self._aa is not None:
            self._aa.set_xlim(self._ax.get_xlim())
            self._aa.set_ylim(self._ax.get_ylim())

        if invert:
            self._ax.invert_xaxis()
            if self._aa is not None:
                self._aa.invert_xaxis()

        return self._ax.get_xlim(), self._ax.get_ylim()

    def _create_axes(self, extent):
        """Create axes and axis artist.

        Parameters
        ----------
        extent : `list`
            Axis extent [lon_min, lon_max, lat_min, lat_max] (degrees).
        """
        extreme_finder = ExtremeFinderWrapped(20, 20, self._wrap)
        if self._wrap == 180.0:
            include_last_lon = True
        else:
            include_last_lon = False
        grid_locator1 = angle_helper.LocatorD(10, include_last=include_last_lon)
        grid_locator2 = angle_helper.LocatorD(6, include_last=True)

        # We always want the formatting to be wrapped at 180 (-180 to 180)
        tick_formatter1 = WrappedFormatterDMS(180.0, self._longitude_ticks)
        tick_formatter2 = angle_helper.FormatterDMS()

        def proj_wrap(lon, lat):
            lon = np.atleast_1d(lon)
            lat = np.atleast_1d(lat)
            lon[np.isclose(lon, self._wrap)] = self._wrap - 1e-10
            proj_xy = self.projection.transform_points(PlateCarree(), lon, lat)
            return proj_xy[..., 0], proj_xy[..., 1]

        grid_helper = GridHelperSkyproj(
            (proj_wrap, self.proj_inverse),
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
            tick_formatter1=tick_formatter1,
            tick_formatter2=tick_formatter2,
            celestial=self.do_celestial
        )

        self._grid_helper = grid_helper

        fig = self._ax.figure
        rect = self._ax.get_position()
        self._aa = axisartist.Axes(fig, rect, grid_helper=grid_helper, frameon=False, aspect=1.0)
        fig.add_axes(self._aa)

        self._aa.format_coord = self._format_coord
        self._aa.axis['left'].major_ticklabels.set_visible(True)
        self._aa.axis['right'].major_ticklabels.set_visible(False)
        self._aa.axis['bottom'].major_ticklabels.set_visible(True)
        self._aa.axis['top'].major_ticklabels.set_visible(True)

        self.set_xlabel('Right Ascension', size=16)
        self.set_ylabel('Declination', size=16)

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
            val = hp.UNSEEN
        elif self._redraw_dict['hspmap'] is not None:
            val = self._redraw_dict['hspmap'].get_values_pos(lon, lat)
        elif self._redraw_dict['hpxmap'] is not None:
            pix = hp.ang2pix(self._redraw_dict['nside'],
                             lon,
                             lat,
                             lonlat=True,
                             nest=self._redraw_dict['nest'])
            val = self._redraw_dict['hpxmap'][pix]
        else:
            return coord_string

        if np.isclose(val, hp.UNSEEN):
            coord_string += ', val=UNSEEN'
        else:
            coord_string += ', val=%f' % (val)
        return coord_string

    def _change_axis(self, ax):
        """Callback for axis change.

        Parameters
        ----------
        ax : `skyproj.SkyAxesSubplot`
        """
        extent = ax.get_extent(lonlat=True)
        if not np.isclose(extent[0], self._extent[0]) or not np.isclose(extent[1], self._extent[1]):
            self._changed_x_axis = True
        if not np.isclose(extent[2], self._extent[2]) or not np.isclose(extent[3], self._extent[3]):
            self._changed_y_axis = True

        if not self._changed_x_axis or not self._changed_y_axis:
            # Nothing to do yet.
            return

        gone_home = False
        if np.all(np.isclose(ax.get_extent(lonlat=False), self._initial_extent_xy)):
            gone_home = True

        # Reset to new extent
        self._changed_x_axis = False
        self._changed_y_axis = False
        self._extent = extent

        # This synchronizes the axis artist to the plot axes after zoom.
        if self._aa is not None:
            self._aa.set_position(self._ax.get_position(), which='original')

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
        if self._autorescale:
            # Recompute scaling
            try:
                vmin, vmax = np.percentile(values_raster.compressed(), (2.5, 97.5))
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

        im = self.pcolormesh(lon_raster, lat_raster, values_raster,
                             vmin=vmin, vmax=vmax,
                             **self._redraw_dict['kwargs_pcolormesh'])
        self._redraw_dict['im'] = im
        self._ax._sci(im)

        if redraw_colorbar:
            mappable = ScalarMappable(Normalize(vmin=vmin, vmax=vmax),
                                      cmap=self._redraw_dict['colorbar'].cmap)
            self._redraw_dict['colorbar'].update_normal(mappable)
        if redraw_inset_colorbar:
            mappable = ScalarMappable(Normalize(vmin=vmin, vmax=vmax),
                                      cmap=self._redraw_dict['inset_colorbar'].cmap)
            self._redraw_dict['inset_colorbar'].update_normal(mappable)

    def _change_size(self, event):
        """Callback for figure resize.

        Parameters
        ----------
        event : `matplotlib.backend_bases.Event`
        """
        # This synchronizes the axis artist to the plot axes after zoom.
        if self._aa is not None:
            self._aa.set_position(self._ax.get_position(), which='original')

    def _draw_callback(self, event):
        # On draw it's sometimes necessary to synchronize the axisartist.
        if self._aa is not None:
            self._aa.set_position(self._ax.get_position(), which='original')

    def set_xlabel(self, text, side='bottom', **kwargs):
        """Set the label on the x axis.

        Parameters
        ----------
        text : `str`
            x label string.
        side : `str`, optional
            Side to set the label.  Can be ``bottom`` or ``top``.
        **kwargs : `dict`
            Additional keyword arguments accepted by ax.set_xlabel().
        """
        return self._aa.axis[side].label.set(text=text, **kwargs)

    def set_ylabel(self, text, side='left', **kwargs):
        """Set the label on the y axis.

        Parameters
        ----------
        text : `str`
            x label string.
        side : `str`, optional
            Side to set the label.  Can be ``left`` or ``right``.
        **kwargs : `dict`
            Additional keyword arguments accepted by ax.set_xlabel().
        """
        return self._aa.axis[side].label.set(text=text, **kwargs)

    @property
    def ax(self):
        return self._ax

    @property
    def aa(self):
        return self._aa

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
        lon_wrap = wrap_values(lon)

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
        while not enclosed and lon_min > (self._lon_0 - 180.0):
            e_x, e_y = self.proj([lon_min, lon_min], [lat_min, lat_max])
            n_out = np.sum(x < e_x.min())
            if n_out == 0:
                enclosed = True
            else:
                lon_min = np.clip(lon_min - lon_step, self._lon_0 - 180., None)

        # Compute lon_max so that it fits all the data
        enclosed = False
        lon_max = lon_cent + lon_step
        while not enclosed and lon_max < (self._lon_0 + 180.0):
            e_x, e_y = self.proj([lon_max, lon_max], [lat_min, lat_max])
            n_out = np.sum(x > e_x.max())
            if n_out == 0:
                enclosed = True
            else:
                lon_max = np.clip(lon_max + lon_step, None, self._lon_0 + 180.)

        return [lon_max, lon_min, lat_min, lat_max]

    def plot(self, *args, **kwargs):
        return self._ax.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        return self._ax.scatter(*args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        return self._ax.pcolormesh(*args, **kwargs)

    def fill(self, *args, **kwargs):
        return self._ax.fill(*args, **kwargs)

    def legend(self, *args, loc='upper left', **kwargs):
        """Add legend to the axis with ax.legend(*args, **kwargs)."""
        return self._ax.legend(*args, loc=loc, **kwargs)

    def draw_polygon(self, lon, lat, edgecolor='red', linestyle='solid',
                     facecolor=None, **kwargs):
        """Plot a polygon from a list of lon, lat coordinates.

        This routine is a convenience wrapper around plot() and fill(), both
        of which work in geodesic coordinates.

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
            self.plot(np.append(lon, lon[0]),
                      np.append(lat, lat[0]),
                      color=edgecolor, linestyle=linestyle, **kwargs)
        if facecolor is not None:
            self.fill(lon, lat, color=facecolor, **kwargs)

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

    def draw_hpxmap(self, hpxmap, nest=False, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healpix map.

        Parameters
        ----------
        hpxmap : `np.ndarray`
            Healpix map to plot, with length 12*nside*nside and UNSEEN for
            illegal values.
        nest : `bool`, optional
            Map in nest ordering?
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        im : `matplotlib.collections.QuadMesh`
            Image that was displayed
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
        """
        nside = hp.npix2nside(hpxmap.size)
        pixels, = np.where(hpxmap != hp.UNSEEN)

        if lon_range is None or lat_range is None:
            if zoom:
                _lon_range, _lat_range = healpix_pixels_range(nside,
                                                              pixels,
                                                              self._wrap,
                                                              nest=nest)
            else:
                extent = self.get_extent()
                lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
                lat_range = [extent[2], extent[3]]

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

        if vmin is None or vmax is None:
            # Auto-scale from visible values
            _vmin, _vmax = np.percentile(values_raster.compressed(), (2.5, 97.5))
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax

        if zoom:
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, vmin=vmin, vmax=vmax, **kwargs)
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
        self._redraw_dict['xsize'] = xsize
        self._redraw_dict['kwargs_pcolormesh'] = kwargs

        return im, lon_raster, lat_raster, values_raster

    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healpix map made of pixels and values.

        Parameters
        ----------
        nside : `int`
            Healpix nside of pixels to plot.
        pixels : `np.ndarray`
            Array of pixels to plot.
        values : `np.ndarray`
            Array of values associated with pixels.
        nest : `bool`, optional
            Map in nest ordering?
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        im : `matplotlib.collections.QuadMesh`
            Image that was displayed
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
        """
        if lon_range is None or lat_range is None:
            if zoom:
                _lon_range, _lat_range = healpix_pixels_range(nside,
                                                              pixels,
                                                              self._wrap,
                                                              nest=nest)
            else:
                extent = self.get_extent()
                lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
                lat_range = [extent[2], extent[3]]

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

        if vmin is None or vmax is None:
            # Auto-scale from visible values
            _vmin, _vmax = np.percentile(values_raster.compressed(), (2.5, 97.5))
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax

        if zoom:
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, vmin=vmin, vmax=vmax, **kwargs)
        self._ax._sci(im)

        return im, lon_raster, lat_raster, values_raster

    def draw_hspmap(self, hspmap, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healsparse map.

        Parameters
        ----------
        hspmap : `healsparse.HealSparseMap`
            Healsparse map to plot.
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        im : `matplotlib.collections.QuadMesh`
            Image that was displayed
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
        """
        self._hspmap = hspmap
        self._hpxmap = None

        if lon_range is None or lat_range is None:
            if zoom:
                # Using the coverage map is much faster even if approximate.
                _lon_range, _lat_range = healpix_pixels_range(hspmap.nside_coverage,
                                                              np.where(hspmap.coverage_mask)[0],
                                                              self._wrap,
                                                              nest=True)
            else:
                extent = self.get_extent()
                lon_range = [min(extent[0], extent[1]), max(extent[0], extent[1])]
                lat_range = [extent[2], extent[3]]

            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = hspmap_to_xy(hspmap,
                                                             lon_range,
                                                             lat_range,
                                                             xsize=xsize)

        if vmin is None or vmax is None:
            # Auto-scale from visible values
            _vmin, _vmax = np.percentile(values_raster.compressed(), (2.5, 97.5))
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax

        if zoom:
            # Watch for masked array here...
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, vmin=vmin, vmax=vmax, **kwargs)

        self._ax._sci(im)

        # Link up callbacks
        self._redraw_dict['hspmap'] = hspmap
        self._redraw_dict['hpxmap'] = None
        self._redraw_dict['lon_range_home'] = lon_range
        self._redraw_dict['lat_range_home'] = lat_range
        self._redraw_dict['im'] = im
        self._redraw_dict['vmin'] = vmin
        self._redraw_dict['vmax'] = vmax
        self._redraw_dict['xsize'] = xsize
        self._redraw_dict['kwargs_pcolormesh'] = kwargs

        return im, lon_raster, lat_raster, values_raster

    def draw_hpxbin(self, lon, lat, C=None, nside=256, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Create a healpix histogram of counts in lon, lat.

        Related to ``hexbin`` from matplotlib.

        If ``C`` array is specified then the mean is taken from the C values.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude values.
        lat : `np.ndarray`
            Array of latitude values.
        C : `np.ndarray`, optional
            Array of values to average in each pixel.
        nside : `int`, optional
            Healpix nside resolution.
        nest : `bool`, optional
            Compute map in nest ordering?
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        hpxmap : `np.ndarray`
            Computed healpix map.
        im : `matplotlib.collections.QuadMesh`
            Image that was displayed.
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
        """
        hpxmap = healpix_bin(lon, lat, C=C, nside=nside, nest=nest)

        im, lon_raster, lat_raster, values_raster = self.draw_hpxmap(
            hpxmap, nest=nest, zoom=zoom, xsize=xsize, vmin=vmin,
            vmax=vmax, rasterized=rasterized, lon_range=lon_range,
            lat_range=lat_range,
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
        im = plt.gci()
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

        cbar = plt.colorbar(cax=cax, orientation=orientation, ticks=ticks, format=format, **kwargs)
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

        plt.sca(ax)

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

        cbar = plt.colorbar(ax=ax, location=location, ticks=ticks,
                            fraction=fraction, pad=pad, **kwargs)

        if location == 'right' or location == 'left':
            cbar_axis = 'y'
        else:
            cbar_axis = 'x'

        cbar.ax.tick_params(axis=cbar_axis, labelsize=fontsize)

        if label is not None:
            cbar.set_label(label, size=fontsize)

        if self._aa is not None:
            self._aa.set_position(self._ax.get_position(), which='original')

        # Reset the "home" position because axis has been shifted.
        self._initial_extent_xy = self._ax.get_extent(lonlat=False)

        plt.sca(ax)

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

        gc = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
        radec = gc.fk5
        ra = radec.ra.degree
        dec = radec.dec.degree

        self.plot(ra, dec, linewidth=linewidth, color=color, linestyle=linestyle, **kwargs)
        # pop any labels
        kwargs.pop('label', None)
        if width > 0:
            for delta in [+width, -width]:
                gc = SkyCoord(l=glon*u.degree, b=(glat + delta)*u.degree, frame='galactic')
                radec = gc.fk5
                ra = radec.ra.degree
                dec = radec.dec.degree
                self.plot(ra, dec, linewidth=1.0, color=color,
                          linestyle='--', **kwargs)


# The following skyprojs include the equal-area projections that are tested
# and known to work.

class McBrydeSkyproj(Skyproj):
    # McBryde-Thomas Flat Polar Quartic
    def __init__(self, **kwargs):
        super().__init__(projection_name='mbtfpq', **kwargs)


class LaeaSkyproj(Skyproj):
    # Lambert Azimuthal Equal Area
    def __init__(self, **kwargs):
        super().__init__(projection_name='laea', **kwargs)


class MollweideSkyproj(Skyproj):
    # Mollweide
    pole_clip = 1.0

    def __init__(self, **kwargs):
        super().__init__(projection_name='moll', **kwargs)


class HammerSkyproj(Skyproj):
    # Hammer-Aitoff
    def __init__(self, **kwargs):
        super().__init__(projection_name='hammer', **kwargs)


class EqualEarthSkyproj(Skyproj):
    # Equal Earth
    def __init__(self, **kwargs):
        super().__init__(projection_name='eqearth', **kwargs)
