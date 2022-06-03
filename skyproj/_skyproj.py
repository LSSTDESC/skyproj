import warnings

import matplotlib.pyplot as plt

import numpy as np
import healpy as hp

import mpl_toolkits.axisartist as axisartist
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .skycrs import get_crs, PlateCarreeCRS, GnomonicCRS
from .hpx_utils import healpix_pixels_range, hspmap_to_xy, hpxmap_to_xy, healpix_to_xy, healpix_bin
from .mpl_utils import ExtremeFinderWrapped, WrappedFormatterDMS, GridHelperSkyproj
from .utils import wrap_values, _get_boundary_poly_xy


class _Skyproj():
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
        Automatically rescale color bars when zoomed?
    galactic : `bool`, optional
        Plotting in Galactic coordinates?  Recommendation for Galactic plots
        is to have longitude_ticks set to ``symmetric`` and celestial = True.
    **kwargs : `dict`, optional
        Additional arguments to send to cartosky/proj4 projection CRS initialization.
    """
    def __init__(self, ax=None, projection_name='cyl', lon_0=0, gridlines=True, celestial=True,
                 extent=None, longitude_ticks='positive', autorescale=True, galactic=False, **kwargs):
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
        crs = get_crs(projection_name, **kwargs)
        self._ax = fig.add_subplot(subspec, projection=crs)
        self._crs_orig = crs
        self._reprojected = False

        self._aa = None

        self.do_celestial = celestial
        self.do_gridlines = gridlines
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
        self._boundary_labels = []

        self._initialize_axes(extent, extent_xy=extent_xy)

        # Set up callbacks on axis zoom.
        self._add_change_axis_callbacks()

        # Set up callback on figure resize.
        self._frc = self.ax.figure.canvas.mpl_connect('resize_event', self._change_size)
        self._dc = self.ax.figure.canvas.mpl_connect('draw_event', self._draw_callback)
        self._initial_extent_xy = [0]*4

        # Set up reproject callback.
        self._rpc = self.ax.figure.canvas.mpl_connect('key_press_event', self._keypress_callback)

        self._draw_aa_bounds_and_labels()

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
        out = ((lat < (-90.0 + self._pole_clip))
               | (lat > (90.0 - self._pole_clip)))
        proj_xy = self.crs.transform_points(PlateCarreeCRS(), lon, lat)
        # FIXME I don't like this, look at the get_extent code instead/as well?
        proj_xy[..., 1][out] = np.nan
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
        proj_lonlat = PlateCarreeCRS().transform_points(self.crs, x, y)
        return proj_lonlat[..., 0], proj_lonlat[..., 1]

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
        # Reset any axis artist if necessary
        if self._aa is not None:
            self._aa.remove()
            self._aa = None

        self._set_axes_limits(extent, extent_xy=extent_xy, invert=False)
        self._create_axes(extent)
        self._set_axes_limits(extent, extent_xy=extent_xy, invert=self.do_celestial)

        self._ax.set_frame_on(False)
        if self.do_gridlines:
            self._aa.grid(True, linestyle=':', color='k', lw=0.5)

        self._aa.axis[:].line.set_visible(False)
        self._aa.axis[:].major_ticks.set_visible(False)

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
        self._set_axes_limits(extent, invert=self.do_celestial)
        self._extent_xy = self._ax.get_extent(lonlat=False)

        self._draw_aa_bounds_and_labels()

    def _draw_aa_bounds_and_labels(self):
        """Set the axisartist bounds and labels."""
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
                                             linewidth=plt.rcParams['axes.linewidth'])

        self._aa.axis[:].line.set_visible(False)
        self._aa.axis[:].major_ticks.set_visible(False)

        # Remove any previous labels
        if self._boundary_labels:
            for label in self._boundary_labels:
                label.remove()
            self._boundary_labels = []

        grid_finder = self._grid_helper.grid_finder
        grid_info = grid_finder.get_grid_info(
            extent_xy[0],
            extent_xy[2],
            extent_xy[1],
            extent_xy[3]
        )

        # Recover lon_level, lat_level
        extremes = grid_finder.extreme_finder(
            grid_finder.inv_transform_xy,
            extent_xy[0],
            extent_xy[2],
            extent_xy[1],
            extent_xy[3]
        )
        _, _, lon_factor = grid_finder.grid_locator1(extremes[0], extremes[1])
        _, _, lat_factor = grid_finder.grid_locator2(extremes[2], extremes[3])

        self._boundary_labels.extend(self._draw_aa_lat_labels(extent_xy, grid_info, lat_factor))
        self._boundary_labels.extend(self._draw_aa_lon_labels(extent_xy, grid_info, lon_factor))

    def _draw_aa_lat_labels(self, extent_xy, grid_info, factor):
        """Draw axis artist latitude labels.

        Parameters
        ----------
        extent_xy : `list`
            Extent in x/y space
        grid_info : `dict`
            Grid info to determine label locations
        factor : `float`
            Multiplicative factor to convert ticks to values.

        Returns
        -------
        labels : `list` [`matplotlib.Text`]
        """
        levels = grid_info['lat']['levels']
        lines = grid_info['lat']['lines']

        inverted = (extent_xy[1] < extent_xy[0])

        # The grid_info will be reversed left/right if the axis is inverted.
        if inverted:
            gi_side_map = {'left': 'right',
                           'right': 'left'}
            x0_index = 1
            x1_index = 0
        else:
            gi_side_map = {side: side for side in ['left', 'right']}
            x0_index = 0
            x1_index = 1

        boundary_labels = []

        for axis_side in ['left', 'right']:
            if not self._aa.axis[axis_side].major_ticklabels.get_visible():
                continue

            tick_levels = grid_info['lat']['tick_levels'][gi_side_map[axis_side]]

            for lat_level, lat_line in zip(levels, lines):
                if np.abs(np.abs(lat_level) - 90.0) < 1.0:
                    continue

                if int(lat_level*factor) in tick_levels:
                    continue

                lat_line_x = lat_line[0][0]
                lat_line_y = lat_line[0][1]

                if gi_side_map[axis_side] == 'right':
                    lat_line_x = lat_line_x[::-1]
                    lat_line_y = lat_line_y[::-1]

                if axis_side == 'left':
                    ha = 'right'
                else:
                    ha = 'left'

                if lat_level < 0.0:
                    va = 'top'
                else:
                    va = 'bottom'

                # Skip any that are out of the y bounding box.
                if lat_line_y[0] < extent_xy[2] or lat_line_y[0] > extent_xy[3]:
                    continue

                if lat_line_x[0] < extent_xy[x0_index] or lat_line_y[0] > extent_xy[x1_index]:
                    continue

                label = self._tick_formatter2(axis_side, factor, [lat_level])[0]
                boundary_labels.append(self._ax.text(lat_line_x[0],
                                                     lat_line_y[0],
                                                     label,
                                                     size=plt.rcParams['ytick.labelsize'],
                                                     lonlat=False,
                                                     clip_on=False,
                                                     ha=ha,
                                                     va=va))
        return boundary_labels

    def _draw_aa_lon_labels(self, extent_xy, grid_info, factor):
        """Draw axis artist latitude labels.

        Parameters
        ----------
        extent_xy : `list`
            Extent in x/y space
        grid_info : `dict`
            Grid info to determine label locations
        factor : `float`
            Multiplicative factor to convert ticks to values.

        Returns
        -------
        labels : `list` [`matplotlib.Text`]
        """
        levels = grid_info['lon']['levels']
        lines = grid_info['lon']['lines']

        # Need to compute maximum extent in the x direction
        x_min = 1e100
        x_max = -1e100
        for line in grid_info['lon_lines']:
            x_min = min((x_min, np.min(line[0])))
            x_max = max((x_max, np.max(line[0])))
        delta_x = x_max - x_min

        # The grid_info will be reversed left/right if the axis is inverted.
        inverted = (extent_xy[1] < extent_xy[0])
        if inverted:
            x0_index = 1
            x1_index = 0
        else:
            x0_index = 0
            x1_index = 1

        boundary_labels = []

        draw_equatorial_labels = False
        if self._equatorial_labels:
            min_lat = np.min(grid_info['lat']['levels'])
            max_lat = np.max(grid_info['lat']['levels'])
            if min_lat < -89.0 and max_lat > 89.0:
                draw_equatorial_labels = True

        if draw_equatorial_labels:
            levels = np.array(grid_info['lon']['levels'])

            x, y = self.proj(levels, np.zeros(len(levels)))

            ok, = np.where((x > extent_xy[x0_index]) & (x < extent_xy[x1_index])
                           & (y > extent_xy[2]) & (y < extent_xy[3]))

            prev_x = None
            for i in ok:
                if prev_x is not None:
                    # Check if too close to last label.
                    if abs(x[i] - prev_x)/delta_x < 0.05:
                        continue
                prev_x = x[i]

                label = self._tick_formatter1('top', factor, [levels[i]])[0]
                boundary_labels.append(self._ax.text(x[i],
                                                     y[i],
                                                     label,
                                                     size=plt.rcParams['xtick.labelsize'],
                                                     lonlat=False,
                                                     clip_on=False,
                                                     ha='right',
                                                     va='bottom'))
        else:
            if self._radial_labels:
                line_index = -1
            else:
                line_index = 0
            for axis_side in ['top', 'bottom']:
                if not self._aa.axis[axis_side].major_ticklabels.get_visible():
                    continue

                tick_levels = grid_info['lon']['tick_levels'][axis_side]

                prev_x = None
                for lon_level, lon_line in zip(levels, lines):
                    if int(lon_level*factor) in tick_levels:
                        continue

                    lon_line_x = lon_line[0][0]
                    lon_line_y = lon_line[0][1]

                    if lon_line_x[line_index] < extent_xy[x0_index] or \
                       lon_line_x[line_index] > extent_xy[x1_index] \
                       or lon_line_y[line_index] < extent_xy[2] or \
                       lon_line_y[line_index] > extent_xy[3]:
                        continue

                    if axis_side == 'top':
                        va = 'bottom'
                        index = -1
                        y_offset = 0.02*(lon_line_y[-1] - lon_line_y[0])
                    else:
                        va = 'top'
                        index = 0
                        y_offset = -0.02*(lon_line_y[-1] - lon_line_y[0])

                    if prev_x is not None:
                        # check if too close to last label.
                        if abs(lon_line_x[index] - prev_x)/delta_x < 0.05:
                            continue

                    prev_x = lon_line_x[index]

                    label = self._tick_formatter1(axis_side, factor, [lon_level])[0]
                    boundary_labels.append(self._ax.text(lon_line_x[index],
                                                         lon_line_y[index] + y_offset,
                                                         label,
                                                         size=plt.rcParams['xtick.labelsize'],
                                                         lonlat=False,
                                                         clip_on=False,
                                                         ha='center',
                                                         va=va))

        return boundary_labels

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
        if self._wrap == 180.0 and not self._full_circle:
            include_last_lon = True
        else:
            include_last_lon = False
        grid_locator1 = angle_helper.LocatorD(10, include_last=include_last_lon)
        grid_locator2 = angle_helper.LocatorD(6, include_last=True)

        # We always want the formatting to be wrapped at 180 (-180 to 180)
        self._tick_formatter1 = WrappedFormatterDMS(180.0, self._longitude_ticks)
        self._tick_formatter2 = angle_helper.FormatterDMS()

        def proj_wrap(lon, lat):
            lon = np.atleast_1d(lon)
            lat = np.atleast_1d(lat)
            lon[np.isclose(lon, self._wrap)] = self._wrap - 1e-10
            proj_xy = self.crs.transform_points(PlateCarreeCRS(), lon, lat)
            return proj_xy[..., 0], proj_xy[..., 1]

        if self.crs.name == 'cyl':
            delta_cut = 80.0
        else:
            delta_cut = 0.5*self.crs.radius

        grid_helper = GridHelperSkyproj(
            (proj_wrap, self.proj_inverse),
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
            tick_formatter1=self._tick_formatter1,
            tick_formatter2=self._tick_formatter2,
            celestial=self.do_celestial,
            equatorial_labels=self._equatorial_labels,
            delta_cut=delta_cut
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

        self.set_xlabel(self._default_xy_labels[0], size=16)
        self.set_ylabel(self._default_xy_labels[1], size=16)

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

        # This synchronizes the axis artist to the plot axes after zoom.
        if self._aa is not None:
            self._aa.set_position(self._ax.get_position(), which='original')

        self._draw_aa_bounds_and_labels()

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
            plt.draw()

            self._reprojected = True

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
        while not enclosed and lon_min > (self.lon_0 - 180.0):
            e_x, e_y = self.proj([lon_min, lon_min], [lat_min, lat_max])
            n_out = np.sum(x < e_x.min())
            if n_out == 0:
                enclosed = True
            else:
                lon_min = np.clip(lon_min - lon_step, self.lon_0 - 180., None)

        # Compute lon_max so that it fits all the data
        enclosed = False
        lon_max = lon_cent + lon_step
        while not enclosed and lon_max < (self.lon_0 + 180.0):
            e_x, e_y = self.proj([lon_max, lon_max], [lat_min, lat_max])
            n_out = np.sum(x > e_x.max())
            if n_out == 0:
                enclosed = True
            else:
                lon_max = np.clip(lon_max + lon_step, None, self.lon_0 + 180.)

        return [lon_max, lon_min, lat_min, lat_max]

    def plot(self, *args, **kwargs):
        return self._ax.plot(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        return self._ax.scatter(*args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        return self._ax.pcolormesh(*args, **kwargs)

    def fill(self, *args, **kwargs):
        return self._ax.fill(*args, **kwargs)

    def circle(self, *args, **kwargs):
        return self._ax.circle(*args, **kwargs)

    def ellipse(self, *args, **kwargs):
        return self._ax.ellipse(*args, **kwargs)

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
            if values_raster.dtype == bool:
                _vmin, _vmax = 0, 1
            else:
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
                    rasterized=True, lon_range=None, lat_range=None, valid_mask=False,
                    **kwargs):
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
            Minimum value for color scale.  Defaults to 2.5th percentile, or 0 for bool.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile, or 1 for bool.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        valid_mask : `bool`, optional
            Plot the valid pixels of the map.
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

        if vmin is None or vmax is None:
            if values_raster.dtype == bool:
                _vmin, _vmax = 0, 1
            else:
                # Auto-scale from visible values
                _vmin, _vmax = np.percentile(values_raster.compressed(), (2.5, 97.5))
            if _vmin == _vmax:
                # This will make the color scaling work decently well when we
                # have a flat integer type map.
                _vmin -= 0.1
                _vmax += 0.1
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

        if not self._galactic:
            gc = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
            radec = gc.fk5
            lon = radec.ra.degree
            lat = radec.dec.degree
        else:
            lon = glon
            lat = glat

        self.plot(lon, lat, linewidth=linewidth, color=color, linestyle=linestyle, **kwargs)
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
                self.plot(lon, lat, linewidth=1.0, color=color,
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
        # Allow clipping of the poles until Mollweide is fixed in proj
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
