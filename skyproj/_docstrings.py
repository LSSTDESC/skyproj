def add_func_docstr(docstring):
    def _doc(func):
        func.__doc__ = docstring
        return func
    return _doc


def skyproj_init_parameters(intro_string, include_projection_name=False, include_lon_0=True):
    par_str = intro_string

    par_str += """

Parameters
----------
ax : `matplotlib.axes.Axes`, optional
    Axis object to replace with a skyproj axis."""

    if include_projection_name:
        par_str += """
projection_name : `str`, optional
    Valid proj4/cartosky projection name."""

    if include_lon_0:
        par_str += """
lon_0 : `float`, optional
    Central longitude of projection."""

    par_str += """
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
rcparams : `dict`, optional
    Dictionary of matplotlib rc parameters to override.  In particular, the code will
    use ``xtick.labelsize`` and ``ytick.labelsize`` for the x and y tick labels, and
    ``axes.linewidth`` for the map boundary.
n_grid_lon : `int`, optional
    Number of gridlines to use in the longitude direction
    (default is axis_ratio * n_grid_lat).
n_grid_lat : `int`, optional
    Number of gridlines to use in the latitude direction (default is 6).
min_lon_ticklabel_delta : `float`, optional
    Minimum relative spacing between longitude tick labels (relative to width
    of axis). Smaller values yield closer tick labels (and potential for clashes)
    and larger values yield further tick labels."""

    return par_str


skyproj_kwargs_par = """
**kwargs : `dict`, optional
    Additional arguments to send to cartosky/proj4 projection CRS initialization.
"""


draw_map_common_pars = """
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
norm : `str` or `matplotlib.colors.Normalize`, optional
    The normalization method used to scale the data. By default a
    linear scaling is used. This may be an instance of `Normalize` or
    a scale name, such as ``linear``, ``log``, ``symlog``, ``logit``.
**kwargs : `dict`
    Additional args to pass to pcolormesh."""


draw_map_returns = """

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

draw_hpxmap_docstr = """Use pcolormesh to draw a healpix map.

Parameters
----------
hpxmap : `np.ndarray`
    Healpix map to plot, with length 12*nside*nside and UNSEEN for
    illegal values.
nest : `bool`, optional
    Map in nest ordering?""" + draw_map_common_pars + draw_map_returns


draw_hpxpix_docstr = """Use pcolormesh to draw a healpix map made of pixels and values.

Parameters
----------
nside : `int`
    Healpix nside of pixels to plot.
pixels : `np.ndarray`
    Array of pixels to plot.
values : `np.ndarray`
    Array of values associated with pixels.""" + draw_map_common_pars + draw_map_returns


draw_hspmap_docstr = """Use pcolormesh to draw a healsparse map.

Parameters
----------
hspmap : `healsparse.HealSparseMap`
    Healsparse map to plot.""" + draw_map_common_pars + draw_map_returns


draw_hpxbin_docstr = """Create a healpix histogram of counts in lon, lat.

Related to ``hexbin`` from matplotlib.

If ``C`` array is specified then the mean is taken from the C values.

Parameters
----------
lon : `np.ndarray`
    Array of longitude values.
lat : `np.ndarray`
    Array of latitude values.
C : `np.ndarray`, optional
    Array of values to average in each pixel.""" + draw_map_common_pars + draw_map_returns
