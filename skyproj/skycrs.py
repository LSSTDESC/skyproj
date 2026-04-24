import numpy as np
import warnings

from ._cskyproj import (
    transform,
    PLATE_CARREE,
    MOLLWEIDE,
    EQUAL_EARTH,
    MBTFPQ,
    HAMMER,
    LAEA,
    GNOMONIC,
    ALBERS,
    OBLIQUE_MOLLWEIDE,
)

__all__ = ["SkyCRS", "PlateCarreeCRS", "McBrydeThomasFlatPolarQuarticCRS", "MollweideCRS",
           "HammerCRS", "EqualEarthCRS", "LambertAzimuthalEqualAreaCRS", "GnomonicCRS",
           "ObliqueMollweideCRS", "AlbersEqualAreaCRS", "get_crs", "get_available_crs",
           "proj", "proj_inverse"]


RADIUS = 1.0


class SkyCRS:
    """Coordinate Reference System (CRS) class describing sky projections.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS type.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name=None, radius=RADIUS, **kwargs):
        self._name = name

        self._plot_geodesics = True

        self._projection_dict = {
            "radius": radius,
        }
        self._projection_dict.update(**kwargs)

        if name == "cyl":
            self._projection_dict["projection"] = PLATE_CARREE
        elif name == "moll":
            self._projection_dict["projection"] = MOLLWEIDE
        elif name == "eqearth":
            self._projection_dict["projection"] = EQUAL_EARTH
        elif name == "mbtfpq":
            self._projection_dict["projection"] = MBTFPQ
        elif name == "hammer":
            self._projection_dict["projection"] = HAMMER
        elif name == "laea":
            self._projection_dict["projection"] = LAEA
        elif name == "gnom":
            self._projection_dict["projection"] = GNOMONIC
        elif name == "aea":
            self._projection_dict["projection"] = ALBERS
        elif name == "obmoll":
            self._projection_dict["projection"] = OBLIQUE_MOLLWEIDE

    def with_new_center(self, lon_0, lat_0=None):
        """Create a new SkyCRS with a new lon_0/lat_0.

        Parameters
        ----------
        lon_0 : `float`
            New longitude center.
        lat_0 : `float`, optional
            New latitude center (for projections that support it.)

        Returns
        -------
        crs : `skyproj.SkyCRS`
            New projection CRS.
        """
        projection_dict = self._projection_dict.copy()
        projection_dict["lon_0"] = lon_0
        if lat_0 is not None:
            projection_dict["lat_0"] = lat_0

        return self.__class__(name=self._name, **projection_dict)

    def transform_points(self, x, y, inverse=False):
        """Transform points from lon/lat to this CRS or the inverse.

        Parameters
        ----------
        x : `np.ndarray`
            Array of x values, may be any number of dimensions.
        y : `np.ndarray`
            Array of y values, may be any number of dimensions.
        inverse : `bool`
            Apply inverse transformation (CRS to lon/lat)?

        Returns
        -------
        result : `np.ndarray`
            Array of transformed points. Dimensions are [dim_x, 2]
            such that the final index gives the transformed x_prime, y_prime.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        result_shape = tuple(x.shape[i] for i in range(x.ndim)) + (2, )

        x = x.ravel()
        y = y.ravel()

        if inverse:
            result = transform(self._projection_dict, x, y, inverse=inverse)
        else:
            result = transform(self._projection_dict, x, y, inverse=inverse)

        if len(result_shape) > 2:
            return result.reshape(result_shape)

        return result

    @property
    def lon_0(self):
        return self._projection_dict["lon_0"]

    @property
    def lat_0(self):
        return self._projection_dict.get("lat_0", None)

    @property
    def lat_1(self):
        return self._projection_dict.get("lat_1", None)

    @property
    def lat_2(self):
        return self._projection_dict.get("lat_2", None)

    @property
    def name(self):
        return self._name

    @property
    def radius(self):
        return self._projection_dict["radius"]

    @property
    def proj4_params(self):
        warnings.warn("Use of proj4_params has been deprecated.", FutureWarning)

        proj4_params = {
            "R": self.radius,
            "proj": self.name,
        }
        for key in ["lon_0", "lat_0", "lat_1", "lat_2"]:
            if key in self._projection_dict:
                proj4_params[key] = self._projection_dict[key]

        if self.name == "obmoll":
            proj4_params["proj"] = "ob_tran"
            proj4_params["o_proj"] = "moll"
            proj4_params["o_lat_p"] = self._projection_dict["lat_p"]
            proj4_params["o_lon_p"] = self._projection_dict["lon_p"]

        return proj4_params

    def set_plot_geodesics(self, plot_geodesics):
        self._plot_geodesics = plot_geodesics

    def _as_mpl_transform(self, axes=None):
        from .transforms import SkyTransform

        if axes is None:
            raise ValueError("help need axes")

        return (SkyTransform(self, plot_geodesics=self._plot_geodesics) + axes.transData)

    def _as_mpl_axes(self):
        from .skyaxes import SkyAxes

        axes = SkyAxes

        return axes, {'sky_crs': self}


class PlateCarreeCRS(SkyCRS):
    """Equirectangular (Plate carree) sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``cyl``.
    lon_0 : `float`, optional
        Central longitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='cyl', lon_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, **kwargs)


class McBrydeThomasFlatPolarQuarticCRS(SkyCRS):
    """McBryde Thomas Flat Polar Quartic sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``mbtfpq``.
    lon_0 : `float`, optional
        Central longitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='mbtfpq', lon_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, **kwargs)


class MollweideCRS(SkyCRS):
    """Mollweide sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``moll``.
    lon_0 : `float`, optional
        Central longitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='moll', lon_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, **kwargs)


class ObliqueMollweideCRS(SkyCRS):
    """Oblique Mollweide sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``moll``.
    lon_0 : `float`, optional
        Central longitude of projection.
    lat_p : `float`, optional
        Latitude of the North Pole of the unrotated coordinate system.
    lon_p : `float`, optional
        Longitude of the North Pole of the unrotated coordinate system.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='obmoll', lon_0=0.0, lat_p=90.0, lon_p=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, lat_p=lat_p, lon_p=lon_p, **kwargs)

    @property
    def lon_0(self):
        return self._projection_dict["lon_0"] + self._projection_dict["lon_p"]


class HammerCRS(SkyCRS):
    """Hammer-Aitoff sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``hammer``.
    lon_0 : `float`, optional
        Central longitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='hammer', lon_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, **kwargs)


class EqualEarthCRS(SkyCRS):
    """Equal Earth sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``eqearth``.
    lon_0 : `float`, optional
        Central longitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='eqearth', lon_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, **kwargs)


class LambertAzimuthalEqualAreaCRS(SkyCRS):
    """Lambert Azimuthal Equal Area sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``laea``.
    lon_0 : `float`, optional
        Central longitude of projection.
    lat_0 : `float`, optional
        Central latitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='laea', lon_0=0.0, lat_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, lat_0=lat_0, **kwargs)


class GnomonicCRS(SkyCRS):
    """Gnomonic sky CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``gnom``.
    lon_0 : `float`, optional
        Central longitude of projection.
    lat_0 : `float`, optional
        Central latitude of projection.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='gnom', lon_0=0.0, lat_0=0.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, lat_0=lat_0, **kwargs)


class AlbersEqualAreaCRS(SkyCRS):
    """Albers Equal Area CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS. Must be ``aea``.
    lon_0 : `float`, optional
        Central longitude of projection.
    lat_1 : `float`, optional
        First standard parallel.
    lat_2 : `float`, optional
        Second standard parallel.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for projection.
    """
    def __init__(self, name='aea', lon_0=0.0, lat_1=15.0, lat_2=45.0, radius=RADIUS, **kwargs):
        super().__init__(name=name, radius=radius, lon_0=lon_0, lat_1=lat_1, lat_2=lat_2, **kwargs)


_crss = {
    'hammer': ('Hammer', HammerCRS),
    'mbtfpq': ('McBryde-Thomas Flat Polar Quartic', McBrydeThomasFlatPolarQuarticCRS),
    'cyl': ('Plate Carree', PlateCarreeCRS),
    'eqearth': ('Equal Earth', EqualEarthCRS),
    'laea': ('Lambert Azimuthal Equal Area', LambertAzimuthalEqualAreaCRS),
    'moll': ('Mollweide', MollweideCRS),
    'obmoll': ('Oblique Mollweide', ObliqueMollweideCRS),
    'gnom': ('Gnomonic', GnomonicCRS),
    'aea': ('Albers Equal Area', AlbersEqualAreaCRS),
}


def get_crs(name, **kwargs):
    """Return a skyproj CRS.

    For list of projections available, use skyproj.get_available_crs().

    Parameters
    ----------
    name : `str`
        Skyproj name of projection CRS.
    **kwargs :
        Additional kwargs appropriate for given projection CRS.

    Returns
    -------
    crs : `skyproj.SkyCRS`
    """
    # Is this a listed projection CRS?
    if name not in _crss:
        raise ValueError(f'{name} CRS name is not recognized.  See get_available_crs()')

    descr, crsclass = _crss[name]

    return crsclass(name=name, **kwargs)


def get_available_crs():
    """Return dict of available projection CRSs.

    Returns
    -------
    available_crs: `dict`
        Available CRSs.  Key is skyproj name, value is brief description.
    """
    available_crs = {}
    for name, (descr, crsclass) in _crss.items():
        available_crs[name] = descr

    return available_crs


def proj(lon, lat, projection=None, pole_clip=None, wrap=None):
    if projection is None:
        raise RuntimeError("Must specify a projection.")

    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    if pole_clip is not None:
        out = ((lat < (-90.0 + pole_clip))
               | (lat > (90.0 - pole_clip)))
    if wrap is not None:
        lon[np.isclose(lon, wrap)] = wrap - 1e-10
    proj_xy = projection.transform_points(lon, lat)
    if pole_clip is not None:
        proj_xy[..., 1][out] = np.nan

    return proj_xy[..., 0], proj_xy[..., 1]


def proj_inverse(x, y, projection=None):
    if projection is None:
        raise RuntimeError("Must specify a projection.")

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    proj_lonlat = projection.transform_points(x, y, inverse=True)
    return proj_lonlat[..., 0], proj_lonlat[..., 1]
