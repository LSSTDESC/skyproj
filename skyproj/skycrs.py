import math
import numpy as np

from pyproj import CRS
from pyproj import Transformer
from pyproj.exceptions import ProjError

from .utils import wrap_values

__all__ = ["SkyCRS", "PlateCarreeCRS", "McBrydeThomasFlatPolarQuarticCRS", "MollweideCRS",
           "HammerCRS", "EqualEarthCRS", "LambertAzimuthalEqualAreaCRS", "GnomonicCRS",
           "ObliqueMollweideCRS", "get_crs", "get_available_crs"]


RADIUS = 1.0


class SkyCRS(CRS):
    """Coordinate Reference System (CRS) class describing sky projections.

    This is a specialized subclass of a `pyproj.CRS` Coordinate Reference
    System object.  Unlike a general earth-based CRS it uses a fixed
    reference radius instead of a generalized ellipsoidal model.  And it
    has additional routines used by SkyProj.

    Parameters
    ----------
    name : `str`, optional
        Name of projection CRS type.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name=None, radius=RADIUS, **kwargs):
        self._name = name
        self.proj4_params = {'ellps': 'sphere',
                             'R': radius}
        self.proj4_params.update(**kwargs)
        super().__init__(self.proj4_params)

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
        proj4_params = self.proj4_params.copy()
        proj4_params['lon_0'] = lon_0
        if lat_0 is not None:
            proj4_params['lat_0'] = lat_0

        return self.__class__(**proj4_params)

    def transform_points(self, src_crs, x, y):
        """Transform points from a source coordinate reference system (CRS)
        to this CRS.

        Parameters
        ----------
        src_crs : `skyproj.SkyCRS`
            Source coordinate reference system describing x/y points.
        x : `np.ndarray`
            Array of x values, may be any number of dimensions.
        y : `np.ndarray`
            Array of y values, may be any number of dimensions.

        Returns
        -------
        result : `np.ndarray`
            Array of transformed points. Dimensions are [dim_x, 2]
            such that the final index gives the transformed x_prime, y_prime.
        """
        result_shape = tuple(x.shape[i] for i in range(x.ndim)) + (2, )

        x = x.ravel()
        y = y.ravel()

        npts = x.shape[0]

        result = np.zeros([npts, 2], dtype=np.float64)
        if npts:
            if isinstance(src_crs, PlateCarreeCRS):
                # We need to wrap to [-180, 180)
                x = wrap_values(x)
            try:
                transformer = Transformer.from_crs(src_crs, self, always_xy=True)
                result[:, 0], result[:, 1] = transformer.transform(x, y, None, errcheck=False)
            except ProjError as err:
                msg = str(err).lower()
                if (
                    "latitude" in msg
                    or "longitude" in msg
                    or "outside of projection domain" in msg
                    or "tolerance condition error" in msg
                ):
                    result[:] = np.nan
                else:
                    raise

            # and set to nans
            result[~np.isfinite(result)] = np.nan

        if len(result_shape) > 2:
            return result.reshape(result_shape)

        return result

    @property
    def lon_0(self):
        return self.proj4_params['lon_0']

    @property
    def lat_0(self):
        if 'lat_0' in self.proj4_params:
            return self.proj4_params['lat_0']
        else:
            return None

    @property
    def name(self):
        return self._name

    @property
    def radius(self):
        return self.proj4_params['R']

    def _as_mpl_transform(self, axes=None):
        from .transforms import SkyTransform

        if axes is None:
            raise ValueError("help need axes")

        return (SkyTransform(self) + axes.transData)

    def _as_mpl_axes(self):
        from .skyaxes import SkyAxes

        return SkyAxes, {'sky_crs': self}


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='cyl', lon_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'eqc',
                        'lon_0': lon_0,
                        'to_meter': math.radians(1)*radius,
                        'vto_meter': 1}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='mbtfpq', lon_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'mbtfpq',
                        'lon_0': lon_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='moll', lon_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'moll',
                        'lon_0': lon_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='obmoll', lon_0=0.0, lat_p=90.0, lon_p=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'ob_tran',
                        'o_proj': 'moll',
                        'o_lat_p': lat_p,
                        'o_lon_p': lon_p,
                        'lon_0': lon_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)

    @property
    def lon_0(self):
        return self.proj4_params['lon_0'] + self.proj4_params['o_lon_p']


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='hammer', lon_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'hammer',
                        'lon_0': lon_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='eqearth', lon_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'eqearth',
                        'lon_0': lon_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='laea', lon_0=0.0, lat_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'laea',
                        'lon_0': lon_0,
                        'lat_0': lat_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


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
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name='gnom', lon_0=0.0, lat_0=0.0, radius=RADIUS, **kwargs):
        proj4_params = {'proj': 'gnom',
                        'lon_0': lon_0,
                        'lat_0': lat_0}
        proj4_params = {**proj4_params, **kwargs}

        super().__init__(name=name, radius=radius, **proj4_params)


_crss = {
    'hammer': ('Hammer', HammerCRS),
    'mbtfpq': ('McBryde-Thomas Flat Polar Quartic', McBrydeThomasFlatPolarQuarticCRS),
    'cyl': ('Plate Carree', PlateCarreeCRS),
    'eqearth': ('Equal Earth', EqualEarthCRS),
    'laea': ('Lambert Azimuthal Equal Area', LambertAzimuthalEqualAreaCRS),
    'moll': ('Mollweide', MollweideCRS),
    'obmoll': ('Oblique Mollweide', ObliqueMollweideCRS),
    'gnom': ('Gnomonic', GnomonicCRS)
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
