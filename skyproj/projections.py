import math
import numpy as np

from pyproj import CRS
from pyproj import Transformer
from pyproj.exceptions import ProjError

from .utils import wrap_values

__all__ = ["SkyProjection", "PlateCarree", "McBrydeThomasFlatPolarQuartic", "Mollweide",
           "Hammer", "EqualEarth", "LambertAzimuthalEqualArea", "Gnomonic",
           "ObliqueMollweide", "get_projection", "get_available_projections"]


RADIUS = 1.0


class SkyProjection(CRS):
    """Projection class describing sky projections.

    This is a specialized subclass of a `pyproj.CRS` Coordinate Reference
    System object.  Unlike a general earth-based CRS it uses a fixed
    reference radius instead of a generalized ellipsoidal model.  And it
    has additional routines used by SkyProj.

    Parameters
    ----------
    name : `str`, optional
        Name of projection type.
    radius : `float`, optional
        Radius of projected sphere.
    **kwargs : `dict`, optional
        Additional kwargs for PROJ4 parameters.
    """
    def __init__(self, name=None, radius=RADIUS, **kwargs):
        self._name = name
        self.proj4_params = {'a': radius,
                             'b': radius}
        self.proj4_params.update(**kwargs)
        super().__init__(self.proj4_params)

    def with_new_center(self, lon_0, lat_0=None):
        """Create a new SkyProjection with a new lon_0/lat_0.

        Parameters
        ----------
        lon_0 : `float`
            New longitude center.
        lat_0 : `float`, optional
            New latitude center (for projections that support it.)

        Returns
        -------
        proj : `skyproj.SkyProjection`
            New projection.
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
        src_crs : `skyproj.SkyProjection`
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
            if isinstance(src_crs, PlateCarree):
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
        return self.proj4_params['a']

    def _as_mpl_transform(self, axes=None):
        from .transforms import SkyTransform

        if axes is None:
            raise ValueError("help need axes")

        return (SkyTransform(self) + axes.transData)

    def _as_mpl_axes(self):
        from .skyaxes import SkyAxes

        return SkyAxes, {'sky_projection': self}


class PlateCarree(SkyProjection):
    """Equirectangular (Plate carree) sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``cyl``.
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


class McBrydeThomasFlatPolarQuartic(SkyProjection):
    """McBryde Thomas Flat Polar Quartic sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``mbtfpq``.
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


class Mollweide(SkyProjection):
    """Mollweide sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``moll``.
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


class ObliqueMollweide(SkyProjection):
    """Oblique Mollweide sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``moll``.
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


class Hammer(SkyProjection):
    """Hammer-Aitoff sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``hammer``.
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


class EqualEarth(SkyProjection):
    """Equal Earth sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``eqearth``.
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


class LambertAzimuthalEqualArea(SkyProjection):
    """Lambert Azimuthal Equal Area sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``laea``.
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


class Gnomonic(SkyProjection):
    """Gnomonic sky projection CRS.

    Parameters
    ----------
    name : `str`, optional
        Name of projection. Must be ``gnom``.
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


_projections = {
    'hammer': ('Hammer', Hammer),
    'mbtfpq': ('McBryde-Thomas Flat Polar Quartic', McBrydeThomasFlatPolarQuartic),
    'cyl': ('Plate Carree', PlateCarree),
    'eqearth': ('Equal Earth', EqualEarth),
    'laea': ('Lambert Azimuthal Equal Area', LambertAzimuthalEqualArea),
    'moll': ('Mollweide', Mollweide),
    'obmoll': ('Oblique Mollweide', ObliqueMollweide),
    'gnom': ('Gnomonic', Gnomonic)
}


def get_projection(name, **kwargs):
    """Return a skyproj projection.

    For list of projections available, use skyproj.get_available_projections().

    Parameters
    ----------
    name : `str`
        Skyproj name of projection.
    **kwargs :
        Additional kwargs appropriate for given projection.

    Returns
    -------
    proj : `skyproj.SkyProjection`
    """
    # Is this a listed projection?
    if name not in _projections:
        raise ValueError(f'{name} projection name is not recognized.  See get_available_projections()')

    descr, projclass = _projections[name]

    return projclass(name=name, **kwargs)


def get_available_projections():
    """Return dict of available projections.

    Returns
    -------
    available_projections: `dict`
        Available projections.  Key is cartosky name, value is brief description.
    """
    available_projections = {}
    for name, (descr, projclass) in _projections.items():
        available_projections[name] = descr

    return available_projections
