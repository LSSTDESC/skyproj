import math
import numpy as np

from pyproj import CRS
from pyproj import Transformer
from pyproj.exceptions import ProjError

from .utils import wrap_values

__all__ = ["SkyProjection", "PlateCarree", "McBrydeThomasFlatPolarQuartic", "Mollweide",
           "Hammer", "EqualEarth", "LambertAzimuthalEqualArea",
           "get_projection", "get_available_projections"]


RADIUS = 6378137.0  # meters


class SkyProjection(CRS):
    def __init__(self, radius=RADIUS, **kwargs):
        self.proj4_params = {'a': radius,
                             'b': radius}
        self.proj4_params.update(**kwargs)
        super().__init__(self.proj4_params)

    def transform_points(self, src_crs, x, y):
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

    def _as_mpl_transform(self, axes=None):
        from .transforms import SkyTransform

        if axes is None:
            raise ValueError("help need axes")

        return (SkyTransform(self) + axes.transData)

    def _as_mpl_axes(self):
        from .skyaxes import SkyAxes

        return SkyAxes, {'sky_projection': self}


class PlateCarree(SkyProjection):
    def __init__(self, lon_0=0.0, radius=RADIUS):
        proj4_params = {'proj': 'eqc',
                        'lon_0': lon_0,
                        'to_meter': math.radians(1)*radius,
                        'vto_meter': 1}

        super().__init__(radius=radius, **proj4_params)


class McBrydeThomasFlatPolarQuartic(SkyProjection):
    def __init__(self, lon_0=0.0, lat_0=0.0, radius=RADIUS):
        proj4_params = {'proj': 'mbtfpq',
                        'lon_0': lon_0,
                        'lat_0': lat_0}

        super().__init__(radius=radius, **proj4_params)


class Mollweide(SkyProjection):
    def __init__(self, lon_0=0.0, radius=RADIUS):
        proj4_params = {'proj': 'moll',
                        'lon_0': lon_0}

        super().__init__(radius=radius, **proj4_params)


class Hammer(SkyProjection):
    def __init__(self, lon_0=0.0, lat_0=0.0, radius=RADIUS):
        proj4_params = {'proj': 'hammer',
                        'lon_0': lon_0,
                        'lat_0': lat_0}

        super().__init__(radius=radius, **proj4_params)


class EqualEarth(SkyProjection):
    def __init__(self, lon_0=0.0, radius=RADIUS):
        proj4_params = {'proj': 'eqearth',
                        'lon_0': lon_0}

        super().__init__(radius=radius, **proj4_params)


class LambertAzimuthalEqualArea(SkyProjection):
    def __init__(self, lon_0=0.0, lat_0=0.0, radius=RADIUS):
        proj4_params = {'proj': 'laea',
                        'lon_0': lon_0,
                        'lat_0': lat_0}

        super().__init__(radius=radius, **proj4_params)


_projections = {
    'hammer': ('Hammer', Hammer),
    'mbtfpq': ('McBryde-Thomas Flat Polar Quartic', McBrydeThomasFlatPolarQuartic),
    'cyl': ('Plate Carree', PlateCarree),
    'eqearth': ('Equal Earth', EqualEarth),
    'laea': ('Lambert Azimuthal Equal Area', LambertAzimuthalEqualArea),
    'moll': ('Mollweide', Mollweide)
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

    return projclass(**kwargs)


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
