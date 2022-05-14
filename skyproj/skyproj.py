import numpy as np

from ._skyproj import _Skyproj

__all__ = ['Skyproj', 'McBrydeSkyproj', 'LaeaSkyproj', 'MollweideSkyproj',
           'HammerSkyproj', 'EqualEarthSkyproj', 'GnomonicSkyproj',
           'ObliqueMollweideSkyproj']


class _Stadium:
    """Extension class to create a stadium-shaped projection boundary.
    """
    def _compute_proj_boundary_xy(self):
        proj_boundary_xy = {}

        edge_offset = 180.0 - 1e-6
        nstep = 1000

        x, y = self.proj(np.linspace(self.lon_0 - edge_offset,
                                     self.lon_0 - edge_offset,
                                     nstep),
                         np.linspace(-90.0 + 1e-6,
                                     90.0 - 1e-6,
                                     nstep))
        proj_boundary_xy['left'] = np.column_stack((x, y))

        x, y = self.proj(np.linspace(self.lon_0 + edge_offset,
                                     self.lon_0 + edge_offset,
                                     nstep),
                         np.linspace(-90.0 + 1e-6,
                                     90.0 - 1e-6,
                                     nstep))
        proj_boundary_xy['right'] = np.column_stack((x, y))

        x, y = self.proj(np.linspace(self.lon_0 - edge_offset,
                                     self.lon_0 + edge_offset,
                                     nstep),
                         np.linspace(90.0 - 1e-6,
                                     90.0 - 1e-6,
                                     nstep))
        proj_boundary_xy['top'] = np.column_stack((x, y))

        x, y = self.proj(np.linspace(self.lon_0 - edge_offset,
                                     self.lon_0 + edge_offset,
                                     nstep),
                         np.linspace(-90.0 + 1e-6,
                                     -90.0 + 1e-6,
                                     nstep))
        proj_boundary_xy['bottom'] = np.column_stack((x, y))

        return proj_boundary_xy


class _Ellipse21:
    """Extension class to create an ellipse-shaped projection boundary.
    """
    def _compute_proj_boundary_xy(self):
        proj_boundary_xy = {}

        nstep = 1000

        t = np.linspace(-np.pi/2., np.pi/2., nstep)
        x = 2*self.crs.radius*np.sqrt(2)*np.cos(t)
        y = self.crs.radius*np.sqrt(2)*np.sin(t)
        proj_boundary_xy['right'] = np.column_stack((x, y))

        t = np.linspace(np.pi/2., 3*np.pi/2., nstep)
        x = 2*self.crs.radius*np.sqrt(2)*np.cos(t)
        y = self.crs.radius*np.sqrt(2)*np.sin(t)
        proj_boundary_xy['left'] = np.column_stack((x, y))

        proj_boundary_xy['top'] = np.zeros((0, 2))
        proj_boundary_xy['bottom'] = np.zeros((0, 2))

        return proj_boundary_xy


class _Circle:
    """Extension class to create a circular projection boundary.
    """
    def _compute_proj_boundary_xy(self):
        proj_boundary_xy = {}

        nstep = 1000

        t = np.linspace(-np.pi/2., np.pi/2., nstep)
        x = 2*self.crs.radius*np.cos(t)
        y = 2*self.crs.radius*np.sin(t)
        proj_boundary_xy['right'] = np.column_stack((x, y))

        t = np.linspace(np.pi/2., 3*np.pi/2., nstep)
        x = 2*self.crs.radius*np.cos(t)
        y = 2*self.crs.radius*np.sin(t)
        proj_boundary_xy['left'] = np.column_stack((x, y))

        proj_boundary_xy['top'] = np.zeros((0, 2))
        proj_boundary_xy['bottom'] = np.zeros((0, 2))

        return proj_boundary_xy


# The default skyproj is a cylindrical Plate Carree projection.

class Skyproj(_Skyproj, _Stadium):
    # docstring inherited
    # Plate Carree
    def __init__(self, **kwargs):
        super().__init__(projection_name='cyl', **kwargs)


# The following skyprojs include the equal-area projections that are tested
# and known to work.

class McBrydeSkyproj(_Skyproj, _Stadium):
    # docstring inherited
    # McBryde-Thomas Flat Polar Quartic
    def __init__(self, **kwargs):
        super().__init__(projection_name='mbtfpq', **kwargs)


class LaeaSkyproj(_Skyproj, _Circle):
    # docstring inherited
    # Lambert Azimuthal Equal Area
    def __init__(self, **kwargs):
        super().__init__(projection_name='laea', **kwargs)

    @property
    def _full_circle(self):
        return True

    @property
    def _init_extent_xy(self):
        return True

    @property
    def _radial_labels(self):
        return True

    @property
    def _default_xy_labels(self):
        return ("", "")

    @property
    def _full_sky_extent_initial(self):
        lon0 = self.lon_0 - 180.0
        lon1 = self.lon_0 + 180.0
        _lat_0 = self.lat_0
        if _lat_0 == -90.0:
            lat0 = -90.0
            lat1 = 90.0 - 1e-5
        elif _lat_0 == 90.0:
            lat0 = -90.0 + 1e-5
            lat1 = 90.0
        else:
            lat0 = -90.0 + 1e-5
            lat1 = 90.0 - 1e-5

        return [lon0, lon1, lat0, lat1]


class MollweideSkyproj(_Skyproj, _Ellipse21):
    # docstring inherited
    # Mollweide
    def __init__(self, **kwargs):
        super().__init__(projection_name='moll', **kwargs)

    @property
    def _pole_clip(self):
        return 1.0

    @property
    def _equatorial_labels(self):
        return True


class HammerSkyproj(_Skyproj, _Ellipse21):
    # docstring inherited
    # Hammer-Aitoff
    def __init__(self, **kwargs):
        super().__init__(projection_name='hammer', **kwargs)

    @property
    def _equatorial_labels(self):
        return True


class EqualEarthSkyproj(_Skyproj, _Stadium):
    # docstring inherited
    # Equal Earth
    def __init__(self, **kwargs):
        super().__init__(projection_name='eqearth', **kwargs)


class ObliqueMollweideSkyproj(_Skyproj, _Ellipse21):
    """Oblique Mollweide Projection.

    Parameters
    ----------
    lon_0 : `float`, optional
        Central longitude of the underlying Mollweide projection.
    lat_p : `float`, optional
        Latitude of the North Pole of the unrotated coordinate system.
    lon_p : `float`, optional
        Longitude of the North Pole of the unrotated coordinate system.
    **kwargs : `dict`, optional
        Additional kwargs for `skyproj._Skyproj`.
    """
    # Oblique Mollweide
    def __init__(self, **kwargs):
        super().__init__(projection_name='obmoll', **kwargs)

    @property
    def _pole_clip(self):
        return 1.0

    @property
    def _equatorial_labels(self):
        return True

    @property
    def _default_xy_labels(self):
        return ("", "")

    @property
    def _init_extent_xy(self):
        return True


# The Gnomonic (tangent plane) projection is not equal-area and
# is not available for full-sky plots.  It is only for small
# zoomed regions
class GnomonicSkyproj(_Skyproj, _Circle):
    # docstring inherited
    # Gnomonic
    def __init__(self, **kwargs):
        super().__init__(projection_name='gnom', **kwargs)

    @property
    def _full_sky_extent_initial(self):
        lon_0 = self.lon_0
        lat_0 = self.lat_0
        cos_lat = np.cos(np.deg2rad(lat_0))
        return [lon_0 - 0.5/cos_lat, lon_0 + 0.5/cos_lat,
                lat_0 - 0.5, lat_0 + 0.5]
