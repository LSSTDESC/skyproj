import numpy as np

from ._skyproj import _Skyproj

from ._docstrings import skyproj_init_parameters, skyproj_kwargs_par


__all__ = ['Skyproj', 'McBrydeSkyproj', 'LaeaSkyproj', 'MollweideSkyproj',
           'HammerSkyproj', 'EqualEarthSkyproj', 'GnomonicSkyproj',
           'ObliqueMollweideSkyproj', 'AlbersSkyproj']


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
    __doc__ = (skyproj_init_parameters("A Plate Carree cylindrical projection Skyproj map.")
               + skyproj_kwargs_par)

    # Plate Carree
    def __init__(
        self,
        ax=None,
        *,
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
        super().__init__(
            ax=ax,
            projection_name='cyl',
            lon_0=lon_0,
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            **kwargs,
        )


# The following skyprojs include the equal-area projections that are tested
# and known to work.

class McBrydeSkyproj(_Skyproj, _Stadium):
    __doc__ = (skyproj_init_parameters("A McBryde-Thomas Flat Polar Quartic projection Skyproj map.")
               + skyproj_kwargs_par)

    # McBryde-Thomas Flat Polar Quartic
    def __init__(
        self,
        ax=None,
        *,
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
        super().__init__(
            ax=ax,
            projection_name='mbtfpq',
            lon_0=lon_0,
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            **kwargs,
        )


class LaeaSkyproj(_Skyproj, _Circle):
    __doc__ = (skyproj_init_parameters("A Lambert Azimuthal Equal Area projection Skyproj map.")
               + skyproj_kwargs_par)
    __doc__ = skyproj_init_parameters(
        "A Lambert Azimuthal Equal Area projection Skyproj map.",
        include_lon_0=False,
    )
    __doc__ += """
lon_0 : `float`, optional
    Central longitude of the LAEA projection.
lat_0 : `float`, optional
    Central latitude of the LAEA projection."""

    # Lambert Azimuthal Equal Area
    def __init__(
        self,
        ax=None,
        *,
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
        lon_0=0.0,
        lat_0=0.0,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
            projection_name='laea',
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            lon_0=lon_0,
            lat_0=lat_0,
            **kwargs,
        )

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
    __doc__ = (skyproj_init_parameters("A Mollweide projection Skyproj map.")
               + skyproj_kwargs_par)

    # Mollweide
    def __init__(
        self,
        ax=None,
        *,
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
        super().__init__(
            ax=ax,
            projection_name='moll',
            lon_0=lon_0,
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            **kwargs,
        )

    @property
    def _pole_clip(self):
        return 1.0

    @property
    def _equatorial_labels(self):
        return True


class HammerSkyproj(_Skyproj, _Ellipse21):
    __doc__ = (skyproj_init_parameters("A Hammer-Aitoff projection Skyproj map.")
               + skyproj_kwargs_par)

    # Hammer-Aitoff
    def __init__(
        self,
        ax=None,
        *,
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
        super().__init__(
            ax=ax,
            projection_name='hammer',
            lon_0=lon_0,
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            **kwargs,
        )

    @property
    def _equatorial_labels(self):
        return True


class EqualEarthSkyproj(_Skyproj, _Stadium):
    __doc__ = (skyproj_init_parameters("An Equal Earth projection Skyproj map.")
               + skyproj_kwargs_par)

    # Equal Earth
    def __init__(
        self,
        ax=None,
        *,
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
        super().__init__(
            ax=ax,
            projection_name='eqearth',
            lon_0=lon_0,
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            **kwargs,
        )


class ObliqueMollweideSkyproj(_Skyproj, _Ellipse21):
    __doc__ = skyproj_init_parameters("An Oblique Mollweide projection Skyproj map.", include_lon_0=False)
    __doc__ += """
lon_0 : `float`, optional
    Central longitude of the underlying Mollweide projection.
lat_p : `float`, optional
    Latitude of the North Pole of the unrotated coordinate system.
lon_p : `float`, optional
    Longitude of the North Pole of the unrotated coordinate system."""
    __doc__ += skyproj_kwargs_par

    # Oblique Mollweide
    def __init__(
        self,
        ax=None,
        *,
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
        lon_0=0.0,
        lat_p=90.0,
        lon_p=0.0,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
            projection_name='obmoll',
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            lon_0=lon_0,
            lat_p=lat_p,
            lon_p=lon_p,
            **kwargs,
        )

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
    __doc__ = skyproj_init_parameters("""A Gnomonic (tangent plane) projection Skyproj map.

This projection is not equal area and is not available for full sky plots.
It should only be used for small zoomed regions.""", include_lon_0=False)
    __doc__ += """
lon_0 : `float`, optional
    Central longitude of the Gnomonic projection.
lat_0 : `float`, optional
    Central latitude of the Gnomonic projection."""
    __doc__ += skyproj_kwargs_par

    # Gnomonic
    def __init__(
        self,
        ax=None,
        *,
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
        lon_0=0,
        lat_0=0,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
            projection_name='gnom',
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            lon_0=lon_0,
            lat_0=lat_0,
            **kwargs,
        )

    @property
    def _full_sky_extent_initial(self):
        lon_0 = self.lon_0
        lat_0 = self.lat_0
        cos_lat = np.cos(np.deg2rad(lat_0))
        return [lon_0 - 0.5/cos_lat, lon_0 + 0.5/cos_lat,
                lat_0 - 0.5, lat_0 + 0.5]


class AlbersSkyproj(_Skyproj, _Stadium):
    __doc__ = skyproj_init_parameters("An Albers Equal Area Skyproj map.", include_lon_0=False)
    __doc__ += """
lon_0 : `float`, optional
    Central longitude of the projection.
lat_1 : `float`, optional
    First standard parallel of the projection.
lat_2 : `float`, optional
    Second standard parallel of the projection."""
    __doc__ += skyproj_kwargs_par

    # Albers Equal Area
    def __init__(
        self,
        ax=None,
        *,
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
        lon_0=0,
        lat_1=15.0,
        lat_2=45.0,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
            projection_name='aea',
            gridlines=gridlines,
            celestial=celestial,
            extent=extent,
            longitude_ticks=longitude_ticks,
            autorescale=autorescale,
            galactic=galactic,
            rcparams=rcparams,
            n_grid_lon=n_grid_lon,
            n_grid_lat=n_grid_lat,
            min_lon_ticklabel_delta=min_lon_ticklabel_delta,
            lon_0=lon_0,
            lat_1=lat_1,
            lat_2=lat_2,
            **kwargs,
        )

    @property
    def _default_xy_labels(self):
        return ("", "")
