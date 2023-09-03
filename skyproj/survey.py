from .skyproj import McBrydeSkyproj, LaeaSkyproj, AlbersSkyproj
from .utils import get_datafile  # fix this

__all__ = ['DESSkyproj', 'BlissSkyproj', 'MaglitesSkyproj',
           'DecalsSkyproj', 'DESMcBrydeSkyproj', 'DESAlbersSkyproj']


class _Survey:
    """Extension class to add routines for drawing survey outlines.
    """
    def draw_des(self, **kwargs):
        """Draw the DES footprint."""
        return self.draw_des19(**kwargs)

    def draw_des19(self, edgecolor='red', lw=2, **kwargs):
        """Draw the DES 2019 footprint."""
        filename = get_datafile('des-round19-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)

    def draw_des17(self, edgecolor='red', lw=2, **kwargs):
        """Draw the DES 2017 footprint."""
        filename = get_datafile('des-round17-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)

    def draw_decals(self, edgecolor='red', lw=2, **kwargs):
        """Draw the DECaLS footprint."""
        filename = get_datafile('decals-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)

    def draw_maglites(self, edgecolor='blue', lw=2, **kwargs):
        """Draw the MagLiteS footprint."""
        filename = get_datafile('maglites-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)

    def draw_bliss(self, edgecolor='magenta', lw=2, **kwargs):
        """Draw the BLISS footprint."""
        filename = get_datafile('bliss-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)

    # Override zoom default for survey maps to keep the default fixed.
    def draw_hpxmap(self, hpxmap, nest=False, zoom=False, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        # docstring inherited
        return super().draw_hpxmap(hpxmap,
                                   nest=nest,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=False, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        # docstring inherited
        return super().draw_hpxpix(nside,
                                   pixels,
                                   values,
                                   nest=nest,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    def draw_hspmap(self, hspmap, zoom=False, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        # docstring inherited
        return super().draw_hspmap(hspmap,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    def draw_hpxbin(self, lon, lat, C=None, nside=256, nest=False, zoom=False, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        # docstring inherited
        return super().draw_hpxmap(lon,
                                   lat,
                                   C=C,
                                   nside=nside,
                                   nest=nest,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)


class DESMcBrydeSkyproj(McBrydeSkyproj, _Survey):
    # docstring inherited
    def __init__(self, ax=None, lon_0=30, gridlines=True,
                 celestial=True, extent=[90, -50, -74, 10],
                 n_grid_lon=10, n_grid_lat=6, **kwargs):
        super().__init__(ax=ax, lon_0=lon_0, gridlines=gridlines,
                         celestial=celestial, extent=extent,
                         n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat, **kwargs)


DESSkyproj = DESMcBrydeSkyproj


class DESAlbersSkyproj(AlbersSkyproj, _Survey):
    # docstring inherited
    def __init__(self, ax=None, lon_0=30.0, lat_1=-15.0, lat_2=-50.0, gridlines=True,
                 celestial=True, extent=[80, -40, -80, 10],
                 n_grid_lon=10, n_grid_lat=6, min_tick_delta=0.05, **kwargs):
        super().__init__(ax=ax, lon_0=lon_0, lat_1=lat_1, lat_2=lat_2, gridlines=gridlines,
                         celestial=celestial, extent=extent,
                         n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat, min_tick_delta=min_tick_delta,
                         **kwargs)

    @property
    def _default_xy_labels(self):
        return ("Right Ascension", "Declination")


class BlissSkyproj(McBrydeSkyproj, _Survey):
    # docstring inherited
    def __init__(self, ax=None, lon_0=100, gridlines=True,
                 celestial=True, extent=[-60, 250, -55, 0],
                 n_grid_lon=10, n_grid_lat=6, **kwargs):
        super().__init__(ax=ax, lon_0=lon_0, gridlines=gridlines,
                         celestial=celestial, extent=extent,
                         n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat, **kwargs)


class MaglitesSkyproj(LaeaSkyproj, _Survey):
    # docstring inherited
    def __init__(self, ax=None, lon_0=0, lat_0=-90, gridlines=True,
                 celestial=True, extent=[-150, 70, -85, -50],
                 n_grid_lon=10, n_grid_lat=6, **kwargs):
        super().__init__(ax=ax, lon_0=lon_0, lat_0=lat_0,
                         gridlines=gridlines, celestial=celestial, extent=extent,
                         n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat, **kwargs)


class DecalsSkyproj(McBrydeSkyproj, _Survey):
    # docstring inherited
    def __init__(self, ax=None, lon_0=105.0, gridlines=True,
                 celestial=True, extent=[180, -180, -30, 40],
                 n_grid_lon=10, n_grid_lat=6, **kwargs):
        super().__init__(ax=ax, lon_0=lon_0, gridlines=gridlines,
                         celestial=celestial, extent=extent,
                         n_grid_lon=n_grid_lon, n_grid_lat=n_grid_lat, **kwargs)
