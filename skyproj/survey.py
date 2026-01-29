from .skyproj import McBrydeSkyproj, LaeaSkyproj, AlbersSkyproj
from .utils import get_datafile  # fix this

from ._docstrings import skyproj_init_parameters, skyproj_kwargs_par
from ._docstrings import (
    add_func_docstr,
    draw_hpxmap_docstr,
    draw_hpxpix_docstr,
    draw_hspmap_docstr,
    draw_hpxbin_docstr,
)

__all__ = ['DESSkyproj', 'BlissSkyproj', 'MaglitesSkyproj',
           'DecalsSkyproj', 'DESMcBrydeSkyproj', 'DESAlbersSkyproj',
           'RomanHLWASSkyproj', 'RomanHLWASMcBrydeSkyproj']


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

    def draw_roman_hlwas(self, edgecolor='red', lw=2, **kwargs):
        filename = get_datafile('roman-hlwas-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)

    # Override zoom default for survey maps to keep the default fixed.
    @add_func_docstr(draw_hpxmap_docstr)
    def draw_hpxmap(self, hpxmap, nest=False, zoom=False, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
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

    @add_func_docstr(draw_hpxpix_docstr)
    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=False, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
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

    @add_func_docstr(draw_hspmap_docstr)
    def draw_hspmap(self, hspmap, zoom=False, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        return super().draw_hspmap(hspmap,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    @add_func_docstr(draw_hpxbin_docstr)
    def draw_hpxbin(self, lon, lat, C=None, nside=256, nest=False, zoom=False, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        return super().draw_hpxbin(lon,
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


class DESMcBrydeSkyproj(_Survey, McBrydeSkyproj):
    __doc__ = skyproj_init_parameters(
        "A projection for the DES Survey using a McBryde-Thomas Flat Polar Quartic projection.",
    )
    __doc__ += skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        lon_0=30.0,
        gridlines=True,
        celestial=True,
        extent=[90, -50, -74, 10],
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=10,
        n_grid_lat=6,
        min_lon_ticklabel_delta=0.1,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
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


DESSkyproj = DESMcBrydeSkyproj


class DESAlbersSkyproj(_Survey, AlbersSkyproj):
    __doc__ = skyproj_init_parameters(
        "A projection for the DES Survey using an Albers Equal Area projection.",
        include_lon_0=False,
    )
    __doc__ += skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        gridlines=True,
        celestial=True,
        extent=[80, -40, -80, 10],
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=10,
        n_grid_lat=6,
        min_lon_ticklabel_delta=0.1,
        lon_0=30.0,
        lat_1=-15.0,
        lat_2=-50.0,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
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
        return ("Right Ascension", "Declination")


class BlissSkyproj(_Survey, McBrydeSkyproj):
    __doc__ = skyproj_init_parameters(
        "A projection for the BLISS Survey using a McBryde-Thomas Flat Polar Quartic projection.",
        include_lon_0=False,
    )
    __doc__ += skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        lon_0=100.0,
        gridlines=True,
        celestial=True,
        extent=[-60, 250, -55, 0],
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=10,
        n_grid_lat=6,
        min_lon_ticklabel_delta=0.1,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
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


class MaglitesSkyproj(_Survey, LaeaSkyproj):
    __doc__ = skyproj_init_parameters(
        "A projection for the MagLiteS Survey using a Lambert Azimuthal Equal Area projection.",
        include_lon_0=False,
    )
    __doc__ += skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        gridlines=True,
        celestial=True,
        extent=[-150, 70, -85, -50],
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=10,
        n_grid_lat=6,
        min_lon_ticklabel_delta=0.1,
        lon_0=0.0,
        lat_0=-90.0,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
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


class DecalsSkyproj(_Survey, McBrydeSkyproj):
    __doc__ = skyproj_init_parameters(
        "A projection for the DECaLS Survey using a McBryde-Thomas Flat Polar Quartic projection.",
        include_lon_0=False,
    )
    __doc__ += skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        lon_0=105.0,
        gridlines=True,
        celestial=True,
        extent=[180, -180, -30, 40],
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=10,
        n_grid_lat=6,
        min_lon_ticklabel_delta=0.1,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
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


class RomanHLWASMcBrydeSkyproj(_Survey, McBrydeSkyproj):
    __doc__ = skyproj_init_parameters(
        """A projection for the Roman High-Latitude Wide-Area Survey using a McBryde-Thomas Flat Polar Quartic
           projection."""
    )
    __doc__ += skyproj_kwargs_par

    def __init__(
        self,
        ax=None,
        *,
        lon_0=-90.0,
        gridlines=True,
        celestial=True,
        extent=[0, 270, -60, 10],
        longitude_ticks='positive',
        autorescale=True,
        galactic=False,
        rcparams={},
        n_grid_lon=10,
        n_grid_lat=6,
        min_lon_ticklabel_delta=0.1,
        **kwargs,
    ):
        super().__init__(
            ax=ax,
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


RomanHLWASSkyproj = RomanHLWASMcBrydeSkyproj
