import numpy as np
import healpy as hp

from .utils import wrap_values

__all__ = ['healpix_pixels_range', 'hspmap_to_xy', 'hpxmap_to_xy', 'healpix_to_xy',
           'healpix_bin']


def healpix_pixels_range(nside, pixels, wrap, nest=False):
    """Find lon/lat range of healpix pixels, using wrap angle.

    Parameters
    ----------
    nside : `int`
        Healpix nside
    pixels : `np.ndarray`
        Array of pixel numbers
    wrap : `float`
        Wrap angle.
    nest : `bool`, optional
        Nest ordering?

    Returns
    -------
    lon_range : `tuple` [`float`, `float`]
        Longitude range of pixels (min, max)
    lat_range : `tuple` [`float`, `float`]
        Latitude range of pixels (min, max)
    """
    lon, lat = hp.pix2ang(nside, pixels, nest=nest, lonlat=True)

    eps = hp.max_pixrad(nside, degrees=True)

    lat_range = (np.clip(np.min(lat) - eps, -90.0 + 1e-5, None),
                 np.clip(np.max(lat) + eps, None, 90.0 - 1e-5))

    # FIXME: the wrap logic here is wrong.

    lon_wrap = (lon + wrap) % 360. - wrap
    lon_range = (np.min(lon_wrap) - eps, np.max(lon_wrap) + eps)

    # Check if we have overrun and need to do the full range
    full_range = False
    if (lon_range[0] < (wrap - 360.0)) and (lon_range[1] > (wrap - 360.0)):
        full_range = True
    elif (lon_range[0] < (wrap + 360.0)) and (lon_range[1] > (wrap + 360.0)):
        full_range = True
    elif (lon_range[1] - lon_range[0]) >= 359.0:
        full_range = True

    if full_range:
        lon_0 = wrap_values((wrap + 180.0) % 360.0)
        lon_range = (lon_0 - 180., lon_0 + 180.0 - 1e-5)

    return lon_range, lat_range


def hspmap_to_xy(hspmap, lon_range, lat_range, xsize=1000, aspect=1.0, valid_mask=False):
    """Convert healsparse map to rasterized x/y positions and values.

    Parameters
    ----------
    hspmap : `healsparse.HealSparseMap`
        Healsparse map
    lon_range : `tuple` [`float`, `float`]
        Longitude range for rasterization, (min, max).
    lat_range : `tuple` [`float`, `float`]
        Latitude range for rasterization, (min, max).
    xsize : `int`, optional
        Number of rasterized pixels in the x direction.
    aspect : `float`, optional
        Aspect ratio for ysize.
    valid_mask : `bool`, optional
        Plot the valid pixels of the map.

    Returns
    -------
    lon_raster : `np.ndarray`
        Rasterized longitude values (length xsize).
    lat_raster : `np.ndarray`
        Rasterized latitude values (length xsize*aspect).
    values_raster : `np.ma.maskedarray`
        Rasterized values (2-d).  Invalid values are masked.
    """
    lon_raster, lat_raster = np.meshgrid(np.linspace(lon_range[0], lon_range[1], xsize),
                                         np.linspace(lat_range[0], lat_range[1], int(aspect*xsize)))

    # For pcolormesh we need the central locations
    clon = (lon_raster[1:, 1:] + lon_raster[:-1, :-1])/2.
    clat = (lat_raster[1:, 1:] + lat_raster[:-1, :-1])/2.

    values = hspmap.get_values_pos(clon, clat, valid_mask=valid_mask)
    if hspmap.is_wide_mask_map:
        # Special case wide masks.  We just display 1 where any bit
        # is defined, and 0 otherwise.
        values = np.any(values, axis=2).astype(np.int32)
        mask = (values == 0)
    elif values.dtype == bool:
        mask = values == hspmap._sentinel
    else:
        mask = np.isclose(values, hspmap._sentinel)

    return lon_raster, lat_raster, np.ma.array(values, mask=mask)


def hpxmap_to_xy(hpxmap, lon_range, lat_range, nest=False, xsize=1000, aspect=1.0):
    """Convert healpix map to rasterized x/y positions and values.

    Parameters
    ----------
    hpxmap : `np.ndarray`
        Healpix map
        lon_range : `tuple` [`float`, `float`]
        Longitude range for rasterization, (min, max).
    lat_range : `tuple` [`float`, `float`]
        Latitude range for rasterization, (min, max).
    xsize : `int`, optional
        Number of rasterized pixels in the x direction.
    aspect : `float`, optional
        Aspect ratio for ysize.

    Returns
    -------
    lon_raster : `np.ndarray`
        Rasterized longitude values (length xsize).
    lat_raster : `np.ndarray`
        Rasterized latitude values (length xsize*aspect).
    values_raster : `np.ma.maskedarray`
        Rasterized values (2-d).  Invalid values are masked.
    """
    lon_raster, lat_raster = np.meshgrid(np.linspace(lon_range[0], lon_range[1], xsize),
                                         np.linspace(lat_range[0], lat_range[1], int(aspect*xsize)))

    # For pcolormesh we need the central locations
    clon = (lon_raster[1:, 1:] + lon_raster[:-1, :-1])/2.
    clat = (lat_raster[1:, 1:] + lat_raster[:-1, :-1])/2.

    pix_raster = hp.ang2pix(hp.npix2nside(hpxmap.size), clon, clat, nest=nest, lonlat=True)
    values = hpxmap[pix_raster]

    mask = np.isclose(values, hp.UNSEEN)

    return lon_raster, lat_raster, np.ma.array(values, mask=mask)


def healpix_to_xy(nside, pixels, values, lon_range, lat_range,
                  nest=False, xsize=1000, aspect=1.0):
    """Convert healpix pixels to rasterized x/y positions and values.

    Parameters
    ----------
    nside : `int`
        Healpix nside.
    pixels : `np.ndarray`
        Array of pixel numbers
    values : `np.ndarray`
        Array of pixel values
    lon_range : `tuple`, optional
        Longitude range to do rasterization (min, max).
    lat_range : `tuple`, optional
        Latitude range to do rasterization (min, max).
    nest : `bool`, optional
        Nest ordering?
    xsize : `int`, optional
        Number of rasterized pixels in the x direction.
    aspect : `float`, optional
        Aspect ratio to compute ysize.

    Returns
    -------
    lon_raster : `np.ndarray`
        Rasterized longitude values (length xsize).
    lat_raster : `np.ndarray`
        Rasterized latitude values (length xsize*aspect).
    values_raster : `np.ndarray`
        Rasterized values (2-d).
    """
    test = np.unique(pixels)
    if test.size != pixels.size:
        raise ValueError("The pixels array must be unique.")

    lon_raster, lat_raster = np.meshgrid(np.linspace(lon_range[0], lon_range[1], xsize),
                                         np.linspace(lat_range[0], lat_range[1], int(aspect*xsize)))

    # For pcolormesh we need the central locations
    clon = (lon_raster[1:, 1:] + lon_raster[:-1, :-1])/2.
    clat = (lat_raster[1:, 1:] + lat_raster[:-1, :-1])/2.

    pix_raster = hp.ang2pix(nside, clon, clat, nest=nest, lonlat=True)

    st = np.argsort(pixels)
    sub1 = np.searchsorted(pixels, pix_raster, sorter=st)

    if pix_raster.max() > pixels.max():
        bad = np.where(sub1 == pixels.size)
        sub1[bad] = pixels.size - 1

    sub2 = np.where(pixels[st[sub1]] == pix_raster)
    sub1 = st[sub1[sub2]]

    mask = np.ones(pix_raster.shape, dtype=bool)
    mask[sub2] = False
    values_raster = np.zeros(pix_raster.shape, dtype=values.dtype)
    values_raster[sub2] = values[sub1]

    return lon_raster, lat_raster, np.ma.array(values_raster, mask=mask)


def healpix_bin(lon, lat, C=None, nside=256, nest=False):
    """Create a healpix histogram of counts in lon/lat space.

    Parameters
    ----------
    lon : `np.ndarray`
        Longitude array (degrees).
    lat : `np.ndarray`
        Latitude array (degrees).
    C : `np.ndarray`, optional
        Array of values to take average, paired with lon/lat.
    nside : `int`, optional
        Healpix nside resolution.
    nest : `bool`, optional
        Map in nest format?

    Returns
    -------
    hpxmap : `np.ndarray`
        Healsparse map of counts.
    """
    pix = hp.ang2pix(nside, lon, lat, lonlat=True, nest=nest)

    count = np.zeros(hp.nside2npix(nside), dtype=np.int32)
    np.add.at(count, pix, 1)
    good = count > 0

    if C is not None:
        hpxmap = np.zeros(hp.nside2npix(nside))
        np.add.at(hpxmap, pix, C)
        hpxmap[good] /= count[good]
    else:
        hpxmap = count.astype(np.float64)

    hpxmap[~good] = hp.UNSEEN

    return hpxmap
