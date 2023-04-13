import os
import pytest
import numpy as np
import healsparse as hsp

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("skyproj", [skyproj.Skyproj,
                                     skyproj.McBrydeSkyproj,
                                     skyproj.MollweideSkyproj,
                                     skyproj.HammerSkyproj,
                                     skyproj.EqualEarthSkyproj,
                                     skyproj.AlbersSkyproj])
@pytest.mark.parametrize("lon_0", [0.0, -100.0, 100.0, 180.0])
def test_skyproj_basic(tmp_path, skyproj, lon_0):
    """Test full sky maps."""
    plt.rcParams.update(plt.rcParamsDefault)

    # Full image
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj(ax=ax, lon_0=lon_0)
    fname = f'{sp.projection_name}_full_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("skyproj", [skyproj.Skyproj,
                                     skyproj.McBrydeSkyproj,
                                     skyproj.MollweideSkyproj,
                                     skyproj.HammerSkyproj,
                                     skyproj.EqualEarthSkyproj,
                                     skyproj.LaeaSkyproj,
                                     skyproj.AlbersSkyproj])
def test_skyproj_zoom(tmp_path, skyproj):
    plt.rcParams.update(plt.rcParamsDefault)

    # Simple zoom
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj(ax=ax, extent=[0, 50, 0, 50])
    fname = f'{sp.projection_name}_zoom.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lonlat", [(0.0, 0.0),
                                    (120.0, -75.0),
                                    (-120.0, 75.0)])
def test_skyproj_gnom(tmp_path, lonlat):
    """Test gnomonic zooms."""
    plt.rcParams.update(plt.rcParamsDefault)

    lon_0, lat_0 = lonlat

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.GnomonicSkyproj(ax=ax, lon_0=lon_0, lat_0=lat_0)
    # draw a square square, make sure it looks square
    delta_lat = 0.1
    delta_lon = delta_lat/np.cos(np.deg2rad(lat_0))
    sp.draw_polygon(
        [lon_0 - delta_lon, lon_0 + delta_lon, lon_0 + delta_lon, lon_0 - delta_lon],
        [lat_0 - delta_lat, lat_0 - delta_lat, lat_0 + delta_lat, lat_0 + delta_lat]
    )
    fname = f'gnom_{lon_0}_{lat_0}.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lonlatplonp", [(0.0, 45.0, -90.0),
                                         (100.0, 80.0, 0.0)])
def test_skyproj_obmoll(tmp_path, lonlatplonp):
    """Test Oblique Mollweide."""
    plt.rcParams.update(plt.rcParamsDefault)

    lon_0, lat_p, lon_p = lonlatplonp

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.ObliqueMollweideSkyproj(ax=ax, lon_0=lon_0, lat_p=lat_p, lon_p=lon_p)
    fname = f'{sp.projection_name}_{lon_0}_{lat_p}_{lon_p}.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lat1lat2", [(15.0, 45.0),
                                      (-20.0, 15.0),
                                      (-15.0, -45.0)])
def test_skyproj_albers(tmp_path, lat1lat2):
    """Test Albers Equal Area."""
    plt.rcParams.update(plt.rcParamsDefault)

    lat_1, lat_2 = lat1lat2

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.AlbersSkyproj(ax=ax, lat_1=lat_1, lat_2=lat_2)
    fname = f'{sp.projection_name}_{lat_1}_{lat_2}.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("skyproj", [skyproj.Skyproj,
                                     skyproj.McBrydeSkyproj,
                                     skyproj.MollweideSkyproj,
                                     skyproj.HammerSkyproj,
                                     skyproj.EqualEarthSkyproj])
@pytest.mark.parametrize("lon_0", [-180.0, -140.0, -100.0, -60.0, -20.0, 0.0,
                                   20.0, 60.0, 100.0, 140.0, 180.0])
def test_skyproj_fullsky_extent(skyproj, lon_0):
    """Test getting the full sky extent."""

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj(ax=ax, lon_0=lon_0)

    lon1, lon0, lat0, lat1 = sp.get_extent()

    # We allow some slop because of the way that clipping works
    assert lat0 < (-90.0 + sp._pole_clip + 0.3)
    assert lat0 >= -90.0
    assert lat1 > (90.0 - sp._pole_clip - 0.3)
    assert lat1 <= 90.0

    # Cover (almost) full 360 deg?
    assert (lon1 - lon0) > (360.0 - 1.5)
    # And the rotated start is close to -180?
    rot = (lon0 - lon_0 + 180.0) % 360.0 - 180.0
    assert rot < (-180.0 + 1.0)


@pytest.mark.parametrize("skyproj", [skyproj.McBrydeSkyproj,
                                     skyproj.MollweideSkyproj])
def test_skyproj_nogap_180(tmp_path, skyproj):
    """Test that there is no mysterious gap when lon_0=180.0"""
    plt.rcParams.update(plt.rcParamsDefault)

    testmap = hsp.HealSparseMap.make_empty(32, 256, dtype=np.int32)
    poly = hsp.geom.Polygon(ra=[-20, 20, 20, -20], dec=[-20, -20, 20, 20], value=1)
    pixels = poly.get_pixels(nside=testmap.nside_sparse)
    testmap[pixels] = 1
    poly = hsp.geom.Polygon(ra=[160, 200, 200, 160], dec=[-20, -20, 20, 20], value=1)
    pixels = poly.get_pixels(nside=testmap.nside_sparse)
    testmap[pixels] = 1

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj(ax=ax, lon_0=180.0)
    sp.draw_hspmap(testmap, zoom=False)
    fname = f'{sp.projection_name}_gaptest.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


def test_skyproj_override_sizes(tmp_path):
    """Test overriding the label/width sizes."""
    plt.rcParams.update(plt.rcParamsDefault)

    rcparams = {'xtick.labelsize': 20,
                'ytick.labelsize': 4,
                'axes.linewidth': 5}

    # Full image
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    _ = skyproj.McBrydeSkyproj(ax=ax, rcparams=rcparams)
    fname = 'skyproj_full_override_sizes.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)

    # And confirm that the changes do not carry over to another plot.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    _ = skyproj.McBrydeSkyproj(ax=ax)
    fname = 'mbtfpq_full_0.0.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)
