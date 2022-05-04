import os
import pytest
import numpy as np

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
                                     skyproj.EqualEarthSkyproj])
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
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("skyproj", [skyproj.Skyproj,
                                     skyproj.McBrydeSkyproj,
                                     skyproj.MollweideSkyproj,
                                     skyproj.HammerSkyproj,
                                     skyproj.EqualEarthSkyproj,
                                     skyproj.LaeaSkyproj])
def test_skyproj_zoom(tmp_path, skyproj):
    plt.rcParams.update(plt.rcParamsDefault)

    # Simple zoom
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj(ax=ax, extent=[0, 50, 0, 50])
    fname = f'{sp.projection_name}_zoom.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
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
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
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
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
