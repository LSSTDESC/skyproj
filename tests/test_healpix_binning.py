import os

import numpy as np
import hpgeom as hpg

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def test_healpix_binning(tmp_path):
    """Test healpix binning functionality."""
    plt.rcParams.update(plt.rcParamsDefault)

    np.random.seed(1234)

    ra = np.random.uniform(low=30.0, high=40.0, size=10000)
    dec = np.random.uniform(low=-55.0, high=-45.0, size=10000)
    C = np.random.uniform(low=0.0, high=10.0, size=10000)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    hpxmap, im, lon_raster, lat_raster, values_raster = sp.draw_hpxbin(ra, dec)

    # Spot-check a pixel
    pix = hpg.angle_to_pixel(hpg.npixel_to_nside(hpxmap.size), ra, dec, nest=False)
    test, = np.where(pix == 671795)
    assert hpxmap[671795] == test.size

    fname = 'hpxbin.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    # Redo with averaging over values
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    hpxmap, im, lon_raster, lat_raster, values_raster = sp.draw_hpxbin(ra, dec, C=C)

    # Spot-check the pixel
    np.testing.assert_approx_equal(hpxmap[671795], np.mean(C[test]))
    plt.close(fig)

    # And do with a survey
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.DESSkyproj(ax=ax)
    hpxmap, im, lon_raster, lat_raster, values_raster = sp.draw_hpxbin(ra, dec)
    fname = 'hpxbin_two.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)
