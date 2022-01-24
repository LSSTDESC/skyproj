import os

import numpy as np
import healsparse as hsp

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def _get_hspmap():
    # Make a square noise field
    np.random.seed(1234)

    hspmap = hsp.HealSparseMap.make_empty(32, 4096, np.float32)
    poly = hsp.geom.Polygon(ra=[0.0, 10.0, 10.0, 0.0], dec=[0.0, 0.0, 10.0, 10.0], value=1.0)
    pixels = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels] = np.random.normal(size=pixels.size).astype(np.float32)
    # Add in a central square of fixed value.
    poly2 = hsp.geom.Polygon(ra=[5, 5.2, 5.2, 5.0], dec=[5, 5.0, 5.2, 5.2], value=3.0)
    pixels2 = poly2.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels2] = 3.0

    return hspmap


def test_healsparse(tmp_path):
    """Test plotting a healsparse map."""
    hspmap = _get_hspmap()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = m.draw_hspmap(hspmap)
    m.draw_inset_colorbar()
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    # Try a different colormap on the zoom.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    m.draw_hspmap(hspmap, lon_range=[4.9, 5.3], lat_range=[4.9, 5.3], cmap=plt.colormaps['rainbow'])
    m.draw_colorbar()
    fname = 'healsparse_two.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax, lon_0=-175.0)
    im, lon_raster, lat_raster, values_raster = m.draw_hspmap(hspmap, zoom=False)
    m.draw_inset_colorbar()
    fname = 'healsparse_three.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healpix(tmp_path):
    """Test plotting a healpix map."""
    hspmap = _get_hspmap()
    hpxmap = hspmap.generate_healpix_map()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = m.draw_hpxmap(hpxmap, nest=True)
    m.draw_inset_colorbar()
    # These should match the healsparse maps, so we can use the same comparison.
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    m.draw_hpxmap(hpxmap, nest=True, lon_range=[4.9, 5.3], lat_range=[4.9, 5.3],
                  cmap=plt.colormaps['rainbow'])
    m.draw_colorbar()
    fname = 'healsparse_two.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_hpxpix(tmp_path):
    """Test plotting healpix pixels."""
    hspmap = _get_hspmap()
    pixels = hspmap.valid_pixels
    values = hspmap[pixels]

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = m.draw_hpxpix(hspmap.nside_sparse,
                                                              pixels,
                                                              values,
                                                              nest=True)
    m.draw_inset_colorbar()
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    m.draw_hpxpix(hspmap.nside_sparse, pixels, values, nest=True,
                  lon_range=[4.9, 5.3], lat_range=[4.9, 5.3],
                  cmap=plt.colormaps['rainbow'])
    m.draw_colorbar()
    fname = 'healsparse_two.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)
