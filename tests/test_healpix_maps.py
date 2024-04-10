import os
import pytest

import numpy as np
import healsparse as hsp

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def _get_hspmap(ra_center=5.0):
    # Make a square noise field
    np.random.seed(1234)

    hspmap = hsp.HealSparseMap.make_empty(32, 4096, np.float32)
    poly = hsp.geom.Polygon(ra=[ra_center-5.0, ra_center+5.0, ra_center+5.0, ra_center-5.0],
                            dec=[0.0, 0.0, 10.0, 10.0], value=1.0)
    pixels = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels] = np.random.normal(size=pixels.size).astype(np.float32)
    # Add in a central square of fixed value.
    poly2 = hsp.geom.Polygon(ra=[ra_center, ra_center + 0.2, ra_center + 0.2, ra_center],
                             dec=[5, 5.0, 5.2, 5.2], value=3.0)
    pixels2 = poly2.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels2] = 3.0

    return hspmap


def _get_hspmap_bool():
    hspmap = hsp.HealSparseMap.make_empty(32, 4096, bool)
    # Create a region of True
    poly = hsp.geom.Polygon(ra=[0.0, 10.0, 10.0, 0.0], dec=[0.0, 0.0, 10.0, 10.0], value=1.0)
    pixels = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels] = np.full(shape=pixels.size, fill_value=True)
    # Create a central square of False within the True region.
    poly = hsp.geom.Polygon(ra=[5, 5.5, 5.5, 5.0], dec=[5, 5.0, 5.5, 5.5], value=3.0)
    pixels2 = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels2] = np.full(shape=pixels2.size, fill_value=False)

    return hspmap


def _get_hspmap_rec_array():
    # Make a square noise field
    np.random.seed(1234)

    dtype = [("A", np.int64), ("B", np.int64)]
    hspmap = hsp.HealSparseMap.make_empty(32, 4096, dtype=dtype, primary="A")
    poly = hsp.geom.Polygon(ra=[0.0, 10.0, 10.0, 0.0], dec=[0.0, 0.0, 10.0, 10.0], value=1.0)
    pixels = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap[pixels] = np.zeros(shape=pixels.size, dtype=dtype)
    hspmap["A"][pixels] = np.random.normal(size=pixels.size).astype(np.int64)
    hspmap["B"][pixels] = np.random.normal(size=pixels.size).astype(np.int64)
    # Add in a central square of fixed value.
    poly2 = hsp.geom.Polygon(ra=[5, 5.2, 5.2, 5.0], dec=[5, 5.0, 5.2, 5.2], value=3.0)
    pixels2 = poly2.get_pixels(nside=hspmap.nside_sparse)
    hspmap["A"][pixels2] = np.full(shape=pixels2.size, fill_value=3)
    hspmap["B"][pixels2] = np.full(shape=pixels2.size, fill_value=1)

    return hspmap


def test_healsparse(tmp_path):
    """Test plotting a healsparse map."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    # Try a different colormap on the zoom.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    sp.draw_hspmap(hspmap, lon_range=[4.9, 5.3], lat_range=[4.9, 5.3], cmap=plt.colormaps['rainbow'])
    sp.draw_colorbar()
    fname = 'healsparse_two.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=-175.0)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, zoom=False)
    sp.draw_inset_colorbar()
    fname = 'healsparse_three.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, valid_mask=True)
    sp.draw_inset_colorbar()
    fname = 'healsparse_valid_pixels.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.DESSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname = 'healsparse_four.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_log_scale(tmp_path):
    """Test plotting a healsparse map with a log scale."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()

    # Check that we get an error if we try a map with log scale and negative
    # range.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    with pytest.raises(ValueError, match="Invalid vmin or vmax"):
        im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, norm="log")
        sp.draw_colorbar()

    # Offset the map so that we can use a log scale.
    hspmap += 10.0

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, norm="log")
    sp.draw_colorbar()
    fname = 'healsparse_five.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_input_norm(tmp_path):
    """Test plotting a healsparse map with input normalization."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap() + 10.0

    # First make sure we get matched vmin/vmax.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    vmin, vmax = skyproj.utils.get_autoscale_vmin_vmax(values_raster.compressed(), None, None)
    plt.close(fig)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # This should ignore the input vmin/vmax if we set the norm.
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, norm=norm, vmin=-100.0, vmax=-50.0)
    sp.draw_colorbar()
    fname = 'healsparse_five.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_nanval(tmp_path):
    """Test plotting a healsparse map that has a nan value."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()

    hspmap[hspmap.valid_pixels[0]] = np.nan

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_rasterize(tmp_path):
    """Test plotting a healsparse map with and without rasterization."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # We use a small xsize here because otherwise the non-rasterized test pdf
    # (below) is 22Mb and the test takes a long time to run.
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, rasterized=True, xsize=200, zoom=False)
    sp.draw_inset_colorbar()
    fname = 'healsparse_rasterized_on.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_on = os.path.getsize(tmp_path / fname)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(
        hspmap,
        rasterized=False,
        xsize=200,
        zoom=False,
    )
    sp.draw_inset_colorbar()
    fname = 'healsparse_rasterized_off.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_off = os.path.getsize(tmp_path / fname)

    # The rasterized map will be smaller than the non-rasterized map.
    assert size_rasterized_on < size_rasterized_off

    # Test with a redraw event.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, rasterized=True, xsize=200, zoom=False)
    sp.set_extent([10.5, -0.5, -0.5, 10.5])
    sp.draw_inset_colorbar()
    fname = 'healsparse_rasterized_on2.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_on2 = os.path.getsize(tmp_path / fname)

    # The rasterized map will be smaller than the non-rasterized map.
    assert size_rasterized_on2 < size_rasterized_off


def test_healsparse_bool(tmp_path):
    """Test plotting a boolean healsparse map.
    """
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap_bool()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname = 'healsparse_bool.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, valid_mask=True)
    sp.draw_inset_colorbar()
    fname = 'healsparse_bool_valid_pixels.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_empty(tmp_path):
    """Test plotting an empty map.
    """
    # This simply checks that the plot can be made to
    # avoid the empty autoscale bug.
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = hsp.HealSparseMap.make_empty(32, 4096, np.float32)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    _ = sp.draw_hspmap(
        hspmap,
        lon_range=[10, 20],
        lat_range=[10, 20],
        zoom=False,
    )


def test_healpix(tmp_path):
    """Test plotting a healpix map."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()
    hpxmap = hspmap.generate_healpix_map()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap, nest=True)
    sp.draw_inset_colorbar()
    # These should match the healsparse maps, so we can use the same comparison.
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    sp.draw_hpxmap(hpxmap, nest=True, lon_range=[4.9, 5.3], lat_range=[4.9, 5.3],
                   cmap=plt.colormaps['rainbow'])
    sp.draw_colorbar()
    fname = 'healsparse_two.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.DESSkyproj(ax=ax)
    sp.draw_hpxmap(hpxmap, nest=True)
    sp.draw_inset_colorbar()
    fname = 'healsparse_four.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healpix_log_scale(tmp_path):
    """Test plotting a healpix map with a log scale."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap() + 10.0
    hpxmap = hspmap.generate_healpix_map()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap, nest=True, norm="log")
    sp.draw_colorbar()
    fname = 'healsparse_five.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healpix_input_norm(tmp_path):
    """Test plotting a healsparse map with input normalization."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap() + 10.0
    hpxmap = hspmap.generate_healpix_map()

    # First make sure we get matched vmin/vmax.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap, nest=True)
    vmin, vmax = skyproj.utils.get_autoscale_vmin_vmax(values_raster.compressed(), None, None)
    plt.close(fig)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # This should ignore the input vmin/vmax if we set the norm.
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap,
                                                               nest=True,
                                                               norm=norm,
                                                               vmin=-100.0,
                                                               vmax=-50.0)
    sp.draw_colorbar()
    fname = 'healsparse_five.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healpix_rasterized(tmp_path):
    """Test plotting a healpix map with and without rasterization."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()
    hpxmap = hspmap.generate_healpix_map()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # We use a small xsize here because otherwise the non-rasterized test pdf
    # (below) is 22Mb and the test takes a long time to run.
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap, nest=True, rasterized=True, xsize=200)
    sp.draw_inset_colorbar()
    fname = 'healpix_rasterized_on.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_on = os.path.getsize(tmp_path / fname)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap, nest=True, rasterized=False, xsize=200)
    sp.draw_inset_colorbar()
    fname = 'healpix_rasterized_off.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_off = os.path.getsize(tmp_path / fname)

    # The rasterized map will be smaller than the non-rasterized map.
    assert size_rasterized_on < size_rasterized_off

    # Test with a redraw event.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(hpxmap, nest=True, rasterized=True, xsize=200)
    sp.set_extent([10.5, -0.5, -0.5, 10.5])
    sp.draw_inset_colorbar()
    fname = 'healsparse_rasterized_on2.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_on2 = os.path.getsize(tmp_path / fname)

    # The rasterized map will be smaller than the non-rasterized map.
    assert size_rasterized_on2 < size_rasterized_off


def test_healsparse_widemask(tmp_path):
    """Test plotting a healsparse wide mask."""
    plt.rcParams.update(plt.rcParamsDefault)

    # Start with a 1-byte width map
    hspmap = hsp.HealSparseMap.make_empty(32, 4096, hsp.WIDE_MASK, wide_mask_maxbits=7)
    poly = hsp.geom.Polygon(ra=[0.0, 10.0, 10.0, 0.0], dec=[0.0, 0.0, 10.0, 10.0], value=1.0)
    pixels = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap.set_bits_pix(pixels, [0])
    poly2 = hsp.geom.Polygon(ra=[5, 5.2, 5.2, 5.0], dec=[5, 5.0, 5.2, 5.2], value=3.0)
    pixels2 = poly2.get_pixels(nside=hspmap.nside_sparse)
    hspmap.set_bits_pix(pixels2, [4])

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname = 'healsparse_wide_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    # And do the same with a 2-byte wide map
    hspmap = hsp.HealSparseMap.make_empty(32, 4096, hsp.WIDE_MASK, wide_mask_maxbits=15)
    poly = hsp.geom.Polygon(ra=[0.0, 10.0, 10.0, 0.0], dec=[0.0, 0.0, 10.0, 10.0], value=1.0)
    pixels = poly.get_pixels(nside=hspmap.nside_sparse)
    hspmap.set_bits_pix(pixels, [0])
    poly2 = hsp.geom.Polygon(ra=[5, 5.2, 5.2, 5.0], dec=[5, 5.0, 5.2, 5.2], value=3.0)
    pixels2 = poly2.get_pixels(nside=hspmap.nside_sparse)
    hspmap.set_bits_pix(pixels2, [10])

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname2 = 'healsparse_wide_one_alt.png'
    fig.savefig(tmp_path / fname2)
    # Compare to the previoues one.
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname2, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_rec_array(tmp_path):
    """Test plotting a healsparse rec_array map."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap_rec_array()

    # Plot one component of map
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap["A"])
    sp.draw_inset_colorbar()
    fname = 'healsparse_rec_array.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    # Plot valid pixels of map
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, valid_mask=True)
    sp.draw_inset_colorbar()
    fname = 'healsparse_valid_pixels.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_healsparse_rec_array_failover(tmp_path):
    """Test plotting the healsparse map without specifying a component
    Should get a UserWarning and a plot matching valid_mask=True.
    """
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap_rec_array()

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    with pytest.warns(UserWarning):
        im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap)
    sp.draw_inset_colorbar()
    fname = 'healsparse_valid_pixels.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_hpxpix(tmp_path):
    """Test plotting healpix pixels."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()
    pixels = hspmap.valid_pixels
    values = hspmap[pixels]

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(hspmap.nside_sparse,
                                                               pixels,
                                                               values,
                                                               nest=True)
    sp.draw_inset_colorbar()
    fname = 'healsparse_one.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    sp.draw_hpxpix(hspmap.nside_sparse, pixels, values, nest=True,
                   lon_range=[4.9, 5.3], lat_range=[4.9, 5.3],
                   cmap=plt.colormaps['rainbow'])
    sp.draw_colorbar()
    fname = 'healsparse_two.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.DESSkyproj(ax=ax)
    sp.draw_hpxpix(hspmap.nside_sparse, pixels, values, nest=True)
    sp.draw_inset_colorbar()
    fname = 'healsparse_four.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_hpxpix_log_scale(tmp_path):
    """Test plotting healpix pixels with a log scale."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap() + 10.0
    pixels = hspmap.valid_pixels
    values = hspmap[pixels]

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(hspmap.nside_sparse,
                                                               pixels,
                                                               values,
                                                               nest=True,
                                                               norm="log")
    sp.draw_colorbar()
    fname = 'healsparse_five.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_hpxpix_input_norm(tmp_path):
    """Test plotting healpix pixels with input normalization."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap() + 10.0
    pixels = hspmap.valid_pixels
    values = hspmap[pixels]

    # First make sure we get matched vmin/vmax.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(hspmap.nside_sparse,
                                                               pixels,
                                                               values,
                                                               nest=True)
    vmin, vmax = skyproj.utils.get_autoscale_vmin_vmax(values_raster.compressed(), None, None)
    plt.close(fig)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # This should ignore the input vmin/vmax if we set the norm.
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(hspmap.nside_sparse,
                                                               pixels,
                                                               values,
                                                               nest=True,
                                                               norm=norm,
                                                               vmin=-100.0,
                                                               vmax=-50.0)
    sp.draw_colorbar()
    fname = 'healsparse_five.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 40.0)
    if err:
        raise ImageComparisonFailure(err)


def test_hpxpix_rasterized(tmp_path):
    """Test plotting healpix pixels with and without rasterization."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap()
    pixels = hspmap.valid_pixels
    values = hspmap[pixels]

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # We use a small xsize here because otherwise the non-rasterized test pdf
    # (below) is 22Mb and the test takes a long time to run.
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(hspmap.nside_sparse,
                                                               pixels,
                                                               values,
                                                               nest=True,
                                                               rasterized=True,
                                                               xsize=200)
    sp.draw_inset_colorbar()
    fname = 'hpxpix_rasterized_on.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_on = os.path.getsize(tmp_path / fname)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    # We use a small xsize here because otherwise the non-rasterized test pdf
    # (below) is 22Mb and the test takes a long time to run.
    im, lon_raster, lat_raster, values_raster = sp.draw_hpxpix(hspmap.nside_sparse,
                                                               pixels,
                                                               values,
                                                               nest=True,
                                                               rasterized=False,
                                                               xsize=200)
    sp.draw_inset_colorbar()
    fname = 'hpxpix_rasterized_off.pdf'
    fig.savefig(tmp_path / fname)

    size_rasterized_off = os.path.getsize(tmp_path / fname)

    # The rasterized map will be smaller than the non-rasterized map.
    assert size_rasterized_on < size_rasterized_off


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_healsparse_gnomonic(tmp_path, lon_0):
    """Test healsparse map with gnomonic projections."""
    plt.rcParams.update(plt.rcParamsDefault)

    hspmap = _get_hspmap(ra_center=lon_0)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.GnomonicSkyproj(ax=ax, lon_0=lon_0, lat_0=0.0)
    im, lon_raster, lat_raster, values_raster = sp.draw_hspmap(hspmap, zoom=True)
