import os

import hpgeom as hpg

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def test_pixel_boundaries(tmp_path):
    """Test pixel boundaries."""
    plt.rcParams.update(plt.rcParamsDefault)

    nside = 32
    ra, dec = (150.1, 2.1)
    pixel = hpg.angle_to_pixel(nside, ra, dec, nest=True, lonlat=True, degrees=True)
    neighbors = hpg.neighbors(nside, pixel, nest=True)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    sp.draw_pixel_boundaries(nside, neighbors)
    sp.draw_pixel_boundaries(nside, pixel, label_pixels=True, facecolor="lightgray")
    sp.set_extent([ra-5.0, ra+5.0, dec-5.0, dec+5.0])
    fname = f'pixel_boundaries.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)
