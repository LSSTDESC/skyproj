import copy
import os
import pytest

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))

BOX1 = ([-10, 30], 80, 20)
BOX2 = ([170, 30], 80, 20)
BOX3 = ([-10, -30], 80, 20)
BOX4 = ([170, -30], 80, 20)


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_patches_mcbryde(tmp_path, lon_0):
    """Test drawing patches."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)

    # Draw two rectangles (geodesic), one of which will wrap around.
    patch1 = matplotlib.patches.Rectangle(*BOX1, color="blue")
    sp.ax.add_patch(patch1)
    patch2 = matplotlib.patches.Rectangle(*BOX2, color="blue")
    sp.ax.add_patch(patch2)

    # Draw two rectangles (non-geodesic), one of which will wrap around.
    patch3 = matplotlib.patches.Rectangle(*BOX3, color="red")
    sp.ax.add_patch(patch3, geodesic=False)
    patch4 = matplotlib.patches.Rectangle(*BOX4, color="red")
    sp.ax.add_patch(patch4, geodesic=False)

    fname = f'patches_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_patch_collection_mcbryde(tmp_path, lon_0):
    """Test drawing patches via collections."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)

    # Draw two rectangles (geodesic), one of which will wrap around.
    patch1 = matplotlib.patches.Rectangle(*BOX1, color="blue")
    patch2 = matplotlib.patches.Rectangle(*BOX2, color="blue")

    coll = matplotlib.collections.PatchCollection([patch1, patch2], match_original=True)
    sp.ax.add_collection(coll)

    # Draw two rectangles (non-geodesic), one of which will wrap around.
    patch3 = matplotlib.patches.Rectangle(*BOX3, color="red")
    patch4 = matplotlib.patches.Rectangle(*BOX4, color="red")
    coll = matplotlib.collections.PatchCollection([patch3, patch4], match_original=True)
    sp.ax.add_collection(coll, geodesic=False)

    fname = f'patches_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_patches_mcbryde_pre_existing(tmp_path, lon_0):
    """Test drawing patches (pre-existing transform)."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)

    # Draw two rectangles (geodesic), one of which will wrap around.
    patch1 = matplotlib.patches.Rectangle(*BOX1, color="blue")
    patch1.set_transform(copy.copy(sp.ax.projection))
    sp.ax.add_patch(patch1)
    patch2 = matplotlib.patches.Rectangle(*BOX2, color="blue")
    patch2.set_transform(copy.copy(sp.ax.projection))
    sp.ax.add_patch(patch2)

    # Draw two rectangles (non-geodesic), one of which will wrap around.
    patch3 = matplotlib.patches.Rectangle(*BOX3, color="red")
    proj = copy.copy(sp.ax.projection)
    proj.set_plot_geodesics(False)
    patch3.set_transform(proj)
    sp.ax.add_patch(patch3)
    patch4 = matplotlib.patches.Rectangle(*BOX4, color="red")
    proj = copy.copy(sp.ax.projection)
    proj.set_plot_geodesics(False)
    patch4.set_transform(proj)
    sp.ax.add_patch(patch4)

    fname = f'patches_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_patch_collection_mcbryde_pre_existing(tmp_path, lon_0):
    """Test drawing patches via collections (pre-existing transform)."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)

    # Draw two rectangles (geodesic), one of which will wrap around.
    patch1 = matplotlib.patches.Rectangle(*BOX1, color="blue")
    patch2 = matplotlib.patches.Rectangle(*BOX2, color="blue")

    coll = matplotlib.collections.PatchCollection([patch1, patch2], match_original=True)
    coll.set_transform(copy.copy(sp.ax.projection))
    sp.ax.add_collection(coll)

    # Draw two rectangles (non-geodesic), one of which will wrap around.
    patch3 = matplotlib.patches.Rectangle(*BOX3, color="red")
    patch4 = matplotlib.patches.Rectangle(*BOX4, color="red")
    coll = matplotlib.collections.PatchCollection([patch3, patch4], match_original=True)
    proj = copy.copy(sp.ax.projection)
    proj.set_plot_geodesics(False)
    coll.set_transform(proj)
    sp.ax.add_collection(coll)

    fname = f'patches_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)
