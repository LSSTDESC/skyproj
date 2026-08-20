import os
import pytest

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_patches_mcbryde(tmp_path, lon_0):
    """Test drawing patches."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)

    # Draw two rectangles (geodesic), one of which will wrap around.
    patch1 = matplotlib.patches.Rectangle([-10, 40], 20, 20, color="blue")
    sp.ax.add_patch(patch1)
    patch2 = matplotlib.patches.Rectangle([170, 40], 20, 20, color="red")
    sp.ax.add_patch(patch2)

    # Draw two rectangles (non-geodesic), one of which will wrap around.
    patch3 = matplotlib.patches.Rectangle([-10, -40], 20, 20, color="blue")
    sp.ax.add_patch(patch3, geodesic=False)
    patch4 = matplotlib.patches.Rectangle([170, -40], 20, 20, color="red")
    sp.ax.add_patch(patch4, geodesic=False)

    fname = f'patches_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)
