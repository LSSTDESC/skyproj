import os
import pytest

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
    # Full image
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj(ax=ax, lon_0=lon_0)
    fname = f'{m.projection_name}_full_{lon_0}.png'
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
    # Simple zoom
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj(ax=ax, extent=[0, 50, 0, 50])
    fname = f'{m.projection_name}_zoom.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
