import os
import pytest

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("skyproj", [skyproj.Skyproj,
                                     skyproj.McBrydeSkyproj])
def test_tissot(tmp_path, skyproj):
    """Test Tissot Indicatrices."""
    plt.rcParams.update(plt.rcParamsDefault)

    # Full image
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj(ax=ax)
    sp.tissot_indicatrices()
    fname = f'{sp.projection_name}_tissot.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
