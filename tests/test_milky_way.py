import os

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def test_draw_milky_way(tmp_path):
    """Test drawing the Milky Way."""
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skyproj.McBrydeSkyproj(ax=ax)
    m.draw_milky_way(label='Milky Way')
    m.legend()
    fname = 'milky_way.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
