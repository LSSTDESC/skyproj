import os

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def test_draw_ecliptic(tmp_path):
    """Test drawing the ecliptic."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax)
    sp.draw_ecliptic(label='Ecliptic')
    sp.ax.legend()
    fname = 'ecliptic.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


def test_draw_ecliptic_galactic(tmp_path):
    """Test drawing the ecliptic (Galactic Coordinates)."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, galactic=True, longitude_ticks='symmetric')
    sp.draw_ecliptic(label='Ecliptic')
    sp.ax.legend()
    fname = 'ecliptic_galactic.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)
