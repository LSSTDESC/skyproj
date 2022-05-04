import os

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def test_skyproj_plotting(tmp_path):
    """Test basic plotting routines."""
    plt.rcParams.update(plt.rcParamsDefault)

    # Tests are consolidated for fewer comparisons.
    # Note that tests of wrapping lines and polygons are in
    # test_lines_polygons.
    # Tests of pcolormesh are in the healsparse/healpix map plots.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, extent=[0, 50, 0, 50])

    # Test ``plot`` with points.
    sp.plot([10, 20, 30, 40], [10, 20, 30, 40], 'k+')
    sp.plot([40, 30, 20, 10], [10, 20, 30, 40], 'r.')

    # Test ``plot`` with lines.
    # Note that the geodesic line segments do not meet the interior
    # points plotted above.
    sp.plot([10, 40], [10, 40], 'k-')
    sp.plot([40, 10], [10, 40], 'r:')

    # Test ``fill``.
    sp.fill([20, 25, 25, 20], [20, 20, 25, 25], color='blue')

    # Test ``scatter`` with points.
    sp.scatter([15, 35], [15, 35], c=['magenta', 'orange'])

    fname = 'plotting_routines.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
