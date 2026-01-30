import os
import pytest

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import skyproj  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("lon_0", [0.0, 180.0])
def test_lines_polygons_mcbryde(tmp_path, lon_0):
    """Test drawing lines and polygons."""
    plt.rcParams.update(plt.rcParamsDefault)

    # This code draws a bunch of geodesics and polygons that are
    # both empty and filled, and cross over the boundary, to ensure
    # that features are working as intended.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=lon_0)

    # Draw two geodesics, one of which will wrap around.
    sp.ax.plot([-10., 45.], [-10., 45.], 'r-', label='One')
    sp.ax.plot([170., 210.], [-10., 45.], 'b--', label='Two')

    # Draw two non-geodesics, one of which will wrap around.
    sp.ax.plot([-10, 45.], [60.0, 60.0], 'm-', geodesic=False)
    sp.ax.plot([170., 210.], [60.0, 60.0], 'm-', geodesic=False)

    # Draw two unfilled polygons, one of which will wrap around.
    sp.draw_polygon([-20, 20, 20, -20], [20, 20, 40, 40],
                    edgecolor='magenta', label='Three')
    sp.draw_polygon([160, 200, 200, 160], [20, 20, 40, 40],
                    edgecolor='black', label='Four')

    # Draw two filled polygons, one of which will wrap around.
    sp.draw_polygon([-20, 20, 20, -20], [-20, -20, -40, -40],
                    edgecolor='black', facecolor='red', linestyle='--', label='Five')
    sp.draw_polygon([160, 200, 200, 160], [-20, -20, -40, -40],
                    edgecolor='red', facecolor='black', linestyle='-', label='Six')

    # Draw two circles, one empty, one filled.
    sp.ax.circle(40.0, -40.0, 5.0, color='blue', label='Seven')
    sp.ax.circle(-40.0, -40.0, 5.0, color='orange', label='Eight', fill=True)

    # Test ``ellipse``.  We can only plot one point per call
    sp.ax.ellipse(60, 15, 10, 4, 0, color='green', label='Nine')
    sp.ax.ellipse(300, 15, 15, 2, 45, fill=True, color='red', label='Ten')

    # Draw two unfilled boxes, one of which will wrap around.
    sp.draw_box([-20, 20, 20, -20], [50, 50, 70, 70], edgecolor='magenta')
    sp.draw_box([160, 200, 200, 160], [50, 50, 70, 70], edgecolor='black')

    # Draw two filled boxes, one of which will wrap around.
    sp.draw_box([-20, 20, 20, -20], [-50, -50, -70, -70],
                edgecolor='black', facecolor='red', linestyle='--')
    sp.draw_box([160, 200, 200, 160], [-50, -50, -70, -70],
                edgecolor='red', facecolor='black', linestyle='-')

    # Draw an annotated arrow.
    sp.ax.annotate("", xytext=(220, 20), xy=(270, 50), arrowprops=dict(arrowstyle="->"))
    sp.ax.annotate("", xytext=(120, -50), xy=(50, -20), arrowprops=dict(arrowstyle="->"))

    sp.ax.legend()
    fname = f'lines_and_polygons_{lon_0}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("lonlatplonp", [(0.0, 45.0, -90.0),
                                         (100.0, 80.0, -90.0)])
def test_lines_polygons_obmoll(tmp_path, lonlatplonp):
    """Test drawing lines and polygons on an Oblique Mollweide map."""
    plt.rcParams.update(plt.rcParamsDefault)

    lon_0, lat_p, lon_p = lonlatplonp

    # This code draws a bunch of geodesics and polygons that are
    # both empty and filled, and cross over the boundary, to ensure
    # that features are working as intended.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.ObliqueMollweideSkyproj(ax=ax, lon_0=lon_0, lat_p=lat_p, lon_p=lon_p)

    # Draw two geodesics, one of which will wrap around.
    sp.ax.plot([-10., 45.], [-10., 45.], 'r-', label='One')
    sp.ax.plot([170., 210.], [-10., 45.], 'b--', label='Two')

    # Draw two unfilled polygons, one of which will wrap around.
    sp.draw_polygon([-20, 20, 20, -20], [20, 20, 40, 40],
                    edgecolor='magenta', label='Three')
    sp.draw_polygon([160, 200, 200, 160], [20, 20, 40, 40],
                    edgecolor='black', label='Four')

    # Draw two filled polygons, one of which will wrap around.
    # Note that wrapped filled polygons do not currently work with oblique transforms.
    sp.draw_polygon([-20, 20, 20, -20], [-20, -20, -40, -40],
                    edgecolor='black', facecolor='red', linestyle='--', label='Five')
    sp.draw_polygon([160, 200, 200, 160], [-20, -20, -40, -40],
                    edgecolor='red', facecolor='black', linestyle='-', label='Six')

    # Draw two unfilled boxes, one of which will wrap around.
    sp.draw_box([-20, 20, 20, -20], [50, 50, 70, 70], edgecolor='magenta')
    sp.draw_box([160, 200, 200, 160], [50, 50, 70, 70], edgecolor='black')

    # Draw two filled boxes, one of which will wrap around.
    # Note that wrapped filled boxes do not currently work with oblique transforms.
    sp.draw_box([-20, 20, 20, -20], [-50, -50, -70, -70],
                edgecolor='black', facecolor='red', linestyle='--')
    sp.draw_box([160, 200, 200, 160], [-50, -50, -70, -70],
                edgecolor='red', facecolor='black', linestyle='-')

    sp.ax.legend()
    fname = f'lines_and_polygons_obmoll_{lon_0}_{lat_p}_{lon_p}.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)


def test_lines_polygons_mcbryde_opaque_legend(tmp_path):
    """Test drawing lines and polygons with an opaque legend."""
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    sp = skyproj.McBrydeSkyproj(ax=ax, lon_0=0.0)

    sp.ax.plot([-10., 45.], [-10., 45.], 'r-', label='One')
    sp.ax.plot([170., 210.], [-10., 45.], 'b--', label='Two')

    sp.draw_polygon([-20, 20, 20, -20], [20, 20, 40, 40],
                    edgecolor='magenta', label='Three')
    sp.draw_polygon([160, 200, 200, 160], [20, 20, 40, 40],
                    edgecolor='black', label='Four')

    sp.ax.legend(loc='lower right', framealpha=1.0)
    fname = 'lines_and_polygons_opaque_legend.png'
    fig.savefig(tmp_path / fname)
    plt.close(fig)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 15.0)
    if err:
        raise ImageComparisonFailure(err)
