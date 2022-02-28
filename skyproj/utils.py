"""
Random utilities
"""
import os
import os.path
import numpy as np

__all__ = ["get_datadir", "get_datafile", "wrap_values"]


def get_datadir():
    from os.path import abspath, dirname, join
    return join(dirname(abspath(__file__)), 'data')


def get_datafile(filename):
    return os.path.join(get_datadir(), filename)


def wrap_values(values, wrap=180.0):
    """Wrap values according to the wrap angle.

    Parameters
    ----------
    values : `float` or `np.ndarray`
        Values to remap.
    wrap : `float`, optional
        Wrap angle to apply.

    Returns
    -------
    wrapped_array
    """
    return (values - wrap) % 360. + (wrap - 360.)


def _get_boundary_poly_xy(bounds_xy, extent_xy, proj, proj_inverse):
    """Get the boundary polygon in x/y space.

    Parameters
    ----------
    bounds_xy : `dict`
        Dictionary with projection boundaries.
    extent_xy : `list`
        Extent of bounding box.
    proj : `func`
        Forward projection function.
    proj_inverse : `func`
        Inverse projection function.

    Returns
    -------
    boundary_poly_xy : `np.ndarray`
        N,2 array of the boundary polygon.
    """
    bounds_xy_clipped_sides = []
    nstep = 1000

    if extent_xy[0] < extent_xy[1]:
        x0_index = 0
        x1_index = 1
    else:
        x0_index = 1
        x1_index = 0

    def _generate_side(side):
        if bounds_xy[side] is not None:
            use_bounds, = np.where((bounds_xy[side][:, 0] >= extent_xy[x0_index])
                                   & (bounds_xy[side][:, 0] <= extent_xy[x1_index])
                                   & (bounds_xy[side][:, 1] >= extent_xy[2])
                                   & (bounds_xy[side][:, 1] <= extent_xy[3]))
            bounds_x = bounds_xy[side][use_bounds, 0]
            bounds_y = bounds_xy[side][use_bounds, 1]
        else:
            bounds_x = []
            bounds_y = []

        if side == 'left':
            box_x = np.linspace(extent_xy[x0_index], extent_xy[x0_index], nstep)
            box_y = np.linspace(extent_xy[2], extent_xy[3], nstep)
        elif side == 'top':
            box_x = np.linspace(extent_xy[x0_index], extent_xy[x1_index], nstep)
            box_y = np.linspace(extent_xy[3], extent_xy[3], nstep)
        elif side == 'right':
            box_x = np.linspace(extent_xy[x1_index], extent_xy[x1_index], nstep)
            box_y = np.linspace(extent_xy[2], extent_xy[3], nstep)
        else:
            box_x = np.linspace(extent_xy[x1_index], extent_xy[x0_index], nstep)
            box_y = np.linspace(extent_xy[2], extent_xy[2], nstep)

        box_lon, box_lat = proj_inverse(box_x, box_y)
        box_x2, box_y2 = proj(box_lon, box_lat)
        use_box, = np.where(np.isclose(box_x2, box_x)
                            & np.isclose(box_y2, box_y))

        x = np.concatenate((bounds_x, box_x[use_box]))
        y = np.concatenate((bounds_y, box_y[use_box]))

        if side == 'left':
            st = np.argsort(y)
        elif side == 'top':
            st = np.argsort(x)
        elif side == 'right':
            st = np.argsort(y)[::-1]
        else:
            st = np.argsort(x)[::-1]

        return np.column_stack((x[st], y[st]))

    bounds_xy_clipped_sides.append(_generate_side('left'))
    bounds_xy_clipped_sides.append(_generate_side('top'))
    bounds_xy_clipped_sides.append(_generate_side('right'))
    bounds_xy_clipped_sides.append(_generate_side('bottom'))

    return np.concatenate(bounds_xy_clipped_sides)
