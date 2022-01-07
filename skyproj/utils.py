"""
Random utilities
"""
import os
import os.path

__all__ = ["get_datadir", "get_datafile", "remap_pm180_values"]


def get_datadir():
    from os.path import abspath, dirname, join
    return join(dirname(abspath(__file__)), 'data')


def get_datafile(filename):
    return os.path.join(get_datadir(), filename)


def remap_pm180_values(values):
    """Remap value(s) to [180, 180).

    Parameters
    ----------
    values : `float` or `np.ndarray`
        Values to remap.

    Returns
    -------
    remapped_array
    """
    return (values + 180.) % 360. - 180.
