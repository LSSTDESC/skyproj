"""
Random utilities
"""
import os
import os.path

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
