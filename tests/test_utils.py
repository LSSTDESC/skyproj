import numpy as np

import skyproj


def test_wrap_values():
    """Test the wrap values.
    """
    # Make a giant array that wraps around 0 to 360 many times.
    arr = np.arange(20000.) - 10000.
    sin_arr = np.sin(np.deg2rad(arr))
    cos_arr = np.cos(np.deg2rad(arr))

    for wrap_angle in np.arange(360.):
        wrapped_arr = skyproj.utils.wrap_values(arr, wrap=wrap_angle)

        # Confirm we have the correct range:
        np.testing.assert_almost_equal(np.min(wrapped_arr), wrap_angle - 360.)
        np.testing.assert_almost_equal(np.max(wrapped_arr), wrap_angle - 1.0)

        np.testing.assert_array_almost_equal(np.sin(np.deg2rad(wrapped_arr)), sin_arr)
        np.testing.assert_array_almost_equal(np.cos(np.deg2rad(wrapped_arr)), cos_arr)
