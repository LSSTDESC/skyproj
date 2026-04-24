import pytest
import math
import numpy as np

import skyproj

try:
    import pyproj
except ImportError:
    pyproj = None


@pytest.mark.parametrize("crsname", ['hammer',
                                     'mbtfpq',
                                     'cyl',
                                     'eqearth',
                                     'laea',
                                     'moll',
                                     'gnom',
                                     'obmoll'])
def test_get_crs(crsname):
    """Test getting a projection CRS and making a new instance."""
    crs = skyproj.get_crs(crsname)

    crs2 = crs.__class__(**crs._projection_dict)

    for key in crs._projection_dict.keys():
        assert crs2._projection_dict[key] == crs._projection_dict[key]


@pytest.mark.parametrize("crsname", ['hammer',
                                     'mbtfpq',
                                     'cyl',
                                     'eqearth',
                                     'laea',
                                     'moll',
                                     'gnom',
                                     'obmoll'])
def test_update_crs_lon_0(crsname):
    """Test updating a projection CRS lon_0."""
    crs = skyproj.get_crs(crsname)

    crs2 = crs.with_new_center(100.0)

    assert crs.__class__ == crs2.__class__

    for key in crs._projection_dict.keys():
        if key == 'lon_0':
            assert crs2.lon_0 == 100.0
        else:
            assert crs2._projection_dict[key] == crs._projection_dict[key]


@pytest.mark.parametrize("crsname", ['laea',
                                     'gnom'])
def test_update_crs_lon_0_lat_0(crsname):
    """Test updating a projection CRS lon_0/lat_0."""
    crs = skyproj.get_crs(crsname)

    crs2 = crs.with_new_center(100.0, lat_0=-45.0)

    assert crs.__class__ == crs2.__class__

    for key in crs._projection_dict.keys():
        if key == 'lon_0':
            assert crs2.lon_0 == 100.0
        elif key == 'lat_0':
            assert crs2.lat_0 == -45.0
        else:
            assert crs2._projection_dict[key] == crs._projection_dict[key]


@pytest.mark.parametrize("crsname", ['hammer',
                                     'mbtfpq',
                                     'cyl',
                                     'eqearth',
                                     'laea',
                                     'moll',
                                     'gnom',
                                     'obmoll'])
def test_deprecated_proj4_params(crsname):
    """Test updating a projection CRS lon_0."""
    crs = skyproj.get_crs(crsname)

    with pytest.warns(FutureWarning):
        params = crs.proj4_params

    assert params["R"] == crs._projection_dict["radius"]

    if crs.name == "obmoll":
        assert params["proj"] == "ob_tran"
        assert params["o_proj"] == "moll"
        assert params["o_lat_p"] == crs._projection_dict["lat_p"]
        assert params["o_lon_p"] == crs._projection_dict["lon_p"]
    else:
        for key in ["lon_0", "lat_0", "lat_1", "lat_2"]:
            if key in crs._projection_dict:
                assert params[key] == crs._projection_dict[key]


@pytest.mark.parametrize("crsname", ['hammer',
                                     'mbtfpq',
                                     'eqearth',
                                     'laea',
                                     'moll',])
@pytest.mark.parametrize("lon_0", [0.0, 100.0, 180.0])
@pytest.mark.skipif(pyproj is None, reason="pyproj not installed.")
def test_compare_to_pyproj(crsname, lon_0):
    from pyproj import CRS
    from pyproj import Transformer

    np.random.seed(1234)

    lon_vals = np.random.uniform(low=0.0, high=360.0, size=100_000)
    lat_vals = np.random.uniform(low=-90.0, high=90.0, size=100_000)

    # Set up skyproj CRS
    skyproj_crs = skyproj.get_crs(crsname, lon_0=lon_0)

    # Set up proj CRS
    radius = skyproj_crs.radius
    plate_carree = CRS(proj="eqc", lon_0=0.0, ellps="sphere", R=radius, to_meter=math.radians(1)*radius)

    proj_crs = CRS(proj=crsname, lon_0=lon_0, ellps="sphere", R=radius)

    proj_transformer_fwd = Transformer.from_crs(plate_carree, proj_crs, always_xy=True)
    proj_transformer_inv = Transformer.from_crs(proj_crs, plate_carree, always_xy=True)

    # Check forward transform.
    skyproj_xy = skyproj_crs.transform_points(lon_vals, lat_vals)

    pyproj_x, pyproj_y = proj_transformer_fwd.transform(
        skyproj.utils.wrap_values(lon_vals),
        lat_vals,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_xy[:, 0], pyproj_x)
    np.testing.assert_array_almost_equal(skyproj_xy[:, 1], pyproj_y)

    # Check inverse transform.
    skyproj_lonlat = skyproj_crs.transform_points(skyproj_xy[:, 0], skyproj_xy[:, 1], inverse=True)

    pyproj_lon, pyproj_lat = proj_transformer_inv.transform(
        pyproj_x,
        pyproj_y,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 0], pyproj_lon)
    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 1], pyproj_lat)


@pytest.mark.parametrize("lon_0", [0.0, 100.0, 180.0])
@pytest.mark.parametrize("lonlat_p", [(90.0, 0.0), (45.0, 45.0), (120.0, -50.0)])
@pytest.mark.skipif(pyproj is None, reason="pyproj not installed.")
def test_compare_obmoll_to_pyproj(lon_0, lonlat_p):
    from pyproj import CRS
    from pyproj import Transformer

    np.random.seed(1234)

    lon_vals = np.random.uniform(low=0.0, high=360.0, size=100_000)
    lat_vals = np.random.uniform(low=-90.0, high=90.0, size=100_000)

    # Set up skyproj CRS
    skyproj_crs = skyproj.get_crs("obmoll", lon_0=lon_0, lon_p=lonlat_p[0], lat_p=lonlat_p[1])

    # Set up proj CRS
    radius = skyproj_crs.radius
    plate_carree = CRS(proj="eqc", lon_0=0.0, ellps="sphere", R=radius, to_meter=math.radians(1)*radius)

    proj_crs = CRS(
        proj="ob_tran",
        o_proj="moll",
        lon_0=lon_0,
        o_lon_p=lonlat_p[0],
        o_lat_p=lonlat_p[1],
        ellps="sphere",
        R=radius,
    )

    proj_transformer_fwd = Transformer.from_crs(plate_carree, proj_crs, always_xy=True)
    proj_transformer_inv = Transformer.from_crs(proj_crs, plate_carree, always_xy=True)

    # Check forward transform.
    skyproj_xy = skyproj_crs.transform_points(lon_vals, lat_vals)

    pyproj_x, pyproj_y = proj_transformer_fwd.transform(
        skyproj.utils.wrap_values(lon_vals),
        lat_vals,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_xy[:, 0], pyproj_x)
    np.testing.assert_array_almost_equal(skyproj_xy[:, 1], pyproj_y)

    # Check inverse transform.
    skyproj_lonlat = skyproj_crs.transform_points(skyproj_xy[:, 0], skyproj_xy[:, 1], inverse=True)

    pyproj_lon, pyproj_lat = proj_transformer_inv.transform(
        pyproj_x,
        pyproj_y,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 0], pyproj_lon)
    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 1], pyproj_lat)


@pytest.mark.parametrize("crsname", ['laea',
                                     'gnom',])
@pytest.mark.parametrize("lon_0", [0.0, 100.0, 180.0])
@pytest.mark.parametrize("lat_0", [0.0, 25.0, -45.0])
@pytest.mark.skipif(pyproj is None, reason="pyproj not installed.")
def test_compare_lonlat_to_pyproj(crsname, lon_0, lat_0):
    from pyproj import CRS
    from pyproj import Transformer

    np.random.seed(1234)

    lon_vals = np.random.uniform(low=lon_0 - 20.0, high=lon_0 + 20.0, size=100_000)
    lat_vals = np.random.uniform(low=lat_0 - 20.0, high=lat_0 + 20.0, size=100_000)

    # Set up skyproj CRS
    skyproj_crs = skyproj.get_crs(crsname, lon_0=lon_0, lat_0=lat_0)

    # Set up proj CRS
    radius = skyproj_crs.radius
    plate_carree = CRS(proj="eqc", lon_0=0.0, ellps="sphere", R=radius, to_meter=math.radians(1)*radius)

    proj_crs = CRS(proj=crsname, lon_0=lon_0, lat_0=lat_0, ellps="sphere", R=radius)

    proj_transformer_fwd = Transformer.from_crs(plate_carree, proj_crs, always_xy=True)
    proj_transformer_inv = Transformer.from_crs(proj_crs, plate_carree, always_xy=True)

    # Check forward transform.
    skyproj_xy = skyproj_crs.transform_points(lon_vals, lat_vals)

    pyproj_x, pyproj_y = proj_transformer_fwd.transform(
        skyproj.utils.wrap_values(lon_vals),
        lat_vals,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_xy[:, 0], pyproj_x)
    np.testing.assert_array_almost_equal(skyproj_xy[:, 1], pyproj_y)

    # Check inverse transform.
    skyproj_lonlat = skyproj_crs.transform_points(skyproj_xy[:, 0], skyproj_xy[:, 1], inverse=True)

    pyproj_lon, pyproj_lat = proj_transformer_inv.transform(
        pyproj_x,
        pyproj_y,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 0], pyproj_lon)
    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 1], pyproj_lat)


@pytest.mark.parametrize("lon_0", [0.0, 100.0, 180.0])
@pytest.mark.parametrize("lat_12", [(15.0, 45.0), (-60.0, -45.0)])
@pytest.mark.skipif(pyproj is None, reason="pyproj not installed.")
def test_compare_albers_to_pyproj(lon_0, lat_12):
    from pyproj import CRS
    from pyproj import Transformer

    np.random.seed(1234)

    lon_vals = np.random.uniform(low=0.0, high=360.0, size=100_000)
    lat_vals = np.random.uniform(low=-90.0, high=90.0, size=100_000)

    # Set up skyproj CRS
    skyproj_crs = skyproj.get_crs("aea", lon_0=lon_0, lat_1=lat_12[0], lat_2=lat_12[1])

    # Set up proj CRS
    radius = skyproj_crs.radius
    plate_carree = CRS(proj="eqc", lon_0=0.0, ellps="sphere", R=radius, to_meter=math.radians(1)*radius)

    proj_crs = CRS(proj="aea", lon_0=lon_0, lat_1=lat_12[0], lat_2=lat_12[1], ellps="sphere", R=radius)

    proj_transformer_fwd = Transformer.from_crs(plate_carree, proj_crs, always_xy=True)
    proj_transformer_inv = Transformer.from_crs(proj_crs, plate_carree, always_xy=True)

    # Check forward transform.
    skyproj_xy = skyproj_crs.transform_points(lon_vals, lat_vals)

    pyproj_x, pyproj_y = proj_transformer_fwd.transform(
        skyproj.utils.wrap_values(lon_vals),
        lat_vals,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_xy[:, 0], pyproj_x)
    np.testing.assert_array_almost_equal(skyproj_xy[:, 1], pyproj_y)

    # Check inverse transform.
    skyproj_lonlat = skyproj_crs.transform_points(skyproj_xy[:, 0], skyproj_xy[:, 1], inverse=True)

    pyproj_lon, pyproj_lat = proj_transformer_inv.transform(
        pyproj_x,
        pyproj_y,
        None,
        errcheck=False,
    )

    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 0], pyproj_lon)
    np.testing.assert_array_almost_equal(skyproj_lonlat[:, 1], pyproj_lat)
