import pytest

import skyproj


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
