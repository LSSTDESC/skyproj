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

    crs2 = crs.__class__(**crs.proj4_params)

    for key in crs.proj4_params.keys():
        assert crs2.proj4_params[key] == crs.proj4_params[key]


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

    for key in crs.proj4_params.keys():
        if key == 'lon_0':
            assert crs2.lon_0 == 100.0
        else:
            assert crs2.proj4_params[key] == crs.proj4_params[key]


@pytest.mark.parametrize("crsname", ['laea',
                                     'gnom'])
def tets_update_crs_lon_0_lat_0(crsname):
    """Test updating a projection CRS lon_0/lat_0."""
    crs = skyproj.get_crs(crsname)

    crs2 = crs.with_new_center(100.0, lat_0=-45.0)

    assert crs.__class__ == crs2.__class__

    for key in crs.proj4_params.keys():
        if key == 'lon_0':
            assert crs2.lon_0 == 100.0
        elif key == 'lat_0':
            assert crs2.lat_0 == -45.0
        else:
            assert crs2.proj4_params[key] == crs.proj4_params[key]
