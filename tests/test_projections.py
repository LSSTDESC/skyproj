import pytest

import skyproj


@pytest.mark.parametrize("projname", ['hammer',
                                      'mbtfpq',
                                      'cyl',
                                      'eqearth',
                                      'laea',
                                      'moll',
                                      'gnom',
                                      'obmoll'])
def test_get_projection(projname):
    """Test getting a projection and making a new instance."""
    proj = skyproj.get_projection(projname)

    proj2 = proj.__class__(**proj.proj4_params)

    for key in proj.proj4_params.keys():
        assert proj2.proj4_params[key] == proj.proj4_params[key]


@pytest.mark.parametrize("projname", ['hammer',
                                      'mbtfpq',
                                      'cyl',
                                      'eqearth',
                                      'laea',
                                      'moll',
                                      'gnom',
                                      'obmoll'])
def test_update_projection_lon_0(projname):
    """Test updating a projection lon_0."""
    proj = skyproj.get_projection(projname)

    proj2 = proj.with_new_center(100.0)

    assert proj.__class__ == proj2.__class__

    for key in proj.proj4_params.keys():
        if key == 'lon_0':
            assert proj2.lon_0 == 100.0
        else:
            assert proj2.proj4_params[key] == proj.proj4_params[key]


@pytest.mark.parametrize("projname", ['laea',
                                      'gnom'])
def tets_update_projection_lon_0_lat_0(projname):
    """Test updating a projection lon_0/lat_0."""
    proj = skyproj.get_projection(projname)

    proj2 = proj.with_new_center(100.0, lat_0=-45.0)

    assert proj.__class__ == proj2.__class__

    for key in proj.proj4_params.keys():
        if key == 'lon_0':
            assert proj2.lon_0 == 100.0
        elif key == 'lat_0':
            assert proj2.lat_0 == -45.0
        else:
            assert proj2.proj4_params[key] == proj.proj4_params[key]
