import pytest

import skyproj


@pytest.mark.parametrize("projname", ['hammer',
                                      'mbtfpq',
                                      'cyl',
                                      'eqearth',
                                      'laea',
                                      'moll',
                                      'gnom'])
def test_get_projection(projname):
    """Test getting a projection and making a new instance."""
    proj = skyproj.get_projection(projname)

    proj2 = proj.__class__(**proj.proj4_params)

    for key in proj.proj4_params.keys():
        assert proj2.proj4_params[key] == proj.proj4_params[key]
