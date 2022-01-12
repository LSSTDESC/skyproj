# skyproj: Sky Projections with matplotlib and proj

The `skyproj` package provides an astronomically oriented interface to ploting sky maps based on [`matplotlib`](https://matplotlib.org/) and [PROJ](https://proj.org/).
This package addresses several issues present in the [`healpy`](https://healpy.readthedocs.io/en/latest/) plotting routines:
1. `healpy` supports a limited set of sky projections (`cartview`, `mollview`, and `gnomview`)
2. `healpy` converts sparse healpix maps to full maps to plot; this is memory intensive for large `nside`

`skyproj` is intended as the primary visualization tool for [`healsparse`](https://healsparse.readthedocs.io/en/latest/) maps, which provide high resolution maps in a memory efficent way.
`skyproj` can create interactive visualizations of `healsparse` and `healpy` maps that dynamically change resolution.
In addition, `skyproj` provides some convenience functionality for large optical surveys.
The `skyproj` package has its origins in `cartosky`, which was built on `cartopy` and some of the features may be familiar to users of `cartopy`.
However, it has diverged significantly from the original code as the needs of mapping the sky are somewhat different than the needs of mapping the Earth.

## Installation

The easiest way to install `skyproj` is from pypi or conda-forge. (Coming soon.)

## Tutorial

If you want to see what you can do with `skyproj`, check out the [tutorial](tutorial/).
