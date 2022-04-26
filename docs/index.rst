.. healsparse documentation master file, created by
   sphinx-quickstart on Fri Jun  5 07:57:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`SkyProj`: Sky Projections with matplotlib and PROJ
===================================================

`SkyProj` provides an astronomically oriented interface to plotting sky maps based on matplotlib_ and PROJ_.
The primary goal of `SkyProj` is to create publication quality projected visualizations of healsparse_ and HEALPix_ maps.
In addition, `SkyProj` can be an interactive tool that allows you to dynamically zoom in on healsparse_ maps showing the full resolution of the map.

The `SkyProj` package addresses multiple issues in the default healpy_ plotting routines:

1. healpy_ supports a limited set of sky projections (`cartview`, `mollview`, and `gnomview`)
2. healpy_ converts sparse HEALPix_ maps to full maps to plot; this is memory intensive for large nside.
3. healpy_ is best at plotting full-sky and not partial-sky maps.

The code is hosted in GitHub_.
Please use the `issue tracker <https://github.com/LSSTDESC/skyproj/issues>`_ to let us know about any problems or questions with the code.
The list of released versions of this package can be found `here <https://github.com/LSSTDESC/skyproj/releases>`_, with the main branch including the most recent (non-released) development.

The `SkyProj` code was written by Eli Rykoff based on work by Alex Drlica-Wagner, and matplotlib/projection integrations derived from the cartopy_ package.
This software was developed under the Rubin Observatory Legacy Survey of Space and Time (LSST) Dark Energy Science Collaboration (DESC) using LSST DESC resources.
The DESC acknowledges ongoing support from the Institut National de Physique Nucléaire et de Physique des Particules in France; the Science & Technology Facilities Council in the United Kingdom; and the Department of Energy, the National Science Foundation, and the LSST Corporation in the United States.
DESC uses resources of the IN2P3 Computing Center (CC-IN2P3--Lyon/Villeurbanne - France) funded by the Centre National de la Recherche Scientifique; the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231; STFC DiRAC HPC Facilities, funded by UK BIS National E-infrastructure capital grants; and the UK particle physics grid, supported by the GridPP Collaboration.
This work was performed in part under DOE Contract DE-AC02-76SF00515.


.. _HEALPix: https://healpix.jpl.nasa.gov/
.. _DESC: https://lsst-desc.org/
.. _healpy: https://healpy.readthedocs.io/en/latest/
.. _GitHub: https://github.com/LSSTDESC/skyproj
.. _healsparse: https://healsparse.readthedocs.io/en/latest/
.. _cartopy: https://scitools.org.uk/cartopy/docs/latest/
.. _PROJ: https://proj.org/
.. _matplotlib: https://matplotlib.org/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quickstart
   basic_interface
   maps
   projections
   surveys


Modules API Reference
=====================

.. toctree::
   :maxdepth: 3

   modules

Search
======

* :ref:`search`


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
