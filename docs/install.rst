Install
=======

`SkyProj` requires the following python packages and their dependencies:

* `numpy <https://github.com/numpy/numpy>`_
* `astropy <https://astropy.org>`_
* `matplotlib <https://matplotlib.org>`_
* `healpy <https://healpy.readthedocs.io/en/latest/>`_
* `healsparse <https://healsparse.readthedocs.io/en/latest/>`_

In addition, it requires:

* `proj <https://proj.org/en/stable/index.html>`_

`SkyProj` is available at `pypi <https://pypi.org/project/skyproj>`_ and `conda-forge <https://anaconda.org/conda-forge/skyproj>`_.
The most convenient way of installing the latest released version is simply:

.. code-block:: python

  conda install -c conda-forge skyproj

or

.. code-block:: python

  pip install skyproj

To install from source, you can run from the root directory:

.. code-block:: python

  pip install .
