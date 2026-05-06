from setuptools import setup, Extension
import numpy
import sys


include_dirs = [numpy.get_include()]
library_dirs = []

extra_link_args = []

libraries = []
if sys.platform == 'win32':
    # Windows: no pthread or math library needed, use native
    pass
else:
    # Linux/macOS: link with pthread
    libraries.append("m")
    libraries.append("pthread")

ext = Extension(
    "skyproj._cskyproj",
    [
        "skyproj/skyproj.c",
        "skyproj/projections.c",
        "skyproj/str_dict.c",
        "skyproj/geodesics.c",
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=[ext],
)
