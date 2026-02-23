from setuptools import setup, Extension
import numpy
import os
import sys
import shutil
from pathlib import Path


include_dirs = [numpy.get_include()]
library_dirs = []

proj_dir = os.environ.get("PROJ_DIR")
if proj_dir is None:
    proj = shutil.which("proj", path=sys.prefix)
    if proj is None:
        proj = shutil.which("proj")
    if proj is None:
        raise RuntimeError("proj not found via PROJ_DIR env var or in path.")

    proj_dir = Path(proj).parent.parent
else:
    proj_dir = Path(proj_dir)

if not proj_dir.exists():
    raise RuntimeError(f"Could not find proj directory {proj_dir}")

proj_include_dir = str(proj_dir / "include")
# Add more logic here.
proj_library_dir = str(proj_dir / "lib")

include_dirs.append(proj_include_dir)
library_dirs.append(proj_library_dir)

extra_link_args = []
if sys.platform == "darwin":
    extra_link_args = [
        f"-Wl,-rpath,{proj_library_dir}",
        f"-L{proj_library_dir}",
    ]

ext = Extension(
    "skyproj._cskyproj",
    [
        "skyproj/skyproj.c",
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["proj", "m"],
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=[ext],
)
