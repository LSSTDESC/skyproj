try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("skyproj")
except PackageNotFoundError:
    # package is not installed
    pass

from .skymap import *
from .projections import *
from .transforms import *
from .skyaxes import *
