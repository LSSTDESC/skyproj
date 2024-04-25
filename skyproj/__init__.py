try:
    from ._version import __version__
except ImportError:
    pass

from .skyproj import *
from .skycrs import *
from .transforms import *
from .skyaxes import *
from .survey import *
