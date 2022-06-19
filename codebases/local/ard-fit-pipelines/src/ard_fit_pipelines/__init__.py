
from importlib.metadata import (
    # import-as with leading _ so that we don't pollute the globals of anyone
    # daring enough to *-import us.
    version as _version,
    PackageNotFoundError as _PNFE
)

try:
    __version__ = _version('ard_fit_pipelines')
except _PNFE:
    # package is not installed
    pass

#import .core
#import .transforms

from .core import Transform
from . import transforms
from . import core