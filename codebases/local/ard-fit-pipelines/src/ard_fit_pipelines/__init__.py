from importlib.metadata import (  # noqa: N814
    # import-as with leading _ so that we don't pollute the globals of anyone
    # daring enough to *-import us.
    version as _version,
    PackageNotFoundError as _PNFE,
)

try:
    __version__ = _version("ard_fit_pipelines")
except _PNFE:
    # package is not installed
    pass

# import .core
# import .transforms

from .core import (
    Transform,
    StatelessTransform,
    FitTransform,
    HP,
    HPFmtStr,
    HPLambda,
    HPCols,
    columns_field,
)
from .graph import (
    Pipeline,
    IfHyperparamIsTrue,
    IfHyperparamLambda,
    IfTrainingDataHasProperty,
)
from .transforms import (
    Identity,
    ColumnsTransform,
    WeightedTransform,
    CopyColumns,
    KeepColumns,
    RenameColumns,
    DropColumns,
    StatelessLambda,
    StatefulLambda,
    Pipe,
    Clip,
    Winsorize,
    ImputeConstant,
    ImputeMean,
    DeMean,
    ZScore,
    Print,
    LogMessage,
)
