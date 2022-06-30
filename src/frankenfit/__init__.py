"""
Frankenfit - it's alive! And fit!
"""
from importlib.metadata import (  # noqa: N814
    # import-as with leading _ so that we don't pollute the globals of anyone
    # daring enough to *-import us.
    version as _version,
    PackageNotFoundError as _PNFE,
)

try:
    __version__ = _version("frankenfit")
except _PNFE:
    # package is not installed
    pass

# Names:
# - flitterbop (so glam)
# - fitbop (too random?)
# - fitapply
#       fitapply.io, fitapp.ly
# - fitbits (too similar to fitbit)
# - fapple (too fappy)
# - ply
# - fapply (too fappy)
# - getfit (too similar to getit? too generic?)
# - aplifit

from frankenfit.core import (
    UnknownDatasetError,
    UnresolvedHyperparameterError,
    Dataset,
    PandasDataset,
    DatasetCollection,
    Data,
    Transform,
    FitTransform,
    StatelessTransform,
    HP,
    HPFmtStr,
    HPLambda,
    HPCols,
    columns_field,
)

from frankenfit.transforms import (
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
    SKLearn,
    Statsmodels,
)

from frankenfit.graph import (
    Pipeline,
    IfHyperparamIsTrue,
    IfHyperparamLambda,
    IfTrainingDataHasProperty,
    Join,
)
