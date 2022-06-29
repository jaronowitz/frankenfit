"""
Fitapply - Get fit and apply yourself!
"""
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

from ard_fit_pipelines.core import (
    UnknownDatasetError,
    UnresolvedHyperparameterError,
    Dataset,
    PandasDataset,
    DatasetCollection,
    Data,
    data_to_dataframe,
    Transform,
    FitTransform,
    StatelessTransform,
    HP,
    HPFmtStr,
    HPLambda,
    HPCols,
    columns_field,
)

from ard_fit_pipelines.transforms import (
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

from ard_fit_pipelines.graph import (
    Pipeline,
    IfHyperparamIsTrue,
    IfHyperparamLambda,
    IfTrainingDataHasProperty,
)
