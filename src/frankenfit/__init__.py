# Unpublished Copyright (c) 2022 Max Bane, all rights reserved.
#
# NOTICE: All information contained herein is, and remains the property of Max Bane.
# The intellectual and technical concepts contained herein are proprietary to Max Bane
# and may be covered by U.S. and Foreign Patents, patents in process, and are protected
# by trade secret or copyright law. Dissemination of this information or reproduction
# of this material is strictly forbidden unless prior written permission is obtained
# from Max Bane. Access to the source code contained herein is hereby forbidden to
# anyone except current employees, contractors, or customers of Max Bane who have
# executed Confidentiality and Non-disclosure agreements explicitly covering such
# access.
#
# The copyright notice above does not evidence any actual or intended publication or
# disclosure of this source code, which includes information that is confidential
# and/or proprietary, and is a trade secret, of Max Bane. ANY REPRODUCTION,
# MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE
# OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF MAX BANE IS STRICTLY
# PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE
# RECEIPT OR POSSESSION OF THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY
# OR IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, NOR TO
# MANUFACTURE, USE, OR SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.

"""
The public Frankenfit API consists of all non-underscore-prefixed names in the
top-level ``frankenfit`` package. Therefore a single import statement pulls in the
complete API::

    import frankenfit

.. TIP::

    As a stylistic convention, and for the sake of brevity, the author of Frankenfit
    recommends importing ``frankenfit`` with the short name ``ff``::

        import frankenfit as ff

    All of the examples in the reference documentation assume that ``frankenfit`` has
    been imported with the short name ``ff`` as above.

In case you use a star-import (:code:`from frankenfit import *`), care is taken to
ensure that all and only the public API names are imported, so that your namespace is
not polluted with unrelated names.
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

from frankenfit.backend import (
    Backend,
    DummyBackend,
    DaskBackend,
)

from frankenfit.params import (
    params,
    HP,
    HPFmtStr,
    HPLambda,
    HPDict,
    HPCols,
    fmt_str_field,
    dict_field,
    columns_field,
    optional_columns_field,
)

from frankenfit.core import (
    Bindings,
    Transform,
    FitTransform,
    StatelessTransform,
    ConstantTransform,
    BasePipeline,
    # Exceptions that users can catch
    UnresolvedHyperparameterError,
    NonInitialConstantTransformWarning,
)

from frankenfit.universal import (
    UniversalTransform,
    UniversalPipeline,
    UniversalPipelineInterface,
    UniversalGrouper,
    Identity,
)

from frankenfit.dataframe import (
    DataFrameTransform,
    DataFramePipeline,
    DataFramePipelineInterface,
    DataFrameGrouper,
    ReadDataFrame,
    ReadPandasCSV,
    ReadDataset,
    fit_group_on_self,
    fit_group_on_all_other_groups,
    # Exceptions that users can catch
    UnfitGroupError,
)
