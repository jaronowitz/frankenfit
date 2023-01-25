# Copyright (c) 2023 Max Bane <max@thebanes.org>
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# Subject to the terms and conditions of this license, each copyright holder and
# contributor hereby grants to those receiving rights under this license a perpetual,
# worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except for failure to
# satisfy the conditions of this license) patent license to make, have made, use, offer
# to sell, sell, import, and otherwise transfer this software, where such license
# applies only to those patent claims, already acquired or hereafter acquired,
# licensable by such copyright holder or contributor that are necessarily infringed by:
#
# (a) their Contribution(s) (the licensed copyrights of copyright holders and
# non-copyrightable additions of contributors, in source or binary form) alone; or
#
# (b) combination of their Contribution(s) with the work of authorship to which such
# Contribution(s) was added by such copyright holder or contributor, if, at the time the
# Contribution is added, such addition causes such combination to be necessarily
# infringed. The patent license shall not apply to any other combinations which include
# the Contribution.
#
# Except as expressly stated above, no rights or licenses from any copyright holder or
# contributor is granted under this license, whether expressly, by implication, estoppel
# or otherwise.
#
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

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
from __future__ import annotations

# import-as with leading _ so that we don't pollute the globals of anyone; daring;
# enough to *-import us.
from importlib.metadata import PackageNotFoundError as _PNFE
from importlib.metadata import version as _version  # noqa: N814

try:
    __version__ = _version("frankenfit")
except _PNFE:  # pragma: no cover
    # package is not installed
    pass

from frankenfit.backend import DaskBackend
from frankenfit.core import (  # Exceptions that users can catch
    Backend,
    Bindings,
    ConstantTransform,
    FitTransform,
    Future,
    LocalBackend,
    NonInitialConstantTransformWarning,
    Pipeline,
    StatelessTransform,
    Transform,
    UnresolvedHyperparameterError,
)
from frankenfit.dataframe import (  # Exceptions that users can catch
    DataFramePipeline,
    DataFrameTransform,
    ReadDataFrame,
    ReadDataset,
    ReadPandasCSV,
    UnfitGroupError,
    fit_group_on_all_other_groups,
    fit_group_on_self,
)
from frankenfit.params import (
    ALL_COLS,
    HP,
    HPCols,
    HPDict,
    HPFmtStr,
    HPLambda,
    columns_field,
    dict_field,
    fmt_str_field,
    optional_columns_field,
    params,
)
from frankenfit.universal import Identity, UniversalPipeline, UniversalTransform
