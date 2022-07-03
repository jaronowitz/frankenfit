# Frankenfit API reference

## Copyright notice

```{warning}
Unpublished Copyright (c) 2022 Max Bane, all rights reserved.

NOTICE: All information contained herein is, and remains the property of Max Bane.
The intellectual and technical concepts contained herein are proprietary to Max Bane
and may be covered by U.S. and Foreign Patents, patents in process, and are protected
by trade secret or copyright law. Dissemination of this information or reproduction
of this material is strictly forbidden unless prior written permission is obtained
from Max Bane. Access to the source code contained or referenced herein is hereby
forbidden to anyone except current employees, contractors, or customers of Max Bane
who have executed Confidentiality and Non-disclosure agreements explicitly covering
such access.

The copyright notice above does not evidence any actual or intended publication or
disclosure of this source code, which includes information that is confidential
and/or proprietary, and is a trade secret, of Max Bane. ANY REPRODUCTION,
MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE
OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF MAX BANE IS STRICTLY
PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE
RECEIPT OR POSSESSION OF THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY
OR IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO
MANUFACTURE, USE, OR SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.
```

## `frankenfit` package
```{eval-rst}
.. automodule:: frankenfit
```

------------

## Core `Transform` base classes

```{eval-rst}
.. autoclass:: Transform
    :members:
    :private-members: _fit, _apply

.. autoclass:: FitTransform
    :members:

.. autoclass:: StatelessTransform
    :show-inheritance:
    :members:
    :exclude-members: FitStatelessTransform
```

## Writing a `Transform` subclass
Foobar.

## Datasets and dataset collections

```{eval-rst}
.. autoclass:: Dataset
    :members:

.. autoclass:: PandasDataset
    :show-inheritance:
    :members:

.. autoclass:: DatasetCollection
    :members:

.. autoexception:: UnknownDatasetError
    :show-inheritance:
```

## Hyperparameters
Foobar.

```{eval-rst}
.. autoclass:: HP
    :members:

.. autoclass:: HPFmtStr
    :show-inheritance:
    :members:

.. autofunction:: fmt_str_field

.. autoclass:: HPCols
    :show-inheritance:
    :members:

.. autofunction:: columns_field

.. autoclass:: HPDict
    :show-inheritance:
    :members:

.. autofunction:: dict_field

.. autoclass:: HPLambda
    :show-inheritance:
    :members:

.. autoexception:: UnresolvedHyperparameterError
    :show-inheritance:
```

## Pipelines
Foobar.

```{eval-rst}
.. autoclass:: Pipeline
    :show-inheritance:
    :members:
    :exclude-members: hyperparams, FitPipeline

.. autoclass:: Join
    :show-inheritance:
    :members:
    :exclude-members: hyperparams, FitJoin

.. autoclass:: IfHyperparamIsTrue
    :show-inheritance:
    :members:
    :exclude-members: hyperparams, FitIfHyperparamIsTrue

.. autoclass:: IfHyperparamLambda
    :show-inheritance:
    :members:
    :exclude-members: hyperparams, FitHyperparamLambda

.. autoclass:: IfTrainingDataHasProperty
    :show-inheritance:
    :members:
    :exclude-members: hyperparams, FitIfTrainingDataHasPorperty
```

## Transform library
Foobar.
