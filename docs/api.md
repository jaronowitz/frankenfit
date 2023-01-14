# Frankenfit API reference

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

..
    # .. autoclass:: FitTransform
    #     :members:
    #
    # .. autoclass:: StatelessTransform
    #     :show-inheritance:
    #     :members:
```

## Writing a `Transform` subclass
Foobar.

## Hyperparameters
Foobar.

```{eval-rst}
..
    # .. autoclass:: HP
    #     :members:
    #
    # .. autoclass:: HPFmtStr
    #     :show-inheritance:
    #     :members:
    #
    # .. autofunction:: fmt_str_field
    #
    # .. autoclass:: HPCols
    #     :show-inheritance:
    #     :members:
    #
    # .. autofunction:: columns_field
    #
    # .. autoclass:: HPDict
    #     :show-inheritance:
    #     :members:
    #
    # .. autofunction:: dict_field
    #
    # .. autoclass:: HPLambda
    #     :show-inheritance:
    #     :members:
    #
    # .. autoexception:: UnresolvedHyperparameterError
    #     :show-inheritance:
```

## Pipelines
Foobar.

```{eval-rst}
.. autoclass:: Pipeline
    :show-inheritance:
    :members:
    :inherited-members:

..
    # .. autoclass:: BasePipeline
    #     :show-inheritance:
    #     :members:
    #     :inherited-members:
    #
    # .. autoclass:: frankenfit.universal.IfHyperparamIsTrue
    #     :show-inheritance:
    #     :members:
    #     :exclude-members: hyperparams
    #
    # .. autoclass:: frankenfit.universal.IfHyperparamLambda
    #     :show-inheritance:
    #     :members:
    #     :exclude-members: hyperparams
    #
    # .. autoclass:: frankenfit.universal.IfFittingDataHasProperty
    #     :show-inheritance:
    #     :members:
    #     :exclude-members: hyperparams
    #
    # .. autoclass:: frankenfit.universal.ForBindings
    #     :show-inheritance:
    #     :members:
```

## Computational backends and futures

## Transform library

### DataFrame Transforms

```{eval-rst}
.. autoclass:: frankenfit.dataframe.DeMean

.. autoclass:: frankenfit.dataframe.Winsorize

..
    # .. autoclass:: Identity
    #     :show-inheritance:
    #
    # .. autoclass:: Select
    #     :show-inheritance:
    #
    # .. autoclass:: Drop
    #     :show-inheritance:
    #
    # .. autoclass:: Copy
    #     :show-inheritance:
    #
    # .. autoclass:: Rename
    #     :show-inheritance:
    #
    # .. autoclass:: SKLearn
    #     :show-inheritance:
    #
    # .. autoclass:: Print
    #     :show-inheritance:
    #
    # .. autoclass:: LogMessage
    #     :show-inheritance:
    #
    # .. autoclass:: Join
    #     :show-inheritance:
    #     :members:
    #
    # .. autoclass:: Correlation
    #     :show-inheritance:
```
