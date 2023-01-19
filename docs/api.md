# Frankenfit API reference

## `frankenfit` package
```{eval-rst}
.. automodule:: frankenfit
```

------------

## Core base classes

```{eval-rst}
.. autoclass:: Transform
    :members:
    :private-members: _fit, _apply

.. autodata:: frankenfit.core.DEFAULT_VISUALIZE_DIGRAPH_KWARGS

.. autoclass:: FitTransform
    :members:

.. autoclass:: StatelessTransform
    :members:
    :show-inheritance:

.. autoclass:: ConstantTransform
    :members:
    :show-inheritance:

.. autoclass:: NonInitialConstantTransformWarning
    :show-inheritance:
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

..
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

(universal-api)=
### Universal Transforms

The module `frankenfit.universal` contains Frankenfit's built-in library of generically
useful Transforms that make no assumptions about the type or shape of the data to which
they are applied.

```{eval-rst}
.. autoclass:: frankenfit.universal.StatelessLambda
```

(dataframe-api)=
### DataFrame Transforms

The module `frankenfit.dataframe` provides a library of broadly useful Transforms on 2-D
Pandas DataFrames.

```{eval-rst}
.. autoclass:: frankenfit.dataframe.Assign

.. autoclass:: frankenfit.dataframe.Clip

.. autoclass:: frankenfit.dataframe.Copy

.. autoclass:: frankenfit.dataframe.Correlation

.. autoclass:: frankenfit.dataframe.DeMean

.. autoclass:: frankenfit.dataframe.Pipe

.. autoclass:: frankenfit.dataframe.SKLearn

.. autoclass:: frankenfit.dataframe.Winsorize

.. autoclass:: frankenfit.dataframe.ZScore

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
