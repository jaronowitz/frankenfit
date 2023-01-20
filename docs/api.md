# Frankenfit API reference

## `frankenfit` package
```{eval-rst}
.. automodule:: frankenfit
```

------------

## Core classes

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
```

### Universal pipelines

```{eval-rst}
.. autoclass:: frankenfit.universal.UniversalCallChain
    :show-inheritance:
    :members:

.. autoclass:: frankenfit.universal.UniversalPipelineInterface
    :show-inheritance:
    :members:

.. autoclass:: UniversalPipeline
    :show-inheritance:
    :members:
```

### `DataFrame` pipelines

```{eval-rst}
.. autoclass:: frankenfit.dataframe.DataFrameCallChain
    :show-inheritance:
    :members:

.. autoclass:: frankenfit.dataframe.DataFramePipelineInterface
    :show-inheritance:
    :members:

.. autoclass:: DataFramePipeline
    :show-inheritance:
    :members:
```

## Computational backends and futures

## Writing a `Transform` subclass
Foobar.

## Transform library

(universal-api)=
### Universal transforms

The module `frankenfit.universal` contains Frankenfit's built-in library of generically
useful Transforms that make no assumptions about the type or shape of the data to which
they are applied.

```{eval-rst}
.. autoclass:: frankenfit.universal.Identity

.. autoclass:: frankenfit.universal.IfFittingDataHasProperty

.. autoclass:: frankenfit.universal.IfHyperparamIsTrue

.. autoclass:: frankenfit.universal.IfHyperparamLambda

.. autoclass:: frankenfit.universal.ForBindings

.. autoclass:: frankenfit.universal.LogMessage

.. autoclass:: frankenfit.universal.Print

.. autoclass:: frankenfit.universal.StatefulLambda

.. autoclass:: frankenfit.universal.StatelessLambda

.. autoclass:: frankenfit.universal.StateOf
```

(dataframe-api)=
### `DataFrame` transforms

The module `frankenfit.dataframe` provides a library of broadly useful Transforms on 2-D
Pandas DataFrames.

```{eval-rst}
.. autoclass:: frankenfit.dataframe.Affix

.. autoclass:: frankenfit.dataframe.Assign

.. autoclass:: frankenfit.dataframe.Clip

.. autoclass:: frankenfit.dataframe.Copy

.. autoclass:: frankenfit.dataframe.Correlation

.. autoclass:: frankenfit.dataframe.DeMean

.. autoclass:: frankenfit.dataframe.Drop

.. autoclass:: frankenfit.dataframe.Filter

.. autoclass:: frankenfit.dataframe.GroupByCols

.. autoclass:: frankenfit.dataframe.ImputeConstant

.. autoclass:: frankenfit.dataframe.ImputeMean

.. autoclass:: frankenfit.dataframe.Join

.. autoclass:: frankenfit.dataframe.Pipe

.. autoclass:: frankenfit.dataframe.Prefix
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.ReadDataFrame
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.ReadDataset
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.ReadPandasCSV
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.Rename

.. autoclass:: frankenfit.dataframe.Select

.. autoclass:: frankenfit.dataframe.SKLearn

.. autoclass:: frankenfit.dataframe.Statsmodels

.. autoclass:: frankenfit.dataframe.Suffix
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.Winsorize

.. autoclass:: frankenfit.dataframe.WriteDataset
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.WritePandasCSV
    :show-inheritance:

.. autoclass:: frankenfit.dataframe.ZScore

```
