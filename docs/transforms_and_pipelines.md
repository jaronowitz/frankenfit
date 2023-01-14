---
jupytext:
  cell_metadata_filter: -all
  formats: notebooks///ipynb,docs///md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: .venv-dev
  language: python
  name: python3
---

# Transforms and pipelines

## Transforms

The basic building blocks of Frankenfit data pipelines are
[`Transforms`](frankenfit.Transform). Conceptually, each Transform represents a
data manipulation that must first be [`fit`](frankenfit.Transform.fit) on some
**fitting data**, yielding some **state**, which the user may then **apply** to transform
some **apply data**.

Frankenfit includes an extensive library of built-in Transforms, and ordinarily one will
create instances of these Transforms by using the so-called "chall-chain API" provided
by [`Pipeline`](frankenfit.Pipeline) objects. For example, a Pipeline (specifically a
`DataFramePipeline`) comprising a [`Winsorize`](frankenfit.dataframe.Winsorize)
Transform followed by a [`DeMean`](frankenfit.dataframe.DeMean) might look like this:

```{code-cell} ipython3
import frankenfit as ff

ff.DataFramePipeline().winsorize(limit=0.01).de_mean();
```

However, it is also possible to instantiate Transform objects directly. For example,
Transforms whose fitting data and apply data are meant to be pandas DataFrames are kept
in the module `frankenfit.dataframe`, and we might instantiate the ``DeMean`` Transform
as follows:

```{code-cell} ipython3
dmn = ff.dataframe.DeMean()
```

Let's load some data:

```{code-cell} ipython3
# Load a dataset of diamond prices and covariates
from pydataset import data
diamonds_df = data("diamonds")[["carat", "depth", "table", "price"]]
diamonds_df.head()
```

```{note}
Throughout the documentation we make use of the
[pydataset](https://pypi.org/project/pydataset/) package for loading example data like
`diamonds`.
```

+++

The `DeMean` Transform instance `dmn` may then be **fit** on the data. By default it
learns to de-mean all columns in the DataFrame.

```{code-cell} ipython3
fit_dmn = dmn.fit(diamonds_df)
```

The [`fit()`](frankenfit.Transform.fit) method returns an instance of `FitTransform`,
which encapsulates the **state** that was learned on the fitting data, and which may be
**applied** to a dataset by calling its `apply()` method.

```{code-cell} ipython3
fit_dmn.apply(diamonds_df).head()
```

In the case of [`DeMean`](frankenfit.dataframe.DeMean), the state consists of the means
of the columns observed in the fitting data. Of course, other Transforms will have
totally different kinds of state (e.g., the state of a fit
[`Winsorize`](frankenfit.dataframe.Winsorize) Transform is the values of the outlier
quantiles observed in the fitting data), and some Transforms may have no state at all
(for example `ImputeConstant` replaces missing values with a constant that is
independent of any fitting data; see `StatelessTransform`).

One may query the the state of a `FitTransform` by calling its `state()` method. The
exact type and value of the state is an implementation detail of the Transform in
question, and in the case of `DeMean` we can see that its state is a pandas `Series` of
means, indexed by column name:

```{code-cell} ipython3
fit_dmn.state()
```

Crucially, the fitting data and apply data need not be the same. For example, we might
de-mean the dataset with respect to the means observed on some subsample of it:

```{code-cell} ipython3
dmn.fit(diamonds_df.sample(100)).apply(diamonds_df).head()
```

Or we might divide the data into disjoint "training" and "test" sets, feeding the former
to `fit()` and the latter to `apply()`. We call this an **out-of-sample** application of
the Transform.

```{code-cell} ipython3
train_df = diamonds_df.sample(frac=0.5)
test_df = diamonds_df.loc[list(set(diamonds_df.index) - set(train_df.index))]

dmn.fit(train_df).apply(test_df).head()
```

### Parameters

+++

### Abstract descriptions and immutability

Transform instances like `dmn` are best thought of as light-weight, abstract, immutable
descriptions of what to do to some unspecified data; they store no data or state in and
of themselves. They are essentially factories for producing `FitTransform` instances by
feeding data to their `fit()` methods, and it's those `FitTransform` instances which
hold the (possibly heavyweight) state of the now-fit Transform, and are actually capable
of transforming data through their `apply()` methods.

Instances of `Transform` and `FitTransform` are both immutable and re-usable.

* A `Transform` instance is a frozen description of a transformation, with fixed
  parameter values (although the use of hyperparameters allows deferring the resolution
  of some or all parameter values until the moment that `fit()` is called; see
  [Hyperparams]). It is re-usable in the sense that a single `Transform` instance may be
  `fit()` on many different datasets, each time returning a new instance of
  `FitTransform`, and never modifying the fitting data in the process.
* A `FitTransform` instance has some state which, once constructed by `Transform.fit()`,
  is fixed and immutable. The instance may be re-used by calling `apply()` on many
  different datasets. The `apply()` method never modifies the data that it is given;
  it returns a copy of the data in which the fit transformation has been applied.

```{note}
It's worth noting that nothing formally prevents a rogue `Transform` implementation from
modifying the fitting data or apply data. This is Python, after all. Immutability is
merely a convention to be followed when implementing a new `Transform`.

Furthermore, once the user has an instance of `Transform` or `FitTransform` in hand,
nothing truly prevents him from modifying its parameters or state. This should be
avoided except for a few extraordinary circumstances (e.g. making a modified copy of a
`Transform` whose type is not known at runtime), and in any case, the Pipeline
call-chain API, which is preferred over direct instantiation of `Transform`s, makes it
difficult to do so.
```

+++

### Transform objects

### FitTransform objects

## Pipelines

```{code-cell} ipython3
import pandas as pd
import frankenfit as ff

@ff.params
class DeMean(ff.Transform):
    """
    De-mean some columns.

    Parameters
    ----------
    cols : list(str)
        The names of the columns to de-mean.
    """
    cols: list[str]

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        # return state as a pandas Series of the columns' means in data_fit, indexed by
        # column name
        return data_fit[self.cols].mean()

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
        # return a new DataFrame in which the columns have been demeaned with respect to
        # the provided state
        return data_apply.assign(
            **{c: data_apply[c] - state[c] for c in self.cols}
        )
```

As authors of a Transform, in most cases we must

```{code-cell} ipython3
dmn  = DeMean(["price", "carat"])
dmn
```

```{code-cell} ipython3
diamonds_df[["price", "carat"]].mean()
```

```{code-cell} ipython3
dmn.visualize()
```

When your book is built, the contents of any `{code-cell}` blocks will be
executed with your default Jupyter kernel, and their outputs will be displayed
in-line with the rest of your content.

+++

Glossary:

* fitting data, training data
* fit-time
* apply data
* apply-time
* data in, data result
* state
