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

```{code-cell} ipython3
:tags: [remove-cell]
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
```

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
directly as follows:

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

Most Transforms have some associated **parameters** that control their behavior. The
values of these parameters are supplied by the user when constructing the Transform
(but as we'll cover in more detail later, the values may be "hyperparameters" with
deferred evaluation; see [Hyperparams]). Parameters may be required or optional,
typically with some reasonable default value in the latter case.

For example, the [`DeMean`](frankenfit.dataframe.DeMean) Transform that we've been using
above has two optional parameters:

* `cols`: A list of the names of the columns to de-mean; by default, all
  columns are de-meaned.

* `w_col`: The name of a column to use as a source of observation weights when
  computing the means; by default, the means are unweighted.

Therefore we can define a `DeMean` Transform that only de-means the `price` and `table` columns of the data, or one which de-means `price` with respect to its `carat`-weighted mean:

```{code-cell} ipython3
dmn_2cols = ff.dataframe.DeMean(["price", "table"])
dmn_2cols.fit(train_df).apply(test_df).head()
```

```{code-cell} ipython3
dmn_price_weighted = ff.dataframe.DeMean(["price"], w_col="carat")
dmn_price_weighted.fit(train_df).apply(test_df).head()
```

```{tip}
Note that parameters have an order and can generally be specified positionally or by
name. So for example `DeMean(["price", "table"])` could also be written as
`DeMean(cols=["price", "table"])`, and `DeMean(["price"], w_col="carat")` could be written as
`DeMean(["price"], "carat")` or `DeMean(cols=["price"], w_col="carat")`.
```

[`Winsorize`](frankenfit.dataframe.Winsorize) is an example of a Transform with a
required parameter, `limit`, which specifies the threshold at which extreme values
should be trimmed. It also accepts an optional `cols` parameter, like `DeMean`.

E.g., winsorizing the top and bottom 1% of values in all columns:

```python
ff.dataframe.Winsorize(limit=0.01)
```

Winsorizing just the `price` column's top and bottom 5% of values:

```python
ff.dataframe.Winsorize(limit=0.01, cols=["price"])
```

```{tip}
`DeMean` and `Winsorize` are part of a larger family of DataFrame Transforms that accept
a `cols` parameter. Others include `ZScore`, `Select`, `Drop`, `ImputeConstant`, and
many more. As a notational convenience, all of these Transforms allow `cols` to be given
as a single string, rather than a list of strings, in the case that the user wants the
Transform to apply to a single column. Under the hood, this is converted to a length-1
list. So our previous example could also be written most succinctly as `Winsorize(0.01,
"price")`.

Furthermore, all of the Transforms in this family follow the convention that omitting
the `cols` argument indicates that the Transform should be applied to all columns in the
data.

When implementing one's own bespoke Transforms on DataFrames, it is possible to get this
same behavior by using the [`columns_field`](frankenfit.params.columns_field)
field-specifier; see TODO.
```

+++

Once constructed, Transform instances carry their parameters as attributes:

```{code-cell} ipython3
dmn_2cols.cols
```

```{code-cell} ipython3
dmn_2cols.w_col is None
```

It is also possible to retrieve the names of a Transform's parameters by calling the
[`params()`](frankenfit.Transform.params) method:

```{code-cell} ipython3
dmn_2cols.params()
```

The `repr` of a Transform instance additionally shows the values of its parameters:

```{code-cell} ipython3
display(dmn_2cols)
display(ff.dataframe.Winsorize(0.01, "price"))
```

The observant reader doubtlessly noticed the presence of a parameter named `"tag"` in
the examples above. This is a special, implicit parameter common to all Transforms. For
now we need only note its existence, and that it automatically receives a value, which
may be overridden by the `tag` keyword-only argument available to all Transforms, e.g.:

```{code-cell} ipython3
win_price = ff.dataframe.Winsorize(0.01, "price", tag="winsorize_price")
win_price
```

Every Transform has a `name` attribute that incorporates its class name and `tag`:

```{code-cell} ipython3
win_price.name
```

```{code-cell} ipython3
dmn_2cols.name
```

This will come in handy later when we wish to refer to specific Transforms embedded in
larger pipelines, as described in [Tagging and selecting Transforms].

```{important}
While `tag` is a parameter, whose value may optionally be supplied when creating a Transform, `name` is *not* a parameter, and cannot be set directly. It's just an attribute whose value is automatically derived from the Transform's class name and `tag`.
```

+++

### Abstract descriptions and immutability

Transform instances like `dmn` are best thought of as light-weight, abstract, immutable
descriptions of what to do to some as-yet unspecified data; they store no data or state
in and of themselves. They are essentially factories for producing `FitTransform`
instances by feeding data to their `fit()` methods, and it's those `FitTransform`
instances which hold the (possibly heavyweight) state of the now-fit Transform, and are
actually capable of transforming data through their `apply()` methods.

Instances of `Transform` and `FitTransform` are both immutable and re-usable:

* A `Transform` instance is an immutable description of a transformation, with fixed
  parameter values provided at the time of instantiation (although the use of
  hyperparameters allows deferring the resolution of some or all parameter values until
  the moment that `fit()` is called; see [Hyperparams]). It is re-usable in the sense
  that a single `Transform` instance may be `fit()` on many different datasets, each
  time returning a new instance of `FitTransform`, and never modifying the fitting data
  in the process.
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

## Pipelines

When modeling or analyzing some dataset, one usually wishes to **compose** many
Transforms. For example, consider the dataset of diamond prices and covariates:

```{code-cell} ipython3
diamonds_df.head()
```

Suppose we want to build a model that predicts diamond prices as a function of their
weight (`carat`), pavilion `depth` (how "tall" the diamond is), and `table` diameter
(how wide the diamond's uppermost facet is; see [this figure](https://en.wikipedia.org/wiki/Diamond_cut#/media/File:Diamond_facets.svg) on wikipedia).

```{code-cell} ipython3
diamonds_df.hist()
diamonds_df.describe()
```

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

As authors of a Transform, in most cases we must implement `_fit` and `_apply` methods.

```{code-cell} ipython3
dmn.visualize()
```

Glossary:

* fitting data, training data
* fit-time
* apply data
* apply-time
* data in, data result
* state
