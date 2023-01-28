---
jupytext:
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

# FIXME: this cell should not appear in docs output
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
```

# Transforms and pipelines

## Transforms

The basic building blocks of Frankenfit data pipelines are
**[`Transforms`](frankenfit.Transform).** Conceptually, each Transform represents a data
manipulation that must first be **[`fit`](frankenfit.Transform.fit)** on some
**fitting data**, yielding some **state**, which the user may then
**[`apply`](frankenfit.FitTransform.apply)** to transform some **apply data**.

Frankenfit includes an extensive library of built-in Transforms, and ordinarily one will
create instances of these Transforms by using the so-called ["chall-chain
API"](call-chain-api) provided by [`Pipeline`](frankenfit.Pipeline) objects. For
example, a Pipeline (specifically a [`DataFramePipeline`](frankenfit.DataFramePipeline))
comprising a [`Winsorize`](frankenfit.dataframe.Winsorize) Transform followed by a
[`DeMean`](frankenfit.dataframe.DeMean) might look like this:

```{code-cell} ipython3
import frankenfit as ff

ff.DataFramePipeline().winsorize(limit=0.01).de_mean();
```

However, it is also possible to instantiate Transform objects directly. For example,
Transforms whose fitting data and apply data are meant to be pandas DataFrames are kept
in the module [`frankenfit.dataframe`](dataframe-api), and we might instantiate the
``DeMean`` Transform directly as follows:

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

The `DeMean` Transform instance `dmn` may then be **fit** on the data. By default it
learns to de-mean all columns in the DataFrame.

```{code-cell} ipython3
fit_dmn = dmn.fit(diamonds_df)
```

The [`fit()`](frankenfit.Transform.fit) method returns an instance of
[`FitTransform`](frankenfit.FitTransform), which encapsulates the **state** that was
learned on the fitting data, and which may be **applied** to a dataset by calling its
[`apply()`](frankenfit.FitTransform.apply) method.

```{code-cell} ipython3
fit_dmn.apply(diamonds_df).head()
```

In the case of [`DeMean`](frankenfit.dataframe.DeMean), the state consists of the means
of the columns observed in the fitting data. When applied, it subtracts these means from
the corresponding columns of the apply data, returning the result. Of course, other
Transforms will have totally different kinds of state (e.g., the state of a fit
[`Winsorize`](frankenfit.dataframe.Winsorize) Transform is the values of the outlier
quantiles observed in the fitting data), and some Transforms may have no state at all
(for example [`ImputeConstant`](frankenfit.dataframe.ImputeConstant) replaces missing
values with a constant that is independent of any fitting data; see [Stateless
transforms](stateless-transforms)).

One may query the the state of a `FitTransform` by calling its
[`state()`](frankenfit.FitTransform.state) method. The exact type and value of the state
is an implementation detail of the Transform in question, and in the case of `DeMean` we
can see that its state is a pandas `Series` of means, indexed by column name:

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
train_df = diamonds_df.sample(frac=0.5, random_state=1337420)
test_df = diamonds_df.loc[list(set(diamonds_df.index) - set(train_df.index))]

dmn.fit(train_df).apply(test_df).head()
```

### Parameters

Most Transforms have some associated **parameters** that control their behavior. The
values of these parameters are supplied by the user when constructing the Transform
(but as we'll cover in more detail later, the values may be "hyperparameters" with
deferred evaluation; see {doc}`hyperparams`). Parameters may be required or optional,
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

:::{tip}
Note that parameters have an order and can generally be specified positionally or by
name. So for example `DeMean(["price", "table"])` could also be written as
`DeMean(cols=["price", "table"])`, and `DeMean(["price"], w_col="carat")` could be written as
`DeMean(["price"], "carat")` or `DeMean(cols=["price"], w_col="carat")`.
:::

[`Winsorize`](frankenfit.dataframe.Winsorize) is an example of a Transform with a
required parameter, `limit`, which specifies the threshold at which extreme values
should be trimmed. It also accepts an optional `cols` parameter, like `DeMean`.

E.g., winsorizing the top and bottom 1% of values in all columns:

```python
ff.dataframe.Winsorize(limit=0.01)
```

Winsorizing just the `price` column's top and bottom 5% of values:

```python
ff.dataframe.Winsorize(limit=0.05, cols=["price"])
```

:::{tip}
`DeMean` and `Winsorize` are part of a larger family of [DataFrame
Transforms](dataframe-api) that accept a `cols` parameter. Others include
[`ZScore`](frankenfit.dataframe.ZScore), [`Select`](frankenfit.dataframe.Select),
[`Drop`](frankenfit.dataframe.Drop),
[`ImputeConstant`](frankenfit.dataframe.ImputeConstant), and [many more](dataframe-api).
As a notational convenience, all of these Transforms allow `cols` to be given as a
single string, rather than a list of strings, in the case that the user wants the
Transform to apply to a single column.  Under the hood, this is converted to a length-1
list. So our previous example could also be written most succinctly as `Winsorize(0.05,
"price")`.

Furthermore, all of the Transforms in this family follow the convention that omitting
the `cols` argument indicates that the Transform should be applied to all columns in the
data, unless otherwise noted in the [API reference](api-reference).

When implementing one's own bespoke Transforms on DataFrames, it is possible to get this
same behavior by using the [`columns_field`](frankenfit.columns_field) field-specifier;
see {doc}`implementing_transforms`.
:::

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
display(
    dmn_2cols,
    ff.dataframe.Winsorize(0.01, "price"),
)
```

(transform-tags)=
#### Tags

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
larger pipelines, as described in [Tagging and selecting
Transforms](tagging-selecting-transforms).

:::{important}
While `tag` is a parameter, whose value may optionally be supplied when creating a
Transform, `name` is *not* a parameter, and cannot be set directly. It's just a
read-only attribute whose value is automatically derived from the Transform's class name
and `tag`.
:::

+++

### Abstract descriptions and immutability

Transform instances like `dmn` are best thought of as light-weight, abstract, immutable
descriptions of what to do to some as-yet unspecified data; they store no data or state
in and of themselves. They are essentially factories for producing
[`FitTransform`](frankenfit.FitTransform) instances by feeding data to their
[`fit()`](frankenfit.Transform.fit) methods, and it's those `FitTransform` instances
which hold the (possibly heavyweight) state of the now-fit Transform, and are actually
capable of transforming data through their [`apply()`](frankenfit.FitTransform.apply)
methods.

Instances of `Transform` and `FitTransform` are both immutable and re-usable:

* A `Transform` instance is an immutable description of a transformation, with fixed
  parameter values provided at the time of instantiation (although the use of
  hyperparameters allows deferring the resolution of some or all parameter values until
  the moment that `fit()` is called; see {doc}`hyperparams`). It is re-usable in the
  sense that a single `Transform` instance may be `fit()` on many different datasets,
  each time returning a new instance of `FitTransform`, never modifying its parameters
  or the fitting data in the process.
* A `FitTransform` instance has some state which, once constructed by `Transform.fit()`,
  is fixed and immutable. The instance may be re-used by calling `apply()` on many
  different datasets. The `apply()` method never modifies the state nor the data that it
  is given; it always returns a copy of the data in which the fit transformation has
  been applied.

:::{note}
It's worth noting that nothing formally prevents a rogue `Transform` implementation from
modifying the fitting data or apply data. This is Python, after all. Immutability is
merely a convention to be followed when implementing a new `Transform`.

Furthermore, once the user has an instance of `Transform` or `FitTransform` in hand,
nothing truly prevents him from modifying its parameters or state. This should be
avoided except for a few extraordinary circumstances (e.g. making a modified copy of a
`Transform` whose type is not known at runtime), and in any case, the Pipeline
[call-chain API](call-chain-api), which is preferred over direct instantiation of
`Transform` objects, makes it inconvenient to do so.
:::

(stateless-transforms)=
### Stateless transforms

A number of [`Transforms`](frankenfit.Transform) don't actually have any state to fit;
Frankenfit calls these [`StatelessTransforms`](frankenfit.StatelessTransform). For these
Transforms, the [`fit()`](frankenfit.Transform.fit) method is essentially a null
operation, and by convention the [`state()`](frankenfit.FitTransform.state) of the
resulting [`FitTransform`](frankenfit.FitTransform) is `None`. All of the action is in
[`apply()`](frankenfit.FitTransform.apply).

For example, suppose we just want to log-transform some column like `price`; that is,
replace each value $x$ with $\log(1+x)$. (The $1$ is there to ensure that we don't try
to compute $\log(0)$.) This is an operation that always gives the same result regardless
of what might be observed on any fitting data, hence stateless. We could of course
accomplish this directly with numpy and pandas:

```{code-cell} ipython3
import numpy as np

np.log1p(diamonds_df[["price"]]).head()
```

But as we'll see in the next section, it is often useful to be able to represent such
stateless data manipulations as Frankenfit `Transforms`, so that we can compose them
easily with other (possibly stateful) Transforms in our data pipelines, and generally
work with a common API for invoking all of our data manipulations.

The Frankenfit library provides a few workhorse `StatelessTransforms` that can be used
to wrap simple stateless operations like `np.log1p` above:

* [`Pipe`](frankenfit.dataframe.Pipe) lets us easily "pipe" (the specified
  columns of) a `DataFrame` through an arbitrary function, as long as it accepts and
  returns a `DataFrame`:

```{code-cell} ipython3
log_price = ff.dataframe.Pipe(np.log1p, cols="price")
```

* [`StatelessLambda`](frankenfit.universal.StatelessLambda) is even more
  generic. It passes the apply-time data (which may or may not be a `DataFrame`)
  directly to a user-supplied function, returning the result.

```{code-cell} ipython3
log_price_lambda = ff.universal.StatelessLambda(
    lambda df: df.assign(price=np.log1p(df["price"]))
)
```

:::{note}
The reader no doubt noticed that
[`StatelessLambda`](frankenfit.universal.StatelessLambda) is kept under
[`frankenfit.universal`](universal-api), rather than under
[`frankenfit.dataframe`](dataframe-api) like the other Transforms introduced so far. The
distinction is that `frankenfit.dataframe` is for Transforms that operate on DataFrames,
while `frankenfit.universal` is for Transforms that make no assumption about the type of
the data.
:::

To illustrate, `log_price` can be fit (always with `None` state) and applied like any
other Transform:

```{code-cell} ipython3
log_price_fit = log_price.fit(diamonds_df)
display(log_price_fit.state())
```

```{code-cell} ipython3
log_price_fit.apply(diamonds_df).head()
```

Because `fit()` is a null operation, the fitting data argument is optional:

```{code-cell} ipython3
log_price.fit().apply(diamonds_df).head()
```

In fact, as a convenience, the `fit()` call can be skipped entirely:

```{code-cell} ipython3
log_price.apply(diamonds_df).head()
```

All `StatlessTransforms` inherit an [`apply()`](frankenfit.StatelessTransform.apply)
method, which is syntactic sugar for ``.fit(...).apply(...)``.

:::{important}
It's important to remember that *stateful* Transforms like
[`DeMean`](frankenfit.dataframe.DeMean) and
[`Winsorize`](frankenfit.dataframe.Winsorize) have no ``apply()`` method! They must
first be `fit()`, which returns a [`FitTransform`](frankenfit.FitTransform) instance,
and it's the `FitTransform` that has [`apply()`](frankenfit.FitTransform.apply).
Stateless Transforms can be fit and applied in the same manner, and
[`StatelessTransform.apply()`](frankenfit.StatelessTransform.apply) is merely a convenience.

Furthermore, the signature of `StatelessTransform.apply()` is a little different than
that of `FitTransform.apply()`, since it allows the specification of hyperparameter
bindings, which can only be given at fit-time.
:::

+++

## Pipelines

### Composing transforms

When modeling or analyzing some dataset, one usually wishes to **compose** many
Transforms. For example, consider the dataset of diamond prices and covariates:

```{code-cell} ipython3
diamonds_df.head()
```

Suppose we want to build a model that predicts diamond prices as a function of their
weight (`carat`), pavilion `depth` (how "tall" the diamond is), and `table` diameter
(how wide the diamond's uppermost facet is; see [this
figure](https://en.wikipedia.org/wiki/Diamond_cut#/media/File:Diamond_facets.svg) on
wikipedia).

To do so, we can imagine fitting a simple linear regression model on these variables.
But first we note that these variables have very different ranges and scales from each
other, as well as outliers:

```{code-cell} ipython3
# Recall that train_df is a random sample of half the observations in diamonds_df
train_df.hist()
train_df.describe()
```

Therefore in practice we'll want to apply several feature-cleaning transformations to
the data before fitting a regression model. Specifically, let's suppose we want to:

1. Winsorize all four variables to trim outliers.
2. Log-transform the `carat` and `price` variables to make them more symmetric.
3. Z-score the three predictor variables to put them on the same scale with zero means.
   (It's important to do this after the previous steps, so that the means and standard
   deviations used for the z-scores are not distorted by outliers.)
4. Fit a linear regression of `price` predicted by `carat`, `table`, and `depth`.
5. Finally, because we log-transformed `price`, exponentiate the predictions of the
   regression model to put them back in the original units.

The [`frankenfit.dataframe`](dataframe-api) module provides Transforms for all of these
operations ([`Winsorize`](frankenfit.dataframe.Winsorize),
[`Pipe`](frankenfit.dataframe.Pipe) [`ZScore`](frankenfit.dataframe.ZScore),
[`SKLearn`](frankenfit.dataframe.SKLearn)), and naively we might manually combine them
to implement our end-to-end model. For example, we could instantiate our transforms like
so:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

# 1. Winsorize all four variables
winsorize = ff.dataframe.Winsorize(0.05)
# 2. Log-transform carat and price
log_price_carat = ff.dataframe.Pipe(np.log1p, ["price", "carat"])
# 3. Z-score the three predictor variables
z_score = ff.dataframe.ZScore(["carat", "table", "depth"])
# 4. Linear regression
regress = ff.dataframe.SKLearn(
    sklearn_class=LinearRegression,
    x_cols=["carat", "table", "depth"],
    response_col="price",
    hat_col="price_hat",
    class_params={"fit_intercept": True}  # additional arguments for LinearRegression
)
# 5. Exponentiate the predictions back to original units
exp_price_hat = ff.dataframe.Pipe(np.expm1, "price_hat")
```

And then, whenever we want to fit our model on some fitting data, we go through a
procedure like that below, where each Transform is fit on the result of fitting and
applying the previous transform to the data:

```{code-cell} ipython3
# start with train_df as the input data
winsorize_fit = winsorize.fit(train_df)
df = winsorize_fit.apply(train_df)

# remember that Pipe is stateless, so we could skip the explicit call to fit(), but
# we'll use it here for symmetry with the other Transforms
log_price_carat_fit = log_price_carat.fit(df)
df = log_price_carat_fit.apply(df)

z_score_fit = z_score.fit(df)
df = z_score_fit.apply(df)

regress_fit = regress.fit(df)
df = regress_fit.apply(df)

exp_price_hat_fit = exp_price_hat.fit(df)
df = exp_price_hat_fit.apply(df)

df.head()
```

At the end of this process, we have a bunch of `FitTransform` instances `winsorize_fit`,
`log_price_cara_fit`, `z_score_fit`, `regress_fit`, `exp_price_hat_fit`, as well as the
DataFrame `df`, which contains the results of applying our whole model to its own
fitting data.

Incidentally, we can see that the model does a reasonable job of predicting its own
fitting data, with a high correlation between `price_hat_dollars` and the original,
un-transformed `price`, though there is clearly some non-random structure to the errors:

```{code-cell} ipython3
eval_df = (
    train_df[["price"]]
    .assign(price_hat=df["price_hat"])
)
eval_df.plot.scatter(x="price_hat", y="price", alpha=0.3)
eval_df.hist(figsize=(5,2))
eval_df.corr()
```

Even more incidentally, the `state()` of `regress_fit` is just a (fit) scikit-learn
[`LinearRegression`](
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
) object, so if we are interested in the betas learned for the predictors, we can access
them in the usual way. Unsurprisingly, it seems that `carat` is the most important by
far for predicting `price` (so much so that if we were doing this for real, we might
consider dropping the other predictors altogether):

```{code-cell} ipython3
regress_fit.state().coef_, regress_fit.state().intercept_
```

To predict the prices of previously unseen diamonds, we must go through a similar
process of applying each `FitTransform` in turn to some new dataset with the same
schema:

```{code-cell} ipython3
# recall that test_df is the other half of diamonds_df
# we use "oos" as an abbreviation of "out-of-sample"
df_oos = winsorize_fit.apply(test_df)
df_oos = log_price_carat_fit.apply(df_oos)
df_oos = z_score_fit.apply(df_oos)
df_oos = regress_fit.apply(df_oos)
df_oos = exp_price_hat_fit.apply(df_oos)

df_oos.head()
```

The virtue of using `FitTransform` objects like this is that our entire end-to-end model
of diamond prices, including feature cleaning and regression, was fit strictly on one
set of data (`train_df`) and is being applied strictly out-of-sample to new data
(`test_df`). The data in `test_df` is winsorized using the quantiles that were observed
in `train_df`, it's z-scored using the means and standard deviations that were observed
in `train_df`, and predicted prices are generated using the regression betas that were
learned on `train_df`.

:::{important}
There is, to invent some terminology, a clean separation between **fit-time** and
**apply-time**.
:::

As expected, the out-of-sample predictions are not as correlated with observed `price`
as the in-sample predictions, although the degradation is miniscule, perhaps suggesting
that our training set was not very biased:

```{code-cell} ipython3
eval_oos_df = (
    test_df[["price"]]
    .assign(price_hat=df_oos["price_hat"])
)
eval_oos_df.corr()
```

### Pipeline transforms

Now, this is generally **not** how one should use Frankenfit to implement data modeling
pipelines. The example above serves merely to introduce the basic principles from the
ground up, so to speak. Rather than manually chaining Transforms together in a laborious
and error-prone way as we saw above, we should use a special Transform, called
[`Pipeline`](frankenfit.Pipeline) (and its subclasses), which *contains other
Transforms.* The `Pipeline` Transform takes a single parameter, `transforms`, which is a
list of Transforms to be composed together sequentially as we did manually above.

Our diamond price-modeling pipeline can be rewritten as an actual `Pipeline` like so:

```{code-cell} ipython3
price_model = ff.Pipeline(
    transforms=[
        ff.dataframe.Winsorize(0.05),
        ff.dataframe.Pipe(np.log1p, ["price", "carat"]),
        ff.dataframe.ZScore(["carat", "table", "depth"]),
        ff.dataframe.SKLearn(
            sklearn_class=LinearRegression,
            x_cols=["carat", "table", "depth"],
            response_col="price",
            hat_col="price_hat",
            class_params={"fit_intercept": True}
        ),
        ff.dataframe.Pipe(np.expm1, "price_hat"),
    ]
)
```

Now when we fit this `Pipeline` object `price_model` (which is just a `Transform` like
any other), it will handle the logic of feeding each constituent Transform's output into
the next Transform in the sequence. The state of the `FitTransform` that results from fitting a
`Pipeline` is the list of the states of all of the constituent `FitTransforms`; applying it applies each constituent `FitTransform` in sequence.

```{code-cell} ipython3
price_model.fit(train_df).apply(test_df).head()
```

As before, every stateful constituent Transform (`Winsorize`, `ZScore`, `SKLearn`) is
applied strictly **out-of-sample** using whatever state it learned on the fitting data.
Wrapping up a composition of Transforms as a single Transform like this is quite
powerful because it allows one easily to re-use the whole end-to-end model on multiple
datasets, to embed it within other Pipelines, and so on. For example, since
`price_model` is just a `Transform`, we could compose it with a
[`Copy`](frankenfit.dataframe.Copy) Transform that preserves the original `price` column
before all of the winsorizing and standardizing, and a
[`Correlation`](frankenfit.dataframe.Correlation) Transform that computes the
correlation between (standardized) `price_orig` and `price_hat`:

```{code-cell} ipython3
price_model_corr = ff.Pipeline(
    transforms=[
        ff.dataframe.Copy("price", "price_orig"),
        price_model,
        ff.dataframe.Correlation(["price_orig"], ["price_hat"])
    ]
)

price_model_corr_fit = price_model_corr.fit(train_df)
price_model_corr_fit.apply(train_df)  # correlation on its own training data
```

```{code-cell} ipython3
price_model_corr_fit.apply(test_df)  # correlation on test data
```

(call-chain-api)=
### The call-chain API

In the previous section we created the `price_model` [`Pipeline`](frankenfit.Pipeline)
by directly supplying the `transforms` parameter as a list of `Transform` objects:

```python
price_model = ff.Pipeline(
    transforms=[
        ff.dataframe.Winsorize(0.05),
        ff.dataframe.Pipe(np.log1p, ["price", "carat"]),
        ff.dataframe.ZScore(["carat", "table", "depth"]),
        ff.dataframe.SKLearn(
            sklearn_class=LinearRegression,
            x_cols=["carat", "table", "depth"],
            response_col="price",
            hat_col="price_hat",
            class_params={"fit_intercept": True}
        ),
        ff.dataframe.Pipe(np.expm1, "price_hat"),
    ]
)
```

While useful for illustrating how Pipelines work internally, this is generally
*not* the preferred way to write data-modeling pipelines with Frankenfit. Instead we
use what we call the **call-chain API**, so named because it involves making a "chain"
of method calls on Pipeline objects to build up the sequence of Transforms
incrementally. This style of writing Pipelines is more concise and readable (effectively
a domain-specific language), taking inspiration from Pandas' own [similar style of
usage](https://tomaugspurger.github.io/posts/method-chaining/).

Using the call-chain API, our `price_model` pipeline can be written more idiomatically
as follows:

```{code-cell} ipython3
price_model = (
    ff.DataFramePipeline()
    .winsorize(0.05)
    .pipe(np.log1p, ["price", "carat"])
    .z_score(["carat", "table", "depth"])
    .sk_learn(
        sklearn_class=LinearRegression,
        x_cols=["carat", "table", "depth"],
        response_col="price",
        hat_col="price_hat",
        class_params={"fit_intercept": True}
    )
    .pipe(np.expm1, "price_hat")
)
```

One can see two main differences from the previous style of writing a Pipeline:

1. Instead of constructing an instance of [`Pipeline`](frankenfit.Pipeline), we begin by
   constructing an instance of [`DataFramePipeline`](frankenfit.DataFramePipeline).

   This is because `Pipeline` is just a generic base class. It provides the core
   functionality common to all pipelines, but no call-chain
   methods.[^footnote-if-fitting] As noted earlier, Frankenfit's library of built-in
   Transforms is organized into submodules:

   * [`frankefit.universal`](universal-api) for generically useful Transforms that make
     no assumptions about the type or shape of the data.
   * [`frankenfit.dataframe`](dataframe-api) for Transforms that operate on `DataFrames`.

   These also provide subclasses of `Pipeline`, available to users directly under the
   `frankenfit` package namespace:

   * [`ff.UniversalPipeline`](frankenfit.UniversalPipeline) defines call-chain methods
     for all of the Transforms in `frankenfit.universal`.
   * [`ff.DataFramePipeline`](frankenfit.DataFramePipeline) defines call-chain methods for all
     of the Transforms in `frankenfit.dataframe`, *in addition to* those in
     `frankenfit.universal`. (`DataFramePipeline` is in fact a subclass of
     `UniversalPipeline`.)

   Thus when writing our pipelines, we'll almost always be using `DataFramePipeline` (or
   `UniversalPipeline`) rather than `Pipeline`, in order to use the call-chain methods.

2. Rather than passing a list of `Transform` objects, we make a chain of method-calls;
   the method names are the [snake_case](https://en.wikipedia.org/wiki/Snake_case)
   transliterations of the corresponding
   [CamelCase](https://en.wikipedia.org/wiki/Camel_case) `Tranform` class names.

   We construct an initially empty pipeline with `DataFramePipeline()` and then proceed
   to append Transforms to it with successive method calls. Each call passes its
   arguments to the corresponding `Transform` class constructor and returns a new
   pipeline with that `Transform` appended, ready for the next method call.

[^footnote-if-fitting]: *Well actually,* the `Pipeline` base class does include
[`then()`](frankenfit.Pipeline.then),
[`apply_fit_transform()`](frankenfit.Pipeline.apply_fit_transform), and
[`if_fitting()`](frankenfit.Pipeline.if_fitting), which act as call-chain methods,
though not specific to any particular data operations.

+++

It's important to note that each method call returns a *new* `DataFramePipeline`
instance with an additional `Transform` at the end of its `transforms` parameter. The
initial `DataFramePipeline` object is never modified:

```{code-cell} ipython3
empty_pipeline = ff.DataFramePipeline()
empty_pipeline
```

```{code-cell} ipython3
pip = empty_pipeline.winsorize(0.05)
pip
```

```{code-cell} ipython3
# empty_pipeline is still as it was
empty_pipeline
```

:::{seealso}
More details and nuances of the call-chain API, having to do with Transforms that group
and nest other Transforms (possibly sub-pipelines), are covered in {doc}`dataframes` and
{doc}`branching_and_grouping`
:::

+++

### Fit-and-apply in one go: `Pipeline.apply()`

A common operation with a Pipeline is to fit it and apply it on the same dataset, as in:

```{code-cell} ipython3
price_model.fit(train_df).apply(train_df).head()
```

However, as written above, this is inefficient. Recall that, under the hood,
`price_model.fit(train_df)` does something like this:

```{code-cell} ipython3
winsorize_fit = winsorize.fit(train_df)
df = winsorize_fit.apply(train_df)

log_price_carat_fit = log_price_carat.fit(df)
df = log_price_carat_fit.apply(df)

z_score_fit = z_score.fit(df)
df = z_score_fit.apply(df)

regress_fit = regress.fit(df)
df = regress_fit.apply(df)

exp_price_hat_fit = exp_price_hat.fit(df)
df = exp_price_hat_fit.apply(df)
```

In order to fit the next `Transform` in the sequence, `Pipeline.fit()` needs the result
of applying the previous `FitTransform` to the fitting data. This means that by the time
it has fit all of the Transforms, it has, as a side-effect, already computed the result
of applying the whole Pipeline to its own fitting data. This is exactly what `df` is by
the end of our example above.

Thus when we call `apply()` on the Pipeline's own fitting data as in:

```python
price_model.fit(train_df).apply(train_df)
```

...we are duplicating work. All of the constituent `FitTransforms` will be applied to
`train_df`, which is exactly what happened already at fit-time.

For this reason, unlike ordinary stateful Transforms, Pipelines come with an
[`apply()`](frankenfit.Pipeline.apply) method (otherwise `apply()` is only available on
`StatelessTransforms` as discussed [above](stateless-transforms), or on the
`FitTransform` object returned by `fit()`). Calling it on some data is equivalent
to fitting and then applying the Pipeline on the same data, but more efficient.

```{code-cell} ipython3
assert (
    price_model.apply(train_df)
    .equals(
        price_model.fit(train_df).apply(train_df)
    )
)
```

In general the only reason to call `Pipeline.fit()` is if one plans to apply the
resulting `FitTransform` to some out-of-sample data, as in:

```python
# note the apply data is different than the fitting data
price_model.fit(train_df).apply(test_df)
```

+++

The following table summarizes the distribution of `fit()` and `apply()` methods:

|                                                       | `fit()` | `apply()` |
|-------------------------------------------------------|---------|-----------|
| [`Transform`](frankenfit.Transform)                   | ✔       |          |
| [`FitTransform`](frankenfit.FitTransform)             |         | ✔        |
| [`StatelessTransform`](frankenfit.StatelessTransform) | ✔       |  ✔       |
| [`Pipeline`](frankenfit.Pipeline)                     | ✔       |  ✔       |

+++

(visualizing-pipelines)=
### Visualizing pipelines

Because `price_model` is just a Transform, we can query its `params()` like any other
Transform, and access their values as attributes:

```{code-cell} ipython3
price_model.params()
```

```{code-cell} ipython3
price_model.transforms
```

However, for a complicated `Pipeline`, it can be difficult to figure out what it is
doing by looking at the raw `transforms` list. The
[`visualize`](frankenfit.Transform.visualize) method uses the [GraphViz
library](https://pypi.org/project/graphviz/) to produce a visualization of the Pipeline
as an ordered sequence of Transforms:

```{code-cell} ipython3
price_model.visualize()
```

It's worth noting that `visualize` is in fact a method available on all `Transform` objects, not just `Pipelines`.

```{code-cell} ipython3
ff.dataframe.Winsorize(0.05).visualize()
```

This becomes especially useful for certain complex Transforms that group or combine
other Transforms, such as those covered in {doc}`branching_and_grouping`.

+++

### Concatenating pipelines

Once defined, Pipelines can be composed together in various ways. The simplest is
**concatenation**, whereby one Pipeline's sequence of Transforms is followed immediately
by that of another.

For example, we could break our `price_model` Pipeline up into smaller parts. Along the
way we'll introduce an additional step,
[`copy("price", "price_train")`](frankenfit.dataframe.Copy), so that we can prepare
a training response column while preserving the original unmodified `price` column for
later evaluation of our predictions:

```{code-cell} ipython3
# we'll fit our regression on a winsorized and log-transformed *copy* of price
prepare_training_response = (
    ff.DataFramePipeline()
    .copy("price", "price_train")
    .winsorize(0.05, "price_train")
    .pipe(np.log1p, "price_train")
)

prepare_features = (
    ff.DataFramePipeline()
    .winsorize(0.05, ["carat", "table", "depth"])
    .pipe(np.log1p, "carat")
    .z_score(["carat", "table", "depth"])
)

predict_price = (
    ff.DataFramePipeline()
    .sk_learn(
        sklearn_class=LinearRegression,
        x_cols=["carat", "table", "depth"],
        response_col="price_train",
        hat_col="price_hat",  # <---
        class_params={"fit_intercept": True}
    )
    .pipe(np.expm1, "price_hat")
)
```

We can concatenate these three smaller Pipelines into a complete model using
[`then()`](frankenfit.Pipeline.then):

```{code-cell} ipython3
combined_model = (
    prepare_training_response
    .then(prepare_features)
    .then(predict_price)
)
combined_model.visualize()
```

```{code-cell} ipython3
fit_model = combined_model.fit(train_df)
fit_model.apply(test_df).head()
```

`p1.then(p2)` does pretty much what you'd expect: a new `Pipeline` (of the same subclass
as `p1`, so in our case a `DataFramePipeline`) is returned whose `transforms` parameter
is `p1.transforms + p2.transforms`.

As a bit of syntactic sugar, the `__add__` operator is overridden so that Pipelines can
also be concatenated using addition syntax:

```{code-cell} ipython3
combined_model = prepare_training_response + prepare_features + predict_price
```

Concatenation enables greater re-usability of our pipelines. For example, a common setup
might be that we have several different predictive data models for some problem, and
separately, we have one or more ways of "scoring" their predictions. We can easily pick
a model, pick a scoring method, and combine them, as in:

```{code-cell} ipython3
score_predictions = ff.DataFramePipeline().correlation(["price_hat"], ["price"])

(combined_model + score_predictions).apply(train_df)  # in-sample score
```

#### Other uses of `then()`

Concatenation via `then()` is also a good way of introducing a `Transform` that doesn't
have a corresponding call-chain method (say because it's a custom `Transform` subclass
of the user's own devising; but see {doc}`implementing_transforms` and in particular
[`Pipeline.with_methods()`](frankenfit.Pipeline.with_methods) for other ways of
addressing this):

```python
(
    ff.DataFramePipeline()
    ...
    .then(MyGreatTransform(...))
    ...
)
```

`then()` can also be used to initiate a call-chain sequence if one is starting with a
bare `Transform` instance outside of a `Pipeline`. Without any arguments, `then()`
returns a Pipeline containing `self`. For example

```{code-cell} ipython3
# suppose we have a bare DeMean object from somewhere...
de_mean = ff.dataframe.DeMean("foo")
# we can start a DataFramePipeline by calling then()
my_pipeline = (
    de_mean
    .then()
    .winsorize(0.05)
    .pipe(np.sqrt, "bar")
)
my_pipeline.visualize()
```

:::{tip}
The `then()` method of Transforms defined in [`frankenfit.universal`](universal-api)
returns a [`UniversalPipeline`](frankenfit.UniversalPipeline), while that of those
defined in [`frankenfit.dataframe`](dataframe-api) returns a
[`DataFramePipeline`](frankenfit.DataFramePipeline).
:::

+++

(including-FitTransforms)=
### Including `FitTransforms` in a pipeline

Pipelines may only contain [`Transforms`](frankenfit.Transform), but often we might wish
to include an already-fit [`FitTransform`](frankenfit.FitTransform) in a Pipeline,
potentially alongside some (unfit) `Transforms`.  This is possible using a built-in
utility Transform called [`ApplyFitTransform`](frankenfit.core.ApplyFitTransform)
(call-chain method [`apply_fit_transform()`](frankenfit.Pipeline.apply_fit_transform)),
whose purpose is to wrap an already-fit `FitTransform` instance as a stateless
Transform. At fit-time, `ApplyFitTransform` does nothing; at apply-time, it applies the
wrapped `FitTransform` instance. This allows one to embed `FitTransform` objects
wherever ``Transform`` is ordinarily required, for example in a
[`Pipeline`](frankenfit.Pipeline).

This is particularly useful in situations where we've already fit some predictive
pipeline, and now we want to layer some additional transformations onto its input or
output. Because those transformations may be stateful themselves, we can even create
"heterogeneously fit" Pipelines, wherein different parts of the Pipeline have been fit
on different datasets.

For example, continuing with our `fit_model` (a `FitTransform` instance) from above, now
that we've fit that model on `train_df`, we might be interested to know how it performs
on the subset of `test_df` with just the largest diamonds, say those weighting at least
one carat. We can use the [`Filter`](frankenfit.dataframe.Filter) Transform (call-chain
method [`filter()`](frankenfit.dataframe.DataFrameCallChain.filter)) to select the rows
in question, and send them to our `fit_model` with `apply_fit_transform()`:

```{code-cell} ipython3
(
    ff.DataFramePipeline()
    .filter(lambda df: df["carat"] >= 1)
    .apply_fit_transform(fit_model)
).apply(test_df).plot.scatter("price_hat", "price", alpha=0.3);
```

As a convenience, both [`then()`](frankenfit.core.PipelineMember.then) and the addition
operator will automatically wrap their argument in an `ApplyFitTransform` if a
`FitTransform` is provided, so we could also write our filtered Pipeline as:

```{code-cell} ipython3
keep_large = ff.DataFramePipeline().filter(lambda df: df["carat"] >= 1)
(keep_large + fit_model);  # or equivalently: keep_large.then(fit_model)
```

If we visualize this Pipeline, we can see the resulting `ApplyFitTransform` transform:

```{code-cell} ipython3
(keep_large + fit_model).visualize()
```

Earlier we mentioned a common use-case in which we have one or more predictive
pipelines, and one or more "scoring" methods, which we'd like to mix and match with each
other. This is another situation in which `ApplyFitTransform` is handy, because we might
want to fit the predictions on one dataset (training data) and apply and score them on
another dataset (test data). Let's revisit out `combined_model` from before, and define
two scoring pipelines, one based on correlation and the other based on mean squared
error:

```{code-cell} ipython3
# predictive pipeline (a Transform)
combined_model = (prepare_training_response + prepare_features + predict_price)

# scoring pipelines
score_corr = ff.DataFramePipeline().correlation(["price_hat"], ["price"])
score_mse = ff.DataFramePipeline().stateless_lambda(
    lambda df: ((df["price_hat"] - df["price"])**2).mean()
)
```

One thing we could do is concatenate `combined_model` with each scoring pipeline, and
then fit the resulting Pipeline on `train_df` and apply it to `test_df` in order to get
the out-of-sample scores:

```{code-cell} ipython3
(combined_model + score_corr).fit(train_df).apply(test_df)
```

```{code-cell} ipython3
(combined_model + score_mse).fit(train_df).apply(test_df)
```

But that is wasteful, because we are fitting the pipeline every time that we wish to
score it. Thanks to `ApplyFitTransform` we can fit the predictive pipline just once,
giving us a `FitTransform`, which we concatenate with the scoring pipelines, and apply
the result to `test_df` to compute the out-of-sample scores. In this way we avoid
re-fitting the Pipeline each time:

```{code-cell} ipython3
# now we only fit the model once
fit_model = combined_model.fit(train_df)
display(
    (fit_model + score_corr).apply(test_df),
    (fit_model + score_mse).apply(test_df),
)
```

We can even include our `keep_large` filtering pipeline from before to get out-of-sample scores on only the largest diamonds (having fit on *all* sizes of diamonds):

```{code-cell} ipython3
# still using the same fit from before
display(
    (keep_large + fit_model + score_corr).apply(test_df),
    (keep_large + fit_model + score_mse).apply(test_df),
)
```

(tagging-selecting-transforms)=
### Tagging and selecting transforms

As mentioned [above](transform-tags), every `Transform` has an optional
[`tag`](frankenfit.Transform.tag) parameter, which determines the
[`name`](frankenfit.Transform.name) of the `Transform` instance. The purpose of `name`
is to make it more convenient to identify and select individual `Transforms` when
embedded in larger `Pipelines`. For example, revisiting the `combined_model` from the
previous sections, we might add a custom tag like `"price_regression"` to the
`sk_learn()` Transform that performs the linear regression:

```{code-cell} ipython3
predict_price_tagged = (
    ff.DataFramePipeline()
    .sk_learn(
        sklearn_class=LinearRegression,
        x_cols=["carat", "table", "depth"],
        response_col="price_train",
        hat_col="price_hat",
        class_params={"fit_intercept": True},
        tag="price_regression"  # <---
    )
    .pipe(np.expm1, "price_hat")
)
```

The resulting [`SKLearn`](frankenfit.dataframe.SKLearn) `Transform` object will have the
name `"SKLearn#price_regression"`, and with this tagged Transform in our
`combined_model`, we can now pull it back out for inspection by using the method
[`find_by_name()`](frankenfit.core.PipelineMember.find_by_name):

```{code-cell} ipython3
combined_model = (prepare_training_response + prepare_features + predict_price_tagged)
combined_model.find_by_name("SKLearn#price_regression")
```

The [`FitTransform`](frankenfit.FitTransform) class also provides a similar
[`find_by_name()`](frankenfit.core.PipelineMember.find_by_name) method, which searches
through the [`state()`](frankenfit.FitTransform.state) of the `FitTransform`. This means
that after fitting the `combined_model` Pipeline on some data, which yields a
`FitTransform` representing the state of the entire fit Pipeline, we can easily pull out
the sub-`FitTransform` corresponding to the regression:

```{code-cell} ipython3
fit_model = combined_model.fit(train_df)
fit_regression = fit_model.find_by_name("SKLearn#price_regression")
fit_regression
```

This is useful, for example, if we want to inspect the estimated betas of the regression:

```{code-cell} ipython3
fit_regression.state().coef_  # coef_ attribute of sklearn.linear_model.LinearRegression
```
