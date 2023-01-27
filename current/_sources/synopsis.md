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

# Synopsis and overview

Frankenfit is a Python library for data scientists that provides a domain-specific
language (DSL) for creating, fitting, and applying predictive data modeling pipelines.
Its key features are:

* A concise and readable **DSL** (inspired by the pandas [method-chaining
  style](https://tomaugspurger.github.io/posts/method-chaining/)) to create data
  modeling **pipelines** from chains of composable building blocks called
  **transforms**. Pipelines themselves are composable, re-usable, and extensible, with
  a thorough [library of transforms](transform-library) available for building,
  grouping, and combining pipelines in useful ways.
* Rigorous separation between, on the one hand, **fitting** the state of your pipeline
  on some training data, and, on the other, **applying** it
  [out-of-sample](https://stats.stackexchange.com/questions/260899/what-is-difference-between-in-sample-and-out-of-sample-forecasts)
  to make predictions on test data. Once fit, a pipeline can be re-used to make
  predictions on many different test datasets, and these predictions are truly
  **out-of-sample**, right down to the quantiles used to winsorize your features
  (for example).
* The ability to specify your pipeline's parameters as **hyperparameters**, whose values
  are bound later. This can make your pipelines more re-usable, and enables powerful
  workflows like hyperparameter search, cross-validation, and other resampling schemes,
  all described in the same DSL used to create pipelines.
* **Parallel computation** on distributed backends (currently
  [Dask](https://www.dask.org)). Frankenfit automatically figures out which parts of your
  pipeline are independent of each other and runs them in parallel on a distributed
  compute cluster.
* A focus on **user ergonomics** and **interactive usage.** Extensive type annotations
  enable smart auto-completions by IDEs. [Visualizations](visualizing-pipelines) help
  you see what your pipelines are doing. You can [implement your own
  transforms](implementing-transforms) with almost zero boilerplate.

Frankenfit takes some inspiration from scikit-learn's [`pipeline`
module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline),
but aims to be much more general-purpose and flexible. It integrates easily with
industry-standard libraries like [pandas](https://pandas.pydata.org),
[scikit-learn](https://scikit-learn.org) and [statsmodels](https://www.statsmodels.org),
or your own in-house library of statistical models and data transformations.

Frankenfit's focus is on 2-D Pandas DataFrames, but the core API is agnostic and could also
be used to implement pipelines on other data types, like text or images.

:::{tip}
As a stylistic convention, and for the sake of brevity, the author of Frankenfit
recommends importing `frankenfit` with the short name `ff`:

```python
import frankenfit as ff
```
:::

With Frankenfit, you can:

* [Create pipelines](synopsis-create) using a DSL of call-chain methods.
* [Fit pipelines and apply them to data](synopsis-fit-apply) to generate predictions.
* [Use hyperparameters](synopsis-hyperparams) to generalize your pipelines and concisely
  execute hyperparameter searches and data batching.
* [Run your pipelines on distributed backends](synopsis-backends), exploiting the
  parallelism inherent to any branching operations in a pipeline.

The remainder of this page summarizes each of these workflows with a running example,
while the subsequent sections of the documentation detail how everything works from the
ground up.

```{code-cell} ipython3
:tags: [remove-cell]

# FIXME: this cell should not be visible in docs output.
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
```

(synopsis-create)=
## Create pipelines

Let's suppose we want to model the prices of round-cut diamonds using the venerable
[diamonds](https://ggplot2.tidyverse.org/reference/diamonds.html) dataset, which is
often used to teach regression. It looks like this:

```{code-cell} ipython3
from pydataset import data
diamonds_df = data('diamonds')
diamonds_df.info()
diamonds_df.head()
```

```{code-cell} ipython3
:tags: [remove-cell]

# FIXME: this cell should not be visible in docs output.
diamonds_df.rename_axis(index="index").to_csv("./diamonds.csv")
```

:::{note}
Throughout the documentation we make use of the
[pydataset](https://pypi.org/project/pydataset/) package for loading example data like
`diamonds`.
:::

Let's divide the data into random training and test sets, and examine the distributions
of the `carat`, `depth`, `table`, and `price` variables on the training set:

```{code-cell} ipython3
# randomly split train and test data
train_df = diamonds_df.sample(frac=0.5, random_state=1337_420_69)
test_df = diamonds_df.loc[list(set(diamonds_df.index) - set(train_df.index))]

(
    train_df
    [["carat", "depth", "table", "price"]]
    .hist()
);
```

Below is our predictive pipeline for modeling `price`. The core of it is a linear
regression of `price` on `carat`, `table`, and `depth` (using the scikit-learn
`LinearRegression` estimator), but first, inspired by the histograms above, our pipeline
creates a training response `price_train` by log-transforming and winsorizing the raw
`price`, and creates features by winsorizing, z-scoring, and clipping the raw predictor
variables (and in the case of `carat`, log-transforming it). Finally we exponeniate the
predictions of the regression to put them in the original units of `price` (dollars).

```{code-cell} ipython3
import numpy as np
import sklearn.linear_model
import frankenfit as ff

# use "do" as shorthand for a new pipeline
do = ff.DataFramePipeline()
diamond_model = (
    do
    .if_fitting(
        # create training response when fitting
        do.assign(
            # We'll train a model on the log-transformed and winsorized price of a
            # diamond.
            price_train=do["price"].pipe(np.log1p).winsorize(0.05),
        )
    )
    # Transform carats feature to log-carats
    .pipe(np.log1p, "carat")
    # Prepare features: trim outliers and standardize
    .assign(
        do[["carat", "depth", "table"]]
        .suffix("_fea")  # name the prepared features with _fea suffix
        .winsorize(0.05)  # trim top and bottom 5% of each
        .z_score()
        .impute_constant(0.0)  # fill missing values with zero (since they are z-scores,
                               # zero is the expected mean)
        .clip(lower=-2, upper=2)  # clip z-scores
    )
    # Fit a linear regression model to predict training response from the prepared
    # features
    .sk_learn(
        sklearn.linear_model.LinearRegression,
        x_cols=["carat_fea", "depth_fea", "table_fea"],
        response_col="price_train",
        hat_col="price_hat",
        class_params=dict(fit_intercept=True),
    )
    # Exponentiate the regression model's predictions back from log-dollars to dollars
    .pipe(np.expm1, "price_hat")
)
```

The pipeline is created by a chained sequence of method calls on a
[`DataFramePipeline`](frankenfit.DataFramePipeline) object, some of which accept
arguments that are themselves `DataFramePipelines` built up by their own call-chains. We
call this the [**call-chain API**](call-chain-api). Each method call adds a
**transform** to the pipeline, and you can peruse the library of built-in transforms [in
the API documentation](transform-library).

+++

(synopsis-fit-apply)=
## Fit pipelines and apply them to data

The pipeline object `diamond_model` is best thought of as a light-weight, abstract, immutable
description of what to do to some as-yet unspecified data; it stores no data or state
in and of itself.

With our pipeline defined, we **fit** it on the training data, obtaining a
[`FitTransform`](frankenfit.FitTransform) object `fit_diamond_model`, which encapsulates
the learned **states** of all of the transforms in the pipeline:

```{code-cell} ipython3
fit_diamond_model = diamond_model.fit(train_df)
```

The fit may then be **applied** to another input `DataFrame`, in this case the test set:

```{code-cell} ipython3
predictions_df = fit_diamond_model.apply(test_df)
predictions_df.head()
```

We can see that the resulting `DataFrame`, `predictions_df`, has a number of new
columns added by our pipeline: the standardized features (`carat_fea`, `depth_fea`,
`table_fea`), and most importantly, our predicted price, `price_hat`. We can confirm
that our transforms have affected the distributions of the features in the expected way,
and check how `price_hat` relates to actual `price`:

```{code-cell} ipython3
(
    predictions_df
    [["carat_fea", "depth_fea", "table_fea", "price_hat"]]
    .hist()
);

predictions_df.plot.scatter("price_hat", "price", alpha=0.3);
```

:::{important}
Our entire end-to-end model of diamond prices, including feature preparation and
regression, was fit **strictly** on one set of data (`train_df`) and applied strictly
**out-of-sample** to new data (`test_df`). The columns in `test_df` are being winsorized
using the quantiles that were observed in `train_df`, z-scored using the means and
standard deviations that were observed in `train_df`, and `price_hat` is generated using
the regression betas that were learned on `train_df`.
:::

The ability to fit a complex pipeline on one set of data and use the fit state to
generate predictions on different data is fundamental to statistical resampling
techniques like cross-validation, as well as many common operations on time series.

### Group and cross-validate pipelines

Frankenfit provides various transforms that fit and apply **child transforms**, which
can be combined to achieve many use cases. For example, we can use
[`group_by_cols()`](frankenfit.dataframe.GroupByCols) (together with
[`correlation()`](frankenfit.dataframe.Correlation)) to check how well our model, which
was fit on the training set, predicts price **on the test set** for each `cut` of
diamond:

```{code-cell} ipython3
(
    do
    # group the next transform by the "cut" column
    .group_by_cols("cut", group_keys=True, as_index=True)
    .then(
        fit_diamond_model.then().correlation("price_hat", "price")
    )
).apply(test_df).head()
```

`group_by_cols()` works similarly to
[`pandas.DataFrame.groupby()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html):
it "consumes" the next method call in the chain and whatever transform that call is
introducing to the pipeline will be fit/applied separately for each distinct value of
the `cut` column. In this case the next call is `then(fit_diamond_model...)`, which is
introducing a pipeline that "scores" the predictions of our already-fit model. The
result of all of this is a `DataFrame` that shows us, for each subset of the test data
corresponding to a different value of `cut`, the correlation  between `price` and
`price_hat` as generated on that subset by our model, which was fit on the entire
training set.

+++

But it's important to note that `group_by_cols()` is very different than pandas'
`groupby()`. It takes a child transform (which as we've seen may be a complex pipeline),
and before it can apply that transform to a group of data $g_i$ we must answer: how
should the transform for $g_i$ be fit? This doesn't matter in the example above because
[fitting is a null-op on a `FitTransform`](including-FitTransforms) like
`fit_diamond_model`, but where it gets interesting is when we apply `group_by_cols()` to
an *unfit* pipeline like our original `diamond_model`. The default "fitting schedule,"
as we call it, is to generate predictions on $g_i$ by fitting on $g_i$, but we can
imagine other fitting schedules, like:

* generate predictions on $g_i$ by fitting on all $g_j$ where $j<i$, for some ordering
  of groups (time series data, for example);
* generate predictions on $g_i$ by fitting on all other groups $g_j$, $i \neq j$; this
  is exactly the definition of
  [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))!

So suppose we want to perform 5-fold cross-validation on the model of diamond prices. We
can create a cross-validation pipeline around `diamond_model` like so:

```{code-cell} ipython3
crossval_pipeline = (
    # read the full-sample data from a file
    do.read_pandas_csv("./diamonds.csv")
    # randomly partition rows into 5 groups
    .assign(
        group=lambda df: np.random.uniform(
            low=0, high=5, size=len(df)
        ).astype("int32")
    )
    # group the next transform by the "group" column
    .group_by_cols(
        "group", fitting_schedule=ff.fit_group_on_all_other_groups,
    )
    .then(diamond_model)  # <-- our diamond_model pipeline is applied to each group
                          # by fitting it on all the other groups
    # score the out-of-sample predictions across all groups
    .correlation("price_hat", "price")
)

crossval_pipeline.apply()
```

We randomly divide the dataset into groups (based on a column that we create with
`assign()`), and  use `group_by_cols()` together with a specified `fitting_schedule` to,
for each group, generate predictions from the `diamond_model` pipeline by fitting it on
the data from all *other* groups, but applying it to the data from the group in
question.

This gives us a dataset of entirely out-of-sample predictions, whose performance we then
score with the `correlation()` transform.

+++

(synopsis-hyperparams)=
## Use hyperparameters

All of the transforms in the `diamond_model` pipeline have **parameters**. For example,
the `0.05` in `winsorize(0.05)`, the `x_cols=`, `response_col=`, `hat_col=` and other
keyword-arguments to `sk_learn`, are all parameters. We can replace some of these
parameters with **hyperparameters**, to indicate that their values are deferred, and
will not be known until `fit()` is called, at which point the caller of `fit()` must
provide them.

The simplest way to introduce a hyperparameter to a pipeline is to replace the desired
parameter with an instance of [`ff.HP`](frankenfit.HP), indicating the **name** of the
hyperparameter that will supply the value of that parameter at fit-time. For example,
let's create a new hyperparameterized diamond pipeline, with the following changes:

* Instead of sklearn's `LinearRegression`, we use `Lasso`, which implements a [lasso
  regression](https://en.wikipedia.org/wiki/Lasso_(statistics)) with shrinkage parameter `alpha`, which we provide as a hyperparameter named `"alpha"`.
* We replace the hard-coded list `["carat", "depth", "table"]` with a hyperparameter
  named `"features"` which is expected to be a list of strings.

```{code-cell} ipython3
diamond_model_hyperparams = (
    do
    .if_fitting(
        do.assign(
            price_train=do["price"].pipe(np.log1p).winsorize(0.05),
        )
    )
    .pipe(np.log1p, "carat")
    .assign(
        do[ff.HP("features")]  # <-- "features" hyperparam
        .suffix("_fea")
        .winsorize(0.05)
        .z_score()
        .impute_constant(0.0)
        .clip(lower=-2, upper=2)
    )
    .sk_learn(
        sklearn.linear_model.Lasso,
        # x_cols is a hyperparameterized list of columns derived from "features"
        x_cols=ff.HPLambda(lambda bindings: [f+"_fea" for f in bindings["features"]]),
        response_col="price_train",
        hat_col="price_hat",
        class_params=dict(
            fit_intercept=True,
            alpha=ff.HP("alpha"),  # <-- "alpha" hyperparam
        ),
    )
    .pipe(np.expm1, "price_hat")
)
```

We can inspect the set of hyperparameters referenced by this new pipeline with `hyperparams()`:

```{code-cell} ipython3
diamond_model_hyperparams.hyperparams()
```

If we attempt to fit `diamond_model_hyperparams` as we did previously, i.e. with:

```python
diamond_model_hyperparams.fit(train_df)
```
...then `fit()` will raise
[`UnresolvedHyperparameterError`](frankenfit.UnresolvedHyperparameterError).

Instead we must supply values for the hyperparameters, which we call **bindings**; they
can be supplied as keyword arguments to `fit()`:

```{code-cell} ipython3
diamond_model_hyperparams.fit(train_df, alpha=0.1, features=["depth", "table"]);
```

```{code-cell} ipython3
(
    diamond_model_hyperparams
    .fit(train_df, alpha=0.1, features=["depth", "table"])
    .then()
    .correlation("price_hat", "price")
    .apply(test_df)
)
```

By using hyperparameters, we can make our pipelines much more re-usable. We can write a
pipeline once, then use it to fit potentially very different models by supplying
different bindings. Frankenfit provides utility transforms like
[`if_hyperparam_is_true()`](frankenfit.universal.IfHyperparamIsTrue) and
[`if_hyperparam_lambda()`](frankenfit.universal.IfHyperparamLambda) that allow for creative uses
of hyperparameters to control the behavior of your pipelines.

+++

### Hyperparameter searches

One transform in particular,
[`group_by_bindings()`](frankenfit.dataframe.GroupByBindings), is like
`group_by_cols()`, but instead of fitting and applying a child transform on different
groups of data, it does so on different hyperparameter bindings, which can be used to
implement hyperparameter searches, among other things.

For example, let's define a search space on our two hyperparameters, `"alpha"` and
`"features"`:

```{code-cell} ipython3
import itertools

alphas = [0.01, 0.05, 0.10]
feature_sets = [
    ["depth"],
    ["table"],
    ["carat"],
    ["depth", "table"],
    ["depth", "carat"],
    ["carat", "table"],
    ["depth", "table", "carat"],
]

search_space = [
    {"alpha": alpha, "features": features}
    for alpha, features in itertools.product(alphas, feature_sets)
]

search_space
```

Suppose we want to know how our model performs on each point in this search space when
trained on `train_df` and evaluated on `test_df`. We can use `group_by_bindings()` on
the search space that we've defined:[^footnote-NaN-corr]

[^footnote-NaN-corr]: Some of the hyperparameter bindings result in a correlation of
`NaN` because `depth` is such a weak predictor that, even with relatively small values
of `alpha`, the lasso basically zeroes out its regression beta. The regression therefore
just predicts its intercept on every observation; thus the predictions have no variance
and the correlation is undefined.

```{code-cell} ipython3
(
    do
    .group_by_bindings(search_space, as_index=True)
    .then(
        diamond_model_hyperparams
        .correlation("price_hat", "price")
        .rename({"price": "correlation"})
    )
).fit(train_df).apply(test_df)
```

We can even go a step further: suppose that for each point in the search space, rather
than fitting the model on `train_df` and evaluating it on `test_df`, we want to perform
5-fold cross validation of the model as we did before. That is, divide the data into 5
groups, then for each binding of hyperparameters, apply the model that results from that
binding to each group by fitting it on every other group, and evaluate all of the
resulting out-of-sample predictions. This will give us a cross-validated evaluation of
each binding in the search space.

```{code-cell} ipython3
cv_diamonds_hyperparams = (
    do
    .group_by_cols(
        "group", fitting_schedule=ff.fit_group_on_all_other_groups,
    )
    .then(diamond_model_hyperparams)
    # score the out-of-sample predictions across all groups
    .correlation("price_hat", "price")
    .rename({"price": "correlation"})
)

search_cv = (
    do.read_pandas_csv("./diamonds.csv")
    # randomly assign rows to groups. note we've put this before group_by_bindings() so
    # that we don't get a different randomization for each binding
    .assign(
        group=lambda df: np.random.uniform(
            low=0, high=5, size=len(df)
        ).astype("int32")
    )
    # run cv_diamonds_hyerparams on each binding in search_space
    .group_by_bindings(search_space, as_index=True)
        .then(cv_diamonds_hyperparams)
)
```

```{code-cell} ipython3
search_cv.apply()
```

This just scratches the surface of what's possible with hyperparameters and grouping
transforms. For example, if we have a dataset that is too large to fit in memory, we
could operate on it in "chunks" by defining a pipeline that begins with a data-reading
transform that has (say) `begin` and `end` parameters that determine what range of data
it will read. If these parameters are written as hyperparameters, then
`group_by_bindings()` on a sequence of (`begin`, `end`)-pairs can be used to apply the
pipeline one chunk at a time, possibly in parallel on a distributed backend.  (TODO:
link to an example of this.)

```{code-cell} ipython3
:tags: [remove-cell]

from myst_nb import glue
glue("how_parallel", len(search_space) * 5)
```

(synopsis-backends)=
## Run on distributed backends

The `search_cv` pipeline that we've defined above has a deeply parallel structure.
`group_by_bindings()` fits and applies the `cv_diamonds_hyperparams` on each binding in
`search_space`, and these are all independent of each other. Furthermore, within each
`cv_diamonds_hyperparams`, `group_by_cols` fits and applies `diamond_model_hyperparams`
on each fold, which are also independent of each other.  Therefore we have something
like `len(search_space) * 5` = {glue:text}`how_parallel` totally independent fits and
applications of `cv_diamonds_hyperparams` being computed.

Frankenfit makes it easy to exploit the parallelism inherent to a pipeline by running it
on a **[`Backend`](backends)**. Currently two `Backends` are available:

* [`LocalBackend`](frankenfit.LocalBackend), which we've already been using implicitly.
  On this backend, pipelines are computed single-threaded in the local Python
  interpreter process.
* [`DaskBackend`](frankenfit.backend.DaskBackend), which computes pipelines on a Dask
  cluster.

To demonstrate, let's start a Dask cluster on the local machine with four worker
processes and two threads per worker:

```{code-cell} ipython3
from dask import distributed
cluster = distributed.LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    scheduler_port=0,
    dashboard_address=":0",
)
client = distributed.Client(cluster)
```

```{code-cell} ipython3
:tags: [remove-cell]

client
```

We then wrap the Dask `client` object with an instance of Frankenfit's `DaskBackend`:

```{code-cell} ipython3
dask = ff.DaskBackend(client)
```

And now it's as simple as setting this backend on the `search_cv` pipeline before we apply it:

```{code-cell} ipython3
search_cv.on_backend(dask).apply()  # runs on dask!
```

The above blocks the local Python interpreter process while waiting for the result. Alternatively, we could call `dask.apply(search_cv)` to immediately obtain a [`Future`](frankenfit.Future) object from which we can later retrieve the result when it is ready. In either case, our Dask cluster's task-stream dashboard shows all of the tasks that have been scheduled in parallel:

+++

:::{figure} _static/sshot-dask-taskstream-search_cv.png
Screenshot of the Dask task-stream when applying `search_cv` on a `DaskBackend`.
:::

```{code-cell} ipython3
:tags: [remove-cell]

client.shutdown()
client.close()
```

## Next steps

This page has been meant to give a broad tour of what you can do with Frankenfit, and
has necessarily omitted or glossed over some details. The rest of the documentation
explains how everything works in detail, and provides a handy [reference for the
complete API](api).
