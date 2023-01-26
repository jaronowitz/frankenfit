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
  [Dask](https://www.dask.org)). Frankenfit automatically figures out what parts of your
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
often used to teach regression, and looks like this:

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

```{code-cell} ipython3
# randomly split train and test data
train_df = diamonds_df.sample(frac=0.5, random_state=1337_420_69)
test_df = diamonds_df.loc[list(set(diamonds_df.index) - set(train_df.index))]
```

```{code-cell} ipython3
(
    train_df
    [["carat", "depth", "table", "price"]]
    .hist()
);
```

:::{note}
Throughout the documentation we make use of the
[pydataset](https://pypi.org/project/pydataset/) package for loading example data like
`diamonds`.
:::

+++

Create concise and readable descriptions of data learning and transformation pipelines
using a callchain-style API. A pipeline is a sequence of transforms, each applying to
the output of the transform that precedes it. For example, here's a pipeline for
predicting diamond prices, including feature preparation and response transformations:

```{code-cell} ipython3
import numpy as np
import sklearn.linear_model
import frankenfit as ff

# "do" as shorthand for a new pipeline
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
        .impute_constant(0.0)  # fill missing values with zero (i.e. mean)
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

(synopsis-fit-apply)=
## Fit pipelines and apply them to data

The pipeline itself is only a lightweight description of what to do to some input data.

+++

*Fit* the pipeline on data, obtaining a `FitTransform` object, which
encapsulates the learned *states* of all of the transforms in the pipeline:

```{code-cell} ipython3
fit_diamond_model = diamond_model.fit(train_df)
```

The fit may then be applied to another input DataFrame:

```{code-cell} ipython3
predictions_df = fit_diamond_model.apply(test_df)
predictions_df.head()
```

```{code-cell} ipython3
(
    predictions_df
    [["carat_fea", "depth_fea", "table_fea", "price_hat"]]
    .hist()
);

predictions_df.plot.scatter("price_hat", "price");
```

The ability to fit a complex pipeline on one set of data and use the fit state to
generate predictions on different data is fundamental to statistical resampling
techniques like cross-validation, as well as many common operations on time series.

Frankenfit provides various transforms that fit and apply *child transforms*, which can
be combined to achieve many use cases. For example, suppose we want to perform 5-fold
cross validation on the model of diamond prices:

```{code-cell} ipython3
(
    do
    .group_by_cols("cut", group_keys=True, as_index=True)
        .then(
            fit_diamond_model
            .then()
            .correlation("price_hat", "price")
        )
).apply(test_df).head()
```

```{code-cell} ipython3
crossval_pipeline = (
    do.read_pandas_csv("./diamonds.csv")
    # randomly assign rows to groups
    .assign(
        group=lambda df, ngroups=5: np.random.uniform(
            low=0, high=ngroups, size=len(df)
        ).astype("int32")
    )
    .group_by_cols(
        "group", fitting_schedule=ff.fit_group_on_all_other_groups,
    )
        .then(diamond_model)  # <-- our diamond_model pipeline is what's being grouped!
    # score the out-of-sample predictions across all groups
    .correlation("price_hat", "price")
)

crossval_pipeline.apply()
```

We use `group_by_cols()` to divide the dataset into groups (based on a column that we
create with `assign()`), and for each group, generate predictions from the
`diamond_model` pipeline by fitting it on the data from all *other* groups, but applying
it to the data from the group in question.

This gives us a dataset of entirely out-of-sample predictions, whose performance we
score by feeding it to a transform that outputs the correlation between observed and
predicted price.

+++

(synopsis-hyperparams)=
## Use hyperparameters

3. Hyperparameters and bindings. Concisely describe and execute hyperparameter searches
and data batching.

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

diamond_model_hyperparams.hyperparams()
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

#(
#    do
#    .group_by_bindings()
#)
```

```{code-cell} ipython3
(
    do
    .group_by_bindings(search_space, as_index=True)
        .then(
            diamond_model_hyperparams
            .correlation("price_hat", "price")
            #.stateless_lambda(lambda df: df.head(3))
        )
).fit(train_df).apply(test_df)
```

```{code-cell} ipython3
def make_crossval_pipeline(model_pipeline, hat_col, response_col):
    return (
        do.read_pandas_csv("./diamonds.csv")
        # randomly assign rows to groups
        # ngroups is a hyperparam with default value of 5
        .assign(
            group=lambda df, ngroups=5: np.random.uniform(
                low=0, high=ngroups, size=len(df)
            ).astype("int32")
        )
        .group_by_cols(
            "group", fitting_schedule=ff.fit_group_on_all_other_groups,
        )
            .then(model_pipeline)
        # score the out-of-sample predictions across all groups
        .correlation(hat_col, response_col)
    )

cv_diamonds_hyperparams = make_crossval_pipeline(
    diamond_model_hyperparams, "price_hat", "price"
)
cv_diamonds_hyperparams.hyperparams()
```

```{code-cell} ipython3
search_cv = (
    do
    .group_by_bindings(search_space, as_index=True)
        .then(cv_diamonds_hyperparams)
)
search_cv.apply(ngroups=5)
```

(synopsis-backends)=
## Run on distributed backends

4. Run on distributed backends, exploiting the parallelism inherent to any branching
operations in the pipeline.

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

```{code-cell} ipython3
dask = ff.DaskBackend(client)
```

```{code-cell} ipython3
search_cv.on_backend(dask).apply()
```

```{code-cell} ipython3
# if we were doing this for real we'd want to make sure that our group randomization is fixed
```

Screenshot of the Dask task-stream:

![task-stream](_static/sshot-dask-taskstream-search_cv.png)

```{code-cell} ipython3
client.shutdown()
client.close()
```
