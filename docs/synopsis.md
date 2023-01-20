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

Frankenfit is a Python library for creating predictive data modeling pipelines from
reusable, composable building blocks called *Transforms*. Once defined, a pipeline may
be fit on some data and hyperparameters, and then applied on some possibly different
data. If the apply-time input data is different than the fit-time input data, then the
pipeline and all of its constituent transforms are applied entirely "out-of-sample"
with respect to the fitting data.

Frankenfit takes some inspiration from scikit-learn's
[`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html),
but the Frankenfit API is intended to be used as a more general domain-specific language
for data learning and transformation pipelines. It's easy to include models and
estimators from third-party libraries like [scikit-learn](https://scikit-learn.org/) and
[statsmodels](https://www.statsmodels.org/) in your Frankenfit pipelines.

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

* [Create pipelines](synopsis-create)
* [Fit pipelines and apply them to data](synopsis-fit-apply)

(synopsis-create)=
## Create pipelines

Create concise and readable descriptions of data learning and transformation pipelines
using a callchain-style API. A pipeline is a sequence of transforms, each applying to
the output of the transform that precedes it. For example, here's a pipeline for
predicting diamond prices, including feature preparation and response transformations:

```{code-cell} ipython3
:tags: [remove-cell]
# FIXME: this cell should not be visible in docs output.
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
```

```{code-cell} ipython3
import numpy as np
import sklearn.linear_model
import frankenfit as ff

do = ff.DataFramePipeline()
diamond_model = (
    do
    .assign(
        # We train a model to predict the log-transformed and winsorized price of a
        # diamond.
        price_train=do["price"].pipe(np.log1p).winsorize(0.05),
        # Transform carats to log-carats
        carat=do["carat"].pipe(np.log1p),
    )
    # Prepare features: trim outliers and standardize
    .assign(
        do[["carat", "depth", "table"]]
        .suffix("_fea")
        .winsorize(0.05)
        .z_score()
        .impute_constant(0.0)
        .clip(lower=-2, upper=2)
    )
    # Fit a linear regression model to predict log-prices from the prepared features
    .sk_learn(
        sklearn.linear_model.LinearRegression,
        x_cols=["carat_fea", "depth_fea", "table_fea"],
        response_col="price_train",
        hat_col="price_hat",
        class_params=dict(fit_intercept=True),
    )
    # Transform the regression model's predictions back from log-dollars to dollars
    .assign(
        price_hat_dollars=do["price_hat"].pipe(np.expm1)
    )
)
```

(synopsis-fit-apply)=
## Fit pipelines and apply them to data

The pipeline itself is only a lightweight description of what to do to some input data.

```{code-cell} ipython3
from pydataset import data
df = data('diamonds')
df.head()
```

*Fit* the pipeline on data, obtaining a `FitTransform` object, which
encapsulates the learned *states* of all of the transforms in the pipeline:

```{code-cell} ipython3
df_in = df.sample(100).reset_index()
fit_diamond_model = diamond_model.fit(df_in)
```

The fit may then be applied to another input DataFrame:

```{code-cell} ipython3
df_out = df.sample(1000).reset_index()
result = fit_diamond_model.apply(df_out)
result[["carat_fea", "depth_fea", "table_fea", "price_hat"]].hist(figsize=(5,5));
result.plot.scatter("price_hat_dollars", "price");
result.head()
```

The ability to fit a complex pipeline on one set of data and use the fit state to
generate predictions on different data is fundamental to statistical resampling
techniques like cross-validation, as well as many common operations on time series.

Frankenfit provides various transforms that fit and apply *child transforms*, which can
be combined to achieve many use cases. For example, suppose we want to perform 5-fold
cross validation on the model of diamond prices:

```{code-cell} ipython3
(
    do.read_pandas_csv("./diamonds.csv")
    .assign(
        # randomly assign rows to groups
        group=lambda df, ngroups=5: np.random.uniform(
            low=0, high=ngroups, size=len(df)
        ).astype("int32")
    )
    .group_by_cols(
        "group", fitting_schedule=ff.fit_group_on_all_other_groups, group_keys=True
    )
        .then(diamond_model)
    .correlation("price", "price_hat_dollars")
).apply()
```

We use `group_by_cols()` to divide the dataset into groups (based on a column that we
create with `assign()`), and for each group, generate predictions from the
`diamond_model` pipeline by fitting it on the data from all *other* groups, but applying
it to the data from the group in question.

This gives us a dataset of entirely out-of-sample predictions, whose performance we
score by feeding it to a transform that outputs the correlation between observed and
predicted price.

+++

## Use hyperparameters

3. Hyperparameters and bindings. Concisely describe and execute hyperparameter searches
and data batching.

## Run on distributed backends

4. Run on distributed backends, exploiting the parallelism inherent to any branching
operations in the pipeline.
