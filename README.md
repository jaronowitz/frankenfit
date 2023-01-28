# 🧟 Frankenfit 📈📊

[![PyPI](https://img.shields.io/pypi/v/frankenfit)](https://pypi.org/project/frankenfit/)
![pyversions](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)
[![pytest](https://github.com/maxbane/frankenfit/actions/workflows/pytest.yml/badge.svg)](https://github.com/maxbane/frankenfit/actions/workflows/pytest.yml)
[![docs](https://github.com/maxbane/frankenfit/actions/workflows/docs.yml/badge.svg)](https://github.com/maxbane/frankenfit/actions/workflows/docs.yml)
[![mypy](https://github.com/maxbane/frankenfit/actions/workflows/mypy.yml/badge.svg)](https://github.com/maxbane/frankenfit/actions/workflows/mypy.yml)
[![license](https://img.shields.io/badge/license-BSD-red)](https://github.com/maxbane/frankenfit/blob/main/LICENSE.txt)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

📖 [Current documentation](https://maxbane.github.io/frankenfit/current/)

Frankenfit is a Python library for data scientists that provides a domain-specific
language (DSL) for creating, fitting, and applying predictive data modeling pipelines.
With Frankenfit, you can:

* [Create
  pipelines](https://maxbane.github.io/frankenfit/current/synopsis.html#create-pipelines)
  using a DSL of call-chain methods.
* [Fit pipelines and apply them to
  data](https://maxbane.github.io/frankenfit/current/synopsis.html#fit-pipelines-and-apply-them-to-data)
  to generate predictions.
* [Use
  hyperparameters](https://maxbane.github.io/frankenfit/current/synopsis.html#use-hyperparameters)
  to generalize your pipelines and concisely execute hyperparameter searches and data
  batching.
* [Run your pipelines on distributed
  backends](https://maxbane.github.io/frankenfit/current/synopsis.html#run-on-distributed-backends)
  (currently [Dask](https://www.dask.org)), exploiting the parallelism inherent to any
  branching operations in a pipeline.

See the [Synopsis and
overview](https://maxbane.github.io/frankenfit/current/synopsis.html) for summaries of
each of these workflows with a running example. Subsequent sections of the documentation
detail how everything works from the ground up.

Frankenfit takes some inspiration from scikit-learn's [`pipeline`
module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline),
but aims to be much more general-purpose and flexible. It integrates easily with
industry-standard libraries like [pandas](https://pandas.pydata.org),
[scikit-learn](https://scikit-learn.org) and [statsmodels](https://www.statsmodels.org),
or your own in-house library of statistical models and data transformations.

## Example

Suppose we want to model the prices of round-cut diamonds using the venerable
[diamonds](https://ggplot2.tidyverse.org/reference/diamonds.html) dataset, which is
often used to teach regression. It looks like this:

|       |   carat | cut       | color   | clarity   |   depth |   table |   price |    x |    y |    z |
|------:|--------:|:--------  |:--------|:----------|--------:|--------:|--------:|-----:|-----:|-----:|
|     1 |    0.23 | Ideal     | E       | SI2       |    61.5 |      55 |     326 | 3.95 | 3.98 | 2.43 |
|     2 |    0.21 | Premium   | E       | SI1       |    59.8 |      61 |     326 | 3.89 | 3.84 | 2.31 |
|     3 |    0.23 | Good      | E       | VS1       |    56.9 |      65 |     327 | 4.05 | 4.07 | 2.31 |
|   ... |     ... | ...       | ...     | ...       |     ... |     ... |     ... |  ... |  ... |  ... |
| 53939 |    0.86 | Premium   | H       | SI2       |    61   |      58 |    2757 | 6.15 | 6.12 | 3.74 |
| 53940 |    0.75 | Ideal     | D       | SI2       |    62.2 |      55 |    2757 | 5.83 | 5.87 | 3.64 |

We can use Frankenfit to define a pipeline for predicting `price` with a linear
regression (from scikit-learn) on `carat`, `depth`, and `table`:

```python
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
    # Transform carat variable to log-carats
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

We can fit this model on some training data:

```python
fit_diamond_model = diamond_model.fit(train_df)
```

And apply it out-of-sample to some test data:

```python
predictions_df = fit_diamond_model.apply(test_df)
```

⭐ When we do so, our entire end-to-end model of diamond prices, including feature
preparation and regression, is fit **strictly** on one set of data (`train_df`) and
applied strictly **out-of-sample** to new data (`test_df`). The columns in `test_df` are
winsorized using the quantiles that were observed in `train_df`, z-scored using the
means and standard deviations that were observed in `train_df`, and predicted prices are
generated using the regression betas that were learned on `train_df`.

The ability to fit a complex pipeline on one set of data and use the fit state to
generate predictions on different data is fundamental to statistical resampling
techniques like cross-validation, and Frankenfit allows you to express many such
use-cases elegantly. For example, we can easily perform 5-fold
[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) of
`diamond_model` by embedding it in an outer pipeline that partitions the data into
groups and uses `group_by_cols()` to apply `diamond_model` to each group using the state
that results from fitting it on every other group:

```python
crossval_pipeline = (
    do
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
    # this gives us a dataset of entirely out-of-sample predictions, which we
    # score with correlation()
    .correlation("price_hat", "price")
)

# Run with:
crossval_pipeline.apply(full_df)
```

See the [Synopsis and
overview](https://maxbane.github.io/frankenfit/current/synopsis.html) section of the
documentation for a more extended walkthrough of this example, and then dive into
[Transforms and
pipelines](https://maxbane.github.io/frankenfit/current/transforms_and_pipelines.html)
to learn how it works from the ground up.

## Getting started

```
$ pip install frankenfit
```

If you want to use the [Dask](https://www.dask.org) backend for distributed computation
of your pipelines:
```
$ pip install "frankenfit[dask]"
```

You may also need to install [GraphViz](https://graphviz.org/) for visualizations to
work. On Ubuntu/Debian:
```
$ sudo apt install graphviz
```

The author of Frankenfit recommends importing it like this:
```python
import frankenfit as ff
```

Everything you need to get going is available in the public
[API](https://maxbane.github.io/frankenfit/current/api.html), `ff.*`. You might want to
start with a [synopsis](https://maxbane.github.io/frankenfit/current/synopsis.html) of
what you can do and proceed from there.

## Documentation

The most up-to-date documentation, corresponding to the unreleased `main` branch of this
repository, is available here: https://maxbane.github.io/frankenfit/current/.

The documentation provides a detailed narrative walkthrough of using the library for
predictive data modeling, as well as a complete API reference.  Please check it out!

## Type annotations

The entire Frankenfit library is meticulously type-annotated and checked with
[`mypy`](https://mypy.readthedocs.io), making use of `Generic` classes where it is
sensible to do so. Aside from catching logical errors, the main benefit of this to users
is that modern IDEs like [Visual Studio Code](https://code.visualstudio.com/) can
interpret the annotations statically to report the types of expressions and provide
intelligent auto-completions.

![screenshot-vscode](docs/_static/sshot-vscode-intellisense-frankenfit-pipeline.png)

Frankenfit includes its own [mypy plugin](src/frankenfit/mypy.py). It is highly
recommended to enable it if you are using mypy to type-check your own code as a
Frankenfit user. Just include the following in your project's
[`pyproject.toml`](https://mypy.readthedocs.io/en/stable/extending_mypy.html#configuring-mypy-to-use-plugins):

```toml
[tool.mypy]
plugins = "frankenfit.mypy"
```

## Development

If you're *not* hacking on the Frankenfit codebase itself, and just want to build or
install it from source, `$ pip build .`, `$ pip sdist .`, `$ pip install .` should all
work out of the box without any special needs, as long as you're using Python 3.8+ and a
recent version of `pip`.

To get started with hacking on Frankenfit itself, make sure that the `python3.10`
binary is on your path, clone this repo, and run:

```
# this just creates a venv and does `pip install -e ".[dev]"` and `pre-commit install`
$ ./setup-venv-dev python3.10
$ source ./.venv-dev/bin/activate
```

From there you may run tests with `$ tox -e py`, `$ tox mypy`, and so on.

The setup script automatically installs git pre-commit hooks. When you run `$ git
commit`, a few linters will run, possibly modifying the source. If files are modified by
the linters, it is necessary to `$ git add` them to staging again and re-run `$ git
commit`.

### Tests

We use the [`pytest`](pytest.org) testing framework together with the [`tox`](tox.wiki)
(v3) test runner. Tests live under `tests/` and are discovered by `pytest` according to
its normal discovery rules. Please be diligent about writing or updating tests for any
new or changed functionality. We use GitHub Actions to run tests on every pull request
and every push to `main`.

### Code style and linters

We follow [`black`](https://github.com/psf/black) to the letter for code formatting and
additionally target [`flake8`](https://flake8.pycqa.org/) compliance, minus a few
exceptions documented in `pyproject.toml`. This is enforced at commit-time by
[`pre-commit`](pre-commit.com) hooks, and checked by post-push continuous integration.

### Dependencies and where they are defined

There are three categories of dependencies for the project:

* Run-time dependencies. These are the dependencies required of users to actually import
  and use the library. They are defined in `pyproject.toml` and will be installed
  automatically by pip when installing the `frankenfit` package. The set of required
  dependencies is small: `attrs`, `pandas`, `pyarrow`, and `graphviz`.

  * When installing Frankenfit, users may specify the optional feature (a.k.a. "extra")
    `dask` (i.e., `"frankenfit[dask]"`) to install `dask-distributed`, enabling use of
    the [`DaskBackend`](https://maxbane.github.io/frankenfit/current/backends.html).
    This is not included by default.

* Test-time and type-checking dependencies. Running the test suite requires additional
  dependencies beyond the run-time dependencies. They are declared by the `tests` extra
  in `pyproject.toml`.  Note that the `tests` extra depends on the `dask` extra, so that
  we may test Dask-related functionality.

* Documentation dependencies. Building the documentation requires [Jupyter
  Book](https://jupyterbook.org) and other additional dependencies, as declared by the
  `docs` extra in `pyproject.toml`.

* Developer dependencies, as declared by the `dev` extra. These are packages that a
  developer hacking on Frankenfit needs to make full use of the repository, including
  `tox` for actually running tox, `pre-commit` for running linters without
  actually making a commit, and so on. The `dev` extra itself depends on all of the
  other extras `[dask,docs,tests]`, so it suffices just to run `pip install -e ".[dev]"`
  on this repo to get started (which is what the setup script does).

### Writing documentation

Documentation lives in `docs/`, and we use [Jupyter Book](https://jupyterbook.org) to
build it as a static HTML site. Documentation content is written in Markdown
(specifically MyST), but the Python docstrings, which are included in the API reference
section of the documentation, must still be in reStructuredText (albeit [NumPy
style](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)), so it's a
bit of a Frankenstein situation. 🧟

The official color scheme of Frankenfit is, of course,
[Dracula](https://draculatheme.com/). 🧛
