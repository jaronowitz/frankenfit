# 🧟 Frankenfit 📈📊

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/maxbane/frankenfit?sort=semver)](https://github.com/maxbane/frankenfit/releases)
[![pytest](https://github.com/maxbane/frankenfit/actions/workflows/pytest.yml/badge.svg)](https://github.com/maxbane/frankenfit/actions/workflows/pytest.yml)
[![docs](https://github.com/maxbane/frankenfit/actions/workflows/docs.yml/badge.svg)](https://github.com/maxbane/frankenfit/actions/workflows/docs.yml)
[![mypy](https://github.com/maxbane/frankenfit/actions/workflows/mypy.yml/badge.svg)](https://github.com/maxbane/frankenfit/actions/workflows/mypy.yml)
[![license](https://img.shields.io/badge/license-BSD-red)](https://github.com/maxbane/frankenfit/blob/main/LICENSE.txt)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Frankenfit is a Python library for data scientists that provides a domain-specific
language (DSL) for creating, fitting, and applying predictive data modeling pipelines.
Its key features are:

* A concise and readable **DSL** (inspired by the pandas [method-chaining
  style](https://tomaugspurger.github.io/posts/method-chaining/)) to create data
  modeling **pipelines** from chains of composable building blocks called
  **transforms**. Pipelines themselves are composable, re-usable, and extensible, with
  a thorough [library of
  transforms](https://maxbane.github.io/frankenfit/current/api.html#transform-library)
  available for building, grouping, and combining pipelines in useful ways.
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
  enable smart auto-completions by IDEs.
  [Visualizations](https://maxbane.github.io/frankenfit/current/transforms_and_pipelines.html#visualizing-pipelines)
  help you see what your pipelines are doing. You can [implement your own
  transforms](https://maxbane.github.io/frankenfit/current/implementing_transforms.html)
  with almost zero boilerplate.

Frankenfit takes some inspiration from scikit-learn's [`pipeline`
module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline),
but aims to be much more general-purpose and flexible. It integrates easily with
industry-standard libraries like [pandas](https://pandas.pydata.org),
[scikit-learn](https://scikit-learn.org) and [statsmodels](https://www.statsmodels.org),
or your own in-house library of statistical models and data transformations.

## Learn more

Visit the [github page](https://github.com/maxbane/frankenfit) for more information
about Frankenfit.

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
