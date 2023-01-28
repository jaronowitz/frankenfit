# 🧟 Frankenfit 📈📊

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/maxbane/frankenfit?sort=semver)](https://github.com/maxbane/frankenfit/releases)
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
