# Frankenfit: it's alive! it's fit! 🧟

Frankenfit is a Python library for describing, fitting, and applying predictive data
modeling pipelines.

## Getting started

### Examples

```python
from sklearn.linear_model import LinearRegression

FEATURES = ["carat", "x", "y", "z", "depth", "table"]

def bake_features(cols):
    return (
        ff.Pipeline()
        .print(fit_msg=f"Baking: {cols}")
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )

pipeline = (
    ff.Pipeline()
    [FEATURES + ["{response_col}"]]
    .copy("{response_col}", "{response_col}_train")
    .winsorize("{response_col}_train", limit=0.05)
    .pipe(["carat", "{response_col}_train"], np.log1p)
    .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
    .sklearn(
        LinearRegression,
        # x_cols=["carat", "depth", "table"],
        x_cols=ff.HP("predictors"),
        response_col="{response_col}_train",
        hat_col="{response_col}_hat",
        class_params={"fit_intercept": True},
    )
    # transform {response_col}_hat from log-dollars back to dollars
    .copy("{response_col}_hat", "{response_col}_hat_dollars")
    .pipe("{response_col}_hat_dollars", np.expm1)
)
```

### Docs

## Development

If you're not hacking on the Frankenfit codebase itself, and just want to build or
install it from source, `$ pip build .`, `$ pip sdist .`, `$ pip install .` should all
work out of the box without creating any special virtual environment, as long as you're
using Python 3.9+ and a recent version of `pip`.

To get started with hacking on Frankenfit itself, make sure that the `python3.9`
binary is on your path, clone this repo, and run:

```
$ ./setup-venv-dev python3.9
$ source ./.venv-dev/bin/activate
```

From there you may run tests with `$ tox` and install the package in editable model with
`$ pip install -e .`

The setup script automatically installs git pre-commit hooks. When you run `$ git commit`, a few linters will run, possibly modifying the source. If files are modified by the linters, it is necessary to `$ git add` them to staging again and re-run `$ git commit`.

### Tests

We use the `pytest` testing framework together with the `tox` test runner. Tests live
under `tests/` and are discovered by `pytest` according to its normal discovery rules.
Please be diligent about writing or updating tests for any new or changed functionality.

### Code style and linters

We follow `black` to the letter and additionally target `flake8` compliance, minus a few
exceptions documented in `.flake8`. This is enforced at commit-time by `pre-commit`
hooks, and checked by `tox` at test-time.

### Dependencies and where they are defined

There are three categories of dependencies for the project:

* Run-time dependencies. These are the dependencies required of users to actually import
  and use the library. They are defined in `pyproject.toml` and will be installed
  automatically by pip when installing the `frankenfit` package.
* Test-time dependencies. Running the test suite requires additional dependencies beyond
  the run-time dependencies. These are defined in `tox.ini`
* Developer dependencies. These are packages that a developer hacking on Frankenfit
  needs to make full use of the repository, including `pre-commit` for running linters
  without actually making a commit, `jupyter` for interacting with example notebooks, as
  well as all of the run-time and test-time dependencies to allow for editor
  autocompletions and ad hoc testing. The developer dependencies are defined in
  `requirements-dev.txt`, and are automatically installed to the environment created by
  the `setup-venv-dev` script.

### Writing documentation

Documentation lives in `docs/`, and we use [Jupyter Book]() to build it as a static HTML
site. Documentation content is written in Markdown (specifically MyST), but the Python
docstrings, which are included in the API reference section of the documentation, must
still be in reStructuredText, so it's a bit of a Frankenstein situation.
