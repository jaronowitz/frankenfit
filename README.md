# `frankenfit`: it's alive! it's fit!

Boomb, that's what it does.

## Getting started

### Examples

### Docs

## Development

If you're not hacking on the `frankenfit` codebase itself, and just want to build or
install it from source, `$ pip build .`, `$ pip sdist .`, `$ pip install .` should all
work out of the box without creating any special virtual environment, as long as you're
using Python 3.9+ and a recent version of `pip`.

To get started with hacking on `frankenfit` itself, make sure that the `python3.9`
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

### Code style

We follow `black` to the letter and additionally target `flake8` compliance, minus a few
exceptions documented in `.flake8`. This is enforced at commit-time by `pre-commit`
hooks, and checked by `tox` at test-time.

### Dependencies and where they are defined

There are three categories of dependencies for the project:

* Run-time dependencies. These are the dependencies required of users to actually import
  and use the library. The are defined in `pyproject.toml` and will be installed
  automatically by pip when installing the `frankenfit` package.
* Test-time dependencies. Running the test suite requires additional dependencies beyond
  the run-time dependencies. These are defined in `tox.ini`
* Developer dependencies. These are packages that a developer hacking on `frankenfit`
  needs to make full use of the repository, including `pre-commit` for running linters
  without actually making a commit, `jupyter` for interacting with example notebooks, as
  well as all of the run-time and test-time dependencies to allow for editor
  autocompletions and ad hoc testing. The developer dependencies are defined in
  `requirements-dev.txt`, and are automatically installed to the environment created by
  the `setup-venv-dev` script.
