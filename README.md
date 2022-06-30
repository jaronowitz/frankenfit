
# `frankenfit`: it's alive! it's fit!

Boomb, that's what it does.

## Getting started

### Examples

### Docs

## Development

We target Python 3.9+. To get started, make sure the `python3.9` binary is on your path,
clone this repo, and run:
```
$ ./setup-venv-dev python3.9
$ source ./.venv-dev/bin/activate
```

From there you may run tests with `$ tox` and install the package in editable model with
`$ pip install -e .`

The setup script automatically installs git pre-commit hooks. When you run `$ git commit`, a few linters will run, possibly modifying the source. If files are modified by the linters, it is necessary to `$ git add` them to staging again and re-run `$ git commit`.

### Tests

We use the `pytest` testing framework together with the `tox` test runner. Tests live
under `tests/` are discovered by `pytest` according to its normal discovery rules.
Please be diligent about writing or updating tests for any new or changed functionality.

### Code style

We follow `black` to the letter and additionally target `flake8` compliance. This is
enforced at commit-time by `pre-commit` hooks, and checked by `tox` at test-time.
