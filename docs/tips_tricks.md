---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tips and tricks

## Type annotations

The whole Frankenfit library is meticulously type-annotated and type-checked (with
`mypy`), making using of generic classes where it makes sense to do so. Benefits:

1. Enhanced productivity in IDEs like vscode that can interpret the type annotations.
   Eample screenshots.
2. Catch logical errors sooner if you type-check your own code (but not obligation to
   do so.)

Call-chain methods are known and auto-completable by IDEs.

![screenshot-pipeline](_static/sshot-vscode-intellisense-frankenfit-pipeline.png)

Return types of `apply()` and `result()` are known, so that even complex expressions can
be auto-completed. Base and universal transforms are generic, `Identity[str]` example.

## Debug your pipelines with `print()` or `log_message()`

TODO: should we make some kind of `breakpoint()` Transform that triggers an actual
breakpoint?

## More concise and readable Pipelines

Use a top-level assignment like `do = ff.DataFramePipeline()` or `pipeline =
ff.DataFramePipeline()`, and use that for initiating all of your pipelines and
sub-pipelines. Examples.

## Break your pipelines into re-usable pieces

"Main model" pipeline, compose with data-reader pipelines and evaluation/scoring
pipeline.

## Use hyperparameters in column names

HPCols functionality for free in `cols` params of built-in transforms.

## Use `assign()` together with `select()`/`[]`, `ALL_COLS`, and affixes

Example from clunky `copy()` to sleek `assign()` + `suffix()`. TODO: `bracket()`.

## Use `if_fitting()` to do certain things only when fitting the pipeline

Example: creating a training response col. It's a waste to do so at apply-time.

## Convert state into data

Example of extracting betas from a regression.

## Working with large datasets

Use `read_dataset` (pyarrow Dataset) index-range as hyperparam; you can compute
summaries without loading the whole dataset in memory.

## `DaskBackend`: traces in the dask dashboard
