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

(implementing-transforms)=
# Implementing your own transforms

```{code-cell} ipython3
import pandas as pd
import frankenfit as ff

@ff.params
class DeMean(ff.Transform):
    """
    De-mean some columns.

    Parameters
    ----------
    cols : list(str)
        The names of the columns to de-mean.
    """
    cols: list[str]

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        # return state as a pandas Series of the columns' means in data_fit, indexed by
        # column name
        return data_fit[self.cols].mean()

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
        # return a new DataFrame in which the columns have been demeaned with respect to
        # the provided state
        return data_apply.assign(
            **{c: data_apply[c] - state[c] for c in self.cols}
        )
```

As authors of a Transform, in most cases we must implement `_fit` and `_apply` methods.

## Consider using `StatelessLambda` or `StatefulLambda`

## Implementing simple transforms with `_fit()` and `_apply()`

## Declaring parameters

## Using custom Transforms in pipelines

`Pipeline.then()`
`Pipeline.with_methods()`.

## Implementing complex transforms with `_submit_fit()` and `_submit_apply()`

## Considerations for complex Transforms

Overriding `hyerparams()`, `_visualize()`.

Using parallel backends. Traces.
