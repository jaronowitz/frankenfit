---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: 'Python 3.10.6 (''.venv-dev'': venv)'
  language: python
  name: python3
---

```{code-cell} ipython3
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.info('Hello from notebook')

from pydataset import data
df = data('diamonds').reset_index().set_index('index')

import frankenfit as ff
from frankenfit import core as ffc
# reload(ffc)

logging.getLogger('frankenfit').setLevel(logging.INFO)

from typing import Optional, Any, TypeVar
```

```{code-cell} ipython3
i = ff.Identity[str]()
z = i.fit("x").apply("x")
y = i.apply("x")
```

```{code-cell} ipython3
@ff.params
class DeMean(ff.Transform[pd.DataFrame, pd.DataFrame]):
    cols: list[str]

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        return data_fit[self.cols].mean()

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
        means = state
        return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})


# reveal_type(DeMean.FitDeMean)

dmn = DeMean(["price"])
ft = dmn.fit(df)
x = ft.resolved_transform()
result = dmn.fit(df).apply(df)
```

```{code-cell} ipython3
DeMean.fit?
```

```{code-cell} ipython3
result.head()
```

```{code-cell} ipython3
ff.Transform[pd.DataFrame, pd.DataFrame]._fit
dmn.fit?
```

```{code-cell} ipython3
ft.apply?
```

```{code-cell} ipython3
ft.resolved_transform?
```

```{code-cell} ipython3
from typing import ClassVar, Type
from abc import abstractmethod

class FitDeMean(ff.FitTransform["DeMean", pd.DataFrame, pd.DataFrame]):
    @abstractmethod
    def apply(
        self,
        data_apply: Optional[pd.DataFrame] = None,
        backend: Optional[ff.Backend] = None,
    ) -> pd.DataFrame:
        """My apply docstr"""
        print("my overridden apply")
        return super().apply(data_apply=data_apply, backend=backend)

def _weighted_means(df, cols, w_col):
    return df[cols].multiply(df[w_col], axis="index").sum() / df[w_col].sum()

@ff.params
class DeMean(ffc.Transform[pd.DataFrame, pd.DataFrame]):
    """
    De-mean some columns.
    """
    cols: list[str]
    w_col: Optional[str] = None

    FitTransformClass: ClassVar[Type[ff.FitTransform]] = FitDeMean


    def _fit(self, data_fit: pd.DataFrame, bindings=None) -> pd.Series:
        if self.w_col is not None:
            return _weighted_means(data_fit, self.cols, self.w_col)
        return data_fit[self.cols].mean()

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
        means = state
        return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})

    _Self = TypeVar("_Self", bound="DeMean")

    def fit(self: _Self, data_fit: Optional[pd.DataFrame] = None, bindings: Optional[ff.Bindings] = None, backend: Optional[ff.Backend] = None) -> ff.FitTransform[_Self, pd.DataFrame, pd.DataFrame]:
        print("my overridden fit")
        return super().fit(data_fit, bindings, backend)
```

```{code-cell} ipython3
DeMean.fit?
```

```{code-cell} ipython3
DeMean.FitTransformClass.apply?
```

```{code-cell} ipython3
DeMean.FitTransformClass?
```

```{code-cell} ipython3
FitDeMean.apply?
```

```{code-cell} ipython3
dmn = DeMean(["price"])
fit_dmn = dmn.fit(df)
print(fit_dmn)
fit_dmn.state()
```

```{code-cell} ipython3
fit_dmn
```

```{code-cell} ipython3
fit_dmn.state?
```

```{code-cell} ipython3
fit_dmn.apply?
```

```{code-cell} ipython3
dmn.fit(df, backend=None)
```

```{code-cell} ipython3
fit_dmn.apply?
```

```{code-cell} ipython3
dmn.fit(df).apply(df, backend=None)
```

```{code-cell} ipython3
dmn_hp = DeMean(ffc.HP("cols"))
dmn_hp.fit(df, bindings={"cols": ["price"]}).apply(df)
```

```{code-cell} ipython3

```
