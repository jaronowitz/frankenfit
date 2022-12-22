---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: frankenfit-venv-dev
  language: python
  name: frankenfit-venv-dev
---

```{code-cell} ipython3
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
import logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.info('Hello from notebook')

from pydataset import data

import frankenfit as ff

import pyarrow.dataset as ds

logging.getLogger('frankenfit').setLevel(logging.INFO)
```

```{code-cell} ipython3
ff.dataframe.Assign(foo=1, bar=2, logprice=lambda df: np.log1p(df['price']), tag='boo')
```

```{code-cell} ipython3
df = data('diamonds').reset_index().set_index('index')

(
    ff.DataFramePipeline()
    .assign(
        intercept=1,
        grp=lambda df: df.index % 5,
        grp_2=lambda self, df: df.index % self.bindings()['k']
    )
).apply(df, bindings={'k': 3})
```

```{code-cell} ipython3
pip = (
    ff.DataFramePipeline()
    .assign({
        'intercept': 1,
        ff.HPFmtStr('grp_{k}'): lambda self, df: df.index % self.bindings()['k']
    }, tag='foo')
)
display(pip)
display(pip.hyperparams())
display(pip.apply(df, bindings={'k': 3}).head(10))
```
