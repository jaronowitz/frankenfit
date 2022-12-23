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
:tags: []

from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')
import logging
LOG = logging.getLogger(__name__)
LOG.info('Hello from notebook')

from pydataset import data

import frankenfit as ff

logging.getLogger('frankenfit').setLevel(logging.INFO)
```

```{code-cell} ipython3
:tags: []

df = data('diamonds')
```

```{code-cell} ipython3
:tags: []

df
```

```{code-cell} ipython3
:tags: []

ff.dataframe.SKLearn?
```

```{code-cell} ipython3
:tags: []

import sklearn
from sklearn.linear_model import LinearRegression
sk = ff.dataframe.SKLearn(LinearRegression, ['carat', 'depth', 'table'], 'price', 'price_hat', class_params={'fit_intercept': True})
sk
```

```{code-cell} ipython3
:tags: []

fit_sk = sk.fit(df)
fit_sk
```

```{code-cell} ipython3
:tags: []

fit_sk._field_names
```

```{code-cell} ipython3
:tags: []

fit_sk.state().coef_
```

```{code-cell} ipython3
:tags: []

sk.params()
```

```{code-cell} ipython3
:tags: []

df[['carat', 'depth', 'table', 'price']].hist(figsize=(8,6));
```

```{code-cell} ipython3
:tags: []

from sklearn.linear_model import LinearRegression

FEATURES = ["carat", "x", "y", "z", "depth", "table"]

def bake_features(cols):
    return (
        ff.DataFramePipeline()
        .print(fit_msg=f"Baking: {cols}")
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.)
        .clip(cols, upper=2, lower=-2)
    )

pipeline = (
    ff.DataFramePipeline()
    .copy("{response_col}", "{response_col}_train")
    .winsorize("{response_col}_train", limit=0.05)
    .pipe(["carat", "{response_col}_train"], np.log1p)
    .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
    .sk_learn(
        LinearRegression,
        #x_cols=["carat", "depth", "table"],
        x_cols=ff.HP('predictors'),
        response_col="{response_col}_train",
        hat_col="{response_col}_hat",
        class_params={"fit_intercept": True}
    )
    # transform {response_col}_hat from log-dollars back to dollars
    .copy("{response_col}_hat", "{response_col}_hat_dollars")
    .pipe("{response_col}_hat_dollars", np.expm1)
)
pipeline
```

```{code-cell} ipython3
:tags: []

pipeline.hyperparams()
```

```{code-cell} ipython3
:tags: []

df = data('diamonds')
index_all = set(df.index)
index_in = set(np.random.choice(df.index, size=int(len(df) / 2), replace=False))
index_out = index_all - index_in
df_in = df.loc[list(index_in)]
df_out = df.loc[list(index_out)]

import itertools

fits = []
for bf in (True, False):
    bindings = {'bake_features': bf, 'response_col': 'price', 'predictors': FEATURES}
    print(bindings)
    fit_pipeline = pipeline.fit(df_in, bindings)
    fits.append(fit_pipeline)
```

```{code-cell} ipython3
:tags: []

for fit in fits:
    # performance on own training data
    pdf = fit.apply(df_in)
    corr_train = pdf[['price_hat', 'price_train']].corr().iloc[0,1]
    corr_raw = pdf[['price_hat_dollars', 'price']].corr().iloc[0,1]
    print(fit.bindings())
    print('\tCorr with in-sample training response:\t%.4f' % corr_train)
    print('\tCorr with in-sample raw response:\t%.4f' % corr_raw)
```

```{code-cell} ipython3
:tags: []

for fit in fits:
    # performance on held-out data
    pdf = fit.apply(df_out)
    corr_train = pdf[['price_hat', 'price_train']].corr().iloc[0,1]
    corr_raw = pdf[['price_hat_dollars', 'price']].corr().iloc[0,1]
    print(fit.bindings())
    print('\tCorr with out-of-sample training response:\t%.4f' % corr_train)
    print('\tCorr with out-of-sample raw response:\t\t%.4f' % corr_raw)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import yfinance
```

```{code-cell} ipython3
:tags: []

x = yfinance.Ticker('MSFT')
```

```{code-cell} ipython3
:tags: []

df = x.history()
```

```{code-cell} ipython3
:tags: []

df.head()
```

```{code-cell} ipython3
:tags: []

x.financials
```

```{code-cell} ipython3
:tags: []

yfinance.download?
```

```{code-cell} ipython3
:tags: []

x = yfinance.download(['msft', 'aapl'], start='2010-01-01', auto_adjust=True)
```

```{code-cell} ipython3
:tags: []

x.head()
```

```{code-cell} ipython3
:tags: []

x.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index(level=1)
```

```{code-cell} ipython3
:tags: []

x['Close'].plot()
```

```{code-cell} ipython3
:tags: []

x.describe()
```
