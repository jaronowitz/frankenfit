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
LOG.setLevel(logging.INFO)
LOG.info('Hello from notebook')

from pydataset import data

import frankenfit as ff
reload(ff.core)
reload(ff.universal)
reload(ff.dataframe)
reload(ff)

ffu = ff.universal
ffdf = ff.dataframe

logging.getLogger('frankenfit').setLevel(logging.INFO)
```

```{code-cell} ipython3
:tags: []

df = data('diamonds').reset_index().drop(['index'], axis=1)
df.head()
```

```{code-cell} ipython3
:tags: []

df.groupby("cut")[["table", "depth"]].mean()
```

```{code-cell} ipython3
:tags: []

df.groupby("cut").apply(len)
```

```{code-cell} ipython3
:tags: []

transform = ffdf.DeMean(["table", "depth"])
```

```{code-cell} ipython3
:tags: []

ffdf.GroupBy("cut", transform)
```

```{code-cell} ipython3
:tags: []

ffdf.GroupBy?
```

```{code-cell} ipython3
:tags: []

fit_gb = ffdf.GroupBy("cut", transform).fit(df)
```

```{code-cell} ipython3
:tags: []

s = fit_gb.state()
s
```

```{code-cell} ipython3
:tags: []

len(df)
```

```{code-cell} ipython3
:tags: []

result_df = fit_gb.apply(df)
result_df.head()
```

```{code-cell} ipython3
:tags: []

result_df[["table", "depth"]].mean()
```

```{code-cell} ipython3
:tags: []

result_df.groupby("cut")[["table", "depth"]].mean()
```

```{code-cell} ipython3
:tags: []

pip = (
    ff.DataFramePipeline()
    .stateless_lambda(len)
)
pip
```

```{code-cell} ipython3
:tags: []

pip.fit(df).apply(df)
```

```{code-cell} ipython3
:tags: []

ffdf.GroupBy("cut", pip).fit(df).apply(df)
```

```{code-cell} ipython3
:tags: []

df.groupby("cut", as_index=False, sort=False).apply(len)
```

```{code-cell} ipython3
:tags: []

pip = (
    ff.DataFramePipeline()
    .group_by("cut")
        .stateless_lambda(len)
)
pip
```

```{code-cell} ipython3
:tags: []

pip.fit(df).apply(df)
```

```{code-cell} ipython3
:tags: []

df.groupby("cut")
```

```{code-cell} ipython3
:tags: []

gb = ff.DataFramePipeline().group_by("cut")
```

```{code-cell} ipython3
:tags: []

gb
```

```{code-cell} ipython3
:tags: []

np.mean(df, axis=0)
```

```{code-cell} ipython3
:tags: []

df.groupby("cut")[["price"]].describe()
```

```{code-cell} ipython3
:tags: []

x = (
    ff.DataFramePipeline()
    .group_by("cut")
    .de_mean(["price"])
    [["cut", "price"]]
).apply(df)
x.groupby("cut")[["price"]].describe()
```

```{code-cell} ipython3
:tags: []

x = (
    ff.DataFramePipeline()
    .group_by("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
        .de_mean(["price"])
    [["cut", "price"]]
).apply(df)
x.groupby("cut")[["price"]].describe()
```

```{code-cell} ipython3
:tags: []

ff.DataFramePipeline.group_by?
```

```{code-cell} ipython3
:tags: []

pip = (
    ff.DataFramePipeline()
    .group_by("cut")
    .de_mean(["price"])
    [["cut", "price"]]
)
pip_fit = pip.fit(df)
```

```{code-cell} ipython3
:tags: []

pip
```

```{code-cell} ipython3
:tags: []

pip_fit
```

```{code-cell} ipython3
:tags: []

pip_fit.find_by_tag("GroupBy#7").state().loc[0,'__state__'].state()
```

```{code-cell} ipython3
:tags: []

s = pip_fit.find_by_tag("GroupBy#7").state().loc[0,'__state__'].state()
type(s)
```

```{code-cell} ipython3
:tags: []

pip_fit.find_by_tag("GroupBy#7").state().assign(
    mean = lambda df: df['__state__'].map(lambda x: x.state()[0])
)
```

```{code-cell} ipython3
:tags: []

pip = (
    ff.DataFramePipeline()
    #.group_by("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
    .group_by("cut")
        .stateless_lambda(lambda df: df[["price"]].mean())
)
result = pip.apply(df).set_index('cut').sort_index().reset_index()#.sort_values('cut')
```

```{code-cell} ipython3
:tags: []

target = df.groupby("cut")[['price']].mean().sort_index().reset_index()
```

```{code-cell} ipython3
:tags: []

result.equals(target)
```

```{code-cell} ipython3
:tags: []

result
```

```{code-cell} ipython3
:tags: []

target
```

```{code-cell} ipython3
:tags: []

pip = (
    ff.DataFramePipeline()
    .group_by("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
    #.group_by("cut")
        .de_mean('price')
    [['cut', 'price']]
)
result = pip.apply(df)
result
```

```{code-cell} ipython3
:tags: []

cuts = pd.Series(df['cut'].unique(), name='cut').sort_values()
cut_means = pd.DataFrame(dict(
    cut=cuts,
    price=cuts.map(lambda v: df.loc[df['cut'] != v]['price'].mean())
))
cut_means
```

```{code-cell} ipython3
:tags: []

target = (
    df
    .merge(cut_means, how='left', on='cut', suffixes=('', '_mean'))
    .assign(price=lambda df: df['price'] - df['price_mean'])
    [['cut', 'price']]
)
target
```

```{code-cell} ipython3
:tags: []

result.equals(target)
```

```{code-cell} ipython3
pip = (
    ff.DataFramePipeline()
    .group_by("cut")
        .de_mean(["price"])
)
d = pip.fit(df.loc[df['cut'] != 'Fair']).apply(df)
d
#pip.fit(df).apply(df)
#result = pip.apply(df).set_index('cut').sort_index().reset_index()#.sort_values('cut')
#result
```

```{code-cell} ipython3

```

```{code-cell} ipython3
df.info()
(df.loc[df['cut'] != 'Fair']).info()
```
