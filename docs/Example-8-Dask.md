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
logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.info('Hello from notebook')

from pydataset import data
df = data('diamonds').reset_index().set_index('index')

import frankenfit as ff
reload(ff)

logging.getLogger('frankenfit').setLevel(logging.INFO)
```

```{code-cell} ipython3
client.shutdown()
```

```{code-cell} ipython3
cluster.close()
```

```{code-cell} ipython3
from dask import distributed
cluster = distributed.LocalCluster(n_workers=4, threads_per_worker=2, host="devbox.homenet.thebanes.org")
print(cluster.dashboard_link)
client = distributed.Client(cluster)
client

#cluster = distributed.LocalCluster(n_workers=1, threads_per_worker=1)
#client = distributed.Client(cluster)
#client = distributed.Client()
#client
```

```{code-cell} ipython3
backend = ff.DaskBackend(client)
backend
```

```{code-cell} ipython3
def foo(x):
    return f"foo({x})"

backend.submit("key_foo", foo, 42).result()
```

```{code-cell} ipython3
t = ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
t.fit(backend=backend)
```

```{code-cell} ipython3
t.apply(backend=backend)
```

```{code-cell} ipython3
t.apply(backend=backend).result()
```

```{code-cell} ipython3
t.fit(backend=backend).apply()
```

```{code-cell} ipython3
t.fit().apply(backend=backend).result()
```

```{code-cell} ipython3
t.fit(backend=backend).apply(backend=backend).result()
```

```{code-cell} ipython3
t.fit(backend=backend).state().result()
```

```{code-cell} ipython3
(
    ff.DataFramePipeline()
    .read_dataset('./diamonds.csv', format='csv', index_col='index')
).fit(backend=backend).state().result()
```

```{code-cell} ipython3
t = ff.dataframe.ZScore(["price"])
#t.fit(df, backend=backend)  # warning about serializing df into task graph
t.fit(df).state()
```

```{code-cell} ipython3
t = ff.dataframe.ZScore(["price"])
t.fit(df, backend=backend).state().result()  # warning about serializing df into task graph
```

```{code-cell} ipython3
from frankenfit.dataframe import ZScore, Clip
```

```{code-cell} ipython3
# Future-passing
read = ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
zscore = ZScore(["price"])
clip = Clip(["price"], upper=2, lower=-2)

d = None
ft = read.fit(d, backend=backend)
d = ft.apply(d, backend=backend)
```

```{code-cell} ipython3
d.unwrap()
```

```{code-cell} ipython3
ft = zscore.fit(d, backend=backend)
d = ft.apply(d, backend=backend)
```

```{code-cell} ipython3
d
```

```{code-cell} ipython3
ft = clip.fit(d, backend=backend)
d = ft.apply(d, backend=backend)
```

```{code-cell} ipython3
ft.state()
```

```{code-cell} ipython3
d.result()
```

```{code-cell} ipython3
t = (
    ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
    .then()  # something wrong with return type of then() here; type params not bound?
    .z_score(["price"])
    .clip(["price"], lower=-2, upper=2)
)
t
```

```{code-cell} ipython3
tf = t.fit(backend=backend)
tf.state()
```

```{code-cell} ipython3
FEATURES = ["carat", "x", "y", "z", "depth", "table"]

def bake_features(cols):
    return (
        ff.DataFramePipeline()
        .print(fit_msg=f"Baking: {cols}")
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )

ff.ReadDataset('./diamonds.csv', format='csv', index_col='index').then(bake_features(FEATURES)).apply().head()
```

```{code-cell} ipython3
ff.ReadDataset('./diamonds.csv', format='csv', index_col='index').then(bake_features(FEATURES)).apply(backend=backend).result().head()
```

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

FEATURES = ["carat", "x", "y", "z", "depth", "table"]

def bake_features(cols):
    return (
        ff.DataFramePipeline()
        .print(fit_msg=f"Baking: {cols}")
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )

complex_pipeline = (
    ff.DataFramePipeline()[FEATURES + ["{response_col}"]]
    .copy("{response_col}", "{response_col}_train")
    .winsorize("{response_col}_train", limit=0.05)
    .pipe(["carat", "{response_col}_train"], np.log1p)
    #.then(bake_features(FEATURES))
    .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
    .sk_learn(
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

complex_pipeline.hyperparams()
```

```{code-cell} ipython3
bindings = {"response_col": "price", "bake_features": True, "predictors": FEATURES}
```

```{code-cell} ipython3
read_diamonds = ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
```

```{code-cell} ipython3
#ff.ReadDataset('./diamonds.csv', format='csv', index_col='index').then(complex_pipeline).fit(bindings=bindings)
read_diamonds.then(complex_pipeline).fit(bindings=bindings).apply().head()
```

```{code-cell} ipython3
fit = read_diamonds.then(complex_pipeline).fit(bindings=bindings, backend=backend)
fit
```

```{code-cell} ipython3
fit.state()
```

```{code-cell} ipython3
fit.apply(backend=backend).result().head()
```

```{code-cell} ipython3
fit.apply().head()
```

```{code-cell} ipython3
read_diamonds.then(complex_pipeline).apply(bindings=bindings, backend=backend).result().head()
```

```{code-cell} ipython3
pip = ff.ReadDataset('./diamonds.csv', format='csv', index_col='index').then(complex_pipeline)
```

```{code-cell} ipython3
pip.fit(bindings=bindings, backend=backend).apply(backend=backend)
```

```{code-cell} ipython3
pip.apply(bindings=bindings, backend=backend)
```

```{code-cell} ipython3
# Good TODOs:
# [x] Optimize StatelessTransform.fit() not to submit/always use DummyBackend?
# Impure Transforms like data readers.
# BasePipeline.fit() and FitTransform[BasePipeline, ...] should chain backend submissions, like BasePipeline.apply()
# Backend.fit()/apply() methods. apply() overloaded for FitTransform, StatelessTransform, BasePipeline.
# ForBindings: combine_fun param, maybe generic in DataResult; DataFramePipeline expects specialized for DataFrames
# Start parallelizing appropriate transforms: ForBindings, Join, GroupByCols, GroupByBindings
```
