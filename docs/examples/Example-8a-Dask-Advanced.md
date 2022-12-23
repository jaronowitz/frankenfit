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

# logging.getLogger('frankenfit').setLevel(logging.DEBUG)
```

```{code-cell} ipython3
try:
    client.shutdown()
    cluster.close()
except:
    pass
```

```{code-cell} ipython3
from dask import distributed
cluster = distributed.LocalCluster(
    n_workers=4, threads_per_worker=2, host="devbox.homenet.thebanes.org",
    scheduler_port=0, dashboard_address=':0'
)
print(cluster.dashboard_link)
client = distributed.Client(cluster)
client

#cluster = distributed.LocalCluster(n_workers=1, threads_per_worker=1)
#client = distributed.Client(cluster)
#client = distributed.Client()
#client
```

```{code-cell} ipython3
dask = ff.DaskBackend(client)
dask
```

```{code-cell} ipython3
p = (
    ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
    .then()
    .z_score(["price"])
    .clip(["price"], lower=-2, upper=2)
)
p
```

```{code-cell} ipython3
p.apply().head()
```

```{code-cell} ipython3
dask.apply(p).result().head()
```

```{code-cell} ipython3
p.on_backend(dask).apply().head()
```

```{code-cell} ipython3
tf = dask.fit(p)
tf.state()
```

```{code-cell} ipython3
dask.apply(tf).result().head()
```

```{code-cell} ipython3
p.on_backend(dask).fit().state()
```

```{code-cell} ipython3
p.on_backend(dask).fit().apply().head()
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
dask.apply(
    ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
    .then(bake_features(FEATURES))
).result().head()
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
    .pipe(["carat", "{response_col}_train"], np.log1p)
    .winsorize("{response_col}_train", limit=0.05)
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
fit = dask.fit(read_diamonds.then(complex_pipeline), bindings=bindings)
fit
```

```{code-cell} ipython3
fit.state()
```

```{code-cell} ipython3
fit.materialize_state().state()
```

```{code-cell} ipython3
dask.apply(fit).result().head()
```

```{code-cell} ipython3
fit.backend
```

```{code-cell} ipython3
fit.apply().head()
```

```{code-cell} ipython3
def dataframable_bindings(bindings):
    result = {}
    for name, val in bindings.items():
        if type(val) not in (float, int, str):
            val = str(val)
        result[name] = val
    return result

def combine_results_as_dataframe(results) -> pd.DataFrame:
    binding_cols: set[str] = set()
    dfs = []
    for x in results:
        dfs.append(x.result.assign(**dataframable_bindings(x.bindings)))
        binding_cols |= x.bindings.keys()
    df = pd.concat(dfs, axis=0)
    df = df.set_index(list(binding_cols))
    return df

pip = (
    ff.DataFramePipeline()
    .for_bindings(
        [
            {"predictors": ["carat"]},
            {"predictors": ["depth"]},
            {"predictors": ["table"]},
        ],
        combine_results_as_dataframe,
    )
        .then(
            read_diamonds
            .then(complex_pipeline)
            .correlation(["{response_col}_hat_dollars"], ["{response_col}"])
        )
    # .rename({"{response_col}": "correlation"})
)

fit = pip.fit(bindings={"bake_features": True, "response_col": "price"})
# pip.hyperparams()
```

```{code-cell} ipython3
fit.apply()
```

```{code-cell} ipython3
pip.apply(bindings={"bake_features": True, "response_col": "price"})
```

```{code-cell} ipython3
# logging.getLogger("frankenfit").setLevel(logging.DEBUG)
r = dask.apply(pip, bindings={"bake_features": True, "response_col": "price"})
r.result()
```

```{code-cell} ipython3
fut = dask.apply(pip, bindings={"bake_features": True, "response_col": "price"})
```

```{code-cell} ipython3

```

```{code-cell} ipython3
dask.fit(pip, bindings={"bake_features": True, "response_col": "price"}).apply()
```

```{code-cell} ipython3
pip = (
    read_diamonds
    .then()
    .for_bindings(
        [
            {"predictors": ["carat"]},
            {"predictors": ["depth"]},
            {"predictors": ["table"]},
        ],
        combine_results_as_dataframe,
    )
        .then(
            complex_pipeline
            .correlation(["{response_col}_hat_dollars"], ["{response_col}"])
        )
)

r = dask.apply(pip, bindings={"bake_features": True, "response_col": "price"})
r.result()
```

```{code-cell} ipython3
df.head()
```

```{code-cell} ipython3
pip = (
    ff.DataFramePipeline()
    .for_bindings(
        [
            {"predictors": ["carat"]},
            {"predictors": ["depth"]},
            {"predictors": ["table"]},
        ],
        combine_results_as_dataframe,
    )
        .then(
            complex_pipeline
            .correlation(["{response_col}_hat_dollars"], ["{response_col}"])
        )
)

dask.apply(pip, df, bindings={"bake_features": True, "response_col": "price"}).result()
```

```{code-cell} ipython3
pip.visualize()
```

```{code-cell} ipython3
fit.state()[-1]
```

```{code-cell} ipython3
# Good TODOs:
# [x] Optimize StatelessTransform.fit() not to submit/always use DummyBackend?
# Impure Transforms like data readers.
# [x] BasePipeline.fit() and FitTransform[BasePipeline, ...] should chain backend submissions, like BasePipeline.apply()
# [x] Backend.fit()/apply() methods. apply() overloaded for FitTransform, StatelessTransform, BasePipeline.
# [x] ForBindings: combine_fun param, maybe generic in DataResult; DataFramePipeline expects specialized for DataFrames
# Start parallelizing appropriate transforms: ForBindings, Join, GroupByCols, GroupByBindings
```

```{code-cell} ipython3
df.head()
```

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

FEATURES = ["carat", "x", "y", "z", "depth", "table"]

def bake_features(cols) -> ff.DataFramePipeline:
    return (
        ff.DataFramePipeline(tag="bake_features")
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )

# per-cut feature means
per_cut_means = (
    ff.DataFramePipeline(tag="per_cut_means")
    .group_by_cols(["cut"])
        .then(
            ff.DataFramePipeline()
            [ff.HP("predictors")]
            .stateful_lambda(
                fit_fun=lambda df: df.mean(),
                apply_fun=lambda df, mean: mean.rename(lambda c: f"cut_mean_{c}")
            )
        )
)

complex_pipeline = (
    ff.DataFramePipeline()
    .select(FEATURES + ["{response_col}", "cut"])
    .copy("{response_col}", "{response_col}_train")
    .pipe(["carat", "{response_col}_train"], np.log1p)
    .winsorize("{response_col}_train", limit=0.05)
    .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
    .join(per_cut_means, how="left", on="cut")
    .sk_learn(
        LinearRegression,
        # x_cols=["carat", "depth", "table"],
        x_cols=ff.HPLambda(lambda bindings: bindings["predictors"] + [f"cut_mean_{c}" for c in bindings["predictors"]]),
        response_col="{response_col}_train",
        hat_col="{response_col}_hat",
        class_params={"fit_intercept": True},
        tag="reg-1",
    )
    # transform {response_col}_hat from log-dollars back to dollars
    .copy("{response_col}_hat", "{response_col}_hat_dollars")
    .pipe("{response_col}_hat_dollars", np.expm1)
)

complex_pipeline.hyperparams()

pip = read_diamonds.then(complex_pipeline)
```

```{code-cell} ipython3
bindings
```

```{code-cell} ipython3
bake_features(FEATURES).then(per_cut_means).apply(df, bindings={"predictors": ["carat"]})
```

```{code-cell} ipython3
bindings
```

```{code-cell} ipython3
pip.apply(bindings=bindings).head()
```

```{code-cell} ipython3
# logging.getLogger("frankenfit").setLevel(logging.DEBUG)
r = dask.apply(pip, bindings=bindings)
r.result().head()
```

```{code-cell} ipython3
pip
```

```{code-cell} ipython3
pip.visualize()
```

```{code-cell} ipython3
pip = (
    ff.DataFramePipeline()
    .group_by_bindings(
        [
            {"predictors": [p]} for p in FEATURES
        ],
    )
        .then(
            read_diamonds
            .then(complex_pipeline)
            .correlation(["{response_col}_hat_dollars"], ["{response_col}"])
        )
)

r = dask.apply(pip, bindings={"response_col": "price", "bake_features": True})
r.result()

# pip.visualize()
```

```{code-cell} ipython3
pip.apply(bindings={"response_col": "price", "bake_features": True})
```

```{code-cell} ipython3
p = read_diamonds.then(complex_pipeline).on_backend(dask)
fit = p.fit(
    bindings={"response_col": "price", "bake_features": True, "predictors": ["depth"]}
)
```

```{code-cell} ipython3
fit.find_by_name("SKLearn#reg-1")
```

```{code-cell} ipython3
df[["price_hat", "price_train", "price"]].corr()
```

```{code-cell} ipython3
df[["price_hat_dollars", "price"]].corr()
```

```{code-cell} ipython3
df_fut = dask.put(df)
complex_pipeline.on_backend(dask).apply(df_fut, bindings=bindings)
```

```{code-cell} ipython3
pip.visualize()
```

```{code-cell} ipython3
# parallel assignment
(
    ff.DataFramePipeline()
    .assign(
        foo="foo",
        bar=lambda df: "bar",
        baz=lambda df: "baz",
        frotz=lambda df: "frotz",
    )
).on_backend(dask).apply(df)
```
