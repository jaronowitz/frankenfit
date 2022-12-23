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
try:
    client.shutdown()
    cluster.close()
except:
    pass
```

```{code-cell} ipython3
from importlib import reload
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./dracula.mplstyle')

from pydataset import data
df = data('diamonds').reset_index().set_index('index')

import frankenfit as ff
reload(ff)

import logging
logging.basicConfig()
logging.getLogger("distributed").setLevel(logging.WARN)
# LOG = logging.getLogger(__name__)
# LOG.setLevel(logging.INFO)
# LOG.info('Hello from notebook')
logging.getLogger('frankenfit').setLevel(logging.DEBUG)

from dask import distributed
# cluster = distributed.LocalCluster()
cluster = distributed.LocalCluster(
    n_workers=4, threads_per_worker=2, host="devbox.homenet.thebanes.org",
    scheduler_port=0, dashboard_address=':0'
)
client = distributed.Client(cluster)

dask = ff.DaskBackend(client)
cluster
```

```{code-cell} ipython3
do = ff.DataFramePipeline()

other_pipeline = do[["price", "carat"]].winsorize(0.1).suffix("_win")
# backend = ff.LocalBackend()
backend = dask

pipeline = do.assign(
    ### multi-column assigments
    do[["price", "carat"]].de_mean().suffix("_dmn"),  # pipeline
    backend.apply(other_pipeline, df),  # future
    # lambda is wrapped in a StatelessLambda transform
    lambda df: pd.DataFrame().assign(uppercut = df["cut"].str.upper()),

    ### named column assignments: transforms with 1-column output
    price_dmn2=do["price"].de_mean(),
    price_win2=backend.apply(other_pipeline["price_win"], df),  # future

    # lambda is wrapped in a StatelessLambda transform
    price_rank=lambda df, price_scale=1.0: price_scale * (
        (df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())
    ),

    intercept=1.0,  # scalar
)
pipeline.on_backend(dask).apply(df, price_scale=2.0).head()
```

```{code-cell} ipython3
ff.universal.StateAsData(
    ff.dataframe.DeMean("price")
).fit(df).apply()
```

```{code-cell} ipython3
ohlc_csv = (
    ff.ReadPandasCSV("../yahoo-2012-2022.csv")
    .then()
    .assign(
        time=lambda df: pd.to_datetime(df["time"])
    )
    .stateless_lambda(lambda df: df.set_index("time"))
)
df = ohlc_csv.apply()
df.head()
```

```{code-cell} ipython3
df.info()
```

```{code-cell} ipython3
df.loc[df["sym"] == "A"]
```

```{code-cell} ipython3
df.groupby("sym")[["sym","Close"]].shift(-1)
```

```{code-cell} ipython3
df.assign(year=df.index.year).head()
df.index.day
```

```{code-cell} ipython3
import pyarrow.dataset as ds
import pyarrow as pa
```

```{code-cell} ipython3
ds.write_dataset(
    pa.Table.from_pandas(
        df
        .assign(
            year=df.index.year,
            month=df.index.month,
            day=df.index.day,
        )
    ),
    "./test3",
    format="parquet",
    partitioning=ds.partitioning(
        pa.schema([("year", pa.int16()), ("month", pa.int8()), ("day", pa.int32())]),
    ),
    existing_data_behavior='delete_matching',
)
```

```{code-cell} ipython3
ff.dataframe.WriteDataset(
    "./test6",
    partitioning_schema=pa.schema([("year", pa.int16()), ("month", pa.int8()), ("day", pa.int32())])
).apply(
    df
    .assign(
        year=df.index.year,
        month=df.index.month,
        day=df.index.day,
    )
)
```

```{code-cell} ipython3
(
    ff.ReadDataFrame(df)
    .then()
    .assign(
        year=lambda df: df.index.year,
        month=lambda df: df.index.month,
        day=lambda df: df.index.day,
    )
    .write_dataset(
        "./test7",
        partitioning_schema=pa.schema(
            [("year", pa.int16()), ("month", pa.int8()), ("day", pa.int32())]
        )
    )
).apply()
```

```{code-cell} ipython3
(
    df
    .assign(
        year=df.index.year,
        month=df.index.month,
        day=df.index.day,
    )
    .assign(
        num=lambda df: df["year"] * 10000 + df["month"] * 100 + df["day"]
    )
).loc["20130620":]
```

```{code-cell} ipython3
datenum = ds.field("year") * 10000 + ds.field("month") * 100 + ds.field("day")
tab = ds.dataset(
    "./test5",
    partitioning=ds.partitioning(
        pa.schema([("year", pa.int64()), ("month", pa.int64()), ("day", pa.int64())]),
    ),
).to_table(filter = datenum > 20120101)
tab.sort_by("time").to_pandas()#.sort_index()
```

```{code-cell} ipython3
tab.sort_by?
```

```{code-cell} ipython3
ff.dataframe.ReadDataset(
    "./test5",
    dataset_kwargs=dict(
        partitioning=pa.schema([("year", pa.int16()), ("month", pa.int8()), ("day", pa.int32())])
    )
).apply().info()
```

```{code-cell} ipython3
x = ds.dataset('./test1')
display(x.to_table().to_pandas().head())
x.files
```

```{code-cell} ipython3
import pyarrow as pa
import pyarrow.dataset as ds
ds.write_dataset?
ds.partitioning
```

```{code-cell} ipython3
add_leadrets = (
    ff.DataFramePipeline()
    .stateless_lambda(
        lambda df: pd.merge(
            df,
            df.groupby("sym")[["sym", "Close"]].shift(-1),
            how="left",
            on=["time", "sym"],
            suffixes=("", "_next")
        )
    )
    .assign(
        leadret_1d=lambda df: df["Close_next"] / df["Close"] - 1.0
    )
    .drop("Close_next")
)
add_leadrets.apply(df)
```

```{code-cell} ipython3
add_leadrets.apply(df).loc[lambda df: df["sym"] == "A"]["leadret_1d"].cumsum().plot()
```

```{code-cell} ipython3
pd.merge?
```

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression  # type: ignore
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
        ff.DataFramePipeline()[ff.HP("predictors")].stateful_lambda(
            fit_fun=lambda df: df.mean(),
            apply_fun=lambda df, mean: mean.rename(lambda c: f"cut_mean_{c}"),
        )
    )
)

complex_pipeline = (
    ff.DataFramePipeline()
    .select(FEATURES + ["{response_col}", "cut"])
    .copy("{response_col}", "{response_col}_train")
    .winsorize("{response_col}_train", limit=0.05)
    .pipe(["carat", "{response_col}_train"], np.log1p)
    .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
    .join(per_cut_means, how="left", on="cut")
    .sk_learn(
        LinearRegression,
        # x_cols=["carat", "depth", "table"],
        x_cols=ff.HPLambda(
            lambda bindings: bindings["predictors"]
            + [f"cut_mean_{c}" for c in bindings["predictors"]]
        ),
        response_col="{response_col}_train",
        hat_col="{response_col}_hat",
        class_params={"fit_intercept": True},
    )
    # transform {response_col}_hat from log-dollars back to dollars
    .copy("{response_col}_hat", "{response_col}_hat_dollars")
    .pipe("{response_col}_hat_dollars", np.expm1)
)

assert complex_pipeline.hyperparams() == {
    "response_col",
    "bake_features",
    "predictors",
}
bindings = {
    "response_col": "price",
    "bake_features": True,
    "predictors": ["carat", "x", "y", "z", "depth", "table"],
}

local_result = complex_pipeline.apply(df, bindings)
assert local_result.equals(
    dask.apply(complex_pipeline, df, bindings).result()
)
```

```{code-cell} ipython3
pip = (
    ff.DataFramePipeline()
    .group_by_bindings(
        [
            {"predictors": ["carat"]},
            {"predictors": ["depth"]},
            {"predictors": ["table"]},
        ],
    )
    .then(
        ff.ReadDataFrame(df)
        .then(complex_pipeline)
        .correlation(["{response_col}_hat_dollars"], ["{response_col}"])
    )
)
bindings = {"response_col": "price", "bake_features": True}
local_result = pip.apply(**bindings)
```

```{code-cell} ipython3
assert local_result.equals(dask.apply(pip, **bindings).result())
```

```{code-cell} ipython3
ff.Identity()
```

```{code-cell} ipython3
ff.Identity().fit("foo")
```

```{code-cell} ipython3
ft = dask.fit(ff.Identity(), "blah")
```

```{code-cell} ipython3
ft.backend
```

```{code-cell} ipython3
ft = ff.Identity().on_backend(dask).fit("bloop")
```

```{code-cell} ipython3
ft.backend
```

```{code-cell} ipython3
ft.apply("floop")
```

```{code-cell} ipython3
ft.materialize_state().state()
```

```{code-cell} ipython3
ff.Identity().apply("blop")
```

```{code-cell} ipython3
dask.apply(ff.Identity(), "blop")
```

```{code-cell} ipython3
ff.Identity().on_backend(dask).apply("blooper")
```

```{code-cell} ipython3
# scatters twice
ff.Identity().on_backend(dask).fit("bloz").apply("bloz")
```

```{code-cell} ipython3
ff.dataframe.ZScore(["price"]).on_backend(dask).fit(df)
```

```{code-cell} ipython3
# dask.apply(ff.dataframe.ZScore(["price"]), df)
# ff.dataframe.ZScore(["price"]).on_backend(dask).apply(df[["price"]])
```

```{code-cell} ipython3
ff.dataframe.ZScore(["price"]).on_backend(dask).fit(df).apply(df[["price"]])
```

```{code-cell} ipython3
dask.apply(ff.Identity(), None).result()
```

```{code-cell} ipython3
import numpy as np
pip = (
    ff.DataFramePipeline()
    .select(["price", "carat"])
    .copy("price", "price_train")
    .pipe(["carat", "price_train"], np.log1p)
    .z_score(["price"])
)
pip.fit(df).materialize_state().state()
```

```{code-cell} ipython3
f = pip.fit(df)
dask.apply(f, df).result().head()
```

```{code-cell} ipython3
pip.apply(df).head()
```

```{code-cell} ipython3
dask.fit(pip, df).materialize_state().state()
```

```{code-cell} ipython3
data = dask.put(df)
pip.on_backend(dask).fit(data).materialize_state().state()#.apply(data)
```

```{code-cell} ipython3
z = ff.dataframe.ZScore(["price", "carat"])
f = dask.fit(z, df)
f.state()
```

```{code-cell} ipython3
dask.apply(f.materialize_state(), df.iloc[1:])
```

```{code-cell} ipython3
f.materialize_state().apply(df.iloc[2:])
```

```{code-cell} ipython3
fp = pip.fit(df)
fp.materialize_state().state()
```

```{code-cell} ipython3
dask.apply(fp.materialize_state(), df.iloc[1:])
```

```{code-cell} ipython3
dfp = dask.fit(pip, df.iloc[2:])
dask.apply(dfp.materialize_state(), df.iloc[1:])
```

```{code-cell} ipython3
dask.apply(dfp, df.iloc[1:])
```

```{code-cell} ipython3
p = (
    pip + (
        ff.DataFramePipeline()
        .if_hyperparam_is_true("dowin", ff.dataframe.Winsorize("price", limit=0.10))
        .pipe(["price"], np.log1p)
    )
)
p.on_backend(dask).apply(df, bindings={"dowin": False})["price"].hist()
p.on_backend(dask).apply(df, bindings={"dowin": True})["price"].hist()
```

```{code-cell} ipython3
logging.getLogger("frankenfit").setLevel(logging.INFO)
```

```{code-cell} ipython3
dask.apply(p, df, bindings={"dowin": True}).result()["price"].hist()
```

```{code-cell} ipython3
from dask.diagnostics import ProgressBar
with ProgressBar():
    dask.apply(p, df, bindings={"dowin": True}).result()["price"].hist()
```

```{code-cell} ipython3
distributed.progress(dask.apply(p, df, bindings={"dowin": True}).fut.unwrap())
```
