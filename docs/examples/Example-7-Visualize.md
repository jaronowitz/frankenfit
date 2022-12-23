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

ff.dataframe.DeMean(["x", "y", "z"]).visualize()
```

```{code-cell} ipython3
:tags: []

pip = ff.DataFramePipeline(transforms=[
    ffdf.DeMean(["x", "y", "z"]),
    ffdf.Winsorize(["x", "y", "z"], limit=0.05),

])
pip.visualize()
#ff.DataFramePipeline()
```

```{code-cell} ipython3
:tags: []

pip
```

```{code-cell} ipython3
:tags: []

ffu.IfHyperparamIsTrue("foo", then=ffdf.DeMean(["xyz"]), otherwise=ffdf.Winsorize(["xyz"], 0.05)).visualize()
```

```{code-cell} ipython3
:tags: []

ffdf.DataFramePipeline(transforms=[
    #ff.IfHyperparamIsTrue("foo", then=ff.Pipeline(transforms=[ff.DeMean(["xyz"])]), otherwise=ff.Winsorize(["xyz"], 0.05))
    ffu.IfHyperparamIsTrue("foo", then=ffdf.DataFramePipeline(transforms=[ffdf.DeMean(["xyz"])])),
]).visualize()
```

```{code-cell} ipython3
:tags: []

pip = ffdf.DataFramePipeline(transforms=[
    ffdf.DeMean(["x", "y", "z"]),
    ffdf.Winsorize(["x", "y", "z"], limit=0.05),
    ffu.IfHyperparamIsTrue("foo", then=ffdf.DeMean(["xyz"]), otherwise=ffdf.Winsorize(["xyz"], 0.05)),
    ffu.IfHyperparamIsTrue("foo", then=ffdf.DeMean(["xyz"])),
    ffdf.Pipe(["carat", "{response_col}_train"], np.log1p),
])
pip.visualize()
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
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )

pipeline = (
    ff.DataFramePipeline()
    [FEATURES + ["{response_col}"]]
    .copy("{response_col}", "{response_col}_train")
    .winsorize("{response_col}_train", limit=0.05)
    .pipe(["carat", "{response_col}_train"], np.log1p)
    .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
    .sk_learn(
        LinearRegression,
        # x_cols=["carat", "depth", "table"],
        x_cols=ff.HP("predictors"),
        response_col="{response_col}_train",
        hat_col="{response_col}_hat",
        class_params={"fit_intercept": True},
        tag='SKLearn - My Amazing Regression Model',
    )
    # transform {response_col}_hat from log-dollars back to dollars
    .copy("{response_col}_hat", "{response_col}_hat_dollars")
    .pipe("{response_col}_hat_dollars", np.expm1)
)
pipeline.visualize()
```

```{code-cell} ipython3
:tags: []

t = ffdf.Join(ff.DataFramePipeline(), ff.DataFramePipeline(), how="left", on="diamond_id")
t.visualize()
```

```{code-cell} ipython3
:tags: []

p = ff.DataFramePipeline(
        transforms=[
            ffdf.Join(ff.DataFramePipeline(tag="xyz"), ff.DataFramePipeline(tag="cut"), how="left", on="diamond_id")
        ]
    )
p.visualize()
```

```{code-cell} ipython3
:tags: []

p = ff.DataFramePipeline(tag="moo").join(ff.DataFramePipeline(tag="cut"), how="left", on="diamond_id").z_score(['price'])
p.visualize()
```

```{code-cell} ipython3
:tags: []

p = ff.DataFramePipeline().join(ff.DataFramePipeline(tag="cut"), how="left", on="diamond_id").z_score(['price'])
p.visualize()
```

```{code-cell} ipython3
:tags: []

x = (
    ff.DataFramePipeline()
    .group_by_cols("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
        .de_mean(["price"])
    [["cut", "price"]]
)
x.visualize()
```

```{code-cell} ipython3
:tags: []

x = (
    ff.DataFramePipeline()
    .group_by_cols("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
        .then(
            ff.DataFramePipeline()
            .winsorize(["price"], limit=0.05)
            .z_score(["price"])
            .clip(["price"], upper=2, lower=-1)
        )
    [["cut", "price"]]
)
x.visualize()
```

```{code-cell} ipython3
:tags: []

def my_score_fn(df):
    pass

pipeline_grouped = (
    ff.DataFramePipeline()
    .print("hi")
    .group_by_cols("cut")
        #.de_mean("price")
        #.then(ff.DeMean(["price"]))
        .then(pipeline)
    .pipe(["{response_col}", "{response_col}_hat_dollars"], my_score_fn)
)
pipeline_grouped.visualize()
```

```{code-cell} ipython3
:tags: []

pipeline_grouped
```

```{code-cell} ipython3
:tags: []

df = data('diamonds').reset_index().drop(['index'], axis=1)
fit = pipeline.fit(df, bindings={'response_col': 'price', 'bake_features': True, 'predictors': FEATURES})
```

```{code-cell} ipython3
:tags: []

pipeline
```

```{code-cell} ipython3
:tags: []

pipeline.find_by_tag(
    'SKLearn - My Amazing Regression Model',
)
```

```{code-cell} ipython3
:tags: []

fit
```

```{code-cell} ipython3
:tags: []

fit.find_by_tag(
    'SKLearn - My Amazing Regression Model',
)
```

```{code-cell} ipython3
:tags: []

fit.find_by_tag(
    'SKLearn - My Amazing Regression Model',
).state().coef_
```
