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

logging.getLogger('frankenfit').setLevel(logging.INFO)
```

```{code-cell} ipython3
:tags: []

df = data('diamonds').reset_index().drop(['index'], axis=1)
df.head()
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
    )
    # transform {response_col}_hat from log-dollars back to dollars
    .copy("{response_col}_hat", "{response_col}_hat_dollars")
    .pipe("{response_col}_hat_dollars", np.expm1)
)
pipeline.hyperparams()
```

```{code-cell} ipython3
:tags: []

pipeline.visualize()
```

```{code-cell} ipython3
:tags: []

ff.dataframe.Correlation?
```

```{code-cell} ipython3
:tags: []

pipcorr = (pipeline + ff.dataframe.Correlation(["{response_col}"], ["{response_col}_hat_dollars"]))
pipcorr.hyperparams()
```

```{code-cell} ipython3
:tags: []

pipcorr.apply(
    df,
    bindings={
        'bake_features': True,
        'response_col': 'price',
        'predictors': ["depth", "table", ]
    }
)
```

```{code-cell} ipython3
:tags: []

print(ff.dataframe.Correlation(["price"], ["carat", "table", "depth"]).apply(df))
```

```{code-cell} ipython3
:tags: []

print(ff.dataframe.Correlation(["table", "depth"], ["x", "y", "z"]).apply(df))
```

```{code-cell} ipython3
:tags: []

n_folds = 5
cv = (
    ff.DataFramePipeline()
    .stateless_lambda(lambda df: df.assign(cv_grp=df.index % n_folds))
    .group_by('cv_grp', fitting_schedule=ff.fit_group_on_all_other_groups)
        .then(
            pipeline
            .correlation(
                ['{response_col}'],
                ['{response_col}_hat_dollars'],
            )
        )
)
cv.apply(
    df,
    bindings={
        'bake_features': True,
        'response_col': 'price',
        'predictors': ["depth", "table", ]
    }
)
```

```{code-cell} ipython3
:tags: []

cv.apply(
    df,
    bindings={
        'bake_features': False,
        'response_col': 'price',
        'predictors': ["depth", "table", ]
    }
)
```

```{code-cell} ipython3
:tags: []

n_folds = 5
def add_cv_grp(df):
    return df.assign(cv_grp = df.index % n_folds)

cv = (
    ff.DataFramePipeline()
    .stateless_lambda(add_cv_grp)
    .group_by('cv_grp', fitting_schedule=ff.fit_group_on_all_other_groups)
        .then(
            pipeline
        )
    .correlation(
        ['{response_col}'],
        ['{response_col}_hat_dollars'],
    )
)
cv.apply(
    df,
    bindings={
        'bake_features': True,
        'response_col': 'price',
        'predictors': ["depth", "table", ]
    }
)
```

```{code-cell} ipython3
:tags: []

cv.apply(
    df,
    bindings={
        'bake_features': False,
        'response_col': 'price',
        'predictors': ["depth", "table", ]
    }
)
```

```{code-cell} ipython3
:tags: []

cv.visualize()
```
