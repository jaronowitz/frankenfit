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
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='whitegrid')
import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.info('Hello from notebook')
```

```{code-cell} ipython3
:tags: []

import frankenfit as ff

logging.getLogger('frankenfit').setLevel(logging.INFO)
#reload(fpc)
```

```{code-cell} ipython3
:tags: []

ff.DataFramePipeline.clip?
```

```{code-cell} ipython3
:tags: []

help(ff.DataFramePipeline.copy)
```

```{code-cell} ipython3
:tags: []

from pydataset import data
```

```{code-cell} ipython3
:tags: []

df = data('diamonds')
data('diamonds', show_doc=True)
```

```{code-cell} ipython3
:tags: []

index_all = set(df.index)
index_in = set(np.random.choice(df.index, size=int(len(df)/2), replace=False))
index_out = index_all - index_in
len(index_all), len(index_in), len(index_out)
df_in = df.loc[list(index_in)]
df_out = df.loc[list(index_out)]
df_in.head()
```

```{code-cell} ipython3
:tags: []

df_out.head()
```

```{code-cell} ipython3
:tags: []

FEATURES = ['carat', 'x', 'y', 'z', 'depth', 'table']
df_in[FEATURES].hist();
df_in[['price']].hist()
```

```{code-cell} ipython3
:tags: []

# call-chain syntax...

def standardize_callchain(cols):
    return (
        ff.DataFramePipeline()
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.)
        .clip(cols, upper=2, lower=-2)
    )

FEATURES = ['carat', 'x', 'y', 'z', 'depth', 'table']

pipeline_callchain = (
    ff.DataFramePipeline()
    .copy('price', 'price_train')
    .pipe(['carat', 'price_train'], np.log1p)
    .then(standardize_callchain(FEATURES + ['price_train']))
)
```

```{code-cell} ipython3
:tags: []

pipeline_callchain
```

```{code-cell} ipython3
:tags: []

fit_pipeline = pipeline_callchain.fit(df_in)
result_in = fit_pipeline.apply(df_in)

result_in[FEATURES].hist();
result_in[['price_train']].hist();
```

```{code-cell} ipython3
:tags: []

fit_pipeline.state()
```

```{code-cell} ipython3
:tags: []

result_out = fit_pipeline.apply(df_out)
result_out[FEATURES].hist();
result_out[['price_train']].hist();
```

```{code-cell} ipython3
:tags: []

help(ff.DataFramePipeline)
```
