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
ff.DataFramePipeline.FitDataFramePipeline.apply?
```

```{code-cell} ipython3
pip = ff.DataFramePipeline(transforms=[ff.dataframe.Select("x"), ff.ReadDataFrame('foo')])
```

```{code-cell} ipython3
df = pd.DataFrame({"x": range(10)})
fit = pip.fit(df)
```

```{code-cell} ipython3
fit.apply(df)
```

```{code-cell} ipython3
ff.ReadDataset('./diamonds.csv')
```

```{code-cell} ipython3
ff.ReadDataset(
    './diamonds.csv',
    format='csv',
    filter=(ds.field('index') > 2) & (ds.field('index') < 7),
    index_col='index',
).apply()
```

```{code-cell} ipython3
bindings = {'filter': (ds.field('index') > 2) & (ds.field('index') < 7)}

ff.ReadDataset(
    './diamonds.csv',
    format='csv',
    filter=ff.HP('filter'),
    index_col='index',
).apply(bindings=bindings)
```

```{code-cell} ipython3
r = ff.ReadDataset('./diamonds-{foo}.csv', columns=['{foo}'])
r
```

```{code-cell} ipython3
type(r.paths), type(r.columns)
```

```{code-cell} ipython3
r = ff.ReadDataset('./diamonds.csv', format='csv')
r.apply()
```

```{code-cell} ipython3
import pyarrow as pa
import pyarrow.parquet as pq
```

```{code-cell} ipython3
df = data('diamonds').reset_index().set_index('index')
df.head()
```

```{code-cell} ipython3
df.loc[3:6]
```

```{code-cell} ipython3
df.to_csv('./diamonds.csv')
```

```{code-cell} ipython3
ff.dataframe.WritePandasCSV(
    './diamonds.csv',
    index_label='index',
).apply(df)

ff.ReadPandasCSV('./diamonds.csv', dict(index_col='index')).apply()
```

```{code-cell} ipython3
r = ff.ReadDataset('./diamonds.csv', format='csv', index_col='index')
r.apply()
```

```{code-cell} ipython3
ff.ReadPandasCSV('./diamonds.csv', dict(index_col=0))
```

```{code-cell} ipython3
ff.ReadPandasCSV('./diamonds.csv', dict(index_col=0)).apply().head()
```

```{code-cell} ipython3
(
    ff.DataFramePipeline()
    [['foo']]
    # should emit RuntimeWarning
    .read_pandas_csv('./diamonds.csv', dict(index_col=0))
).visualize()
```

```{code-cell} ipython3
ff.DataFramePipeline() + ff.ReadPandasCSV('./diamonds.csv', dict(index_col=0))
```

```{code-cell} ipython3
df = data('diamonds').reset_index().drop(['index'], axis=1)
```

```{code-cell} ipython3
r = ff.ReadPandasCSV('./diamonds.csv', dict(index_col=0))
isinstance(r, ff.Transform)
```

```{code-cell} ipython3
df
```

```{code-cell} ipython3
table = pa.Table.from_pandas(df)
table
```

```{code-cell} ipython3
table.to_pandas()
```

```{code-cell} ipython3
from attrs import define

@define
class GenerateRandomData(ff.StatelessTransform):
    cols: dict[str, dict]
    nrows: int

    def _apply(self, df_apply, state=None):
        return pd.DataFrame({
            c: np.random.normal(size=self.nrows, **cparams) for (c, cparams) in self.cols.items()
        })
```

```{code-cell} ipython3
grd = GenerateRandomData(
    {'x': {'loc': 5.0, 'scale': 2.0}, 'y': {'loc': -10.0, 'scale': 0.5}},
    1000
)
grd.apply().hist();
```

```{code-cell} ipython3
ff.Pipeline().then(grd).apply().hist();
```
