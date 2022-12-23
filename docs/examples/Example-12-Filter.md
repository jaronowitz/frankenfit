---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: .venv-dev
  language: python
  name: python3
---

```{code-cell} ipython3
import frankenfit as ff
from pydataset import data
df = data('diamonds').reset_index().set_index('index')
```

```{code-cell} ipython3
ff.DataFramePipeline().filter(
    lambda df, max_price: df["price"] <= max_price
).apply(df, bindings={"max_price": 400})
```

```{code-cell} ipython3
ff.DataFramePipeline().filter(
    lambda df, max_price, *args: df["price"] <= max_price
).apply(df, bindings={"max_price": 400})
```
