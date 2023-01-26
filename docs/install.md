# Installation and getting started

## Installing Frankenfit

It's recommended to install Frankenfit to a [virtual
environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments).
With the environment activated, install Frankenfit from the [Python Package
Index](https://pypi.org/project/frankenfit/) with `pip`.

```
$ pip install frankenfit
```

### Dask

If you want to use the [Dask](https://www.dask.org) backend for distributed computation
of your pipelines, you can use the "dask" extra:
```
$ pip install "frankenfit[dask]"
```

This just adds [`distributed`](https://pypi.org/project/distributed/) to the set of
dependencies, so you can skip it if you already know that you have `distributed` in your
environment, or if you decide to add it on its own later.

### GraphViz

All of Frankenfit's Python dependencies are installed automatically by `pip` (see
above). However, one of those dependencies is the Python [`graphviz`
package](https://pypi.org/project/graphviz/), which is used for
[visualizing](visualizing-pipelines) Frankenfit pipelines and transforms. This in turn
depends on the [GraphViz executables](https://graphviz.org/download/) being available on
your system path. The simplest way to get these is to install the `graphviz` package
using your operating system's package manager. For example, in Ubuntu:

```
$ sudo apt install graphviz
```

## Importing Frankenfit

The author of Frankenfit recommends importing it like this:
```python
import frankenfit as ff
```

Everything you need to get going is available in the public [API](api-reference),
`ff.*`. You might want to start with a [synopsis](synopsis) of what you can do and
proceed from there.

## Notebooks

To use Frankenfit interactively, it is recommended to do so in a [Jupyter
notebook](https://jupyter.org). The author of Frankenfit highly recommends using a
typing-aware IDE when authoring your notebooks, rather than stock JupyterLab, because
Frankenfit's type annotations enable more advanced code analysis and autocompletion
abilities than JupyterLab is (currently) capable of. The screenshot below, for example,
shows a notebook in [Visual Studio
Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

:::{figure} _static/sshot-vscode-intellisense-frankenfit-apply.png
VS Code showing the type signature and pydoc of a pipeline's
[`apply()`](frankenfit.Pipeline.apply) method, while also suggesting methods on the
resulting pandas `DataFrame` object.
:::
