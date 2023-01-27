---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [remove-cell]

from datetime import datetime
from myst_nb import glue
import frankenfit
glue("frankenfit_version", frankenfit.__version__)
glue("build_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
```

# Frankenfit Documentation

Copyright © 2023 Max Bane (max@thebanes.org)

```{epigraph}
“For instance, when bread is baked some parts are split at the surface, and these
parts which thus open, and have a certain fashion contrary to the purpose of the
baker’s art, are beautiful in a manner, and in a peculiar way excite a desire for
eating.”

--Marcus Aurelius, *Meditations*, as translated by George Long in 1862
```

🏠 **Project homepage:** https://github.com/maxbane/frankenfit

🐍 **Python package:** https://pypi.org/project/frankenfit

📅 This documentation was built on {glue:text}`build_time` for Frankenfit version
{glue:text}`frankenfit_version`.

## Table of contents

```{tableofcontents}
```

## License ([BSD-2-Clause-Patent](https://spdx.org/licenses/BSD-2-Clause-Patent.html))

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list
of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

Subject to the terms and conditions of this license, each copyright holder and
contributor hereby grants to those receiving rights under this license a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except for failure to
satisfy the conditions of this license) patent license to make, have made, use, offer to
sell, sell, import, and otherwise transfer this software, where such license applies
only to those patent claims, already acquired or hereafter acquired, licensable by such
copyright holder or contributor that are necessarily infringed by:

* (a) their Contribution(s) (the licensed copyrights of copyright holders and
non-copyrightable additions of contributors, in source or binary form) alone; or

* (b) combination of their Contribution(s) with the work of authorship to which such
Contribution(s) was added by such copyright holder or contributor, if, at the time the
Contribution is added, such addition causes such combination to be necessarily
infringed. The patent license shall not apply to any other combinations which include
the Contribution.

Except as expressly stated above, no rights or licenses from any copyright holder or
contributor is granted under this license, whether expressly, by implication, estoppel
or otherwise.

:::{admonition} DISCLAIMER
:class: warning
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
:::
