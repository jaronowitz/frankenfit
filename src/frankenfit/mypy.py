# Copyright (c) 2023 Max Bane <max@thebanes.org>
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.
#
# Subject to the terms and conditions of this license, each copyright holder and
# contributor hereby grants to those receiving rights under this license a perpetual,
# worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except for failure to
# satisfy the conditions of this license) patent license to make, have made, use, offer
# to sell, sell, import, and otherwise transfer this software, where such license
# applies only to those patent claims, already acquired or hereafter acquired,
# licensable by such copyright holder or contributor that are necessarily infringed by:
#
# (a) their Contribution(s) (the licensed copyrights of copyright holders and
# non-copyrightable additions of contributors, in source or binary form) alone; or
#
# (b) combination of their Contribution(s) with the work of authorship to which such
# Contribution(s) was added by such copyright holder or contributor, if, at the time the
# Contribution is added, such addition causes such combination to be necessarily
# infringed. The patent license shall not apply to any other combinations which include
# the Contribution.
#
# Except as expressly stated above, no rights or licenses from any copyright holder or
# contributor is granted under this license, whether expressly, by implication, estoppel
# or otherwise.
#
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
# SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""
A Mypy plugin for use when typechecking code that uses the frankenfit library.
Piggy-backs on the built-in attrs plugin to make Mypy aware of the automagic
that the @params decorator does, and also expands the constructor signatures of
Transform subclasses to allow hyperparameters (`HP` instances) for all
parameters, in addition to their annotated types.

Example mypy config in `pyroject.toml`::

    [tool.mypy]
    plugins = "frankenfit.mypy"

"""
from __future__ import annotations

from typing import Callable

from mypy.plugin import ClassDefContext, FunctionSigContext, Plugin
from mypy.plugins.attrs import attr_attrib_makers, attr_define_makers
from mypy.typeops import make_simplified_union
from mypy.types import FunctionLike, Instance

PARAMS_DECORATOR = "frankenfit.params.params"
TRANSFORM_BASE_CLASS = "frankenfit.core.Transform"
TRANSFORM_FIELD_MAKERS = {
    "frankenfit.params.fmt_str_field",
    "frankenfit.params.dict_field",
    "frankenfit.params.columns_field",
    "frankenfit.params.optional_columns_field",
}

# Make @transform type-check like @define
# See: https://github.com/python/mypy/issues/5406
attr_define_makers.add(PARAMS_DECORATOR)

# Make fmt_str_field, columns_field, etc. behave like attrs.field
for maker in TRANSFORM_FIELD_MAKERS:
    attr_attrib_makers.add(maker)

known_transform_subclasses: set[str] = {TRANSFORM_BASE_CLASS}


def transform_base_class_callback(ctx: ClassDefContext) -> None:
    """
    Keeps track of Transform subclasses.
    """
    known_transform_subclasses.add(ctx.cls.fullname)
    return


def transform_constructor_sig_callback(ctx: FunctionSigContext) -> FunctionLike:
    """
    Adjust the signature of every Transform subclass's constructor such that all
    non-`tag` arguments have their types unioned with `HP`.
    """
    sig = ctx.default_signature
    new_arg_types = []
    # For some reason ctx.api.lookup_typeinfo() raises an AssertionError, so we
    # to dig into the modules ourselves to find frankenfit.params.HP
    params_module = ctx.api.modules["frankenfit.params"]  # type: ignore [attr-defined]
    hp_typeinfo = params_module.names["HP"].node
    hp_type = Instance(hp_typeinfo, [])
    for arg_name, arg_type in zip(sig.arg_names, sig.arg_types):
        if arg_name == "tag":
            # don't allow special "tag" param to be hyperparameterized
            new_arg_types.append(arg_type)
        else:
            new_arg_types.append(make_simplified_union([arg_type, hp_type]))

    return sig.copy_modified(arg_types=new_arg_types)


class FrankenfitPlugin(Plugin):
    def get_function_signature_hook(
        self, fullname: str
    ) -> Callable[[FunctionSigContext], FunctionLike] | None:
        if fullname in known_transform_subclasses:
            return transform_constructor_sig_callback
        else:
            return None

    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        if fullname in known_transform_subclasses:
            return transform_base_class_callback
        else:
            return None


def plugin(version: str):
    return FrankenfitPlugin
