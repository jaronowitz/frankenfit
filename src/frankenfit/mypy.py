# Unpublished Copyright (c) 2022 Max Bane, all rights reserved.
#
# NOTICE: All information contained herein is, and remains the property of Max Bane.
# The intellectual and technical concepts contained herein are proprietary to Max Bane
# and may be covered by U.S. and Foreign Patents, patents in process, and are protected
# by trade secret or copyright law. Dissemination of this information or reproduction
# of this material is strictly forbidden unless prior written permission is obtained
# from Max Bane. Access to the source code contained herein is hereby forbidden to
# anyone except current employees, contractors, or customers of Max Bane who have
# executed Confidentiality and Non-disclosure agreements explicitly covering such
# access.
#
# The copyright notice above does not evidence any actual or intended publication or
# disclosure of this source code, which includes information that is confidential
# and/or proprietary, and is a trade secret, of Max Bane. ANY REPRODUCTION,
# MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE
# OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN CONSENT OF MAX BANE IS STRICTLY
# PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND INTERNATIONAL TREATIES. THE
# RECEIPT OR POSSESSION OF THIS SOURCE CODE AND/OR RELATED INFORMATION DOES NOT CONVEY
# OR IMPLY ANY RIGHTS TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, NOR TO
# MANUFACTURE, USE, OR SELL ANYTHING THAT IT MAY DESCRIBE, IN WHOLE OR IN PART.

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

from mypy.plugin import ClassDefContext, Plugin, FunctionSigContext
from mypy.plugins.attrs import attr_attrib_makers, attr_define_makers
from mypy.types import Instance, FunctionLike
from mypy.typeops import make_simplified_union

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
