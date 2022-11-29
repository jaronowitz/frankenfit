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
that the @param decorator does, and also expands the constructor signatures of
Transform subclasses to allow hyperparameters (`HP` instances) for all
parameters, in addition to their annotated types.

Example mypy config in `pyroject.toml`::

    [tool.mypy]
    plugins = "frankenfit.mypy"

"""
from __future__ import annotations
from typing import Callable, cast
from warnings import warn

from mypy.nodes import ClassDef, TypeInfo
from mypy.plugin import ClassDefContext, Plugin, SemanticAnalyzerPluginInterface
from mypy.plugins.attrs import attr_attrib_makers, attr_define_makers
from mypy.plugins.common import add_attribute_to_class
from mypy.types import AnyType, Instance, TypeOfAny, TypeType

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


def make_fit_transform_subclass_type_info(
    api: SemanticAnalyzerPluginInterface, transform_classdef: ClassDef
) -> TypeInfo:
    transform_class_name = transform_classdef.name
    fit_transform_subclass_name = f"Fit{transform_class_name}"

    base_type = api.lookup_fully_qualified("frankenfit.core.FitTransform").node
    assert base_type is not None
    base_type = cast("TypeInfo", base_type)
    sub_type = api.basic_new_typeinfo(
        fit_transform_subclass_name, Instance(base_type, []), -1
    )

    return sub_type


def transform_base_class_callback(ctx: ClassDefContext) -> None:
    """
    Callback invoked when mypy encounters a subclass deriving from
    frankenfit.core.Transform. Makes mypy aware of the metaclass automagic
    that Transform subclasses do.
    """

    # TODO: patch every __init__ arg to be a Union with HP

    transform_class_name = ctx.cls.name
    fit_transform_class_name = f"Fit{transform_class_name}"

    # Make a new TypeInfo for the FitTransform subclass that gets generated for
    # this Transform
    fit_transform_class_type: AnyType | TypeType
    try:
        fit_transform_type_info = make_fit_transform_subclass_type_info(
            ctx.api, ctx.cls
        )
    except Exception as e:
        warn(f"Unable to FitTransform subclass typeinfo with mypy API: {e}")
        fit_transform_class_type = AnyType(TypeOfAny.unannotated)
    else:
        fit_transform_class_type = TypeType(Instance(fit_transform_type_info, []))

    # Make mypy aware of the dynamically created class variable for the
    # FitTransform subclass. E.g., DeMean.FitDemean
    add_attribute_to_class(
        ctx.api,
        ctx.cls,
        fit_transform_class_name,
        fit_transform_class_type,
        is_classvar=True,
    )

    # Tell mypy that e.g. DeMean.fit() returns a DeMean.FitDemean instance


class FrankenfitPlugin(Plugin):
    def get_base_class_hook(
        self, fullname: str
    ) -> Callable[[ClassDefContext], None] | None:
        return None
        if fullname == TRANSFORM_BASE_CLASS:
            # print("get_base_class_hook:", fullname)
            return transform_base_class_callback


def plugin(version: str):
    return FrankenfitPlugin
