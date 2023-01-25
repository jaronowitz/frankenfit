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
This module provides the main `@param` decorator (a warpper around
`attrs.define`) for creating Transofrm subclasses with parameters, as well as a
library of classes and field functions for different types of hyperparameterss.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ItemsView,
    KeysView,
    Mapping,
    Tuple,
    TypeVar,
    Union,
    ValuesView,
)

import attrs
from attrs import define, field

T = TypeVar("T")
_T = TypeVar("_T")


class UnresolvedHyperparameterError(NameError):
    """
    Exception raised when a :class:`Transform` is not able to resolve all of its
    hyperparameters at fit-time.
    """


@define
class HP:
    """
    A hyperparameter: that is, a :class:`Transform` parameter whose concrete value is
    deferred until fit-time, at which point its value is "**resolved**" by a dict of
    "**bindings**" provided to the :meth:`~Transform.fit()` call.

    A :class:`FitTransform` cannot be created unless all of its parent
    ``Transform``'s hyperparameters resolved to concrete values. The resolved
    parameter set, together with the fit state, are then used by the
    :meth:`~FitTransform.apply()` method.

    From the perspective of user-defined :meth:`~Transform._fit()` and
    :meth:`~Transform._apply()` methods, all parameters on ``self`` have already
    been resolved to concrete values if they were initially specified as
    hyperparameters, and the fit-time bindings dict itself is available as
    ``self.bindings()``.

    .. NOTE::
        The author of ``frankenfit`` has attempted to strike a balance between
        clarity and brevity in the naming of classes and functions. ``HP`` was
        chosen instead of ``Hyperparameter``, and similarly brief names given to
        its subclasses, because of the anticipated frequency with which
        hyperparameters are written into pipelines in the context of an
        interactive research environment.

    :param name: All hyperparameters have a name. By default (i.e., for instances
        of the ``HP`` base class) this is interepreted as the key mapping to a
        concrete value in the bindings dict.
    :type name: ``str``

    .. SEEALSO::
        Subclasses: :class:`HPFmtStr`, :class:`HPCols`, :class:`HPLambda`,
        :class:`HPDict`.
    """

    name: str

    def resolve(self, bindings: Mapping[str, Any]) -> Any | HP:
        """
        Return the concrete value of this hyperparameter according to the
        provided fit-time bindings. Exactly how the bindings determine the
        concrete value will vary among subclasses of ``HP``. By default, the name of the
        hyperparam (its ``self.name``) is treated as a key in the ``bindings`` dict,
        whose value is the concrete value.

        .. TIP::
            It's generally not expected that users will call this method directly
            themselves, instead relying on :meth:`Transform.fit()`,
            :meth:`StatelessTransform.apply()`, :meth:`Pipeline.apply()`,
            :meth:`Backend.fit()`, and :meth:`Backend.apply()` to do so for them as
            appropriate, as part of their hyperparameter resolution logic.

        .. SEEALSO::
            :class:`UnresolvedHyperparameterError`, :meth:`Transform.fit`,
            :meth:`StatelessTransform.apply`, :meth:`Pipeline.apply`,
            :meth:`Backend.fit`, :meth:`Backend.apply`.

        Parameters
        ----------
        bindings: dict[str, object]
            The fit-time bindings dictionary with respect to which to resolve this
            hyperparameter.

        Returns
        -------
        Any | HP
            Either the concrete value, or ``self`` (i.e., the still-unresolved
            hyperparameter) if resolution is not possible with the given bindings. After
            ``resolve()``-ing all of its hyperparameters, a caller may check for any
            parameters that are still HP objects to determine which, if any,
            hyperparameters could not be resolved. The base implementation of
            :meth:`Transform.fit()` raises an :class:`UnresolvedHyperparameterError` if
            any of the Transform's (or its children's) hyperparameters fail to resolve.
        """
        # default: treat hp name as key into bindings
        return bindings.get(self.name, self)

    X = TypeVar("X")

    @staticmethod
    def resolve_maybe(v: X, bindings: Mapping[str, Any]) -> X | Any:
        """
        A static utility method, that, if ``v`` is a hyperparameter (:class:`HP`
        instance or subclass), returns the result of resolving it on the given
        ``bindings``, otherwise returns ``v`` itself, as it must already be
        concrete.
        """
        if isinstance(v, HP):
            return v.resolve(bindings)
        return v

    def __hash__(self):
        return hash(repr(self))


class HPFmtStr(HP):
    def resolve(self, bindings: Mapping[str, T]) -> str:
        # treate name as format string to be formatted against bindings
        return self.name.format_map(bindings)

    C = TypeVar("C", bound="HPFmtStr")

    @classmethod
    def maybe_from_value(cls: type[C], x: str | HP) -> C | HP | str:
        if isinstance(x, HP):
            return x
        if isinstance(x, str):
            if x != "":
                return cls(x)
            return x
        raise TypeError(
            f"Unable to create a HPFmtStr from {x!r} which has type {type(x)}"
        )


def fmt_str_field(**kwargs):
    return field(converter=HPFmtStr.maybe_from_value, **kwargs)


@define
class HPLambda(HP):
    # TODO: UserLambdaHyperparams!

    resolve_fun: Callable
    name: str = "<lambda>"

    def resolve(self, bindings: Mapping[str, Any]) -> Any:
        return self.resolve_fun(bindings)


@define
class HPDict(HP):
    mapping: Mapping
    name: str = "<dict>"

    def resolve(self, bindings: Mapping[str, Any]) -> dict:
        return {
            (k.resolve(bindings) if isinstance(k, HP) else k): (
                v.resolve(bindings) if isinstance(v, HP) else v
            )
            for k, v in self.mapping.items()
        }

    C = TypeVar("C", bound="HPDict")

    @classmethod
    def maybe_from_value(cls: type[C], x: dict | HP) -> C | dict | HP:
        if isinstance(x, HP):
            return x
        if not isinstance(x, dict):
            raise TypeError(
                f"HPDict.maybe_from_value requires an HP or a dict, but got {x} which "
                f"has type {type(x)}"
            )
        # it's a dict
        if all(map(lambda k: not isinstance(k, HP), x.keys())) and all(
            map(lambda v: not isinstance(v, HP), x.values())
        ):
            return x
        return cls(x)

    def values(self) -> ValuesView:
        return self.mapping.values()

    def keys(self) -> KeysView:
        return self.mapping.keys()

    def items(self) -> ItemsView:
        return self.mapping.items()


def dict_field(**kwargs):
    return field(converter=HPDict.maybe_from_value, **kwargs)


class ALL_COLS:
    """
    Subclasses of :class:`frankenfit.dataframe.ColumnsTransform` use instances of this
    class to indicate that a parameter should refer to all columns found in the data.
    """

    def __repr__(self) -> str:
        return "<ALL_COLS>"


@define
class HPCols(HP):
    cols: list[str | HP]
    name: str = "<cols>"

    C = TypeVar("C", bound="HPCols")

    @classmethod
    def maybe_from_value(
        cls: type[C], x: str | HP | ALL_COLS | None
    ) -> C | str | HP | ALL_COLS | None:
        """_summary_

        :param x: _description_
        :type x: str | HP | Iterable[str  |  HP]
        :return: _description_
        :rtype: HPCols | HP
        """
        if isinstance(x, (HP, ALL_COLS)):
            return x
        if isinstance(x, str):
            return cls([x])
        if x is None:
            return None
        return cls(list(x))

    def resolve(self, bindings):
        try:
            return [
                c.resolve(bindings)
                if isinstance(c, HP)
                else c.format_map(bindings)
                if isinstance(c, str)
                else c
                for c in self.cols
            ]
        except KeyError as e:
            raise UnresolvedHyperparameterError(e)

    def __repr__(self):
        return repr(self.cols)

    def __len__(self):
        return len(self.cols)

    def __iter__(self):
        return iter(self.cols)


def _validate_not_empty(instance, attribute, value):
    """
    attrs field validator that throws a ValueError if the field value is empty.
    """
    if hasattr(value, "__len__"):
        if len(value) < 1:
            raise ValueError(f"{attribute.name} must not be empty but got {value}")
    elif isinstance(value, (HP, ALL_COLS)):
        return
    else:
        raise TypeError(f"{attribute.name} value has no length: {value}")


def columns_field(**kwargs):
    return field(
        validator=_validate_not_empty, converter=HPCols.maybe_from_value, **kwargs
    )


def optional_columns_field(**kwargs):
    return field(converter=HPCols.maybe_from_value, **kwargs)


if TYPE_CHECKING:  # pragma: no cover
    # This is so that pylance/pyright can autocomplete Transform constructor
    # arguments and instance variables.
    # See: https://www.attrs.org/en/stable/extending.html#pyright
    # And: https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md
    def __dataclass_transform__(
        *,
        eq_default: bool = True,
        order_default: bool = False,
        kw_only_default: bool = False,
        field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
    ) -> Callable[[_T], _T]:
        ...

else:
    # At runtime the __dataclass_transform__ decorator should do nothing
    def __dataclass_transform__(**kwargs):
        def identity(f):
            return f

        return identity


@__dataclass_transform__(
    field_descriptors=(
        attrs.field,
        fmt_str_field,
        dict_field,
        columns_field,
        optional_columns_field,
    )
)
def params(*args, **kwargs):
    """
    @transform docstr.
    """
    return define(*args, **{**kwargs, **{"slots": False, "eq": False}})
