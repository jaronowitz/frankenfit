from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple, TypeVar, Union

import attrs
from attrs import define, field

T = TypeVar("T")
_T = TypeVar("_T")


class UnresolvedHyperparameterError(NameError):
    """
    Exception raised when a Transform is not able to resolve all of its
    hyperparameters at fit-time.
    """


@define
class HP:
    """
    A hyperparameter; that is, a transformation parameter whose concrete value
    is deferred until fit-time, at which point its value is "resolved" by a dict
    of "bindings" provided to the :meth:`~Transform.fit()` call.

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
        concrete value will vary among subclasses of HP. By default, the name of
        the hyperparam (its ``self.name``) is treated as a key in the ``bindings``
        dict, whose value is the concrete value.

        :param bindings: The fit-time bindings dictionary with respect to which
            to resolve this hyperparameter.
        :type bindings: dict[str, object]
        :return: Either the concrete value, or ``self`` (i.e., the
            still-unresolved hyperparameter) if resolution is not possible with
            the given bindings. After ``resolve()``-ing all of its
            hyperparameters, a caller may check for any parameters that are
            still HP objects to determine which, if any, hyperparameters could
            not be resolved. The base implementation of :meth:`Transform.fit()`
            raises an :class:`UnresolvedHyperparameterError` if any of the
            Transform's (or its children's) hyperparameters fail to resolve.
        :rtype: object

        .. SEEALSO::
            :class:`UnresolvedHyperparameterError`, :meth:`Transform.fit`,
            :meth:`StatelessTransform.apply`, :meth:`Pipeline.apply`.
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
            (k.resolve(bindings) if isinstance(k, HP) else k): v.resolve(bindings)
            if isinstance(v, HP)
            else v
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


def dict_field(**kwargs):
    return field(converter=HPDict.maybe_from_value, **kwargs)


@define
class HPCols(HP):
    cols: list[str | HP]
    name: str = "<cols>"

    C = TypeVar("C", bound="HPCols")
    X = str | HP | None

    @classmethod
    def maybe_from_value(cls: type[C], x: X) -> C | X:
        """_summary_

        :param x: _description_
        :type x: str | HP | Iterable[str  |  HP]
        :return: _description_
        :rtype: HPCols | HP
        """
        if isinstance(x, HP):
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
    elif isinstance(value, HP):
        return
    else:
        raise TypeError(f"{attribute.name} value has no length: {value}")


def columns_field(**kwargs):
    return field(
        validator=_validate_not_empty, converter=HPCols.maybe_from_value, **kwargs
    )


def optional_columns_field(**kwargs):
    return field(converter=HPCols.maybe_from_value, **kwargs)


if TYPE_CHECKING:
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
def transform(*args, **kwargs):
    """
    @transform docstr.
    """
    return define(*args, **(kwargs | {"slots": False}))
