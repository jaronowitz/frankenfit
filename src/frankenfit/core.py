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
The core building blocks and utilities of the Frankenfit library, including
Transform, FitTransform, HP, and friends.

Ordinarily, users should never need to import this module directly. Instead, they access
the classes and functions defined here through the public API exposed as
``frankenfit.*``.
"""

from __future__ import annotations

import copy
import inspect
import logging
import operator
import types
import warnings
from abc import ABC, abstractmethod
from functools import reduce, wraps
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import attrs
import graphviz  # type: ignore
from attrs import NOTHING, Factory, define, field, fields_dict

from .backend import Backend, DummyBackend, DummyFuture

_LOG = logging.getLogger(__name__)

T = TypeVar("T")
_T = TypeVar("_T")
P = TypeVar("P", bound="BasePipeline")
R = TypeVar("R", bound="Transform")
R_co = TypeVar("R_co", covariant=True, bound="Transform")
State = TypeVar("State")
State_co = TypeVar("State_co", covariant=True)
DataIn = TypeVar("DataIn", contravariant=True)
DataResult = TypeVar("DataResult", covariant=True)

Bindings = dict[str, Any]


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


@__dataclass_transform__(field_descriptors=(attrs.field,))
def transform(*args, **kwargs):
    """
    @transform docstr.
    """
    return define(*args, **(kwargs | {"slots": False}))


def is_iterable(obj):
    """
    Utility function to test if an object is iterable.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def flatten_tuples(xs):
    for x in xs:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


def copy_function(f):
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    # NOTE: we must not use functools.update_wrapper, because that sets the
    # __wrapped__ attribute, which we DON'T want, because it causes help() and
    # friends to do the wrong thing. E.g., we won't be able to change the
    # signature reported by help(). PyPI package `makefun` could also work, but
    # does more than we need.
    g.__name__ = f.__name__
    g.__qualname__ = f.__qualname__
    g.__doc__ = f.__doc__
    g.__module__ = f.__module__
    g.__dict__.update(f.__dict__)
    g.__kwdefaults__ = f.__kwdefaults__
    g.__annotations__ = dict(inspect.get_annotations(f))
    return g


# last auto-generated tag number for a given Transform class name. Used for
# autogenerated tags.
_id_num: dict[str, int] = {}


def _next_id_num(class_name: str) -> int:
    n = _id_num.get(class_name, 0)
    n += 1
    _id_num[class_name] = n
    return n


def _next_auto_tag(partial_self: Transform) -> str:
    """
    Autogenerate a tag for the given Transform object, which is presumably only
    partially constructed. Ensures that Transforms receive unique default
    tags when not specified by the user.

    The current implementation returns the object's class qualname with ``"#N"``
    appended, where N is incremented (per class name) on each call.
    """
    class_name = partial_self.__class__.__qualname__
    nonce = str(_next_id_num(class_name))
    return f"{class_name}#{nonce}"


DEFAULT_VISUALIZE_DIGRAPH_KWARGS = {
    "node_attr": {
        "shape": "box",
        "fontsize": "10",
        "fontname": "Monospace",
    },
    "edge_attr": {"fontsize": "10", "fontname": "Monospace"},
}


class UnresolvedHyperparameterError(NameError):
    """
    Exception raised when a Transform is not able to resolve all of its
    hyperparameters at fit-time.
    """


# TODO: Strictly speaking, could/should not be freely specified, but rather
# follow from the other type params as Transform[DataIn, DataResult]
class FitTransform(Generic[R_co, DataIn, DataResult]):
    """
    The result of fitting a :class:`{transform_class_name}` Transform. Call this
    object's :meth:`apply()` method on some data to get the result of applying the
    now-fit transformation.

    All parameters of the fit {transform_class_name} are available as instance
    variables, with any hyperparameters fully resolved against whatever bindings were
    provided at fit-time.

    The fit state of the transformation, as returned by {transform_class_name}'s
    ``_fit()`` method at fit-time, is available from :meth:`state()`, and this is the
    state that will be used at apply-time (i.e., passed as the ``state``
    argument of the user's ``_fit()`` method).
    """

    def __init__(
        self,
        resolved_transform: R_co,
        state: Any,
        bindings: Optional[Bindings] = None,
    ):
        self.__resolved_transform = resolved_transform
        self.__state = state
        self.__bindings = bindings or {}
        self.tag: str = resolved_transform.tag

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"resolved_transform={self.__resolved_transform!r}, "
            f"state={type(self.__state)!r}, "
            f"bindings={self.__bindings!r}, "
            f")"
        )

    # TODO: shouldn't return type be Optional? Like, what happens when
    # data_apply is None?
    def apply(
        self,
        data_apply: Optional[DataIn] = None,
        backend: Optional[Backend] = None,
    ) -> DataResult:
        """
        Return the result of applying this FitTransform to the given data.
        """
        if isinstance(data_apply, Sized):
            data_len = len(data_apply)
        else:
            data_len = None

        _LOG.debug(
            f"Applying {self.tag} on {type(data_apply)}"
            f"{f' (len={data_len})' if data_len is not None else ''} "
        )

        if backend is None:
            backend = DummyBackend()

        # run user _apply function on backend
        tf = self.resolved_transform()
        sig = inspect.signature(tf._apply).parameters
        if len(sig) not in (2, 3):
            raise TypeError(
                f"I don't know how to invoke user _apply() method with "
                f"{len(sig)} arguments: {tf._apply} with signature {str(sig)}"
            )
        args: tuple[Any, ...] = (data_apply, self.state())
        # pass bindings if _apply has the signature for it
        if len(sig) == 3:
            args += (self.bindings(),)

        result = backend.submit(f"{self.tag}._apply", tf._apply, *args)
        if isinstance(result, DummyFuture):
            return result.result()
        return result

    # TODO: refit()? incremental_fit()?

    def resolved_transform(self) -> R_co:
        """
        Return the Transform that was fit to produce this FitTransform, with all
        hyperparameters resolved to their fit-time bindings.
        """
        return self.__resolved_transform

    def bindings(self) -> Bindings:
        """
        Return the bindings dict according to which the transformation's hyperparameters
        were resolved.
        """
        if self.__bindings is None:
            return {}
        return self.__bindings

    def state(self) -> Any:
        """
        Return the fit state of the transformation, which is an arbitrary object
        determined by the implementation of ``{transform_class_name}._fit()``.
        """
        return self.__state

    def find_by_tag(self, tag: str):
        # TODO: address implementation of find_by_tag on FitTransform. We should
        # consider both FitTransform-valued params and FitTransform objects in
        # our state.
        if self.tag == tag:
            return self

        val = self.state()
        if isinstance(val, FitTransform):
            try:
                return val.find_by_tag(tag)
            except KeyError:
                pass
        elif is_iterable(val):
            for x in cast(Iterable[Any], val):
                if isinstance(x, FitTransform):
                    try:
                        return x.find_by_tag(tag)
                    except KeyError:
                        pass

        raise KeyError(f"No child Transform found with tag: {tag}")


# TODO: remove need for Transform subclasses to write @transform?
@transform
class Transform(ABC, Generic[DataIn, DataResult]):
    """
    The abstract base class of all (unfit) Transforms. Subclasses must implement the
    :meth:`_fit()` and :meth:`_apply()` methods (but see :class:`StatelessTransform`,
    which removes the requirement to implement :meth:`_fit()`).

    Subclasses should use `attrs <https://www.attrs.org>`_ field variables to hold
    parameters (but not fit state) of the transformation being implemented, with the
    expectation that these parameters will be provided by the user of the subclass as
    constructor arguments.  Thanks to ``attrs``, in most cases no constructor needs to
    be written explicitly by the subclass author, and in any case only ``attrs``-managed
    field variables will be treated as potential hyperparameters at fit-time (i.e., to
    potentially get their values from the ``bindings=`` kwarg to :meth:`fit()`).

    The implementations of :meth:`_fit()` and :meth:`_apply()` may refer freely to any
    ``attrs`` fields (generally understood as parameters of the transformation) as
    instance variables on ``self``. If any fields were given as hyperparameters at
    construction time, they are resolved to concrete bindings before ``_fit()`` and
    ``_apply()`` are invoked.

    ``_fit()`` should accept some training data and return an arbitrary object
    representing fit state, which will be passed to ``_apply()`` at apply-time.
    Generally speaking, ``_fit()`` should *not* mutate anything about ``self``.

    ``_apply()`` should then accept a state object as returned by ``_fit()`` and return
    the result of applying the transformation to some given apply-time data, also
    without mutating ``self``.

    Once implemented, the subclass is used like any ``Transform``, which is to say by
    constructing an instance with some parameters (which may be hypeparameters), and
    then calling its ``fit()`` and ``apply()`` methods (note no leading underscores).

    A subclass ``C`` will automatically find itself in possession of an inner class
    ``FitC``, which derives from ``FitTransform``.  ``C.fit()`` will then return a
    ``C.FitC`` instance (encapsulating the state returned by the subclasser's ``_fit()``
    implementation), whose ``apply()`` method (i.e., ``C.FitC.apply()``) employs the
    subclasser's ``_apply()`` implementation.

    .. WARNING::
        Subclasses must not keep parameters in fields named ``fit``, ``apply``,
        ``state``, ``params``, or ``bindings`` as these would break functionality by
        overriding expected method names.

    An example of writing a ``Transform`` subclass::

        from attrs import define
        import pandas as pd
        import frankenfit as ff

        # A simple stateful transform from scratch, subclassing Transform directly.
        @transform
        class DeMean(ff.Transform):
            "De-mean some columns."

            cols: list[str] # Or, get this for free by subclassing ColumnsTransform

            def _fit(self, df_fit: pd.DataFrame) -> object:
                return df_fit[self.cols].mean()

            def _apply(self, df_apply: pd.DataFrame, state: object):
                means = state
                return df_apply.assign(**{
                    c: df_apply[c] - means[c]
                    for c in self.cols
                })

    An example of a stateless Transform whose only parameter is a list of columns; the
    implementation is simplified by subclassing two "convenience base classes":
    :class:`StatelessTransform` for the common case of a transform with no state to fit,
    and :class:`ColumnsTransform`, for the common case of operating on a parameterized
    list of columns, which is made available as an attrs-managed field ``self.cols``::

        class KeepColumns(ff.StatelessTransform, ff.ColumnsTransform):
            def _apply(
                self, df_apply: pd.DataFrame, state: object=None
            ) -> pd.DataFrame:
                return df_apply[self.cols]
    """

    # TODO: do we really want tag to be hyperparameterizable? shouldn't it be
    # invariant wrt fit-time data and bindings?
    tag: str = field(
        init=True,
        eq=False,
        kw_only=True,
        default=Factory(_next_auto_tag, takes_self=True),
    )
    """
    The ``tag`` attribute is the one parameter common to all ``Transforms``. used for
    identifying and selecting Transform instances within Pipelines. Ignored when
    comparing Transforms. It is an optional kwarg to the constructor of ``Transform``
    and all of its subclasses. If not provided, a default value is derived from the
    subclass's ``__qualname__``. It's up to the user to keep tags unique.

    .. SEEALSO::
        :meth:`find_by_tag`, :meth:`FitTransform.find_by_tag`.

    :type: ``str``
    """

    @tag.validator
    def _check_tag(self, attribute, value):
        if not isinstance(value, str):
            raise TypeError(
                f"tag must be a str-like; but got a {type(value)}: {value!r}"
            )

    @abstractmethod
    # def _fit(self, data_fit: Any, bindings: Optional[Bindings] = None) -> Any:
    def _fit(self, data_fit: DataIn) -> Any:
        """
        Implements subclass-specific fitting logic.

        .. NOTE::
            ``_fit()`` is one of two methods (the other being ``_apply()``) that any
            subclass of :class:`Transform` must implement. (But see
            :class:`StatelessTransform` as a way to avoid this for transforms that don't
            have state to fit.)

        Here are some useful points to keep in mind whilst writing your ``_fit()``
        function, which you can consider part of Frankenfit's API contract:

        - When your ``_fit()`` function is executed, ``self`` actually refers to an
          instance of :class:`FitTransform` (in fact a subclass of ``FitTransform`` that
          is specific to your :class:`Transform` subclass), which is being constructed
          and will store the state that your method returns.
        - TODO: Params all available on self, concrete values, hyperparams resolved.
        - You have access to hyperparameter bindings via :meth:`self.bindings()
          <FitTransform.bindings>`.

        :param df_fit: A pandas ``DataFrame`` of training data.
        :type df_fit: ``pd.DataFrame``
        :raises NotImplementedError: If not implemented by the subclass.
        :return: An arbitrary object, the type and meaning of which are specific to the
            subclass. This object will be passed as the ``state`` argument to
            :meth:`_apply()` at apply-time.
        :rtype: ``object``
        """
        raise NotImplementedError

    FitTransformClass: ClassVar[Type[FitTransform]] = FitTransform

    _Self = TypeVar("_Self", bound="Transform")

    def fit(
        self: _Self,
        data_fit: Optional[DataIn] = None,
        bindings: Optional[Bindings] = None,
        backend: Optional[Backend] = None,
    ) -> FitTransform[_Self, DataIn, DataResult]:
        """
        Fit this Transform on some data and hyperparam bindings, and return a
        :class:`FitTransform` object. The actual return value will be some
        subclass of ``FitTransform`` that depends on what class this Transform
        is; for example, :meth:`ZScore.fit()` returns a :class:`FitZScore`
        object.
        """
        if isinstance(data_fit, Sized):
            data_len = len(data_fit)
        else:
            data_len = None

        _LOG.debug(
            f"Fitting {self.tag} on {type(data_fit)}"
            f"{f' (len={data_len})' if data_len is not None else ''} "
            f"with bindings={bindings!r}"
        )

        if backend is None:
            backend = DummyBackend()

        # resolve hyperparams and run user _fit function on backend
        resolved_transform = self._resolve_hyperparams(bindings)
        sig = inspect.signature(resolved_transform._fit).parameters
        if len(sig) not in (1, 2):
            raise TypeError(
                f"I don't know how to invoke user _fit() method with "
                f"{len(sig)} arguments: {resolved_transform._fit} with signature "
                f"{str(sig)}"
            )
        args: tuple[Any, ...] = (data_fit,)
        # pass bindings if _fit has a second argument
        if len(sig) == 2:
            args += (bindings or {},)
        state = backend.submit(f"{self.tag}._fit", resolved_transform._fit, *args)
        if isinstance(state, DummyFuture):
            state = state.result()

        # fit_class: type[FitTransform] = getattr(self, self._fit_class_name)
        return type(self).FitTransformClass(resolved_transform, state, bindings)

    @abstractmethod
    def _apply(self, data_apply: DataIn, state: Any) -> DataResult:
        """
        Implements subclass-specific logic to apply the tansformation after being fit.

        .. NOTE::
            ``_apply()`` is one of two methods (the other being ``_fit()``) that any
            subclass of :class:`Transform` must implement.

        TODO.
        """
        raise NotImplementedError

    def params(self) -> list[str]:
        """
        Return the names of all of the parameters

        :return: list of parameter names.
        :rtype: list[str]
        """
        field_names = list(fields_dict(self.__class__).keys())
        return field_names

    def annotations(self) -> dict[str, str]:
        def _get_annos(t: type) -> tuple[dict, ...]:
            return tuple(inspect.get_annotations(x) for x in (t,) + t.__bases__)

        dicts = list(flatten_tuples(_get_annos(type(self))))
        return reduce(operator.or_, dicts)

    def hyperparams(self) -> set[str]:
        """
        Return the set of hyperparameter names that this :class:`Transform` expects to
        be bound at fit-time. If this Transform contains other Transforms (for example
        if it's a :class:`Pipeline` or a :class:`Join`), then the set of hyperparameter
        names is collected recursively.

        :return: list of hyperparameter names.
        :rtype: list[str]
        """
        # TODO: is there some way to report what type each hyperparam is expected to
        # have?
        # TODO: is there some way to determine automatically which of our params() are,
        # or contain, Transforms, and thereby handle the recursion here rather than
        # expecting subclasses to implement it correctly?

        sd = SentinelDict()
        sub_transform_results = set()
        for name in self.params():
            unbound_val = getattr(self, name)
            if isinstance(unbound_val, HP):
                HP.resolve_maybe(unbound_val, sd)
            elif isinstance(unbound_val, Transform):
                sub_transform_results |= unbound_val.hyperparams()
            elif isinstance(unbound_val, list) and len(unbound_val) > 0:
                for x in unbound_val:
                    if isinstance(x, Transform):
                        sub_transform_results |= x.hyperparams()

        return (sd.keys_checked or set()) | sub_transform_results

    def _resolve_hyperparams(self: R, bindings: Optional[Bindings] = None) -> R:
        """
        Returns a shallow copy of self with all HP-valued params resolved (but
        not recursively), or raises UnresolvedHyperparameter if unable to do so
        with the given bindings.
        """
        bindings = bindings or {}
        params = self.params()
        # TODO someday: check types of bound values against annotations
        # annos = self.annotations()
        resolved_transform = copy.copy(self)
        unresolved = []
        for name in params:
            unbound_val = getattr(self, name)
            bound_val = HP.resolve_maybe(unbound_val, bindings)
            # print("%s: Bound %r -> %r" % (name, unbound_val, bound_val))
            # NOTE: we cannot use setattr because that will trigger attrs'
            # converter machinery, which is only desirable before hyperparams
            # are resolved (columns_field, etc.). So we write to the __dict__
            # directly.
            resolved_transform.__dict__[name] = bound_val
            if isinstance(bound_val, HP):
                unresolved.append(bound_val)

        # freak out if any hyperparameters failed to bind
        if unresolved:
            raise UnresolvedHyperparameterError(
                f"One or more hyperparameters of {self.__class__.__qualname__} were "
                f"not resolved at fit-time: {unresolved}. Bindings were: "
                f"{bindings}"
            )

        return resolved_transform

    def then(
        self: Transform, other: Optional[Transform | list[Transform]] = None
    ) -> BasePipeline:
        if other is None:
            transforms = [self]
        elif isinstance(other, list):
            transforms = [self] + other
        elif isinstance(other, Transform):
            transforms = [self, other]
        else:
            raise TypeError(f"then(): other must be Transform or list, got: {other!r}")

        return BasePipeline(transforms=transforms)

    def __add__(self, other: Optional[Transform | list[Transform]]) -> BasePipeline:
        return self.then(other)

    def _children(self) -> Iterable[Transform]:
        """
        Base implementation checks params that are transforms or iterables of
        transforms. Subclasses should override this if they have other ways of keeping
        child transforms.
        """
        # TODO: address implementation on FitTransform. we need a _children()
        # method that can be overridden like _apply().  ... or actually
        # _children() and _fit_children()... the latter needs to iterate through state
        yield self
        for name in self.params():
            val = getattr(self, name)
            if isinstance(val, Transform):
                yield from val._children()
            elif is_iterable(val) and not isinstance(val, str):
                for x in val:
                    if isinstance(x, Transform):
                        yield from x._children()

    def find_by_tag(self, tag: str) -> Transform:
        """
        Recurse through child transforms (i.e., transforms that are, or are
        contained in, this transform's params) and return the first one with the
        given tag. If not found, raise KeyError.
        """
        for child in self._children():
            if child.tag == tag:
                return child
        raise KeyError(f"No child Transform found with tag: {tag}")

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        # out of the box, handle three common cases:
        # - we are a simple transform with no child transforms
        # - we have one or more child transforms as Transform-valued params
        # - we have one or more child transforms as elements of a list-valued param
        # Subclasses override for their own cases not covered by the above
        # TODO: this function has gotten too big and needs refactoring
        children_as_params: dict[str, Transform] = {}
        children_as_elements_of_params: dict[str, list[Transform]] = {}
        param_reprs: dict[str, str] = {}

        for name in self.params():
            if name == "tag":
                continue
            val = getattr(self, name)
            tvals = []
            has_children = False
            # collect each Transform-type param
            if isinstance(val, Transform):
                children_as_params[name] = val
                has_children = True
            # same for each Transform-type element of a list param
            elif isinstance(val, list) and len(val) > 0:
                for x in val:
                    if isinstance(x, Transform):
                        tvals.append(x)
                        has_children = True
            if tvals:
                children_as_elements_of_params[name] = tvals
            # for non-Transform params, collect their values to be displayed in the
            # label of the node for this Transform
            if (not has_children) and (val is not None):
                param_reprs[name] = repr(val)

        param_reprs_fmt = ",\n".join(
            [" = ".join([k, v]) for k, v in param_reprs.items()]
        )
        self_label = f"{self.tag}\n{param_reprs_fmt}"

        if not (children_as_params or children_as_elements_of_params):
            digraph.node(self.tag, label=self_label)
            return ([self.tag], [(self.tag, "")])

        # we gon' need a cartouche
        my_exits = []
        with digraph.subgraph(name=f"cluster_{self.tag}") as sg:
            bg, fg = bg_fg
            bg_fg = fg, bg
            sg.attr(style="filled", color=bg)
            sg.node_attr.update(style="filled", color=fg)
            sg.node(self.tag, label=self_label)

            for t_name, t in children_as_params.items():
                t_entries, t_exits = t._visualize(sg, bg_fg)
                for t_entry in t_entries:
                    sg.edge(self.tag, t_entry, label=t_name)
                my_exits.extend(t_exits)

            for tlist_name, tlist in children_as_elements_of_params.items():
                prev_exits = None
                for t in tlist:
                    t_entries, t_exits = t._visualize(sg, bg_fg)
                    if prev_exits is None:
                        # edges from self to first transform's entries
                        for t_entry in t_entries:
                            sg.edge(self.tag, t_entry, label=tlist_name)
                    else:
                        # edge from prvious transform's exits node to this
                        # transform's entries node
                        for prev_exit, prev_exit_label in prev_exits:
                            for t_entry in t_entries:
                                sg.edge(prev_exit, t_entry, label=prev_exit_label)
                    prev_exits = t_exits
                # last transform in tlist becomes one of our exits
                my_exits.append((t.tag, ""))

        return [self.tag], my_exits

    def visualize(self, **digraph_kwargs):
        """
        Return a visualization of this Transform as a ``graphviz.DiGraph``
        object. The caller may render it to file or screen.
        """
        digraph = graphviz.Digraph(
            **(DEFAULT_VISUALIZE_DIGRAPH_KWARGS | digraph_kwargs)
        )
        self._visualize(digraph, ("lightgrey", "white"))
        return digraph


class SentinelDict(dict):
    """
    Utility class that behaves exactly like an ordinary did, but keeps track of
    which keys have been read, available in the ``keys_checked`` instance
    attribute, which is either None if no keys have been read, or the set of keys.
    """

    keys_checked: Optional[set] = None

    def _record_key(self, key):
        if self.keys_checked is None:
            self.keys_checked = set()
        self.keys_checked.add(key)

    def __getitem__(self, key):
        self._record_key(key)
        return None

    def get(self, key, *args, **kwargs):
        self._record_key(key)
        return None


class StatelessTransform(Generic[DataIn, DataResult], Transform[DataIn, DataResult]):
    """
    Abstract base class of Transforms that have no state to fit. ``fit()`` is a
    null op on a ``StatelessTransform``, and the ``state()`` of its fit is
    always ``None``. Subclasses must not implement ``_fit()``.

    As a convenience, ``StatelessTransform`` has an ``apply()`` method
    (ordinarily only the corresponding fit would). For any
    ``StatelessTransform`` ``t``, ``t.apply(df, bindings)`` is equivalent to
    ``t.fit(df, bindings=bindings).apply(df)``.
    """

    def _fit(self, data_fit: DataIn) -> None:
        return None

    # TODO: subclasses should automagically get a return type consistent with
    # the return type of self.fit().apply()
    def apply(
        self,
        data_apply: Optional[DataIn] = None,
        bindings: Optional[Bindings] = None,
        backend: Optional[Backend] = None,
    ) -> DataResult:
        """
        Convenience function allowing one to apply a StatelessTransform without
        an explicit preceding call to fit. Implemented by calling fit() on no
        data (but with optional hyperparameter bindings as provided) and then
        returning the result of applying the resulting FitTransform to the given
        object.
        """
        return self.fit(None, bindings=bindings, backend=backend).apply(
            data_apply, backend=backend
        )


class NonInitialConstantTransformWarning(RuntimeWarning):
    """
    An instance of :class:`ConstantTransform` was found to be non-initial in a
    :class:`Pipeline`, or the user provided it with non-empty input data. This
    is usually unintentional.

    .. SEEALSO::
        :class:`ConstantTransform`
    """


class ConstantTransform(
    Generic[DataIn, DataResult], StatelessTransform[DataIn, DataResult]
):
    """
    Abstract base class of Transforms that have no state to fit, and, at apply
    time, produce output data that is independent of the input data.
    Usually, a ``ConstantTransform`` is some kind of data reader or data generator.
    Its parameters and bindings may influence its output, but it takes no input
    data to be transformed per se.

    Because it has the effect of discarding the output of all preceding
    computations in a :class:`Pipeline`, a warning is emited
    (:class:`NonInitialConstantTransformWarning`) whenever a
    ``ConstantTransform`` is fit on non-empty input data, or found to be
    non-initial in a Pipeline.
    """

    Self = TypeVar("Self", bound="ConstantTransform")

    def fit(
        self: Self,
        data_fit: Optional[DataIn] = None,
        bindings: Optional[Bindings] = None,
        backend: Optional[Backend] = None,
    ) -> FitTransform[Self, DataIn, DataResult]:
        if data_fit is not None:
            warning_msg = (
                "A ConstantTransform's fit method received non-empty input data. "
                "This is likely unintentional because that input data will be "
                "ignored and discarded.\n"
                f"transform={self!r}\n"
                f"data_fit=\n{data_fit!r}"
            )
            _LOG.warning(warning_msg)
            warnings.warn(
                warning_msg,
                NonInitialConstantTransformWarning,
            )
        return super().fit(data_fit, bindings, backend=backend)

    # TODO: emit a similar warning from apply(), but that requires futzing with
    # FitTransform


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
    """_summary_

    :param HP: _description_
    :type HP: _type_
    :raises TypeError: _description_
    :return: _description_
    :rtype: _type_
    """

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
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    return field(converter=HPDict.maybe_from_value, **kwargs)


C = TypeVar("C", bound="Callable[..., Any]")


def callchain(transform_class: type[R]) -> Callable[[C], C]:
    def inner(f: C) -> C:
        f.__doc__ = dedent(f.__doc__ or "") + dedent(transform_class.__doc__ or "")

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            return self + transform_class(*args, **kwargs)

        return cast(C, wrapper)

    return inner


def method_wrapping_transform(
    class_qualname: str, method_name: str, transform_class: type[R]
) -> Callable[..., BasePipeline]:
    def method_impl(self, *args, **kwargs) -> BasePipeline:
        return self + transform_class(*args, **kwargs)

    method_impl.__annotations__.update(
        transform_class.__init__.__annotations__ | {"return": class_qualname}
    )
    method_impl.__name__ = method_name
    method_impl.__qualname__ = ".".join((class_qualname, method_name))
    sig = inspect.signature(transform_class.__init__).replace(
        return_annotation=class_qualname
    )
    # Workaround mypy bug: https://github.com/python/mypy/issues/12472
    # method_impl.__signature__ = sig
    setattr(method_impl, "__signature__", sig)
    method_impl.__doc__ = f"""
    Return the result of appending a new :class:`{transform_class.__name__}` transform
    constructed with the given parameters to this pipeline.
    This method's arguments are passed directly to
    ``{transform_class.__name__}.__init__()``.

    .. SEEALSO:: :class:`{transform_class.__qualname__}`
    """
    # if transform_class.__doc__ is not None:
    #     transform_class.__doc__ += f"""

    # .. SEEALSO:: :meth:`{class_qualname}.{method_name}`
    # """

    return method_impl


def _convert_pipeline_transforms(value):
    result = []
    if isinstance(value, BasePipeline):
        # "coalesce" Pipelines
        tf_seq = value.transforms
    elif isinstance(value, Transform):
        tf_seq = [value]
    elif value is None:
        tf_seq = []
    else:
        tf_seq = list(value)

    for tf_elem in tf_seq:
        if isinstance(tf_elem, BasePipeline):
            # "coalesce" Pipelines
            result.extend(tf_elem.transforms)
        elif isinstance(tf_elem, Transform):
            result.append(tf_elem)
        else:
            raise TypeError(f"Pipeline cannot contain a non-Transform: {tf_elem!r}")

    return result


P_co = TypeVar("P_co", bound="BasePipeline", covariant=True)


class Grouper(Generic[P_co]):
    def __init__(
        self,
        pipeline_upstream: P_co,
        wrapper_class: type,
        wrapper_kwarg_name_for_wrappee: str,
        **wrapper_other_kwargs,
    ):
        self._pipeline_upstream = pipeline_upstream
        self._wrapper_class = wrapper_class
        self._wrapper_kwarg_name_for_wrappee = wrapper_kwarg_name_for_wrappee
        self._wrapper_other_kwargs = wrapper_other_kwargs

    def then(self, other: Optional[Transform | list[Transform]]) -> P_co:
        if not isinstance(self._pipeline_upstream, BasePipeline):
            raise TypeError(
                f"Grouper cannot be applied to non-BasePipeline upstream: "
                f"{self._pipeline_upstream} with type "
                f"{type(self._pipeline_upstream)}"
            )

        if not isinstance(other, Transform):
            other = type(self._pipeline_upstream)(transforms=other)

        wrapping_transform = self._wrapper_class(
            **(
                {
                    self._wrapper_kwarg_name_for_wrappee: other,
                }
                | self._wrapper_other_kwargs
            )
        )
        return self._pipeline_upstream + cast(Transform, wrapping_transform)

    def __add__(self, other: Optional[Transform | list[Transform]]) -> P_co:
        return self.then(other)


DataInOut = TypeVar("DataInOut")


@transform
class BasePipeline(Generic[DataInOut], Transform[DataInOut, DataInOut]):
    _pipeline_methods: ClassVar[list[str]] = []

    class _Grouper(Grouper):
        pass

    @classmethod
    def with_methods(
        cls: type[P], subclass_name: Optional[str] = None, **kwargs
    ) -> type[P]:
        if subclass_name is None:
            subclass_name = f"{cls.__name__}WithMethods"

        # workaround for mypy bug: https://github.com/python/mypy/issues/5865
        klass: Any = cls

        class Subclass(klass):
            class _Grouper(klass._Grouper):
                pass

            pass

        Subclass.__name__ = subclass_name
        Subclass.__qualname__ = subclass_name
        Subclass._Grouper.__qualname__ = Subclass.__qualname__ + ".Grouper"

        pipeline_methods = []
        if hasattr(cls, "_pipeline_methods"):
            pipeline_methods.extend(cls._pipeline_methods)
        Subclass._pipeline_methods = pipeline_methods
        for method_name, transform_class in kwargs.items():
            setattr(
                Subclass,
                method_name,
                method_wrapping_transform(
                    Subclass.__qualname__, method_name, transform_class
                ),
            )
            Subclass._pipeline_methods.append(method_name)
            setattr(
                Subclass._Grouper,
                method_name,
                method_wrapping_transform(
                    Subclass.__qualname__, method_name, transform_class
                ),
            )

        return Subclass

    transforms: list[Transform[DataInOut, DataInOut]] = field(
        factory=list, converter=_convert_pipeline_transforms
    )

    @transforms.validator
    def _check_transforms(self, attribute, value):
        t_is_first = True
        for t in value:
            if not isinstance(t, Transform):
                raise TypeError(
                    "Pipeline sequence must comprise Transform instances; found "
                    f"non-Transform {t!r} (type {type(t)})"
                )
            # warning if a ConstantTransform is non-initial
            if (not t_is_first) and isinstance(t, ConstantTransform):
                warning_msg = (
                    f"A ConstantTransform is non-initial in a Pipeline: {t!r}. "
                    "This is likely unintentional because the output of all "
                    "preceding Transforms, once computed, will be discarded by "
                    "the ConstantTransform."
                )
                _LOG.warning(warning_msg)
                warnings.warn(
                    warning_msg,
                    NonInitialConstantTransformWarning,
                )
            t_is_first = False

    def __init__(self, tag=NOTHING, transforms=NOTHING):
        self.__attrs_init__(tag=tag, transforms=transforms)

    def _fit(
        self, data_fit: Any, bindings: Optional[Bindings] = None
    ) -> list[FitTransform]:
        # TODO: run on backend
        fit_transforms = []
        for t in self.transforms:
            ft = t.fit(data_fit, bindings=bindings)
            data_fit = ft.apply(data_fit)
            fit_transforms.append(ft)
        return fit_transforms

    def _apply(self, data_apply: Any, state: list[FitTransform]) -> Any:
        # TODO: run on backend
        df = data_apply
        for fit_transform in state:
            df = fit_transform.apply(df)
        return df

    def __len__(self):
        return len(self.transforms)

    # TODO: should return type be optional?
    def apply(
        self,
        data_fit: Optional[DataInOut] = None,
        bindings: Optional[Bindings] = None,
        backend: Optional[Backend] = None,
    ) -> DataInOut:
        """
        An efficient alternative to ``Pipeline.fit(...).apply(...)``.  When the
        fit-time data and apply-time data are identical, it is more efficient to
        use a single call to ``apply()`` than it is to call
        :meth:`~Transform.fit()` followed by a separate call to
        :meth:`~FitTransform.apply()`, both on the same data argument. This is
        because ``fit()`` itself must already apply every transform in the
        pipeline, in orer to produce the fitting data for the following
        transform. ``apply()`` captures the result of these fit-time
        applications, avoiding their unnecessary recomputation.

        :return: The result of fitting this :class:`Pipeline` and applying it to its own
            fitting data.
        """
        for t in self.transforms:
            ft = t.fit(data_fit, bindings=bindings, backend=backend)
            data_fit = ft.apply(data_fit, backend=backend)
        return data_fit  # type: ignore [return-value]

    def __add__(
        self: P,
        other: Optional[
            Transform[DataInOut, DataInOut] | list[Transform[DataInOut, DataInOut]]
        ],
    ) -> P:
        return self.then(other)

    def then(
        self: P,
        other: Optional[
            Transform[DataInOut, DataInOut] | list[Transform[DataInOut, DataInOut]]
        ] = None,
    ) -> P:
        """
        Return the result of appending the given :class:`Transform` instance(s) to this
        :class:`Pipeline`. The addition operator on Pipeline objects is an alias for
        this method, meaning that the following are equivalent pairs::

            pipeline + ff.DeMean(...) == pipeline.then(ff.DeMean(...))
            pipeline + other_pipeline == pipeline.then(other_pipeline)
            pipeline + [ff.Winsorize(...), ff.DeMean(...)] == pipeline.then(
                [ff.Winsorize(...), ff.DeMean(...)]
            )

        In the case of appending built-in ``Transform`` classes it is usually not
        necessary to call ``then()`` because the ``Pipeline`` object has a more specific
        method for each built-in ``Transform``. For example, the last pipeline in the
        example above could be written more idiomatically as::

            pipeline.winsorize(...).de_mean(...)

        The main use cases for ``then()`` are to append user-defined ``Transform``
        subclasses that don't have built-in methods like the above, and to append
        separately constructed ``Pipeline`` objects when writing a pipeline in the
        call-chain style. For example::

            def bake_features(cols):
                # using built-in methods for Winsorize and ZScore transforms
                return ff.Pipeline().winsorize(cols, limit=0.05).z_score(cols)

            class MyCustomTransform(ff.Transform):
                ...

            pipeline = (
                ff.Pipeline()
                .pipe(['carat'], np.log1p)  # built-in method for Pipe transform
                .then(bake_features(['carat', 'table', 'height']))  # append Pipeline
                .then(MyCustomTransform(...))  # append a user-defined transform
            )

        Another common use case for ``then()`` is when you want :meth:`group_by()` to
        group a complex sub-pipeline, not just a single transform, e.g.::

            pipeline = (
                ff.Pipeline()
                .group_by("cut")
                    .then(
                        # This whole Pipeline of transforms will be fit and applied
                        # independently per distinct value of cut
                        ff.Pipeline()
                        .zscore(["carat", "table", "depth"])
                        .winsorize(["carat", "table", "depth"])
                    )
            )

        :param other: The Transform instance to append, or a list of Transforms, which
            will be appended in the order in which in they appear in the list.
        :type other: :class:`Transform` | ``list[Transform]``
        :raises ``TypeError``: If ``other`` is not a ``Transform`` or list of
            ``Transform``\\ s.
        :rtype: :class:`Pipeline`
        """
        if other is None:
            transforms = self.transforms
        elif isinstance(other, BasePipeline):
            # coalesce pass-through pipeline
            transforms = self.transforms + other.transforms
        elif isinstance(other, Transform):
            transforms = self.transforms + [other]
        elif isinstance(other, list):
            transforms = self.transforms + other
        else:
            raise TypeError(
                f"I don't know how to extend a Pipeline with {other}, which is of "
                f"type {type(other)}, bases = {type(other).__bases__}. "
            )
        return self.__class__(tag=self.tag, transforms=transforms)
