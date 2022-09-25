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
Transform, FitTransform, Dataset, HP, and friends.

Ordinarily, users should never need to import this module directly. Instead, they access
the classes and functions defined here through the public API exposed as
``frankenfit.*``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging
from typing import Callable, Optional, TypeVar
import warnings

from attrs import define, field, fields_dict, Factory
import graphviz
import pandas as pd

_LOG = logging.getLogger(__name__)

T = TypeVar("T")


def is_iterable(obj):
    """
    Utility function to test if an object is iterable.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def _next_auto_tag(partial_self):
    class_name = partial_self.__class__.__qualname__
    nonce = str(_next_id_num(class_name))
    return f"{class_name}#{nonce}"


# TODO: remove need for Transform subclasses to write @define
@define(slots=False)
class Transform(ABC):
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
        @define
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

    # Note the following are regular attributes, NOT managed by attrs

    is_constant = False

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

    # def __attrs_post_init__(self):
    #     class_name = self.__class__.__qualname__
    #     node_id = str(_next_id_num(class_name))
    #     self.tag = f"{class_name}#{node_id}"

    @abstractmethod
    def _fit(self, df_fit: pd.DataFrame) -> object:
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

    @abstractmethod
    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        """
        Implements subclass-specific logic to apply the tansformation after being fit.

        .. NOTE::
            ``_apply()`` is one of two methods (the other being ``_fit()``) that any
            subclass of :class:`Transform` must implement.

        TODO.

        :param df_apply: _description_
        :type df_apply: ``pd.DataFrame``
        :param state: _description_, defaults to None
        :type state: ``object``, optional
        :raises NotImplementedError: _description_
        :return: _description_
        :rtype: pd.DataFrame
        """
        raise NotImplementedError

    def find_by_tag(self, tag: str):
        # base implementation checks params that are transforms or iterables of
        # transforms. Subclasses should override if they have other ways of keeping
        # child transforms.
        if self.tag == tag:
            return self
        for name in self.params():
            val = getattr(self, name)
            if isinstance(val, Transform):
                try:
                    return val.find_by_tag(tag)
                except KeyError:
                    pass
            elif is_iterable(val):
                for x in val:
                    if isinstance(x, Transform):
                        try:
                            return x.find_by_tag(tag)
                        except KeyError:
                            pass

        raise KeyError(f"No child Transform found with tag: {tag}")

    def fit(
        self,
        data_fit: pd.DataFrame = None,
        bindings: Optional[dict[str, object]] = None,
    ) -> FitTransform:
        """
        Fit this Transform on some data and return a :class:`FitTransform` object. The
        actual return value will be some subclass of ``FitTransform`` that depends on
        what class this Transform is; for example, :meth:`ZScore.fit()` returns a
        :class:`FitZScore` object.

        :param data_fit: _description_
        :type data_fit: Data
        :param bindings: _description_, defaults to None
        :type bindings: Optional[dict[str, object]], optional
        :return: _description_
        :rtype: FitTransform
        """
        if data_fit is None:
            data_fit = pd.DataFrame()
        fit_class: FitTransform = getattr(self, self._fit_class_name)
        return fit_class(self, data_fit, bindings)

    def params(self) -> list[str]:
        """
        Return the names of all of the parameters

        :return: list of parameter names.
        :rtype: list[str]
        """
        field_names = list(fields_dict(self.__class__).keys())
        return field_names

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

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        # out of the box, handle three common cases:
        # - we are a simple transform with no child transforms
        # - we have one or more child transforms as Transform-valued params
        # - we have one or more child transforms as elements of a list-valued param
        # Subclasses override for their own cases not covered by the above
        # TODO: this function has gotten too big and needs refactoring
        children_as_params: dict[str, Transform] = {}
        children_as_elements_of_params: dict[str, list[Transform]] = {}
        param_vals: dict[str, object] = {}

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
                param_vals[name] = repr(val)

        param_vals_fmt = ",\n".join([" = ".join([k, v]) for k, v in param_vals.items()])
        self_label = f"{self.tag}\n{param_vals_fmt}"

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

    VISUALIZE_DEFAULT_DIGRAPH_KWARGS = {
        "node_attr": {
            "shape": "box",
            "fontsize": "10",
            "fontname": "Monospace",
        },
        "edge_attr": {"fontsize": "10", "fontname": "Monospace"},
    }

    def visualize(self, **digraph_kwargs):
        # TODO: rework with a _visualize() method that does the actual recursion and can
        # be overridden. have a notion of entry and exit tags for each subgraph
        digraph = graphviz.Digraph(
            **(self.VISUALIZE_DEFAULT_DIGRAPH_KWARGS | digraph_kwargs)
        )
        self._visualize(digraph, ("lightgrey", "white"))
        return digraph

    def __init_subclass__(cls, /, no_magic=False, **kwargs):
        """
        Implements black magic to help with writing Transform subclasses.
        """
        super().__init_subclass__(**kwargs)
        if no_magic:
            return

        class DerivedFitTransform(FitTransform, transform_class=cls):
            pass

        # we should freak out if the subclass has any attribute named 'state' or
        # 'bindings', because those will collide at apply-time with fit_class method
        # names.
        # TODO: generalize this check a bit -- we should be able to determine
        # automatically which parameter names would collide with attributes of
        # FitTransform.
        if hasattr(cls, "state"):
            raise AttributeError(
                "Subclasses of Transform are not allowed to have an attribute "
                'named "state". Deal with it.'
            )
        if hasattr(cls, "bindings"):
            raise AttributeError(
                "Subclasses of Transform are not allowed to have an attribute "
                'named "bindings". Deal with it.'
            )

        fit_class = DerivedFitTransform
        fit_class_name = fit_class.__name__
        fit_class.__qualname__ = ".".join((cls.__qualname__, fit_class_name))
        cls.fit.__annotations__["return"] = fit_class.__qualname__
        setattr(cls, fit_class_name, fit_class)
        cls._fit_class_name = fit_class_name


class SentinelDict(dict):
    keys_checked = None

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


class UnresolvedHyperparameterError(NameError):
    """
    Exception raised when a Transform is not able to resolve all of its
    hyperparameters at fit-time.
    """


class FitTransform(ABC):
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

    def __init__(self, transform: Transform, df_fit: pd.DataFrame, bindings=None):
        "Docstr for FitTransform.__init__"
        bindings = bindings or {}
        self._field_names = transform.params()
        for name in self._field_names:
            unbound_val = getattr(transform, name)
            bound_val = HP.resolve_maybe(unbound_val, bindings)
            # print("%s: Bound %r -> %r" % (name, unbound_val, bound_val))
            setattr(self, name, bound_val)
        self.__bindings = bindings
        self.tag: str = transform.tag
        # freak out if any hyperparameters failed to bind
        self._check_hyperparams()

        # materialize data for user _fit function.
        df_fit = df_fit
        self.__nrows = len(df_fit)
        # but also keep the original collection around (temporarily) in case the user
        # _fit function wants it
        _LOG.debug(
            "Fitting %s on %d rows: %r", self.__class__.__name__, len(df_fit), self
        )
        # run user _fit function
        self.__state = transform._fit.__func__(self, df_fit)

    def _check_hyperparams(self):
        unresolved = []
        for name in self._field_names:
            val = getattr(self, name)
            if isinstance(val, HP):
                unresolved.append(val)
        if unresolved:
            raise UnresolvedHyperparameterError(
                f"One or more hyperparameters of {self.__class__.__qualname__} were "
                f"not resolved at fit-time: {unresolved}. Bindings were: "
                f"{self.__bindings}"
            )

    def __repr__(self):
        fields_str = ", ".join(
            ["%s=%r" % (name, getattr(self, name)) for name in self._field_names]
        )
        data_str = f"<{self.__nrows} rows of fitting data>"
        if fields_str:
            return f'{self.__class__.__name__}({", ".join([fields_str, data_str])})'
        return f"{self.__class__.__name__}({data_str})"

    @abstractmethod
    def _apply(self, df_apply: pd.DataFrame, state=None) -> pd.DataFrame:
        raise NotImplementedError

    def apply(self, df_apply: pd.DataFrame = None) -> pd.DataFrame:
        """
        Return the result of applying this fit Transform to the given DataFrame.
        """
        # materialize data for user _apply function.
        if df_apply is None:
            df_apply = pd.DataFrame()
        # but also keep the original collection around (temporarily) in case the user
        # _apply function wants it
        _LOG.debug(
            "Applying %s to %d rows: %r",
            self.__class__.__qualname__,
            len(df_apply),
            self,
        )
        result = self._apply(df_apply, state=self.__state)
        return result

    # TODO: refit()

    def bindings(self) -> dict[str, object]:
        """
        Return the bindings dict according to which the transformation's hyperparameters
        were resolved.
        """
        return self.__bindings

    def state(self) -> object:
        """
        Return the fit state of the transformation, which is an arbitrary object
        determined by the implementation of ``{transform_class_name}._fit()``.
        """
        return self.__state

    def find_by_tag(self, tag: str):
        # Base implementation checks the state object if it is a FitTransform or an
        # iterable of FitTransforms. Subclasses should override if they have other ways
        # of keeping child FitTransforms.
        # TODO: how can subclasses override this, given that they are created implicitly
        # by metprogramming?
        if self.tag == tag:
            return self

        val = self.state()
        if isinstance(val, FitTransform):
            try:
                return val.find_by_tag(tag)
            except KeyError:
                pass
        elif is_iterable(val):
            for x in val:
                if isinstance(x, FitTransform):
                    try:
                        return x.find_by_tag(tag)
                    except KeyError:
                        pass

        raise KeyError(f"No child Transform found with tag: {tag}")

    def __init_subclass__(cls, /, transform_class: type = None, **kwargs):
        # TODO: futz with base classes so that super() works like normal in the user's
        # _fit() and _apply() methods when subclassing another Transform.
        super().__init_subclass__(**kwargs)
        if transform_class is None:
            return
        cls._apply = transform_class._apply
        cls.__name__ = f"Fit{transform_class.__name__}"
        cls.__doc__ = FitTransform.__doc__.format(
            transform_class_name=transform_class.__name__
        )
        cls.state.__doc__ = FitTransform.state.__doc__.format(
            transform_class_name=transform_class.__name__
        )
        cls.__init__.__annotations__["transform"] = transform_class.__name__

        field_names = list(fields_dict(transform_class).keys())
        cls._field_names = field_names


class StatelessTransform(Transform):
    """
    Abstract base class of Transforms that have no state to fit. ``fit()`` is a
    null op on a ``StatelessTransform``, and the ``state()`` of its fit is
    always ``None``. Subclasses must not implement ``_fit()``.

    As a convenience, ``StatelessTransform`` has an ``apply()`` method
    (ordinarily only the corresponding fit would). For any
    ``StatelessTransform`` ``t``, ``t.apply(df, bindings)`` is equivalent to
    ``t.fit(df, bindings=bindings).apply(df)``.
    """

    def _fit(self, df_fit: pd.DataFrame):
        return None

    def apply(
        self, df_apply: pd.DataFrame = None, bindings: dict[str, object] = None
    ) -> pd.DataFrame:
        """
        Convenience function allowing one to apply a StatelessTransform without an
        explicit preceding call to fit. Implemented by calling fit() on no data (but
        with optional hyperparameter bindings as provided) and then returning the result
        of applying the resulting FitTransform to the given DataFrame.
        """
        return self.fit(None, bindings=bindings).apply(df_apply)


class NonInitialConstantTransformWarning(RuntimeWarning):
    """
    An instance of :class:`ConstantTransform` was found to be non-initial in a
    :class:`Pipeline`, or the user provided it with non-empty input data. This
    is usually unintentional.

    .. SEEALSO::
        :class:`ConstantTransform`
    """


class ConstantTransform(StatelessTransform):
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

    def fit(
        self,
        data_fit: pd.DataFrame = None,
        bindings: Optional[dict[str, object]] = None,
    ) -> FitTransform:
        if data_fit is not None and not data_fit.empty:
            warning_msg = (
                "A ConstantTransform's fit method received non-empty input data. "
                "Tihs is likely unintentional because that input data will be "
                "ignored and discarded.\n"
                f"transform={self!r}\n"
                f"data_fit.head(5)=\n{data_fit.head(5)!r}"
            )
            _LOG.warning(warning_msg)
            warnings.warn(
                warning_msg,
                NonInitialConstantTransformWarning,
            )
        return super().fit(data_fit, bindings)

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

    Within the implementations of user-defined :meth:`~Transform._fit()` and
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

    def resolve(self, bindings: dict[str, T]) -> T | HP:
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
            hyperparameters, a the caller may check for any parameters that are
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

    @staticmethod
    def resolve_maybe(v: object, bindings: dict[str, T]) -> T:
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
    """_summary_

    :param HP: _description_
    :type HP: _type_
    """

    def resolve(self, bindings: dict[str, object]) -> object:
        # treate name as format string to be formatted against bindings
        return self.name.format_map(bindings)

    @classmethod
    def maybe_from_value(cls, x: str | HP):
        """_summary_

        :param x: _description_
        :type x: str | HP
        :raises TypeError: _description_
        :return: _description_
        :rtype: _type_
        """
        if isinstance(x, HP):
            return x
        if isinstance(x, str):
            if x != "":
                return HPFmtStr(x)
            return x
        raise TypeError(
            f"Unable to create a HPFmtStr from {x!r} which has type {type(x)}"
        )


def fmt_str_field(**kwargs):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    return field(converter=HPFmtStr.maybe_from_value, **kwargs)


# Valid column list specs (routed by field converter):
# hp('which_cols') -> plain old hp
# ['x', 'y', 'z'] -> hp_cols (plain old list?)
# ['x', hp('some_col'), 'z'] -> hp_cols
# ['x', '{som_col}', 'z'] -> hp_cols
# Scalars rewritten to lists of one:
#   'z' -> ['z'] -> hp_cols
#   '{some_col}' - ['{som_col}'] -> hp_cols


@define
class HPLambda(HP):
    """_summary_

    :param HP: _description_
    :type HP: _type_
    :return: _description_
    :rtype: _type_
    """

    resolve_fun: Callable
    name: str = None

    def resolve(self, bindings: dict[str, object]) -> object:
        return self.resolve_fun(bindings)


@define
class HPCols(HP):
    """_summary_

    :param HP: _description_
    :type HP: _type_
    :return: _description_
    :rtype: _type_
    """

    cols: list[str | HP]
    name: str = None

    @classmethod
    def maybe_from_value(cls, x: str | HP | Iterable[str | HP]) -> HPCols | HP:
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
        return [
            c.resolve(bindings)
            if isinstance(c, HP)
            else c.format_map(bindings)
            if isinstance(c, str)
            else c
            for c in self.cols
        ]

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
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    return field(
        validator=_validate_not_empty, converter=HPCols.maybe_from_value, **kwargs
    )


def optional_columns_field(**kwargs):
    """_summary_

    :return: _description_
    :rtype: _type_
    """
    return field(factory=list, converter=HPCols.maybe_from_value, **kwargs)


@define
class HPDict(HP):
    """_summary_

    :param HP: _description_
    :type HP: _type_
    :raises TypeError: _description_
    :return: _description_
    :rtype: _type_
    """

    mapping: dict
    name: str = None

    def resolve(self, bindings: dict[str, object]) -> object:
        return {
            (k.resolve(bindings) if isinstance(k, HP) else k): v.resolve(bindings)
            if isinstance(v, HP)
            else v
            for k, v in self.mapping.items()
        }

    @classmethod
    def maybe_from_value(cls, x: dict | HP):
        """_summary_

        :param x: _description_
        :type x: dict | HP
        :raises TypeError: _description_
        :return: _description_
        :rtype: _type_
        """
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


_id_num = {}


def _next_id_num(class_name):
    n = _id_num.get(class_name, 0)
    n += 1
    _id_num[class_name] = n
    return n
