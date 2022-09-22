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
from typing import Callable, Optional, Union

from attrs import define, field, fields_dict, Factory
import graphviz
import pandas as pd

_LOG = logging.getLogger(__name__)


def is_iterable(obj):
    """
    Utility function to test if an object is iterable.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


class Dataset(ABC):
    """
    Abstract base class of a dataset.
    """

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame.
        """
        raise NotImplementedError

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> Dataset:
        """
        Convenienice static method returns a :class:`PandasDataset` wrapping the given
        pandas DataFrame.
        """
        return PandasDataset(df)


@define
class PandasDataset(Dataset):
    df: pd.DataFrame
    """
    That darn dataframe.
    """

    def to_dataframe(self):
        """
        Returns the wrapped `DataFrame` instance directly.
        """
        return self.df


class UnknownDatasetError(KeyError):
    """
    Thrown by DatasetCollection.get_dataset() if the requested dataset name is not
    found.
    """


def _dataset_collection_converter(map: dict[str, pd.DataFrame | Dataset]):
    map = {
        name: PandasDataset(value) if isinstance(value, pd.DataFrame) else value
        for name, value in map.items()
    }
    # let's say that every dsc implicitly has a __pass__ dataset, which is an empty
    # dataframe if no dataset was provided with that name
    if "__pass__" not in map:
        map["__pass__"] = PandasDataset(pd.DataFrame())
    return map


@define
class DatasetCollection:

    map: dict[str, Dataset] = field(converter=_dataset_collection_converter)

    def get_dataset(self, name):
        if name not in self.map:
            if name == "__pass__":
                return PandasDataset(pd.DataFrame())
            raise UnknownDatasetError(
                f"Asked for a Dataset named {name!r} but this DatasetCollection only "
                f"has: {list(self.map.keys())}"
            )
        return self.map[name]

    def to_dataframe(self, name="__pass__"):
        return self.get_dataset(name).to_dataframe()

    @classmethod
    def from_data(cls, data: Optional[Data]):
        if data is None:
            dsc = cls({"__pass__": PandasDataset(pd.DataFrame())})
        elif isinstance(data, pd.DataFrame):
            dsc = cls({"__pass__": PandasDataset(data)})
        elif isinstance(data, Dataset):
            dsc = cls({"__pass__": data})
        elif isinstance(data, cls):
            dsc = data
        else:
            raise TypeError(
                f"Expected {Data} but got {data} which has type {type(data)}"
            )
        return dsc

    def __getitem__(self, dataset_name):
        return self.get_dataset(dataset_name)

    def __or__(self, other):
        if isinstance(other, dict):
            return DatasetCollection(self.map | other)
        if isinstance(other, DatasetCollection):
            return DatasetCollection(self.map | other.map)
        raise TypeError(
            f"Don't know how to union a DatasetCollection with a {type(other)}"
        )


# Type alias for the primary argument to Transform.fit() and .apply()
Data = Union[pd.DataFrame, Dataset, DatasetCollection]


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

    # Special (and default) value "__pass__" means give me give me the output of the
    # preceding Transform in a Pipeline, or the user's unnamed DataFrame/Dataset arg to
    # fit()/apply()
    dataset_name = "__pass__"
    """
    When part of a larger pipeline of transformations, the ``__dataset_name__``
    attribute determines how data is passed to a Transform at fit- and
    apply-time.

    The default value ``"__pass__"`` means that this ``Transform``'s
    :meth:`_fit()` and :meth:`_apply()` methods will receive the output of the Transform
    preceding it, or the user-provided data if being called outside of a Pipeline.

    Any other value will cause the :meth:`_fit()` and :meth:`_apply()` methods to
    receive the dataset with that name in the active :class:`DatasetCollection`.

    .. NOTE::
        Most subclasses of :class:`Transform` don't need to worry about doing anything
        with this attribute. The main exceptions would be if you are writing your own
        customized kind of :class:`Pipeline` class, or any Transform that needs to be
        able to introduce a new "branch" of dataflow into a Pipeline, originating from
        some data other than the output of the preceding Transform.

    .. SEEALSO::
        :attr:`~FitTransform.dataset_collection`, :meth:`_fit`, :meth:`_apply`,
        :class:`Pipeline`.

    :type: ``str``
    """

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
        - You have access to additional information beyond the training data
          (``df_fit``) via the attributes :attr:`self.dataset_name
          <Transform.dataset_name>`, :attr:`self.dataset_collection
          <FitTransform.dataset_collection>` and method :meth:`self.bindings()
          <FitTransform.bindings>`.

        TODO: examples of using dataset_name, etc.

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
        self, data_fit: Data = None, bindings: Optional[dict[str, object]] = None
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
        dsc = DatasetCollection.from_data(data_fit)
        fit_class: FitTransform = getattr(self, self._fit_class_name)
        return fit_class(self, dsc, bindings)

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
    """Exception thrown when a Transform is not able to resolve all of its
    hyperparameters at fit-time."""


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
    state that will be used at apply-time.
    """

    dataset_collection: DatasetCollection = None
    """
    Some docs about this.
    """

    def __init__(self, transform: Transform, dsc_fit: DatasetCollection, bindings=None):
        "Docstr for FitTransform.__init__"
        bindings = bindings or {}
        self._field_names = transform.params()
        for name in self._field_names:
            unbound_val = getattr(transform, name)
            bound_val = HP.resolve_maybe(unbound_val, bindings)
            # print("%s: Bound %r -> %r" % (name, unbound_val, bound_val))
            setattr(self, name, bound_val)
        self.__bindings = bindings
        self.dataset_name: str = transform.dataset_name
        self.tag: str = transform.tag
        # freak out if any hyperparameters failed to bind
        self._check_hyperparams()

        # materialize data for user _fit function.
        df_fit = dsc_fit.to_dataframe(self.dataset_name)
        self.__nrows = len(df_fit)
        # but also keep the original collection around (temporarily) in case the user
        # _fit function wants it
        self.dataset_collection = dsc_fit
        _LOG.debug(
            "Fitting %s on %d rows: %r", self.__class__.__name__, len(df_fit), self
        )
        # run user _fit function
        self.__state = transform._fit.__func__(self, df_fit)
        # now that fitting is done, we don't want to carry around a reference to all the
        # data
        self.dataset_collection = None

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

    def apply(self, data_apply: Data = None) -> pd.DataFrame:
        """
        Return the result of applying this fit Transform to the given DataFrame.
        """
        # materialize data for user _apply function.
        if data_apply is None:
            data_apply = pd.DataFrame()
        dsc_apply = DatasetCollection.from_data(data_apply)
        df_apply = dsc_apply.to_dataframe(self.dataset_name)
        # but also keep the original collection around (temporarily) in case the user
        # _apply function wants it
        self.dataset_collection = dsc_apply
        _LOG.debug(
            "Applying %s to %d rows: %r",
            self.__class__.__qualname__,
            len(df_apply),
            self,
        )
        result = self._apply(df_apply, state=self.__state)
        # now that application is done, we don't want to carry around a reference to all
        # the data
        self.dataset_collection = None
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
    """_summary_

    :param Transform: _description_
    :type Transform: _type_
    """

    def _fit(self, df_fit: pd.DataFrame):
        return None

    def apply(
        self, data_apply: Data = None, bindings: dict[str, object] = None
    ) -> pd.DataFrame:
        """
        Convenience function allowing one to apply a StatelessTransform without an
        explicit preceding call to fit. Implemented by calling fit() on no data (but
        with optional hyperparameter bindings as provided) and then returning the result
        of applying the resulting FitTransform to the given DataFrame.
        """
        return self.fit(None, bindings=bindings).apply(data_apply)


@define
class HP:
    """
    A transformation parameter whose concrete value is deferred until fit-time, at which
    point its value is "resolved" by a dict of "bindings" provided to the fit() call.
    ...
    """

    name: str

    def resolve(self, bindings: dict[str, object]) -> object:
        """_summary_

        :param bindings: _description_
        :type bindings: dict[str, object]
        :return: _description_
        :rtype: object
        """
        # default: treat hp name as key into bindings
        return bindings.get(self.name, self)

    @staticmethod
    def resolve_maybe(v, bindings: dict[str, object]) -> object:
        """_summary_

        :param v: _description_
        :type v: _type_
        :param bindings: _description_
        :type bindings: dict[str, object]
        :return: _description_
        :rtype: object
        """
        if isinstance(v, HP):
            return v.resolve(bindings)
        return v


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
