"""
The graph module provides :class:`~frankenft.Transform` subclasses (notably
:class:`~frankenfit.Pipeline`, :class:`~frankenfit.Join`, and friends) that take other
Transforms as parameters, allowing the creation of sequences and tree-like structures of
Transforms that process each other's output (i.e., composition).

Ordinarily, users should never need to import this module directly. Instead, they access
the classes and functions defined here through the public API exposed as
``frankenfit.*``.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import partial, reduce
import inspect
import operator
import warnings

import logging
from attrs import define, field
import pandas as pd
import numpy as np

from typing import Callable, Optional

from . import transforms as fft
from . import core as ffc
from .core import Transform

_LOG = logging.getLogger(__name__)

# TODO: figure out how to make type annotations look like ``ff.Transform`` instead of
# ``ffc.Transform``.

# - graph-making transforms (pipeline, joins, ifs)
# - the notion that a pipeline may fit and apply on a collection of datasets, not just
# one. graph "branches" may begin with a Dataset node, whose fit/apply methods take a
# DataSpec rather than DFs and produce a DF. User can then fit/apply the whole pipeline
# ona DataSpec to employ a collection of datasets at fit/apply-time.
# ... kinda neat, then you could imagine a Timeseries subclass of Dataset which allows
# users to call fit/apply just on time-ranges (like context)

# Timeseries?
# Graph-making transforms:
# Pipeline, Join, JoinAsOf (time series), IfHyperparamTrue, IfTrainingDataHasProperty,
# GroupedBy, Longitudinally (time series), CrossSectionally (time seires),
# Sequentially (time series), AcrossHyperParamGrid


@define
class IfHyperparamIsTrue(Transform):
    name: str
    then: Transform
    otherwise: Optional[Transform] = None
    allow_unresolved: Optional[bool] = False

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()
        if (not self.allow_unresolved) and self.name not in bindings:
            raise ffc.UnresolvedHyperparameterError(
                f"IfHyperparamIsTrue: no binding for {self.name!r} but "
                "allow_unresolved is False"
            )
        if bindings.get(self.name):
            return self.then.fit(df_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        result.add(self.name)
        return result


@define
class IfHyperparamLambda(Transform):
    fun: Callable  # dict[str, object] -> bool
    then: Transform
    otherwise: Optional[Transform] = None

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()
        if self.fun(bindings):
            return self.then.fit(df_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        # find out what bindings our lambda function queries
        sd = ffc.SentinelDict()
        self.fun(sd)
        result |= sd.keys_checked or set()
        return result


@define
class IfTrainingDataHasProperty(Transform):
    fun: Callable  # df -> bool
    then: Transform
    otherwise: Optional[Transform] = None

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.fun(df_fit):
            return self.then.fit(df_fit, bindings=self.bindings())
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=self.bindings())
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity


@define
class Join(Transform):
    left: Pipeline
    right: Pipeline
    how: str

    on: Optional[str] = None
    left_on: Optional[str] = None
    right_on: Optional[str] = None
    suffixes: tuple[str] = ("_x", "_y")

    # TODO: more merge params like left_index etc.
    # TODO: (when on distributed compute) context extension

    def _fit(self, df_fit: pd.DataFrame) -> object:
        dsc = self.dataset_collection | {"__pass__": df_fit}
        bindings = self.bindings()
        return (
            self.left.fit(dsc, bindings=bindings),
            self.right.fit(dsc, bindings=bindings),
        )

    def _apply(
        self, df_apply: pd.DataFrame, state: tuple[ffc.FitTransform]
    ) -> pd.DataFrame:
        fit_left, fit_right = state
        dsc = self.dataset_collection | {"__pass__": df_apply}
        df_left, df_right = fit_left.apply(dsc), fit_right.apply(dsc)
        return pd.merge(
            left=df_left,
            right=df_right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            suffixes=self.suffixes,
        )


def method_wrapping_transform(
    class_qualname: str, method_name: str, transform_class: type
) -> Callable[..., Pipeline]:
    def method_impl(self, *args, **kwargs) -> Pipeline:
        return self + transform_class(*args, **kwargs)

    method_impl.__annotations__.update(
        transform_class.__init__.__annotations__ | {"return": class_qualname}
    )
    method_impl.__name__ = method_name
    method_impl.__qualname__ = ".".join((class_qualname, method_name))
    method_impl.__signature__ = inspect.signature(transform_class.__init__).replace(
        return_annotation=class_qualname
    )
    method_impl.__doc__ = f"""
    Return the result of appending a new :class:`{transform_class.__name__}` transform
    constructed with the given parameters to this :class:`Pipeline`. This method's
    arguments are passed directly to ``{transform_class.__name__}.__init__()``.

    .. SEEALSO:: :class:`{transform_class.__qualname__}`
    """
    if transform_class.__doc__ is not None:
        transform_class.__doc__ += f"""

    .. SEEALSO:: :meth:`{class_qualname}.{method_name}`
    """

    return method_impl


def _convert_pipeline_transforms(value):
    # "coalesce" Pipelines if they are pass-through
    if isinstance(value, Pipeline) and value.dataset_name == "__pass__":
        return list(value.transforms)

    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], Pipeline)
        and value[0].dataset_name == "__pass__"
    ):
        return list(value[0].transforms)

    if isinstance(value, Transform):
        return [value]

    return list(value)


_pipeline_method_wrapping_transform = partial(method_wrapping_transform, "Pipeline")


class CallChainingMixin(ABC):
    """
    Abstract base class used internally to implement Frankenfit classes that provide a
    standardized call-chaining API, e.g. :class:`Pipeline` and :class:`PipelineGrouper`.
    Concrete subclasses must implement :meth:`then()`.
    """

    @abstractmethod
    def then(self, other: Transform | list[Transform]) -> Pipeline:
        """
        Return the result of appending the given :class:`Transform` to the current
        call-chaining object.
        """
        raise NotImplementedError

    def __add__(self, other):
        return self.then(other)

    ####################
    # call-chaining API:

    copy = _pipeline_method_wrapping_transform("copy", fft.Copy)
    select = _pipeline_method_wrapping_transform("select", fft.Select)
    __getitem__ = select
    rename = _pipeline_method_wrapping_transform("rename", fft.Rename)
    drop = _pipeline_method_wrapping_transform("drop", fft.Drop)
    stateless_lambda = _pipeline_method_wrapping_transform(
        "stateless_lambda", fft.StatelessLambda
    )
    stateful_lambda = _pipeline_method_wrapping_transform(
        "stateful_lambda", fft.StatefulLambda
    )
    pipe = _pipeline_method_wrapping_transform("pipe", fft.Pipe)
    clip = _pipeline_method_wrapping_transform("clip", fft.Clip)
    winsorize = _pipeline_method_wrapping_transform("winsorize", fft.Winsorize)
    impute_constant = _pipeline_method_wrapping_transform(
        "impute_constant", fft.ImputeConstant
    )
    impute_mean = _pipeline_method_wrapping_transform("impute_mean", fft.ImputeMean)
    de_mean = _pipeline_method_wrapping_transform("de_mean", fft.DeMean)
    z_score = _pipeline_method_wrapping_transform("z_score", fft.ZScore)
    print = _pipeline_method_wrapping_transform("print", fft.Print)
    log_message = _pipeline_method_wrapping_transform("log_message", fft.LogMessage)

    if_hyperparam_is_true = _pipeline_method_wrapping_transform(
        "if_hyperparam_is_true", IfHyperparamIsTrue
    )
    if_hyperparam_lambda = _pipeline_method_wrapping_transform(
        "if_hyperparam_lambda", IfHyperparamLambda
    )
    if_training_data_has_property = _pipeline_method_wrapping_transform(
        "if_training_data_has_property", IfTrainingDataHasProperty
    )

    sklearn = _pipeline_method_wrapping_transform("sklearn", fft.SKLearn)
    statsmodels = _pipeline_method_wrapping_transform("statsmodels", fft.Statsmodels)
    correlation = _pipeline_method_wrapping_transform("correlation", fft.Correlation)


EFFACING_TRANSFORM_CLASSES = [
    Join,
]


def is_effacing(transform: Transform):
    if transform.dataset_name != "__pass__":
        return True
    return any(isinstance(transform, C) for C in EFFACING_TRANSFORM_CLASSES)


@define
class Pipeline(Transform, CallChainingMixin):
    # Already defined in the Transform base class, but declare again here so that attrs
    # makes it the first (optional) __init__ argument.
    dataset_name: str = "__pass__"

    transforms: list[Transform] = field(
        factory=list, converter=_convert_pipeline_transforms
    )

    @transforms.validator
    def _check_transforms(self, attribute, value):
        t_is_first = True
        for t in value:
            if not isinstance(t, Transform):
                raise TypeError(
                    "Pipeline sequence must comprise Transform instances; found "
                    f"non-Transform {t} (type {type(t)})"
                )
            # warning if an "effacing Transform" is non-initial. E.g., Join, or
            # any Transform with dataset_name != "__pass__"
            if (not t_is_first) and is_effacing(t):
                warnings.warn(
                    f"An effacing Transform is non-initial in a Pipeline: {t!r}. "
                    "This is likely unintentional because the effects of all "
                    "preceding Transforms will be discarded by the effacing "
                    "Transform.",
                    RuntimeWarning,
                )
            t_is_first = False

    def _fit(self, df_fit: pd.DataFrame) -> object:
        dsc = self.dataset_collection | {"__pass__": df_fit}
        fit_transforms = []
        bindings = self.bindings()
        for t in self.transforms:
            ft = t.fit(dsc, bindings=bindings)
            df = ft.apply(dsc)
            dsc |= {"__pass__": df}
            fit_transforms.append(ft)
        return fit_transforms

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        dsc = self.dataset_collection | {"__pass__": df_apply}
        df = df_apply  # in case we are an empty Pipeline
        for fit_transform in state:
            df = fit_transform.apply(dsc)
            dsc |= {"__pass__": df}
        return df

    def __len__(self):
        return len(self.transforms)

    def fit_and_apply(
        self, data_fit: ffc.Data, bindings: Optional[dict[str, object]] = None
    ) -> pd.DataFrame:
        """
        An efficient alternative to ``self.fit(df).apply(df)`` specific to
        :class:`Pipeline` objects. When the fit-time data and apply-time data are
        identical, it is more efficient to use a single call to ``fit_and_apply()`` than
        it is to call :meth:`~Transform.fit()` followed by a separate call to
        :meth:`~FitTransform.apply()`, both on the same data argument. This is because
        ``fit()`` itself must already apply every transform in the pipeline, in orer to
        produce the fitting data for the following transform. ``fit_and_apply()``
        captures the result of these fit-time applications, avoiding their unnecessary
        recomputation.

        :return: The result of fitting this :class:`Pipeline` and applying it to its own
            fitting data.
        """
        dsc = ffc.DatasetCollection.from_data(data_fit)
        for t in self.transforms:
            ft = t.fit(dsc, bindings=bindings)
            df = ft.apply(dsc)
            dsc |= {"__pass__": df}
        return df

    def then(self, other: Transform | list[Transform]) -> Pipeline:
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
        if isinstance(other, Pipeline) and other.dataset_name == "__pass__":
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
        return Pipeline(dataset_name=self.dataset_name, transforms=transforms)

    def join(
        self, right, how, on=None, left_on=None, right_on=None, suffixes=("_x", "_y")
    ) -> Pipeline:
        """
        Return a new :class:`Pipeline` containing a new :class:`Join` transform with
        this ``Pipeline`` as the ``Join``'s ``left`` argument.

        .. SEEALSO:: :class:`Join`:
        """
        join = Join(
            self,
            right,
            how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
        )
        return Pipeline(transforms=join)

    def group_by(self, cols, fitting_schedule=None) -> PipelineGrouper:
        """
        Return a :class:`PipelineGrouper` object, which will consume the next Transform
        in the call-chain by wrapping it in a :class:`GroupBy` transform and returning
        the result of appending that ``GroupBy`` to this Pipeline. It enables
        Pandas-style call-chaining with ``GroupBy``.

        For example, grouping a single Transform::

            (
                ff.Pipeline()
                # ...
                .group_by("cut")  # -> PipelineGrouper
                    .z_score(cols)  # -> Pipeline
            )

        Grouping a sequence of Transforms::

            (
                ff.Pipeline()
                # ...
                .group_by("cut")
                    .then(
                        ff.Pipeline()
                        .winsorize(cols, limit=0.01)
                        .z_score(cols)
                        .clip(cols, upper=2, lower=-2)
                    )
            )

        .. NOTE::
            When using ``group_by()``, by convention we add a level of indentation to
            the next call in the call-chain, to indicate visually that it is being
            consumed by the preceding ``group_by()`` call.

        :param cols: The column(s) by which to group. The next Transform in the
            call-chain will be fit and applied separately on each subset of data with a
            distinct combination of values in ``cols``.
        :type cols: str | HP | list[str | HP]

        :param fitting_schedule: How to determine the fitting data of each group. The
            default schedule is :meth:`fit_group_on_self`. Use this to implement
            workflows like cross-validation and sequential fitting.
        :type fitting_schedule: Callable[dict[str, object], np.array[bool]]

        :rtype: :class:`PipelineGrouper`
        """
        return PipelineGrouper(cols, self, fitting_schedule or fit_group_on_self)


Pipeline.join.__doc__ = Pipeline.join.__doc__.format(join_docs=Join.__doc__)


def fit_group_on_self(group_col_map):
    """
    The default fitting schedule for :class:`GroupBy`: for each group, the grouped-by
    transform is fit on the data belonging to that group.
    """
    return lambda df: reduce(
        operator.and_, (df[c] == v for (c, v) in group_col_map.items())
    )


def fit_group_on_all_other_groups(group_col_map):
    """
    A built-in fitting schedule for :class:`GroupBy`: for each group, the grouped-by
    transform is fit on the data belonging to all other groups. This is similar to
    k-fold cross-validation if the groups are viewed as folds.
    """
    return lambda df: reduce(
        operator.or_, (df[c] != v for (c, v) in group_col_map.items())
    )


@define
class GroupBy(ffc.Transform):
    """
    Group the fitting and application of a :class:`Transform` by the distinct values of
    some column or combination of columns.

    :param cols: The column(s) by which to group. ``transform`` will be fit and applied
        separately on each subset of data with a distinct combination of values in
        ``cols``.
    :type cols: str | HP | list[str | HP]

    :param transform: The :class:`Transform` to group.
    :type transform: HP | Transform

    :param fitting_schedule: How to determine the fitting data of each group. The
        default schedule is :meth:`fit_group_on_self`. Use this to implement workflows
        like cross-validation and sequential fitting.
    :type fitting_schedule: Callable[[dict[str, object]], np.array[bool]]

    .. SEEALSO::
        :meth:`Pipeline.group_by`

    """

    # TODO: what about grouping by index?
    cols: str | ffc.HP | list[str | ffc.HP] = ffc.columns_field()
    transform: ffc.HP | ffc.Transform = field()
    # TODO: what about hyperparams in the fitting schedule? that's a thing.
    fitting_schedule: Callable[[dict[str, object]], np.array[bool]] = field(
        default=fit_group_on_self
    )

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()

        def fit_on_group(df_group: pd.DataFrame):
            # select the fitting data for this group
            group_col_map = {c: df_group[c].iloc[0] for c in self.cols}
            df_group_fit = df_fit.loc[self.fitting_schedule(group_col_map)]
            # fit the transform on the fitting data for this group
            dsc = self.dataset_collection | {"__pass__": df_group_fit}
            return self.transform.fit(dsc, bindings=bindings)

        return (
            df_fit.groupby(self.cols, as_index=False, sort=False)
            .apply(fit_on_group)
            .rename(columns={None: "__state__"})
        )

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        def apply_on_group(df_group: pd.DataFrame):
            dsc = self.dataset_collection | {
                "__pass__": df_group.drop(["__state__"], axis=1)
            }
            # values of __state__ ought to be identical within the group
            group_state: ffc.FitTransform = df_group["__state__"].iloc[0]
            # TODO: what if this group was not seen at fit-time?
            return group_state.apply(dsc)

        return (
            df_apply.merge(state, how="left", on=self.cols)
            .groupby(self.cols, as_index=False, sort=False)
            .apply(apply_on_group)
        )


class PipelineGrouper(CallChainingMixin):
    """
    An intermediate "grouper" object returned by :meth:`Pipeline.group_by()` (analogous
    to pandas ``DataFrameGroupBy`` objects), which is not a :class:`Pipeline`, but has
    the same call-chain methods as a Pipeline, and consumes the next call to finally
    create the :class:`GroupBy` Transform and return the result of appending that to the
    matrix Pipeline. It enables this style of ``group_by()`` call-chaining syntax::

        (
            ff.Pipeline()
            # ...
            .group_by("cut")  # -> PipelineGrouper
                .z_score(cols)  # -> Pipeline
        )
    """

    def __init__(self, groupby_cols, pipeline_upstream, fitting_schedule):
        self.groupby_cols = groupby_cols
        self.pipeline_upstream = pipeline_upstream
        self.fitting_schedule = fitting_schedule

    def __repr__(self):
        return "PipelineGrouper(%r, %r)" % (self.groupby_cols, self.pipeline_upstream)

    def then(self, other: Transform | list[Transform]) -> Pipeline:
        if isinstance(other, list):
            other = Pipeline(other)
        groupby = GroupBy(
            self.groupby_cols,
            other,
            fitting_schedule=self.fitting_schedule,
        )
        return self.pipeline_upstream + groupby


# @define
# class CrossValidateKFold(Transform):
#     """
#     Cross-validate it, bro.
#
#     Behavior at fit- and apply-times... think of analogy to Sequentially.
#
#     fit-time: df_fit ->
#     """
#
#     transform: Transform = field()
#     score_transform: Transform = field()
#     k: int | ffc.HP = field()
#     by = optional_columns_field()
