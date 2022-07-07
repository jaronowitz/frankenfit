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
from functools import partial
import inspect

import logging
from attrs import define, field
import pandas as pd

from typing import Callable, Optional

from . import transforms as fft
from . import core as ffc
from .core import (
    Transform,
)

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

    Class docs for :class:`{transform_class.__qualname__}`:
    {transform_class.__doc__ or ''}
    """
    return method_impl


def _convert_pipeline_transforms(value):
    if isinstance(value, Pipeline):
        return list(value.transforms)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Pipeline):
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
        for t in value:
            if not isinstance(t, Transform):
                raise TypeError(
                    "Pipeline sequence must comprise Transform instances; found "
                    f"non-Transform {t} (type {type(t)})"
                )

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

    # TODO: fit_and_apply()

    def then(self, other: Transform | list[Transform]) -> Pipeline:
        """
        Return the result of appending the given :class:`Transform` instance(s) to this
        :class:`Pipeline`. The addition operator on Pipeline objects is an alias for
        this method, meaning that the following are equivalent pairs::

            pipeline + ff.DeMean(...) == pipeline.then(ff.DeMean(...))
            pipeline + other_pipeline == pipeline.then(other_pipeline)
            pipeline + [ff.Winsorize(...), ff.DeMean(...)] == pipeline.then(
                [ff.Winsorize, ff.DeMean(...)]
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

        :param other: The Transform instance to append, or a list of Transforms, which
            will be appended in the order in which in they appear in the list.
        :type other: :class:`Transform` | ``list[Transform]``
        :raises ``TypeError``: If ``other`` is not a ``Transform`` or list of
            ``Transform``\\ s.
        :rtype: :class:`Pipeline`
        """
        if isinstance(other, Pipeline):
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

    # Pipeline()...join(left_pipeline, right_pipeline)...
    # join = _pipeline_method_wrapping_transform("join", Join)

    def join(
        self, right, how, on=None, left_on=None, right_on=None, suffixes=("_x", "_y")
    ):
        # joining self as left with arg as right
        join = Join(
            self,
            right,
            how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
        )
        return self + join

    def groupby(self, cols) -> PipelineGrouper:
        return PipelineGrouper(cols, self)


@define
class GroupBy(ffc.Transform):
    """
    Group the fitting and application of a :class:`Transform` by the distinct values of
    some column or combination of columns.
    """

    # TODO: what about grouping by index?
    cols: str | ffc.HP | list[str | ffc.HP] = ffc.columns_field()
    transform: ffc.HP | ffc.Transform = field()

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()

        def fit_on_group(df_group: pd.DataFrame):
            dsc = self.dataset_collection | {"__pass__": df_group}
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
            # values of __state__ auto to be identical within the group
            group_state: ffc.FitTransform = df_group["__state__"].iloc[0]
            return group_state.apply(dsc)

        return (
            df_apply.merge(state, how="left", on=self.cols)
            .groupby(self.cols, as_index=False, sort=False)
            .apply(apply_on_group)
        )


class PipelineGrouper(CallChainingMixin):
    """
    An intermediate "grouper" object returned by :meth:`Pipeline.groupby()` (analogous
    to pandas ``DataFrameGroupBy`` objects), which is not a :class:`Pipeline`, but has
    the same call-chain methods as a Pipeline, and consumes the next call to finally
    create the :class:`GroupBy` Transform and return the result of appending that to the
    matrix Pipeline. It enables this style of ``groupby()`` call-chaining syntax::

        (
            ff.Pipeline()
            # ...
            .groupby("cut")  # -> PipelineGrouper
                .z_score(cols)  # -> Pipeline
        )
    """

    def __init__(self, groupby_cols, pipeline_upstream):
        self.groupby_cols = groupby_cols
        self.pipeline_upstream = pipeline_upstream

    def __repr__(self):
        return "PipelineGrouper(%r, %r)" % (self.groupby_cols, self.pipeline_upstream)

    def then(self, other: Transform | list[Transform]) -> Pipeline:
        if isinstance(other, list):
            other = Pipeline(other)
        groupby = GroupBy(self.groupby_cols, other)
        return self.pipeline_upstream + groupby
