"""
The graph module provides Transform subclasses (notably Pipeline, Join, and friends)
that take other Transforms as parameters, allowing the creation of sequences and
tree-like structures of Transforms that process each other's output (i.e., composition).
"""
from __future__ import annotations
from functools import partial
import inspect

import logging
from attrs import define, field
import pandas as pd

from typing import Callable, Optional

from . import transforms as fft
from . import core as ffc

_LOG = logging.getLogger(__name__)

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
class IfHyperparamIsTrue(fft.Transform):
    name: str
    then: fft.Transform
    otherwise: Optional[fft.Transform] = None

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()
        if bindings.get(self.name):
            return self.then.fit(df_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity


@define
class IfHyperparamLambda(fft.Transform):
    fun: Callable  # dict[str, object] -> bool
    then: fft.Transform
    otherwise: Optional[fft.Transform] = None

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


@define
class IfTrainingDataHasProperty(fft.Transform):
    fun: Callable  # dict[str, object] -> bool
    then: fft.Transform
    otherwise: Optional[fft.Transform] = None

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
class Join(fft.Transform):
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
    Return the result of appending a new {transform_class.__name__} transform
    constructed with the given parameters to this Pipeline. This method's arguments are
    passed directly to {transform_class.__name__}.__init__().

    Class docs for {transform_class.__qualname__}
    ---------------{'-' * len(transform_class.__qualname__)}
    {transform_class.__doc__ or ''}
    Constructor docs for {transform_class.__qualname__}.__init__
    ---------------------{'-' * len(transform_class.__qualname__ + '.__init__')}
    {transform_class.__init__.__doc__}
    """
    return method_impl


def _convert_pipeline_transforms(value):
    if isinstance(value, Pipeline):
        return list(value.transforms)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Pipeline):
        return list(value[0].transforms)
    if isinstance(value, fft.Transform):
        return [value]
    return list(value)


_pipeline_method_wrapping_transform = partial(method_wrapping_transform, "Pipeline")


@define
class Pipeline(fft.Transform):
    dataset_name: str = "__pass__"

    transforms: list[fft.Transform] = field(
        factory=list, converter=_convert_pipeline_transforms
    )

    @transforms.validator
    def _check_transforms(self, attribute, value):
        for t in value:
            if not isinstance(t, fft.Transform):
                raise TypeError(
                    "Pipeline sequence must comprise Transform instances; found "
                    f"non-Transform {t} (type {type(t)})"
                )

    # TODO: should we override hyperparams() to return some kind of collection across
    # transforms?

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

    def __add__(self, other):
        return self.then(other)

    ####################
    # call-chaining API:

    def then(self, other):
        if isinstance(other, Pipeline):
            transforms = self.transforms + other.transforms
        elif isinstance(other, fft.Transform):
            transforms = self.transforms + [other]
        elif isinstance(other, list):
            transforms = self.transforms + other
        else:
            raise TypeError(
                f"I don't know how to extend a Pipeline with {other}, which is of "
                f"type {type(other)}, bases = {type(other).__bases__}. "
            )
        return Pipeline(dataset_name=self.dataset_name, transforms=transforms)

    copy_columns = _pipeline_method_wrapping_transform("copy_columns", fft.CopyColumns)
    keep_columns = _pipeline_method_wrapping_transform("keep_columns", fft.KeepColumns)
    rename_columns = _pipeline_method_wrapping_transform(
        "rename_columns", fft.RenameColumns
    )
    drop_columns = _pipeline_method_wrapping_transform("drop_columns", fft.DropColumns)
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
    statsmodels = _pipeline_method_wrapping_transform("statsmodels", fft.SKLearn)

    # Pipeline()...join(left_pipeline, right_pipeline)...
    # join = _pipeline_method_wrapping_transform("join", Join)

    def join(
        self, *others, how, on=None, left_on=None, right_on=None, suffixes=("_x", "_y")
    ):
        if len(others) == 2:
            # joining two other pipelines (unrelated to self) and appending them to self
            # (which is possibly weird because it may efface all of the data up to this
            # point in the pipeline?)
            join = Join(
                *others,
                how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                suffixes=suffixes,
            )
        elif len(others) == 1:
            # joining self as left with other as right
            join = Join(
                self,
                *others,
                how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                suffixes=suffixes,
            )
        else:
            raise TypeError("Wrong number of arguments.")
        return self + join
