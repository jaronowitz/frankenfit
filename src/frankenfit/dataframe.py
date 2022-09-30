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
Provides a library of basic Transforms on Pandas DataFrames.

Ordinarily, users should never need to import this module directly. Instead, they access
the classes and functions defined here through the public API exposed as
``frankenfit.*``.
"""
from __future__ import annotations
from abc import abstractmethod
from functools import partial, reduce
import inspect
import logging
import operator
from typing import Callable, Iterable, Optional, TypeVar

from attrs import field, NOTHING
import numpy as np
import pandas as pd
from pyarrow import dataset

from .core import (
    transform,
    Transform,
    FitTransform,
    StatelessTransform,
    HP,
    UnresolvedHyperparameterError,
    dict_field,
    fmt_str_field,
    ConstantTransform,
)

from .universal import Pipeline, Identity, ForBindings

_LOG = logging.getLogger(__name__)


@transform
class DataFrameTransform(Transform):
    def then(
        self: DataFrameTransform, other: Optional[Transform | list[Transform]] = None
    ) -> "DataFramePipeline":
        result = super().then(other)
        return DataFramePipeline(transforms=result.transforms)

    @abstractmethod
    def _fit(self, data_fit: Optional[pd.DataFrame] = None) -> object:
        raise NotImplementedError


@transform
class StatelessDataFrameTransform(StatelessTransform, DataFrameTransform):
    def _fit(self, data_fit: Optional[pd.DataFrame] = None) -> object:
        return None


@transform
class ConstantDataFrameTransform(ConstantTransform, DataFrameTransform):
    def _fit(self, data_fit: Optional[pd.DataFrame] = None) -> object:
        return None


@transform
class ReadDataFrame(ConstantDataFrameTransform):
    df: pd.DataFrame

    def _apply(self, data_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return self.df


@transform
class ReadPandasCSV(ConstantDataFrameTransform):
    filepath: str | HP = fmt_str_field()
    read_csv_args: Optional[dict] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return pd.read_csv(self.filepath, **(self.read_csv_args or {}))


@transform
class WritePandasCSV(Identity, DataFrameTransform):
    path: str | HP = fmt_str_field()
    index_label: str | HP = fmt_str_field()
    to_csv_kwargs: Optional[dict] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None):
        df_apply.to_csv(
            self.path, index_label=self.index_label, **(self.to_csv_kwargs or {})
        )
        return df_apply


@transform
class HPCols(HP):
    """_summary_

    :param HP: _description_
    :type HP: _type_
    :return: _description_
    :rtype: _type_
    """

    cols: list[str | HP]
    name: str = None

    C = TypeVar("C", bound="HPCols")
    X = TypeVar("X", bound=str | HP | None | Iterable[str | HP])

    @classmethod
    def maybe_from_value(cls: C, x: X) -> C | X:
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


@transform
class ReadDataset(ConstantDataFrameTransform):
    paths: list[str] = columns_field()
    format: Optional[str] = None
    columns: list[str] = field(default=None, converter=HPCols.maybe_from_value)
    filter: Optional[HP | dataset.Expression] = None
    index_col: Optional[str | int] = None
    dataset_kwargs: Optional[dict] = None
    scanner_kwargs: Optional[dict] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        ds = dataset.dataset(
            self.paths, format=self.format, **(self.dataset_kwargs or {})
        )
        df_out = ds.to_table(
            columns=self.columns, filter=self.filter, **(self.scanner_kwargs or {})
        ).to_pandas()
        # can we tell arrow this?
        if self.index_col is not None:
            df_out = df_out.set_index(self.index_col)
        return df_out


@transform
class Join(DataFrameTransform):
    left: DataFrameTransform
    right: DataFrameTransform
    how: str

    on: Optional[str] = None
    left_on: Optional[str] = None
    right_on: Optional[str] = None
    suffixes: tuple[str] = ("_x", "_y")

    # TODO: more merge params like left_index etc.
    # TODO: (when on distributed compute) context extension

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()
        return (
            self.left.fit(df_fit, bindings=bindings),
            self.right.fit(df_fit, bindings=bindings),
        )

    def _apply(
        self, df_apply: pd.DataFrame, state: tuple[FitTransform]
    ) -> pd.DataFrame:
        fit_left, fit_right = state
        # TODO: parallelize
        df_left, df_right = fit_left.apply(df_apply), fit_right.apply(df_apply)
        return pd.merge(
            left=df_left,
            right=df_right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            suffixes=self.suffixes,
        )


@transform
class ColumnsTransform(DataFrameTransform):
    """
    Abstract base clase of all Transforms that require a list of columns as a parameter
    (cols).  Subclasses acquire a mandatory `cols` argument to their constructors, which
    can be supplied as a list of any combination of:

    - string column names,
    - hyperparameters exepcted to resolve to column names,
    - format strings that will be evaluated as hyperparameters formatted on the bindings
      dict.

    A scalar value for cols (e.g. a single string) will be converted automatically to a
    list of one element at construction time.

    Subclasses may define additional parameters with the same behavior as cols by using
    columns_field().
    """

    cols: list[str | HP] = columns_field()


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


class UnfitGroupError(ValueError):
    """Exception raised when a group-by-like transform is applied to data
    containing groups on which it was not fit."""


# TODO: GroupByRows
@transform
class GroupByCols(DataFrameTransform):
    """
    Group the fitting and application of a :class:`DataFrameTransform` by the
    distinct values of some column or combination of columns.

    :param cols: The column(s) by which to group. ``transform`` will be fit and applied
        separately on each subset of data with a distinct combination of values in
        ``cols``.
    :type cols: str | HP | list[str | HP]

    :param transform: The :class:`DataFrameTransform` to group.
    :type transform: HP | DataFrameTransform

    :param fitting_schedule: How to determine the fitting data of each group. The
        default schedule is :meth:`fit_group_on_self`. Use this to implement workflows
        like cross-validation and sequential fitting.
    :type fitting_schedule: Callable[[dict[str, object]], np.array[bool]]

    .. SEEALSO::
        :meth:`DataFramePipeline.group_by_cols()`

    """

    cols: str | HP | list[str | HP] = columns_field()
    transform: HP | DataFrameTransform = field()  # type: ignore
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
            # TODO: new per-group tags for the FitTransforms? How should find_by_tag()
            # work on FitGroupBy? (By overriding _children())
            return self.transform.fit(df_group_fit, bindings=bindings)

        return (
            df_fit.groupby(self.cols, as_index=False, sort=False)
            .apply(fit_on_group)
            .rename(columns={None: "__state__"})
        )

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        def apply_on_group(df_group: pd.DataFrame):
            df_group_apply = df_group.drop(["__state__"], axis=1)
            # values of __state__ ought to be identical within the group
            group_state: FitTransform = df_group["__state__"].iloc[0]
            if not isinstance(group_state, FitTransform):
                # if this group was not seen at fit-time
                raise UnfitGroupError(
                    f"GroupBy: tried to apply to a group not seen at fit-time:\n"
                    f"{df_group_apply[self.cols].iloc[0]}"
                )
                # returning untransformed group data is undesirable because
                # corruption will silently propagate through a pipeline
                # return df_group_apply
            return group_state.apply(df_group_apply)

        return (
            df_apply.merge(state, how="left", on=self.cols)
            .groupby(self.cols, as_index=False, sort=False, group_keys=False)
            .apply(apply_on_group)
        )


@transform
class GroupByBindings(DataFrameTransform):
    bindings_sequence: iter[dict[str, object]]
    transform: DataFrameTransform
    as_index: bool = True

    def _fit(
        self, data_fit: Optional[pd.DataFrame] = None
    ) -> ForBindings.FitForBindings:
        return ForBindings(self.bindings_sequence, self.transform).fit(data_fit)

    def _apply(
        self, df_apply: pd.DataFrame, state: ForBindings.FitForBindings
    ) -> pd.DataFrame:
        results = state.apply(df_apply)
        binding_cols = set()
        dfs = []
        for x in results:
            dfs.append(x.result.assign(**x.bindings))
            binding_cols |= x.bindings.keys()
        df = pd.concat(dfs, axis=0)
        if self.as_index:
            df = df.set_index(list(binding_cols))
        return df


@transform
class Filter(StatelessDataFrameTransform):
    filter_fun: Callable[[pd.DataFrame], pd.Series[bool]]

    def _apply(
        self, data_apply: Optional[object], state: Optional[object] = None
    ) -> object:
        sig = inspect.signature(self.filter_fun).parameters
        if len(sig) == 1:
            return data_apply.loc[self.filter_fun(data_apply)]
        elif len(sig) == 2:
            return data_apply.loc[self.filter_fun(data_apply, self.bindings())]
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(
                f"Expected callable with 1 or 2 parameters, found {len(sig)}"
            )


@transform
class WeightedTransform(DataFrameTransform):
    """
    Abstract base class of Transforms that accept an optional weight column as a
    parameter (w_col).
    """

    w_col: Optional[str] = None


@transform
class Copy(ColumnsTransform, StatelessDataFrameTransform):
    """
    A stateless Transform that copies values from one or more source columns into
    corresponding destination columns, either creating them or overwriting their
    contents.
    """

    dest_cols: list[str | HP] = columns_field()

    # FIXME: we actually may not be able to validate this invariant until after
    # hyperparams are bound
    @dest_cols.validator
    def _check_dest_cols(self, attribute, value):
        lc = len(self.cols)
        lv = len(value)
        if lc == 1 and lv > 0:
            return

        if lv != lc:
            raise ValueError(
                "When copying more than one source column, "
                f"cols (len {lc}) and dest_cols (len {lv}) must have the same "
                "length."
            )

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if len(self.cols) == 1:
            src_col = self.cols[0]
            return df_apply.assign(
                **{dest_col: df_apply[src_col] for dest_col in self.dest_cols}
            )

        return df_apply.assign(
            **{
                dest_col: df_apply[src_col]
                for src_col, dest_col in zip(self.cols, self.dest_cols)
            }
        )


@transform
class Select(ColumnsTransform, StatelessDataFrameTransform):
    """
    Select the given columns from the data.

    .. TIP::
        As syntactic sugar, :class:`Pipeline` overrides the index operator (via a custom
        ``__getitem__`` implementatino) as a synonym for appending a ``Select``
        transform to the pipeline. For example, the following three pipelines are
        equivalent::

            ff.Pipeline([..., Select(["col1", "col2"]), ...])

            (
                ff.Pipeline()
                ...
                .select(["col1", "col2"]
                ...
            )

            (
                ff.Pipeline()
                ...
                ["col1", "col2"]
                ...
            )
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply[self.cols]


class Drop(ColumnsTransform, StatelessDataFrameTransform):
    """
    Drop the given columns from the data.
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.drop(columns=self.cols)


@transform
class Rename(StatelessDataFrameTransform):
    """
    Rename columns.

    :param how: Either a function that, given a column name, returns what it should be
        renamed do, or a dict from old column names to corresponding new names.
    """

    how: Callable | dict[str, str]

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.rename(columns=self.how)


@transform
class Pipe(ColumnsTransform, StatelessDataFrameTransform):
    apply_fun: Callable[[pd.DataFrame], pd.DataFrame]  # type: ignore

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        result = self.apply_fun(df_apply[self.cols])
        return df_apply.assign(**{c: result[c] for c in self.cols})


# TODO Rank, MapQuantiles


@transform
class Clip(ColumnsTransform, StatelessDataFrameTransform):
    upper: Optional[float] = None
    lower: Optional[float] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{
                col: df_apply[col].clip(upper=self.upper, lower=self.lower)
                for col in self.cols
            }
        )


@transform
class Winsorize(ColumnsTransform):
    # assume symmetric, i.e. trim the upper and lower `limit` percent of observations
    limit: float  # type: ignore

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if not isinstance(self.limit, float):
            raise TypeError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )
        if self.limit < 0 or self.limit > 1:
            raise ValueError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )

        return {
            "lower": df_fit[self.cols].quantile(self.limit, interpolation="nearest"),
            "upper": df_fit[self.cols].quantile(
                1.0 - self.limit, interpolation="nearest"
            ),
        }

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{
                col: df_apply[col].clip(
                    upper=state["upper"][col], lower=state["lower"][col]
                )
                for col in self.cols
            }
        )


@transform
class ImputeConstant(ColumnsTransform, StatelessDataFrameTransform):
    value: object  # type: ignore

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{col: df_apply[col].fillna(self.value) for col in self.cols}
        )


def _weighted_means(df, cols, w_col):
    return df[cols].multiply(df[w_col], axis="index").sum() / df[w_col].sum()


@transform
class DeMean(WeightedTransform, ColumnsTransform):
    """
    De-mean some columns.
    """

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            return _weighted_means(df_fit, self.cols, self.w_col)
        return df_fit[self.cols].mean()

    def _apply(self, df_apply: pd.DataFrame, state: pd.DataFrame):
        means = state
        return df_apply.assign(**{c: df_apply[c] - means[c] for c in self.cols})


@transform
class ImputeMean(WeightedTransform, ColumnsTransform):
    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            return _weighted_means(df_fit, self.cols, self.w_col)
        return df_fit[self.cols].mean()

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        means = state
        return df_apply.assign(**{c: df_apply[c].fillna(means[c]) for c in self.cols})


@transform
class ZScore(WeightedTransform, ColumnsTransform):
    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            means = _weighted_means(df_fit, self.cols, self.w_col)
        else:
            means = df_fit[self.cols].mean()
        return {"means": means, "stddevs": df_fit[self.cols].std()}

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        means, stddevs = state["means"], state["stddevs"]
        return df_apply.assign(
            **{c: (df_apply[c] - means[c]) / stddevs[c] for c in self.cols}
        )


@transform
class SKLearn(DataFrameTransform):
    """
    Wrap a scikit-learn ("sklearn") model. At fit-time, the given sklearn model class
    is instantiated (with arguments from ``class_params``) and trained on the fitting
    data by calling its ``fit()`` method. At apply-time, the now-fit sklearn model
    object is used to generated predictions by calling its `predict()` method, which are
    assigned to the apply-time data as a new column, ``hat_col``.

    :param sklearn_class: The sklearn class to wrap.
    :param x_cols: The predictor columns. These are selected from the fit/apply-data to
        create the ``X`` argument to the sklearn model's ``fit()`` and ``predict()``
        methods.
    :param response_col: The response column. At fit-time, this is selected from the
        fitting data to create the ``y`` argument to the sklearn model's ``fit()``
        method.
    :param hat_col: The name of the new column to create at apply-time containing
        predictions from the sklearn model.
    :param class_params: Parameters to pass as kwargs to the ``sklearn_class``
        constructor when instantiating it.
    :param w_col: The sample weight column. If specified, this is selected at fit-time
        from the fitting data to create the ``sample_weight`` keyword argument to the
        sklearn model's ``fit()`` method.

        .. WARNING:: Not every sklearn model accepts a ``sample_weight`` keyword
            argument to its ``fit()`` method. Consult the documentation of whichever
            sklearn model you are using.
    """

    sklearn_class: type | HP
    x_cols: list[str] = columns_field()
    response_col: str = fmt_str_field()
    hat_col: str = fmt_str_field()
    class_params: dict[str, object] = dict_field(factory=dict)
    w_col: Optional[str] = fmt_str_field(factory=str)

    def _fit(self, df_fit: pd.DataFrame) -> object:
        model = self.sklearn_class(**self.class_params)
        X = df_fit[self.x_cols]
        y = df_fit[self.response_col]
        if self.w_col:
            w = df_fit[self.w_col]
            # TODO: raise exception if model.fit signature has no sample_weight arg
            model = model.fit(X, y, sample_weight=w)
        else:
            model = model.fit(X, y)

        return model

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        model = state
        return df_apply.assign(**{self.hat_col: model.predict(df_apply[self.x_cols])})


@transform
class Statsmodels(DataFrameTransform):
    """
    Wrap a statsmodels model.
    """

    sm_class: type | HP
    x_cols: list[str] = columns_field()
    response_col: str = fmt_str_field()
    hat_col: str = fmt_str_field()
    class_params: dict[str, object] = dict_field(factory=dict)

    def _fit(self, df_fit: pd.DataFrame) -> object:
        X = df_fit[self.x_cols]
        y = df_fit[self.response_col]
        model = self.sm_class(y, X, **self.class_params)
        return model.fit()

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        model = state
        return df_apply.assign(**{self.hat_col: model.predict(df_apply[self.x_cols])})


@transform
class Correlation(StatelessDataFrameTransform):
    """
    Compute the correlation between each pair of columns in the cross-product of
    ``left_cols`` and ``right_cols``.

    :param left_cols: List of "left" correlands. Result will have one row per element of
        ``left_cols``.
    :param right_cols: List of "right" correlands. Result will have one column per
        element of ``right_cols``.
    :param method: One of ``"pearson"``, ``"spearman"``, or ``"kendall"``, specifying
        which type of correlation coefficient to compute.
    :param min_obs: The minimum number of non-missing values for each pair of columns.
        If a pair has fewer than this many non-missing observations, then the
        correlation for that pair will be missing in the result.

    Example::

        from pydataset import data
        df = data("diamonds")
        ff.Correlation(["price"], ["carat"]).apply(df)
        # -->           carat
        # --> price  0.921591

        ff.Correlation(["table", "depth"], ["x", "y", "z"]).apply(df)
        # -->               x         y         z
        # --> table  0.195344  0.183760  0.150929
        # --> depth -0.025289 -0.029341  0.094924

    .. SEEALSO::
        The parameters of :meth:`pandas.DataFrame.corr()`.
    """

    left_cols: list[str] = columns_field()
    right_cols: list[str] = columns_field()
    method: Optional[str | HP] = "pearson"
    min_obs: Optional[int] = 1

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        cm = df_apply[self.left_cols + self.right_cols].corr(
            method=self.method, min_periods=self.min_obs
        )
        return cm.loc[self.left_cols, self.right_cols]


@transform
class Assign(StatelessDataFrameTransform):
    # TODO: keys as fmt str hyperparams
    assignments: dict[str, Callable] = dict_field()

    # Assign([assignment_dict][, tag=][, kwarg1=][, kwarg2][...])
    # ... with only one of assigment_dict or kwargs
    def __init__(self, *args, tag=NOTHING, **kwargs):
        if len(args) > 0:
            if len(kwargs) > 0:
                raise ValueError(
                    f"Expected only one of args or kwargs to be non-empty, "
                    f"but both are: args={args!r}, kwargs={kwargs!r}"
                )
            if len(args) > 1:
                raise ValueError(
                    "Expected only a single dict-typed positional argument"
                )
            assignments = args[0]
        else:
            assignments = kwargs
        self.__attrs_init__(tag=tag, assignments=assignments)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        kwargs = {}
        for k, v in self.assignments.items():
            kwargs[k] = v
            if callable(v):
                sig = inspect.signature(v).parameters
                if len(sig) == 2:
                    # expose self to bivalent lambdas as first arg
                    kwargs[k] = partial(v, self)
                elif len(sig) > 2:
                    raise TypeError(
                        f"Expected lambda with 1 or 2 parameters, found {len(sig)}"
                    )

        return df_apply.assign(**kwargs)


DP = TypeVar("DP", bound="DataFramePipeline")


class DataFramePipeline(
    Pipeline.with_methods(
        "DataFramePipeline",
        read_data_frame=ReadDataFrame,
        read_pandas_csv=ReadPandasCSV,
        write_pandas_csv=WritePandasCSV,
        read_dataset=ReadDataset,
        select=Select,
        __getitem__=Select,
        filter=Filter,
        copy=Copy,
        drop=Drop,
        rename=Rename,
        assign=Assign,
        pipe=Pipe,
        clip=Clip,
        winsorize=Winsorize,
        impute_constant=ImputeConstant,
        de_mean=DeMean,
        impute_mean=ImputeMean,
        z_score=ZScore,
        sk_learn=SKLearn,
        statsmodels=Statsmodels,
        correlation=Correlation,
    )
):
    def join(
        self: DP,
        right: DataFrameTransform,
        how: str,
        on=None,
        left_on=None,
        right_on=None,
        suffixes=("_x", "_y"),
    ) -> DP:
        """
        Return a new :class:`DataFramePipeline` (of the same subclass as
        ``self``) containing a new :class:`Join` transform with this
        pipeline as the ``Join``'s ``left`` argument.

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
        return type(self)(transforms=join)

    def group_by_cols(self: DP, cols, fitting_schedule=None) -> DP.Grouper:
        """
        Return a :class:`Grouper` object, which will consume the next Transform
        in the call-chain by wrapping it in a :class:`GroupBy` transform and returning
        the result of appending that ``GroupBy`` to this pipeline. It enables
        Pandas-style call-chaining with ``GroupBy``.

        For example, grouping a single Transform::

            (
                ff.DataFramePipeline()
                # ...
                .group_by("cut")  # -> PipelineGrouper
                    .z_score(cols)  # -> Pipeline
            )

        Grouping a sequence of Transforms::

            (
                ff.DataFramePipeline()
                # ...
                .group_by("cut")
                    .then(
                        ff.DataFramePipeline()
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
            call-chain will be fit and applied separately on each subset of
            data with a distinct combination of values in ``cols``.
        :type cols: str | HP | list[str | HP]

        :param fitting_schedule: How to determine the fitting data of each group. The
            default schedule is :meth:`fit_group_on_self`. Use this to implement
            workflows like cross-validation and sequential fitting.
        :type fitting_schedule: Callable[dict[str, object], np.array[bool]]

        :rtype: :class:`DataFramePipeline.Grouper`
        """
        return type(self).Grouper(
            self,
            GroupByCols,
            "transform",
            cols=cols,
            fitting_schedule=(fitting_schedule or fit_group_on_self),
        )

    def group_by_bindings(
        self: DP, bindings_sequence: iter[dict[str, object]], as_index: bool = False
    ) -> DP.Grouper:
        return type(self).Grouper(
            self,
            GroupByBindings,
            "transform",
            bindings_sequence=bindings_sequence,
            as_index=as_index,
        )
