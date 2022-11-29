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
Provides a library of broadly useful Transforms on 2-D Pandas DataFrames.

Ordinarily, users should never need to import this module directly. Instead, they access
the classes and functions defined here through the public API exposed as
``frankenfit.*``.
"""
from __future__ import annotations

import inspect
import logging
import operator
from abc import abstractmethod
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from attrs import NOTHING, field
from pyarrow import dataset  # type: ignore

from .params import (
    HP,
    dict_field,
    fmt_str_field,
    columns_field,
    optional_columns_field,
    params,
)
from .core import (
    Bindings,
    ConstantTransform,
    FitTransform,
    P_co,
    StatelessTransform,
    Transform,
    callchain,
)
from .universal import (
    ForBindings,
    Identity,
    UniversalGrouper,
    UniversalPipeline,
    UniversalPipelineInterface,
)

_LOG = logging.getLogger(__name__)

T = TypeVar("T")


class DataFrameTransform(Transform[pd.DataFrame, pd.DataFrame]):
    def then(
        self, other: Optional[Transform | list[Transform]] = None
    ) -> "DataFramePipeline":
        result = super().then(other)
        return DataFramePipeline(transforms=result.transforms)

    # Stubs below are purely for convenience of autocompletion when the user
    # implements subclasses

    @abstractmethod
    def _fit(self, data_fit: pd.DataFrame) -> Any:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _apply(self, data_apply: pd.DataFrame, state: Any) -> pd.DataFrame:
        raise NotImplementedError  # pragma: no cover


class StatelessDataFrameTransform(
    StatelessTransform[pd.DataFrame, pd.DataFrame], DataFrameTransform
):
    def _fit(self, data_fit: pd.DataFrame) -> None:
        return None


class ConstantDataFrameTransform(
    ConstantTransform[pd.DataFrame, pd.DataFrame], DataFrameTransform
):
    pass


@params
class ReadDataFrame(ConstantDataFrameTransform):
    df: pd.DataFrame

    def _apply(self, data_apply: pd.DataFrame, _) -> pd.DataFrame:
        return self.df


@params
class ReadPandasCSV(ConstantDataFrameTransform):
    filepath: str = fmt_str_field()
    read_csv_args: Optional[dict] = None

    def _apply(self, data_apply, _: None) -> pd.DataFrame:
        return pd.read_csv(self.filepath, **(self.read_csv_args or {}))


@params
class WritePandasCSV(StatelessDataFrameTransform, Identity[pd.DataFrame]):
    path: str = fmt_str_field()
    index_label: str = fmt_str_field()
    to_csv_kwargs: Optional[dict] = None

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        data_apply.to_csv(
            self.path, index_label=self.index_label, **(self.to_csv_kwargs or {})
        )
        return data_apply

    # Because Identity derives from UniversalTransform, we have to say which
    # then() to use on instances of WritePandasCSV
    then = DataFrameTransform.then


@params
class ReadDataset(ConstantDataFrameTransform):
    paths: list[str] = columns_field()
    columns: Optional[list[str]] = optional_columns_field(default=None)
    format: Optional[str] = None
    filter: Optional[dataset.Expression] = None
    index_col: Optional[str | int] = None
    dataset_kwargs: Optional[dict] = None
    scanner_kwargs: Optional[dict] = None

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        ds = dataset.dataset(
            self.paths, format=self.format, **(self.dataset_kwargs or {})
        )
        columns = self.columns or None
        df_out = ds.to_table(
            columns=columns, filter=self.filter, **(self.scanner_kwargs or {})
        ).to_pandas()
        # can we tell arrow this?
        if self.index_col is not None:
            df_out = df_out.set_index(self.index_col)
        return df_out


@params
class Join(DataFrameTransform):
    left: Transform[pd.DataFrame, pd.DataFrame]
    right: Transform[pd.DataFrame, pd.DataFrame]
    how: Literal["left", "right", "outer", "inner"]

    on: Optional[str] = None
    left_on: Optional[str] = None
    right_on: Optional[str] = None
    suffixes: tuple[str, str] = ("_x", "_y")

    # TODO: more merge params like left_index etc.
    # TODO: (when on distributed compute) context extension

    def _fit(
        self, data_fit: pd.DataFrame, bindings: Optional[Bindings] = None
    ) -> tuple[FitTransform, FitTransform]:
        bindings = bindings or {}
        return (
            self.left.fit(data_fit, bindings=bindings),
            self.right.fit(data_fit, bindings=bindings),
        )

    def _apply(
        self, data_apply: pd.DataFrame, state: tuple[FitTransform, FitTransform]
    ) -> pd.DataFrame:
        fit_left, fit_right = state
        # TODO: parallelize
        df_left, df_right = fit_left.apply(data_apply), fit_right.apply(data_apply)
        return pd.merge(
            left=df_left,
            right=df_right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            suffixes=self.suffixes,
        )


@params
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

    cols: list[str] = columns_field()


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


DfLocIndex = Union[pd.Series, List, slice, np.ndarray]
DfLocPredicate = Callable[[pd.DataFrame], DfLocIndex]


# TODO: GroupByRows
@params
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

    cols: str | list[str] = columns_field()
    transform: Transform[pd.DataFrame, pd.DataFrame] = field()
    # TODO: what about hyperparams in the fitting schedule? that's a thing.
    fitting_schedule: Callable[[dict[str, Any]], DfLocIndex | DfLocPredicate] = field(
        default=fit_group_on_self
    )

    def _fit(
        self, data_fit: pd.DataFrame, bindings: Optional[Bindings] = None
    ) -> pd.DataFrame:
        def fit_on_group(df_group: pd.DataFrame):
            # select the fitting data for this group
            group_col_map = {c: df_group[c].iloc[0] for c in self.cols}
            df_group_fit: pd.DataFrame = data_fit.loc[
                # pandas-stubs seems to be broken here, see:
                # https://github.com/pandas-dev/pandas-stubs/issues/256
                self.fitting_schedule(group_col_map)  # type: ignore
            ]
            # fit the transform on the fitting data for this group
            # TODO: new per-group tags for the FitTransforms? How should find_by_tag()
            # work on FitGroupBy? (By overriding _children())
            return self.transform.fit(df_group_fit, bindings=bindings)

        return (
            data_fit.groupby(self.cols, as_index=False, sort=False)
            .apply(fit_on_group)
            .rename(columns={None: "__state__"})
        )

    def _apply(self, data_apply: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
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
            data_apply.merge(state, how="left", on=self.cols)
            .groupby(self.cols, as_index=False, sort=False, group_keys=False)
            .apply(apply_on_group)
        )


@params
class GroupByBindings(DataFrameTransform):
    bindings_sequence: Iterable[Bindings]
    transform: Transform[pd.DataFrame, pd.DataFrame]
    as_index: bool = True

    def _fit(
        self, data_fit: pd.DataFrame, bindings: Optional[Bindings] = None
    ) -> FitTransform:
        return ForBindings(self.bindings_sequence, self.transform).fit(
            data_fit, bindings=bindings or {}
        )

    def _apply(
        self,
        data_apply: pd.DataFrame,
        state: FitTransform,  # TODO: actual FitTransform type for ForBindings
    ) -> pd.DataFrame:
        results = state.apply(data_apply)
        binding_cols = set()
        dfs = []
        for x in results:
            dfs.append(x.result.assign(**x.bindings))
            binding_cols |= x.bindings.keys()
        df = pd.concat(dfs, axis=0)
        if self.as_index:
            df = df.set_index(list(binding_cols))
        return df


@params
class Filter(StatelessDataFrameTransform):
    filter_fun: (
        Callable[[pd.DataFrame], pd.Series[bool]]
        | Callable[[pd.DataFrame, Bindings], pd.Series[bool]]
    )

    def _apply(
        self, data_apply: pd.DataFrame, state: None, bindings: Optional[Bindings] = None
    ) -> pd.DataFrame:
        sig = inspect.signature(self.filter_fun).parameters
        if len(sig) == 1:
            filter_fun_monovalent = cast(
                Callable[[pd.DataFrame], "pd.Series[bool]"], self.filter_fun
            )
            return data_apply.loc[filter_fun_monovalent(data_apply)]
        elif len(sig) == 2:
            filter_fun_bivalent = cast(
                Callable[[pd.DataFrame, Bindings], "pd.Series[bool]"], self.filter_fun
            )
            return data_apply.loc[filter_fun_bivalent(data_apply, bindings or {})]
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(
                f"Expected callable with 1 or 2 parameters, found {len(sig)}"
            )


@params
class Copy(ColumnsTransform, StatelessDataFrameTransform):
    """
    A stateless Transform that copies values from one or more source columns into
    corresponding destination columns, either creating them or overwriting their
    contents.
    """

    dest_cols: list[str] = columns_field()

    def _check_cols(self):
        # TODO: maybe in general we should provide some way to check that
        # hyperparemters resolved to expected types
        if not isinstance(self.cols, list):
            raise TypeError("Parameter 'cols' resolved to non-list: {self.cols!r}")
        if not isinstance(self.dest_cols, list):
            raise TypeError(
                "Parameter 'dest_cols' resolved to non-list: {self.dest_cols!r}"
            )
        lc = len(self.cols)
        lv = len(self.dest_cols)
        if lc == 1 and lv > 0:
            return

        if lv != lc:
            raise ValueError(
                "When copying more than one source column, "
                f"cols (len {lc}) and dest_cols (len {lv}) must have the same "
                "length."
            )

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        # Now that hyperparams are bound, we can validate parameter shapes
        self._check_cols()
        if len(self.cols) == 1:
            src_col = self.cols[0]
            return data_apply.assign(
                **{dest_col: data_apply[src_col] for dest_col in self.dest_cols}
            )

        return data_apply.assign(
            **{
                dest_col: data_apply[src_col]
                for src_col, dest_col in zip(self.cols, self.dest_cols)
            }
        )


@params
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

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        return data_apply[self.cols]


class Drop(ColumnsTransform, StatelessDataFrameTransform):
    """
    Drop the given columns from the data.
    """

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        return data_apply.drop(columns=self.cols)


@params
class Rename(StatelessDataFrameTransform):
    """
    Rename columns.

    :param how: Either a function that, given a column name, returns what it should be
        renamed do, or a dict from old column names to corresponding new names.
    """

    how: Callable | dict[str, str]

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        return data_apply.rename(columns=self.how)


@params
class Pipe(ColumnsTransform, StatelessDataFrameTransform):
    apply_fun: Callable[[pd.DataFrame], pd.DataFrame]

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        result = self.apply_fun(data_apply[self.cols])
        return data_apply.assign(**{c: result[c] for c in self.cols})


# TODO Rank, MapQuantiles


@params
class Clip(ColumnsTransform, StatelessDataFrameTransform):
    upper: Optional[float] = None
    lower: Optional[float] = None

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        return data_apply.assign(
            **{
                col: data_apply[col].clip(upper=self.upper, lower=self.lower)
                for col in self.cols
            }
        )


@params
class Winsorize(ColumnsTransform):
    # assume symmetric, i.e. trim the upper and lower `limit` percent of observations
    limit: float

    def _fit(self, data_fit: pd.DataFrame) -> Mapping[str, pd.Series]:
        if not isinstance(self.limit, float):
            raise TypeError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )
        if self.limit < 0 or self.limit > 1:
            raise ValueError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )

        return {
            "lower": data_fit[self.cols].quantile(self.limit, interpolation="nearest"),
            "upper": data_fit[self.cols].quantile(
                1.0 - self.limit, interpolation="nearest"
            ),
        }

    def _apply(
        self, data_apply: pd.DataFrame, state: Mapping[str, pd.Series]
    ) -> pd.DataFrame:
        return data_apply.assign(
            **{
                col: data_apply[col].clip(
                    upper=state["upper"][col], lower=state["lower"][col]
                )
                for col in self.cols
            }
        )


@params
class ImputeConstant(ColumnsTransform, StatelessDataFrameTransform):
    value: Any

    def _apply(self, data_apply: pd.DataFrame, state) -> pd.DataFrame:
        return data_apply.assign(
            **{col: data_apply[col].fillna(self.value) for col in self.cols}
        )


def _weighted_means(df: pd.DataFrame, cols: list[str], w_col: str) -> pd.Series:
    df = df.loc[df[w_col].notnull()]
    w = df[w_col]
    wsums = df[cols].multiply(w, axis="index").sum()
    return pd.Series(
        [wsums[col] / w.loc[df[col].notnull()].sum() for col in wsums.index],
        index=wsums.index,
    )


@params
class DeMean(ColumnsTransform):
    """
    De-mean some columns.
    """

    w_col: Optional[str] = None

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        if self.w_col is not None:
            return _weighted_means(data_fit, self.cols, self.w_col)
        return data_fit[self.cols].mean()

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series):
        means = state
        return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})


@params
class ImputeMean(ColumnsTransform):
    w_col: Optional[str] = None

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        if self.w_col is not None:
            return _weighted_means(data_fit, self.cols, self.w_col)
        return data_fit[self.cols].mean()

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
        means = state
        return data_apply.assign(
            **{c: data_apply[c].fillna(means[c]) for c in self.cols}
        )


@params
class ZScore(ColumnsTransform):
    w_col: Optional[str] = None

    def _fit(self, data_fit: pd.DataFrame) -> dict[str, pd.Series]:
        if self.w_col is not None:
            means = _weighted_means(data_fit, self.cols, self.w_col)
        else:
            means = data_fit[self.cols].mean()
        return {"means": means, "stddevs": data_fit[self.cols].std()}

    def _apply(
        self, data_apply: pd.DataFrame, state: dict[str, pd.Series]
    ) -> pd.DataFrame:
        means, stddevs = state["means"], state["stddevs"]
        return data_apply.assign(
            **{c: (data_apply[c] - means[c]) / stddevs[c] for c in self.cols}
        )


@params
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

    sklearn_class: type  # TODO: protocol?
    x_cols: list[str] = columns_field()
    response_col: str = fmt_str_field()
    hat_col: str = fmt_str_field()
    class_params: dict[str, Any] = dict_field(factory=dict)
    w_col: Optional[str] = fmt_str_field(factory=str)

    def _fit(self, data_fit: pd.DataFrame) -> Any:
        model = self.sklearn_class(**self.class_params)
        X = data_fit[self.x_cols]
        y = data_fit[self.response_col]
        if self.w_col:
            w = data_fit[self.w_col]
            # TODO: raise exception if model.fit signature has no sample_weight arg
            model = model.fit(X, y, sample_weight=w)
        else:
            model = model.fit(X, y)

        return model

    def _apply(self, data_apply: pd.DataFrame, state: Any) -> pd.DataFrame:
        model = state
        return data_apply.assign(
            **{self.hat_col: model.predict(data_apply[self.x_cols])}
        )


@params
class Statsmodels(DataFrameTransform):
    """
    Wrap a statsmodels model.
    """

    sm_class: type  # TODO: protocol?
    x_cols: list[str] = columns_field()
    response_col: str = fmt_str_field()
    hat_col: str = fmt_str_field()
    class_params: dict[str, Any] = dict_field(factory=dict)

    def _fit(self, data_fit: pd.DataFrame) -> Any:
        X = data_fit[self.x_cols]
        y = data_fit[self.response_col]
        model = self.sm_class(y, X, **self.class_params)
        return model.fit()

    def _apply(self, data_apply: pd.DataFrame, state: Any) -> pd.DataFrame:
        model = state
        return data_apply.assign(
            **{self.hat_col: model.predict(data_apply[self.x_cols])}
        )


@params
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
    method: Literal["pearson", "kendall", "spearman"] = "pearson"
    min_obs: int = 2

    def _apply(self, data_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        cm = data_apply[self.left_cols + self.right_cols].corr(
            method=self.method, min_periods=self.min_obs
        )
        return cm.loc[self.left_cols, self.right_cols]


@params
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

    def _apply(
        self, data_apply: pd.DataFrame, state: None, bindings: Optional[Bindings] = None
    ) -> pd.DataFrame:
        kwargs = {}
        for k, v in self.assignments.items():
            kwargs[k] = v
            if callable(v):
                sig = inspect.signature(v).parameters
                if len(sig) == 2:
                    # expose self to bivalent lambdas as first arg
                    kwargs[k] = partial(v, self)
                elif len(sig) == 3:
                    # expose self and bindings to trivalent bindings
                    kwargs[k] = partial(v, self, bindings)
                elif len(sig) > 3:
                    raise TypeError(
                        f"Expected lambda with 1 or 2 parameters, found {len(sig)}"
                    )

        return data_apply.assign(**kwargs)


Cols = Union[str, HP, Sequence[Union[str, HP]]]


class DataFrameCallChain(Generic[P_co]):
    @callchain(ReadDataFrame)
    def read_data_frame(  # type: ignore [empty-body]
        self, df: pd.DataFrame | HP, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`ReadDataFrame` transform to this pipeline.
        """

    @callchain(ReadPandasCSV)
    def read_pandas_csv(  # type: ignore [empty-body]
        self,
        filepath: str | HP,
        read_csv_args: Optional[dict | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`ReadPandasCSV` transform to this pipeline.
        """

    @callchain(WritePandasCSV)
    def write_pandas_csv(  # type: ignore [empty-body]
        self,
        path: str | HP,
        index_label: str | HP,
        to_csv_kwargs: Optional[dict | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`WritePandasCSV` transform to this pipeline.
        """

    @callchain(ReadDataset)
    def read_dataset(  # type: ignore [empty-body]
        self,
        paths: Cols,
        columns: Optional[list[str]] = None,
        format: Optional[str] = None,
        filter: Optional[dataset.Expression] = None,
        index_col: Optional[str | int] = None,
        dataset_kwargs: Optional[dict] = None,
        scanner_kwargs: Optional[dict] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`ReadDataset` transform to this pipeline.
        """

    @callchain(Select)
    def select(  # type: ignore [empty-body]
        self, cols: Cols, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`Select` transform to this pipeline.
        """

    __getitem__ = select

    @callchain(Filter)
    def filter(  # type: ignore [empty-body]
        self,
        filter_fun: (
            Callable[[pd.DataFrame], pd.Series[bool]]
            | Callable[[pd.DataFrame, Bindings], pd.Series[bool]]
            | HP
        ),
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Filter` transform to this pipeline.
        """

    @callchain(Copy)
    def copy(  # type: ignore [empty-body]
        self, cols: Cols, dest_cols: Cols, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`Copy` transform to this pipeline.
        """

    @callchain(Drop)
    def drop(  # type: ignore [empty-body]
        self, cols: Cols, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`Drop` transform to this pipeline.
        """

    @callchain(Rename)
    def rename(  # type: ignore [empty-body]
        self, how: Callable | dict[str, str] | HP, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`Rename` transform to this pipeline.
        """

    @callchain(Assign)
    def assign(  # type: ignore [empty-body]
        self,
        *args,
        tag: Optional[str] = None,
        **kwargs: Any,
    ) -> P_co:
        """
        Append a :class:`Assign` transform to this pipeline.
        """

    @callchain(Pipe)
    def pipe(  # type: ignore [empty-body]
        self,
        cols: Cols,
        apply_fun: Callable | HP,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Pipe` transform to this pipeline.
        """

    @callchain(Clip)
    def clip(  # type: ignore [empty-body]
        self,
        cols: Cols,
        upper: Optional[float | HP] = None,
        lower: Optional[float | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Clip` transform to this pipeline.
        """

    @callchain(Winsorize)
    def winsorize(  # type: ignore [empty-body]
        self, cols: Cols, limit: float | HP, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`Winsorize` transform to this pipeline.
        """

    @callchain(ImputeConstant)
    def impute_constant(  # type: ignore [empty-body]
        self, cols: Cols, value: Any, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`ImputeConstant` transform to this pipeline.
        """

    @callchain(DeMean)
    def de_mean(  # type: ignore [empty-body]
        self, cols: Cols, w_col: Optional[str | HP] = None, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`DeMean` transform to this pipeline.
        """

    @callchain(ImputeMean)
    def impute_mean(  # type: ignore [empty-body]
        self, cols: Cols, w_col: Optional[str | HP] = None, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`ImputeMean` transform to this pipeline.
        """

    @callchain(ZScore)
    def z_score(  # type: ignore [empty-body]
        self, cols: Cols, w_col: Optional[str | HP] = None, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`ZScore` transform to this pipeline.
        """

    @callchain(SKLearn)
    def sk_learn(  # type: ignore [empty-body]
        self,
        sklearn_class: type | HP,
        x_cols: Cols,
        response_col: str | HP,
        hat_col: str | HP,
        class_params: Optional[dict[str, Any]] = None,
        w_col: Optional[str | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`SKLearn` transform to this pipeline.
        """

    @callchain(Statsmodels)
    def statsmodels(  # type: ignore [empty-body]
        self,
        sm_class: type | HP,
        x_cols: Cols,
        response_col: str | HP,
        hat_col: str | HP,
        class_params: Optional[dict[str, Any]] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Statsmodels` transform to this pipeline.
        """

    @callchain(Correlation)
    def correlation(  # type: ignore [empty-body]
        self,
        left_cols: Cols,
        right_cols: Cols,
        method: Literal["pearson", "kendall", "spearman"] | HP = "pearson",
        min_obs: int | HP = 2,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Correlation` transform to this pipeline.
        """


class DataFrameGrouper(Generic[P_co], UniversalGrouper[P_co], DataFrameCallChain[P_co]):
    ...


G_co = TypeVar("G_co", bound=DataFrameGrouper, covariant=True)
SelfDPI = TypeVar("SelfDPI", bound="DataFramePipelineInterface")


class DataFramePipelineInterface(
    Generic[G_co, P_co],
    DataFrameCallChain[P_co],
    UniversalPipelineInterface[pd.DataFrame, G_co, P_co],
):
    _Grouper = DataFrameGrouper

    def join(
        self: SelfDPI,
        right: Transform[pd.DataFrame, pd.DataFrame],
        how: Literal["left", "right", "outer", "inner"] | HP,
        on: Optional[str | HP] = None,
        left_on: Optional[str | HP] = None,
        right_on: Optional[str | HP] = None,
        suffixes=("_x", "_y"),
    ) -> SelfDPI:
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

    def group_by_cols(
        self,
        cols: Cols,
        fitting_schedule: Optional[
            Callable[[dict[str, Any]], DfLocIndex | DfLocPredicate] | HP
        ] = None,
    ) -> G_co:
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
        grouper = type(self)._Grouper(
            self,
            GroupByCols,
            "transform",
            cols=cols,
            fitting_schedule=(fitting_schedule or fit_group_on_self),
        )
        return cast(G_co, grouper)

    def group_by_bindings(
        self, bindings_sequence: Iterable[Bindings], as_index: bool | HP = False
    ) -> G_co:
        grouper = type(self)._Grouper(
            self,
            GroupByBindings,
            "transform",
            bindings_sequence=bindings_sequence,
            as_index=as_index,
        )
        return cast(G_co, grouper)


class DataFramePipeline(
    DataFramePipelineInterface[
        DataFrameGrouper["DataFramePipeline"], "DataFramePipeline"
    ],
    UniversalPipeline,
):
    ...
