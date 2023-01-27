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
Provides a library of broadly useful Transforms on 2-D Pandas DataFrames.

Ordinarily, users should never need to import this module directly. Instead, they access
the classes and functions defined here through the public API exposed as
``frankenfit.*``.
"""
from __future__ import annotations

import inspect
import logging
import operator
from functools import partial, reduce
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import pyarrow  # type: ignore
from attrs import NOTHING, field
from pyarrow import dataset

from .core import (
    Bindings,
    ConstantTransform,
    FitTransform,
    Future,
    P_co,
    PipelineMember,
    R,
    R_co,
    StatelessTransform,
    Transform,
    callchain,
)
from .params import (
    ALL_COLS,
    HP,
    columns_field,
    dict_field,
    fmt_str_field,
    optional_columns_field,
    params,
)
from .universal import (
    FitUniversalTransform,
    ForBindings,
    Identity,
    StatelessLambda,
    UniversalGrouper,
    UniversalPipeline,
    UniversalPipelineInterface,
    UserLambdaHyperparams,
)

_LOG = logging.getLogger(__name__)

T = TypeVar("T")


class FitDataFrameTransform(Generic[R_co], FitUniversalTransform[R_co, pd.DataFrame]):
    def then(
        self,
        other: PipelineMember | Sequence[PipelineMember] | None = None,
    ) -> "DataFramePipeline":
        result = super().then(other)
        return DataFramePipeline(transforms=result.transforms)


class DataFrameTransform(Transform[pd.DataFrame]):
    fit_transform_class: ClassVar[Type[FitTransform]] = FitDataFrameTransform

    def then(
        self,
        other: PipelineMember | Sequence[PipelineMember] | None = None,
    ) -> "DataFramePipeline":
        result = super().then(other)
        return DataFramePipeline(transforms=result.transforms)

    # Stubs below are purely for type specialization, convenience of autocompletion when
    # the user implements subclasses (type annotations)

    def fit(
        self: R,
        data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        bindings: Optional[Bindings] = None,
        /,
        **kwargs,
    ) -> FitDataFrameTransform[R]:
        return cast(FitDataFrameTransform[R], super().fit(data_fit, bindings, **kwargs))

    def _fit(self, data_fit: pd.DataFrame) -> Any:
        raise NotImplementedError  # pragma: no cover

    def _submit_fit(
        self,
        data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        bindings: Optional[Bindings] = None,
    ) -> Any:
        return super()._submit_fit(data_fit, bindings)

    def _apply(self, data_apply: pd.DataFrame, state: Any) -> pd.DataFrame:
        raise NotImplementedError  # pragma: no cover

    def _submit_apply(
        self,
        data_apply: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        state: Any = None,
    ) -> Future[pd.DataFrame] | None:
        return super()._submit_apply(data_apply, state)


class StatelessDataFrameTransform(StatelessTransform[pd.DataFrame], DataFrameTransform):
    def _fit(self, data_fit: pd.DataFrame) -> None:
        return None

    def _submit_fit(
        self,
        data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        bindings: Optional[Bindings] = None,
    ) -> Any:
        # optimize away the fitting of StatelessDataFrameTransforms
        # N.B. this means we cannot have any fit-time side effects.
        return None


class ConstantDataFrameTransform(ConstantTransform[pd.DataFrame], DataFrameTransform):
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
    no_cache: bool = False

    def __attrs_post_init__(self):
        if self.no_cache:
            self.pure = False

    def _apply(self, data_apply, _: None) -> pd.DataFrame:
        return pd.read_csv(self.filepath, **(self.read_csv_args or {}))


@params
class WritePandasCSV(StatelessDataFrameTransform, Identity[pd.DataFrame]):
    path: str = fmt_str_field()
    index_label: str = fmt_str_field()
    to_csv_kwargs: Optional[dict] = None

    pure = False

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        data_apply.to_csv(
            self.path, index_label=self.index_label, **(self.to_csv_kwargs or {})
        )
        return data_apply

    # Because Identity derives from UniversalTransform, we have to say which
    # then() and fit() to use on instances of WritePandasCSV
    then = DataFrameTransform.then
    fit = DataFrameTransform.fit


@params
class ReadDataset(ConstantDataFrameTransform):
    paths: list[str] = columns_field()
    columns: Optional[list[str]] = optional_columns_field(default=None)
    format: Optional[str] = None
    filter: Optional[dataset.Expression] = None
    index_col: Optional[str | int] = None
    partitioning_schema: Optional[list[tuple]] = None
    dataset_kwargs: dict = field(factory=dict)
    scanner_kwargs: dict = field(factory=dict)
    no_cache: bool = False

    def __attrs_post_init__(self):
        if self.no_cache:
            self.pure = False

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        if len(self.paths) == 1:
            paths = self.paths[0]
        dataset_kwargs = self.dataset_kwargs
        if self.partitioning_schema is not None:
            dataset_kwargs = {
                **dataset_kwargs,
                **{
                    "partitioning": dataset.partitioning(
                        pyarrow.schema(self.partitioning_schema)
                    )
                },
            }
        ds = dataset.dataset(paths, format=self.format, **dataset_kwargs)
        columns = self.columns or None
        df_out = ds.to_table(
            columns=columns, filter=self.filter, **self.scanner_kwargs
        ).to_pandas()
        # can we tell arrow this?
        if self.index_col is not None:
            df_out = df_out.set_index(self.index_col)
        return df_out


@params
class WriteDataset(Identity[pd.DataFrame]):
    base_dir: str = fmt_str_field()
    format: str = "parquet"
    partitioning_schema: Optional[list[tuple]] = None
    write_dataset_args: dict = field(factory=dict)
    existing_data_behavior: str = "delete_matching"

    pure = False

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        table = pyarrow.Table.from_pandas(data_apply)
        write_kwargs = self.write_dataset_args
        if self.partitioning_schema is not None:
            write_kwargs = {
                **write_kwargs,
                **{
                    "partitioning": dataset.partitioning(
                        pyarrow.schema(self.partitioning_schema)
                    )
                },
            }
        dataset.write_dataset(
            table,
            self.base_dir,
            format=self.format,
            existing_data_behavior=self.existing_data_behavior,
            **write_kwargs,
        )
        return data_apply


@params
class Join(DataFrameTransform):
    left: Transform[pd.DataFrame]
    right: Transform[pd.DataFrame]
    how: Literal["left", "right", "outer", "inner"]

    on: Optional[str] = None
    left_on: Optional[str] = None
    right_on: Optional[str] = None
    suffixes: tuple[str, str] = ("_x", "_y")

    # TODO: more merge params like left_index etc.

    def _submit_fit(
        self,
        data_fit: pd.DataFrame | Future[pd.DataFrame] | None = None,
        bindings: Optional[Bindings] = None,
    ) -> tuple[FitTransform, FitTransform]:
        bindings = bindings or {}
        with self.parallel_backend() as backend:
            fit_left, fit_right = (
                backend.push_trace("left").fit(self.left, data_fit, bindings),
                backend.push_trace("right").fit(self.right, data_fit, bindings),
            )
            return (fit_left, fit_right)

    def _materialize_state(self, state: Any) -> Any:
        fit_left, fit_right = super()._materialize_state(state)
        return (fit_left.materialize_state(), fit_right.materialize_state())

    def _do_merge(self, df_left, df_right):
        return pd.merge(
            left=df_left,
            right=df_right,
            how=self.how,
            on=self.on,
            left_on=self.left_on,
            right_on=self.right_on,
            suffixes=self.suffixes,
        )

    def _submit_apply(
        self,
        data_apply: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        state: tuple[FitTransform, FitTransform] | None = None,
    ) -> Future[pd.DataFrame]:
        assert state is not None
        fit_left, fit_right = state
        with self.parallel_backend() as backend:
            fut_left, fut_right = (
                backend.push_trace("left").apply(fit_left, data_apply),
                backend.push_trace("right").apply(fit_right, data_apply),
            )
            return backend.submit("_do_merge", self._do_merge, fut_left, fut_right)


class ColumnsTransform(DataFrameTransform):
    """
    Abstract base clase of all Transforms that require a list of columns as a parameter
    (typically ``cols``). Provides a utility method, :meth:`resolve_cols` for resolving
    the list of column names to which such a parameter refers in the case that it has
    the special type :class:`frankenfit.ALL_COLS`.
    """

    def resolve_cols(
        self,
        cols: list[str] | ALL_COLS,
        df: pd.DataFrame,
        ignore: str | list[str] | None = None,
    ) -> list[str]:
        """
        Utilty method returning the list of all column names in ``df`` if ``cols`` is an
        instance of :class:`frankenfit.ALL_COLS`, otherwise returns `cols` unmodified.

        Parameters
        ----------
        cols :
            A list of column names or instance of ``ALL_COLS``. Presumably the value of
            some ``columns_field()``-like parameter in the subclass.
        df :
            The DataFrame from which to get column names if ``cols`` is ``ALL_COLS()``.
            Presumably the fit-time or apply-time data of the subclass.
        ignore :
            Optional column name or list of column names *not* to include in the result
            if ``cols`` is ``ALL_COLS``.
        """
        if ignore is None:
            ignore = []
        elif isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(cols, ALL_COLS):
            return list(c for c in df.columns if c not in ignore)
        return cols


def fit_group_on_self(group_col_map):
    """
    The default fitting schedule for :class:`GroupByCols`: for each group, the
    grouped-by transform is fit on the data belonging to that group.
    """
    return lambda df: reduce(
        operator.and_, (df[c] == v for (c, v) in group_col_map.items())
    )


def fit_group_on_all_other_groups(group_col_map):
    """
    A built-in fitting schedule for :class:`GroupByCols`: for each group, the grouped-by
    transform is fit on the data belonging to all other groups. This is similar to
    k-fold cross-validation if the groups are viewed as folds.
    """
    return lambda df: reduce(
        operator.or_, (df[c] != v for (c, v) in group_col_map.items())
    )


class UnfitGroupError(ValueError):
    """
    Exception raised when a group-by-like transform is applied to data containing groups
    on which it was not fit.
    """


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
        The corresponding call-chain method is :meth:`group_by_cols()
        <frankenfit.dataframe.DataFramePipelineInterface.group_by_cols>`

    """

    cols: str | list[str] = columns_field()
    transform: Transform[pd.DataFrame] = field()
    # TODO: what about hyperparams in the fitting schedule? that's a thing.
    fitting_schedule: Callable[[dict[str, Any]], DfLocIndex | DfLocPredicate] = field(
        default=fit_group_on_self
    )
    as_index: bool = False
    group_keys: bool = False
    sort: bool = False

    # TODO: parallelize
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
            return self.transform.fit(df_group_fit, bindings)

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
            .groupby(
                self.cols,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
            )
            .apply(apply_on_group)
        )


@params
class GroupByBindings(ForBindings[pd.DataFrame], DataFrameTransform):
    bindings_sequence: Sequence[Bindings]
    transform: Transform[pd.DataFrame]
    as_index: bool = True
    # TODO: the proper way to do this would be with an AbstractForBindings that both
    # ForBindings and GroupByBindings derive from
    combine_fun: None = None  # type: ignore[assignment]

    def _combine_results(self, bindings_seq, *results) -> pd.DataFrame:
        binding_cols: set[str] = set()
        dfs = []
        for bindings, result in zip(bindings_seq, results):
            dfs.append(result.assign(**self._dataframable_bindings(bindings)))
            binding_cols |= bindings.keys()
        df = pd.concat(dfs, axis=0)
        if self.as_index:
            df = df.set_index(list(binding_cols))
        return df

    def _dataframable_bindings(self, bindings):
        result = {}
        for name, val in bindings.items():
            if type(val) not in (float, int, str):
                val = str(val)
            result[name] = val
        return result

    then = DataFrameTransform.then
    fit = DataFrameTransform.fit


F = TypeVar("F", bound="Filter")


@params(auto_attribs=False)
class Filter(StatelessDataFrameTransform):
    filter_fun: Callable = field()

    filter_fun_bindings: Bindings
    filter_fun_hyperparams: UserLambdaHyperparams

    def __attrs_post_init__(self):
        if isinstance(self.filter_fun, HP):
            raise TypeError(
                f"StatelessLambda.filter_fun must not be a hyperparameter; got:"
                f"{self.filter_fun!r}. Instead consider supplying a function that "
                f"requests hyperparameter bindings in its parameter signature."
            )
        self.filter_fun_bindings = {}
        self.filter_fun_hyperparams = UserLambdaHyperparams.from_function_sig(
            self.filter_fun, 1
        )

    def hyperparams(self) -> set[str]:
        return (
            super()
            .hyperparams()
            .union(self.filter_fun_hyperparams.required_or_optional())
        )

    def resolve(self: F, bindings: Optional[Bindings] = None) -> F:
        # override _resolve_hyperparams() to collect hyperparam bindings at fit-time
        # so we don't need bindings arg to _apply.
        resolved_self = super().resolve(bindings)
        resolved_self.filter_fun_bindings = (
            self.filter_fun_hyperparams.collect_bindings(bindings or {})
        )
        return resolved_self

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        fun_params = inspect.signature(self.filter_fun).parameters
        positional_args = (data_apply,) if len(fun_params) > 0 else tuple()
        return data_apply.loc[
            self.filter_fun(*positional_args, **self.filter_fun_bindings)
        ]


@params
class Copy(StatelessDataFrameTransform):
    """
    Copy values from one or more source columns into corresponding destination columns,
    either creating them or overwriting their contents.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    cols: list[str] | str
        The names of the source columns. Note that unlike for many other DataFrame
        transforms, ``cols`` **may not** be omitted to indicate all columns. It may,
        however, be a string as short-hand for a length-1 list.

    dest_cols: list[str] | str
        The names of the destination columns. If ``cols`` is length-1, then
        ``dest_cols`` may have any length greater than zero; potentially many copies of
        the source column will be created. Otherwise, ``cols`` and ``dest_cols`` must
        have the same length.

    Raises
    ------
    ValueError
        If ``len(cols) > 1`` and ``len(cols) != len(dest_cols)``.
    """

    cols: list[str] = columns_field()
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

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    .. TIP::
        As syntactic sugar, :class:`DataFramePipeline` overrides the index operator (via
        a custom ``__getitem__`` implementatino) as a synonym for appending a ``Select``
        transform to the pipeline. For example, the following two pipelines are
        equivalent::

            (
                ff.DataFramePipeline()
                ...
                .select(["col1", "col2"]
                ...
            )

            (
                ff.DataFramePipeline()
                ...
                [["col1", "col2"]]
                ...
            )

    Parameters
    ----------
    cols: list[str] | str
        The names of the source columns. Note that unlike for many other DataFrame
        transforms, ``cols`` **may not** be omitted to indicate all columns. It may,
        however, be a string as short-hand for a length-1 list.
    """

    cols: list[str] = columns_field()

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        return data_apply[self.cols]


@params
class Drop(ColumnsTransform, StatelessDataFrameTransform):
    """
    Drop the given columns from the data.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    cols: list[str] | str
        The names of the columns to drop. Note that unlike for many other DataFrame
        transforms, ``cols`` **may not** be omitted to indicate all columns. It may,
        however, be a string as short-hand for a length-1 list.
    """

    cols: list[str] = columns_field()

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        return data_apply.drop(columns=self.cols)


@params
class Rename(StatelessDataFrameTransform):
    """
    Rename columns.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    how: Callable[[str], str] | dict[str, str]
        Either a function that, given a column name, returns what it should be renamed
        do, or a dict from old column names to corresponding new names.
    """

    # TODO: support hp format-strs in keys and values
    # TODO: support UserLambdaHyperparams for callable how

    how: Callable | dict[str, str]

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        return data_apply.rename(columns=self.how)


@params
class Affix(ColumnsTransform, StatelessDataFrameTransform):
    """
    Affix a prefix and suffix to the names of the specified columns. At apply-time,
    return a new ``DataFrame`` in which, for every column ``c`` selected by ``cols``,
    that column has been renamed to ``prefix + c + suffix``.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    prefix: str
        The prefix string.

    suffix: str
        The suffix string.

    cols: list[str] | str | ALL_COLS, optional
        The names of the columns to rename. If omitted, all columns are renamed.
    """

    prefix: str
    suffix: str
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)

    def _apply(self, data_apply: pd.DataFrame, state: Any) -> pd.DataFrame:
        cols = self.resolve_cols(self.cols, data_apply)
        return data_apply.rename(
            columns={c: self.prefix + c + self.suffix for c in cols}
        )


@params
class Prefix(Affix):
    """
    Prepend a prefix to the names of the specified columns. At apply-time, return a new
    ``DataFrame`` in which, for every column ``c`` selected by ``cols``, that column has
    been renamed to ``prefix + c``. Implemented as a subclass of :class:`Affix` in which
    ``suffix`` is always the empty string.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    prefix: str
        The prefix string.

    cols: list[str] | str | ALL_COLS, optional
        The names of the columns to rename. If omitted, all columns are renamed.
    """

    prefix: str
    suffix: str = field(default="", init=False)
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)


@params
class Suffix(Affix):
    """
    Append a suffix to the names of the specified columns.  At apply-time, return a new
    ``DataFrame`` in which, for every column ``c`` selected by ``cols``, that column has
    been renamed to ``c + suffix``. Implemented as a subclass of :class:`Affix` in which
    ``prefix`` is always the empty string.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    suffix: str
        The suffix string.

    cols: list[str] | str | ALL_COLS, optional
        The names of the columns to rename. If omitted, all columns are renamed.
    """

    suffix: str
    prefix: str = field(default="", init=False)
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)


@params
class Pipe(ColumnsTransform, StatelessDataFrameTransform):
    """
    Select the specified columns of a DataFrame and pipe the resulting sub-DataFrame
    through a function.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    apply_fun: Callable[[pd.DataFrame], pd.DataFrame]
        The function through which to pipe the selected data. At apply-time, ``Pipe``
        returns the result of ``apply_fun(data_apply[cols])``.

    cols: list[str] | str | ALL_COLS, optional
        The names of the columns to select. If omitted, the entire apply-time DataFrame
        is passed to ``apply_fun``.
    """

    # TODO: support UserLambdaHyperparams for apply_fun
    apply_fun: Callable[[pd.DataFrame], pd.DataFrame]
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)

    def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
        cols = self.resolve_cols(self.cols, data_apply)
        result = self.apply_fun(data_apply[cols])
        return data_apply.assign(**{c: result[c] for c in cols})


# TODO Rank, MapQuantiles


@params
class Clip(ColumnsTransform, StatelessDataFrameTransform):
    """
    Clip the specified columns of a pandas DataFrame.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    upper: float, optional
        Values greater than `upper` are replaced by `upper`.

    lower: float, optional
        Values less than `lower` are replaced by `lower`.

    cols: list(str) | ALL_COLS, optional
        The names of the columns to clip. If omitted, all columns found in the DataFrame
        are clipped.
    """

    upper: Optional[float]
    lower: Optional[float]
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)

    def _apply(self, data_apply: pd.DataFrame, _: None) -> pd.DataFrame:
        cols = self.resolve_cols(self.cols, data_apply)
        return data_apply.assign(
            **{
                col: data_apply[col].clip(upper=self.upper, lower=self.lower)
                for col in cols
            }
        )


@params
class Winsorize(ColumnsTransform):
    """
    Symmetrically winsorize the specified columns of a pandas DataFrame, i.e., trim the
    upper and lower ``limit`` percent of values ("outliers") by replacing them with the
    ``limit``-th percentile of their respective column.

    Parameters
    ----------
    limit : float
        The percentile threshold beyond which values are trimmed. More precisely, for
        each target column, values less than the ``limit``-th percentile of that column
        are replaced by the ``limit``-th percentile, and values greater than the ``1 -
        limit``-th percentile are replaced by the ``1 - limit``-th percentile.

    cols: list(str) | ALL_COLS, optional
        The names of the columns to winsorize. If omitted, all columns found in the
        DataFrame are winsorized.
    """

    limit: float
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)

    def _fit(self, data_fit: pd.DataFrame) -> Mapping[str, pd.Series]:
        if not isinstance(self.limit, float):
            raise TypeError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )
        if self.limit < 0 or self.limit > 1:
            raise ValueError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )

        cols = self.resolve_cols(self.cols, data_fit)
        return {
            "lower": data_fit[cols].quantile(self.limit, interpolation="nearest"),
            "upper": data_fit[cols].quantile(1.0 - self.limit, interpolation="nearest"),
        }

    def _apply(
        self, data_apply: pd.DataFrame, state: Mapping[str, pd.Series]
    ) -> pd.DataFrame:
        cols = self.resolve_cols(self.cols, data_apply)
        return data_apply.assign(
            **{
                col: data_apply[col].clip(
                    upper=state["upper"][col], lower=state["lower"][col]
                )
                for col in cols
            }
        )


@params
class ImputeConstant(ColumnsTransform, StatelessDataFrameTransform):
    """
    Replace missing values (``NaN``, per pandas convention) in the specified columns
    with a constant value.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    value: Any
        The constant value.

    cols: list(str) | ALL_COLS, optional
        The names of the columns in which to replace missing values. If omitted, all
        columns found in the ``DataFrame`` are affected.
    """

    value: Any
    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)

    def _apply(self, data_apply: pd.DataFrame, state) -> pd.DataFrame:
        cols = self.resolve_cols(self.cols, data_apply)
        return data_apply.assign(
            **{col: data_apply[col].fillna(self.value) for col in cols}
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
    De-mean the specified columns of a pandas DataFrame.

    Parameters
    ----------
    cols: list(str) | ALL_COLS, optional
        A list of the names of the columns to de-mean, or else the :class:`ALL_COLS`
        type to indicate that all columns should be de-meaned. Optional; by default, all
        columns are de-meaned.

    w_col: str, optional
        Optional name of the column to use as a source of observation weights when
        computing the means. If omitted, the means are unweighted.
    """

    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)
    w_col: Optional[str] = None

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        cols = self.resolve_cols(self.cols, data_fit, ignore=self.w_col)
        if self.w_col is not None:
            return _weighted_means(data_fit, cols, self.w_col)
        return data_fit[cols].mean(numeric_only=True)

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series):
        means = state
        cols = self.resolve_cols(self.cols, data_apply, ignore=self.w_col)
        return data_apply.assign(**{c: data_apply[c] - means[c] for c in cols})


@params
class ImputeMean(ColumnsTransform):
    """
    Replace missing values (``NaN``, per pandas convention) in the specified columns
    with the columns' respective fit-time means (optionally weighted by another column).

    Parameters
    ----------
    cols: list(str) | ALL_COLS, optional
        The names of the columns in which to replace missing values. If omitted, all
        columns found in the ``DataFrame`` are affected.

    w_col: str, optional
        Optional name of the column to use as a source of observation weights when
        computing the mean of each column selected by ``cols``. If omitted, the means
        are unweighted.
    """

    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)
    w_col: Optional[str] = None

    def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
        cols = self.resolve_cols(self.cols, data_fit, ignore=self.w_col)
        if self.w_col is not None:
            return _weighted_means(data_fit, cols, self.w_col)
        return data_fit[cols].mean(numeric_only=True)

    def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
        means = state
        cols = self.resolve_cols(self.cols, data_apply, ignore=self.w_col)
        return data_apply.assign(**{c: data_apply[c].fillna(means[c]) for c in cols})


@params
class ZScore(ColumnsTransform):
    """
    Z-score the specified columns of a pandas DataFrame.

    Parameters
    ----------
    cols: list(str) | ALL_COLS, optional
        A list of the names of the columns to z-score, or else the :class:`ALL_COLS`
        type to indicate that all columns should be z-scored. Optional; by default, all
        columns are z-scored.

    w_col: str, optional
        Optional name of the column to use as a source of observation weights when
        computing the means to use for z-scoring. If omitted, the means are unweighted.
        In any case, the standard deviations are always unweighted.
    """

    cols: list[str] | ALL_COLS = columns_field(factory=ALL_COLS)
    w_col: Optional[str] = None

    def _fit(self, data_fit: pd.DataFrame) -> dict[str, pd.Series]:
        cols = self.resolve_cols(self.cols, data_fit, ignore=self.w_col)
        if self.w_col is not None:
            means = _weighted_means(data_fit, cols, self.w_col)
        else:
            means = data_fit[cols].mean(numeric_only=True)
        return {"means": means, "stddevs": data_fit[cols].std()}

    def _apply(
        self, data_apply: pd.DataFrame, state: dict[str, pd.Series]
    ) -> pd.DataFrame:
        cols = self.resolve_cols(self.cols, data_apply, ignore=self.w_col)
        means, stddevs = state["means"], state["stddevs"]
        return data_apply.assign(
            **{c: (data_apply[c] - means[c]) / stddevs[c] for c in cols}
        )


@params
class SKLearn(DataFrameTransform):
    """
    Wrap a ``scikit-learn`` ("sklearn") model. At fit-time, the given sklearn model
    class is instantiated (with arguments from ``class_params``) and trained on the
    fitting data by calling its ``fit()`` method. At apply-time, the now-fit sklearn
    model object is used to generated predictions by calling its ``predict()`` method,
    which are assigned to the apply-time data as a new column, ``hat_col``.

    Parameters
    ----------
    sklearn_class: type
        The sklearn class to wrap.
    x_cols: list[str]
        The predictor columns. These are selected from the fit/apply-data to create the
        ``X`` argument to the sklearn model's ``fit()`` and ``predict()`` methods.
    response_col: str
        The response column. At fit-time, this is selected from the fitting data to
        create the ``y`` argument to the sklearn model's ``fit()`` method. This column
        is only needed at fit-time, hence its creation may be wrapped in
        :class:`~frankenfit.core.IfPipelineIsFitting` if desired.
    hat_col: str
        The name of the new column to create at apply-time containing predictions from
        the sklearn model.
    class_params: dict[str, Any], optional
        Optional parameters to pass as named arguments to the ``sklearn_class``
        constructor when instantiating it.
    w_col: str, optional
        Optional name of a sample weight column. If specified, this is selected at
        fit-time from the fitting data to create the ``sample_weight`` named argument to
        the sklearn model's ``fit()`` method.

        .. WARNING:: Not every sklearn model accepts a ``sample_weight`` keyword
            argument to its ``fit()`` method. Consult the documentation of whichever
            sklearn model you are using.

    Examples
    --------
    ::

        regress = ff.dataframe.SKLearn(
            sklearn_class=LinearRegression,
            x_cols=["carat", "table", "depth"],
            response_col="price",
            hat_col="price_hat",
            class_params={"fit_intercept": True}  # additional kwargs LinearRegression
        )
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
    Wrap a ``statsmodels`` model.  At fit-time, the given model class is instantiated
    (with arguments from ``class_params``) and trained on the fitting data by calling
    its ``fit()`` method. At apply-time, the now-fit model object is used to generated
    predictions by calling its ``predict()`` method, which are assigned to the
    apply-time data as a new column, ``hat_col``.

    Parameters
    ----------
    sm_class: type
        The ``statsmodels`` model class to wrap.
    x_cols: list[str]
        The predictor columns. These are selected from the fit/apply-data to create the
        ``X`` argument to the model's ``fit()`` and ``predict()`` methods.
    response_col: str
        The response column. At fit-time, this is selected from the fitting data to
        create the ``y`` argument to the model's ``fit()`` method. This column is only
        needed at fit-time, hence its creation may be wrapped in
        :class:`~frankenfit.core.IfPipelineIsFitting` if desired.
    hat_col: str
        The name of the new column to create at apply-time containing predictions from
        the ``statsmodels`` model.
    class_params: dict[str, Any], optional
        Optional parameters to pass as named arguments to the ``sm_class`` constructor
        when instantiating it.
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

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    left_cols: list[str]
        List of "left" correlands. Result will have one row per element of
        ``left_cols``.
    right_cols: list[str]
        List of "right" correlands. Result will have one column per element of
        ``right_cols``.
    method: str
        One of ``"pearson"``, ``"spearman"``, or ``"kendall"``, specifying which type of
        correlation coefficient to compute.
    min_obs: int
        The minimum number of non-missing values for each pair of columns.  If a pair
        has fewer than this many non-missing observations, then the correlation for that
        pair will be missing in the result.

    Examples
    --------
    ::

        from pydataset import data
        df = data("diamonds")
        ff.dataframe.Correlation(["price"], ["carat"]).apply(df)
        # -->           carat
        # --> price  0.921591

        ff.dataframe.Correlation(["table", "depth"], ["x", "y", "z"]).apply(df)
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


@params(auto_attribs=False)
class Assign(DataFrameTransform):
    """
    Assign the results of other Transforms (or scalar values) to columns.

    Examples
    --------
    ::

        pipeline = ff.DataFramePipeline().assign(
            # ---multi-column assigments---
            do[["price", "carat"]].de_mean().suffix("_dmn"),  # pipeline
            backend.apply(other_pipeline, diamonds_df),  # future

            # lambda is wrapped in a StatelessLambda transform
            lambda df: pd.DataFrame().assign(uppercut=df["cut"].str.upper()),

            # ---named column assignments: transforms with 1-column output---
            price_dmn2=do["price"].de_mean(),

            # future with 1-column result
            price_win2=backend.apply(other_pipeline["price_win"], diamonds_df),

            # lambda is wrapped in a StatelessLambda transform
            price_rank=lambda df, price_scale=1.0: price_scale * (
                (df["price"] - df["price"].min())
                / (df["price"].max() - df["price"].min())
            ),

            intercept=1.0,  # scalar
        )
    """

    assignments: dict[
        int | str,  # int key indicates multi-column assignment, str indicates named
        (
            Transform[pd.DataFrame]
            | Future[pd.DataFrame]
            | Callable
            | float  # scalars...
            | int
            | str
        ),
    ]

    multi_column_assignments: Sequence[
        Transform[pd.DataFrame] | Future[pd.DataFrame] | Callable
    ] = field(factory=list)

    named_column_assignments: Mapping[
        str,
        (
            Transform[pd.DataFrame]
            | Future[pd.DataFrame]
            | Callable
            | float  # scalars...
            | int
            | str
        ),
    ] = field(factory=dict)

    # Assign([assignment_dict][, tag=][, kwarg1=][, kwarg2][...])
    # ... with only one of assigment_dict or kwargs
    def __init__(self, *args, tag=NOTHING, **kwargs):
        self.__attrs_init__(
            multi_column_assignments=args,
            named_column_assignments=kwargs,
            tag=tag,
        )

    def __attrs_post_init__(self):
        self.assignments = {}
        for i, ass in enumerate(self.multi_column_assignments):
            if callable(ass):
                self.assignments[i] = StatelessLambda[pd.DataFrame](ass)
            else:
                self.assignments[i] = ass
        for k, ass in self.named_column_assignments.items():
            if callable(ass):
                self.assignments[k] = StatelessLambda[pd.DataFrame](ass)
            else:
                self.assignments[k] = ass

    def hyperparams(self) -> set[str]:
        return (
            super()
            .hyperparams()
            .union(
                *(
                    ass.hyperparams()
                    for k, ass in self.assignments.items()
                    if isinstance(ass, Transform)
                )
            )
        )

    def _submit_fit(
        self,
        data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        bindings: Optional[Bindings] = None,
    ) -> dict[int | str, FitTransform | Future]:
        with self.parallel_backend() as backend:
            data_fit = backend.maybe_put(data_fit)
            state: dict[int | str, FitTransform | Future] = {}
            for k, ass in self.assignments.items():
                if isinstance(k, int):
                    trace = f"[{k}]"
                else:
                    trace = k

                if isinstance(ass, Transform):
                    state[k] = backend.push_trace(trace).fit(ass, data_fit, bindings)
                elif isinstance(ass, Future):
                    state[k] = ass
                else:
                    state[k] = backend.put(ass)

            return state

    def _materialize_state(self, state: dict[int | str, FitTransform | Future]) -> Any:
        state_dict = super()._materialize_state(state)
        for k, v in state_dict.items():
            if isinstance(v, FitTransform):
                state_dict[k] = v.materialize_state()
            elif isinstance(v, Future):
                state_dict[k] = v.result()
        return state_dict

    def _do_assign(
        self,
        data_apply: pd.DataFrame,
        *multi_cols: pd.DataFrame,
        **named_cols: pd.DataFrame | pd.Series | int | float | str,
    ):
        result = data_apply
        for mc_df in multi_cols:
            result = result.assign(**{c: mc_df[c] for c in mc_df.columns})

        for c, obj in named_cols.items():
            val = obj
            if isinstance(obj, pd.DataFrame):
                if len(obj.columns) != 1:
                    raise ValueError(
                        f"_do_assign: named column assignment {c!r} expected a 1-"
                        f"column DataFrame, but got: {obj.columns}"
                    )
                val = obj[obj.columns[0]]
            result = result.assign(**{c: val})

        return result

    def _submit_apply(
        self,
        data_apply: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        state: dict[int | str, FitTransform | Future] | None = None,
    ) -> Future[pd.DataFrame] | None:
        assert state is not None
        if data_apply is None:  # pragma: no cover
            data_apply = pd.DataFrame()
        with self.parallel_backend() as backend:
            data_apply = backend.maybe_put(data_apply)
            multi_col_futures: list[Future[pd.DataFrame]] = []
            named_col_futures: dict[str, Future[pd.DataFrame]] = {}
            for k, ass in state.items():
                if isinstance(k, int):
                    stash = multi_col_futures.append
                    trace = f"[{k}]"
                else:
                    stash = partial(named_col_futures.__setitem__, k)
                    trace = k

                if isinstance(ass, FitTransform):
                    fut = backend.push_trace(trace).apply(ass, data_apply)
                elif isinstance(ass, Future):
                    fut = ass
                else:  # pragma: no cover
                    raise TypeError(
                        f"Assign internal error: non-FitTransform, non-Future "
                        f"assignment: k={k!r}, ass={ass!r}"
                    )
                stash(fut)

            return backend.submit(
                "_do_assign",
                self._do_assign,
                data_apply,
                *multi_col_futures,
                pure=True,  # FIXME? what if impure children?
                **named_col_futures,
            )


Cols = Union[str, HP, ALL_COLS, Sequence[Union[str, HP]]]


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

    @callchain(WriteDataset)
    def write_dataset(  # type: ignore [empty-body]
        self,
        base_dir: str | HP,
        format: str | HP = "parquet",
        partitioning_schema: Optional[list[tuple]] = None,
        write_dataset_args: dict | None = None,
        existing_data_behavior: str = "delete_matching",
    ) -> P_co:
        """
        Applend a :class:`WriteDataset` transform to this pipeline.
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
        filter_fun: Callable,
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

    @callchain(Affix)
    def affix(  # type: ignore [empty-body]
        self,
        prefix: str,
        suffix: str,
        cols: Cols | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`Affix` transform to this pipeline.
        """

    @callchain(Prefix)
    def prefix(  # type: ignore [empty-body]
        self,
        prefix: str,
        cols: Cols | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`Prefix` transform to this pipeline.
        """

    @callchain(Suffix)
    def suffix(  # type: ignore [empty-body]
        self,
        suffix: str,
        cols: Cols | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`Suffix` transform to this pipeline.
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
        # Multi-column assignments
        *args: Transform[pd.DataFrame] | Future[pd.DataFrame] | Callable,
        tag: Optional[str] = None,
        # Named-column assignments
        **kwargs: (
            Transform[pd.DataFrame]
            | Future[pd.DataFrame]
            | Callable
            | float  # scalars...
            | int
            | str
        ),
    ) -> P_co:
        """
        Append an :class:`Assign` transform to this pipeline.
        """

    @callchain(Pipe)
    def pipe(  # type: ignore [empty-body]
        self,
        apply_fun: Callable | HP,
        cols: Cols | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Pipe` transform to this pipeline.
        """

    @callchain(Clip)
    def clip(  # type: ignore [empty-body]
        self,
        upper: Optional[float | HP] = None,
        lower: Optional[float | HP] = None,
        cols: Cols | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Clip` transform to this pipeline.
        """

    @callchain(Winsorize)
    def winsorize(  # type: ignore [empty-body]
        self, limit: float | HP, cols: Cols | None = None, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`Winsorize` transform to this pipeline.
        """

    @callchain(ImputeConstant)
    def impute_constant(  # type: ignore [empty-body]
        self, value: Any, cols: Cols | None = None, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append an :class:`ImputeConstant` transform to this pipeline.
        """

    @callchain(DeMean)
    def de_mean(  # type: ignore [empty-body]
        self,
        cols: Cols | None = None,
        w_col: Optional[str | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`DeMean` transform to this pipeline.
        """

    @callchain(ImputeMean)
    def impute_mean(  # type: ignore [empty-body]
        self,
        cols: Cols | None = None,
        w_col: Optional[str | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`ImputeMean` transform to this pipeline.
        """

    @callchain(ZScore)
    def z_score(  # type: ignore [empty-body]
        self,
        cols: Cols | None = None,
        w_col: Optional[str | HP] = None,
        *,
        tag: Optional[str] = None,
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
        right: Transform[pd.DataFrame],
        how: Literal["left", "right", "outer", "inner"] | HP,
        on: Optional[str | HP] = None,
        left_on: Optional[str | HP] = None,
        right_on: Optional[str | HP] = None,
        suffixes=("_x", "_y"),
    ) -> SelfDPI:
        """
        Return a new :class:`DataFramePipeline` (of the same subclass as
        ``self``) containing a new :class:`frankenfit.dataframe.Join` transform with
        this pipeline as the ``Join``'s ``left`` argument.
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
        as_index: bool = False,
        group_keys: bool = False,
        sort: bool = False,
    ) -> G_co:
        """
        Return a :class:`Grouper` object, which will consume the next Transform in the
        call-chain by wrapping it in a :class:`~frankenfit.dataframe.GroupByCols`
        transform and returning the result of appending that ``GroupByCols`` to this
        pipeline. It enables Pandas-style call-chaining with ``GroupByCols``.

        For example, grouping a single Transform::

            (
                ff.DataFramePipeline()
                # ...
                .group_by_cols("cut")  # -> PipelineGrouper
                    .z_score(cols)  # -> Pipeline
            )

        Grouping a sequence of Transforms::

            (
                ff.DataFramePipeline()
                # ...
                .group_by_cols("cut")
                    .then(
                        ff.DataFramePipeline()
                        .winsorize(limit=0.01)
                        .z_score()
                        .clip(upper=2, lower=-2)
                    )
            )

        .. NOTE::
            When using ``group_by_cols()``, by convention we add a level of indentation
            to the next call in the call-chain, to indicate visually that it is being
            consumed by the preceding ``group_by_cols()`` call.

        .. SEEALSO:: See :class:`~frankenfit.dataframe.GroupByCols` for parameters.
        """
        grouper = type(self)._Grouper(
            self,
            GroupByCols,
            "transform",
            cols=cols,
            fitting_schedule=(fitting_schedule or fit_group_on_self),
            as_index=as_index,
            group_keys=group_keys,
            sort=sort,
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
    fit_transform_class: ClassVar[Type[FitTransform]] = FitDataFrameTransform

    def _empty_constructor(self) -> pd.DataFrame:
        return pd.DataFrame()

    def fit(
        self: R,
        data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        bindings: Optional[Bindings] = None,
        /,
        **kwargs,
    ) -> FitDataFrameTransform[R]:
        return cast(FitDataFrameTransform[R], super().fit(data_fit, bindings, **kwargs))
