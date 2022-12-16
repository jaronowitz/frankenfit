# Copyright (c) 2022 Max Bane <max@thebanes.org>
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

from typing import Any, Callable, Generic, Optional, TypeVar, cast

import pandas as pd
from attrs import define, field

from frankenfit.core import (
    Backend,
    Bindings,
    FitTransform,
    Future,
    LocalBackend,
    Transform,
)
from frankenfit.dataframe import (
    DataFramePipeline,
    DataFrameTransform,
    ReadDataFrame,
    StatelessDataFrameTransform,
)
from frankenfit.params import fmt_str_field, params

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class TimeseriesTransform(DataFrameTransform):
    def then(
        self, other: Optional[Transform | list[Transform]] = None
    ) -> "TimeseriesPipeline":
        result = super().then(other)
        return TimeseriesPipeline(transforms=result.transforms)


@params
class AsTimeseries(TimeseriesTransform, StatelessDataFrameTransform):
    begin: str = fmt_str_field(default="{begin}")
    end: str = fmt_str_field(default="{end}")
    time_col: str = "time"
    keep_old_index: bool = False

    def _apply(self, data_apply: pd.DataFrame, _) -> pd.DataFrame:
        if isinstance(data_apply.index, pd.DatetimeIndex):
            data_apply = data_apply.rename_axis("time")
        elif self.time_col not in data_apply.columns:
            raise ValueError(
                f"{self.name}: time_col {self.time_col!r} not found among input "
                f"columns: {data_apply.columns!r}"
            )
        else:
            # parse/convert time_col
            data_apply = data_apply.assign(
                **{self.time_col: pd.to_datetime(data_apply[self.time_col])}
            )
            if self.keep_old_index:
                data_apply = data_apply.reset_index().set_index(self.time_col)
            else:
                data_apply = data_apply.set_index(self.time_col)

        begin = pd.Timestamp(self.begin)
        # we want to be exclusive on the right, even though pandas .loc[] is inclusive
        # on the right, hence we subtract 1ns from self.end
        end = pd.Timestamp(self.end) - pd.Timedelta("1ns")

        return data_apply.loc[begin:end]  # type: ignore [misc]


@params
class TimeseriesDataFrame(ReadDataFrame, AsTimeseries):
    df: pd.DataFrame = field(factory=pd.DataFrame)

    def _apply(self, data_apply: pd.DataFrame, _) -> pd.DataFrame:
        df = ReadDataFrame._apply(self, None, _)  # type: ignore [arg-type]
        df = AsTimeseries._apply(self, df, _)
        return df


@params
class GroupByInterval(TimeseriesTransform):
    transform: TimeseriesTransform
    freq: str
    begin: str = fmt_str_field(default="{begin}")
    end: str = fmt_str_field(default="{end}")

    def steps(self, **bindings) -> pd.DatetimeIndex:
        self = self.resolve(bindings)
        begin = pd.Timestamp(self.begin)
        end = pd.Timestamp(self.end)  # - pd.Timedelta("1ns")

        interp = pd.date_range(begin, end, freq=self.freq)
        return cast(
            pd.DatetimeIndex,
            pd.DatetimeIndex([begin]).union(interp).union(pd.DatetimeIndex([end])),
        )

    def _submit_fit(
        self,
        data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        bindings: Optional[Bindings] = None,
    ) -> Any:
        bindings = bindings or {}
        steps = self.steps(**bindings)
        assert len(steps) >= 2

        ctx_seq = []
        fits = []

        with self.parallel_backend() as backend:
            data_fit = backend.maybe_put(data_fit)

            for i in range(len(steps) - 1):
                begin, end = steps[i], steps[i + 1]
                ctx_seq.append((begin, end))
                ctx_bindings = {**bindings, **{"begin": begin, "end": end}}
                fits.append(backend.fit(self.transform, data_fit, ctx_bindings))

        return ctx_seq, fits

    def _materialize_state(self, state: Any) -> Any:
        ctx_seq, fits = state
        return ctx_seq, [f.materialize_state() for f in fits]

    def _do_stitch(self, ctx_seq, *results) -> pd.DataFrame:
        assert len(ctx_seq) == len(results)
        dfs = []
        for ctx, df in zip(ctx_seq, results):
            begin, end = ctx
            end = end - pd.Timedelta("1ns")
            dfs.append(df.loc[begin:end])

        return pd.concat(dfs, axis="index")

    def _submit_apply(
        self,
        data_apply: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        state: Any = None,
    ) -> Future[pd.DataFrame] | None:
        ctx_seq: list[tuple]
        fits: list[FitTransform]
        ctx_seq, fits = state
        results: list[Future[pd.DataFrame]] = []
        with self.parallel_backend() as backend:
            data_apply = backend.maybe_put(data_apply)
            for fit in fits:
                results.append(backend.apply(fit, data_apply))
            return backend.submit("_do_stitch", self._do_stitch, ctx_seq, *results)


class ReadTimeseriesDataset:
    ...


class WriteTimeseriesDataset:
    ...


class CrossSectionally:
    ...


class TimeseriesPipeline(DataFramePipeline):
    ...


@define
class TimeseriesFuture(Generic[T_co], Future[T_co]):
    fut: Future[T_co]

    def result(self) -> T_co:
        return self.fut.result()

    def belongs_to(self, backend: Backend) -> bool:
        if isinstance(backend, TimeseriesBackend):
            return self.fut.belongs_to(backend.on)
        return False

    def __eq__(self, o):
        if isinstance(o, TimeseriesFuture):
            return self.fut == o.fut
        return self.fut == o


@define
class TimeseriesBackend(Backend):
    on: Backend = field(factory=LocalBackend)

    def put(self, data: T) -> TimeseriesFuture[T]:
        return TimeseriesFuture(self.on.put(data))

    def maybe_unwrap_future(self, data: Any) -> Any:
        if isinstance(data, TimeseriesFuture):
            return data.fut
        return data

    def submit(
        self,
        key_prefix: str,
        function: Callable,
        *function_args,
        pure: bool = True,
        **function_kwargs,
    ) -> TimeseriesFuture[Any]:
        args = tuple(self.maybe_unwrap_future(a) for a in function_args)
        kwargs = {k: self.maybe_unwrap_future(v) for k, v in function_kwargs.items()}
        fut = self.on.submit(key_prefix, function, *args, pure=pure, **kwargs)
        return TimeseriesFuture(fut)
