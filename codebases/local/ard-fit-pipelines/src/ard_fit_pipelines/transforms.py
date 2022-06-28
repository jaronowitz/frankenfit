from __future__ import annotations
import inspect
import logging
from logging import Logger

from typing import Callable, Optional, TextIO
from attrs import define
import pandas as pd

from .core import (
    Transform,
    WeightedTransform,
    columns_field,
    StatelessTransform,
    ColumnsTransform,
    HP,
)

LOG = logging.getLogger(__name__)


class Identity(StatelessTransform):
    """
    The stateless Transform that, at apply-time, simply returns the input data
    unaltered.
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None):
        return df_apply


@define
class CopyColumns(StatelessTransform, ColumnsTransform):
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


class KeepColumns(StatelessTransform, ColumnsTransform):
    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply[self.cols]


@define
class RenameColumns(StatelessTransform):
    how: Callable | dict[str, str]

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.rename(columns=self.how)


class DropColumns(StatelessTransform, ColumnsTransform):
    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.drop(columns=self.cols)


# Inliners: StatelessLambda, StatefulLambda


@define
class StatelessLambda(StatelessTransform):
    apply_fun: Callable  # df[, bindings] -> df

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 1:
            return self.apply_fun(df_apply)
        elif len(sig) == 2:
            return self.apply_fun(df_apply, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")


@define
class StatefulLambda(Transform):
    fit_fun: Callable  # df[, bindings] -> state
    apply_fun: Callable  # df, state[, bindings] -> df

    def _fit(self, df_fit: pd.DataFrame) -> object:
        sig = inspect.signature(self.fit_fun).parameters
        if len(sig) == 1:
            return self.fit_fun(df_fit)
        elif len(sig) == 2:
            return self.fit_fun(df_fit, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 2:
            return self.apply_fun(df_apply, state)
        elif len(sig) == 3:
            return self.apply_fun(df_apply, state, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 2 or 3 parameters, found {len(sig)}")


# TODO ImputeMean, DeMean, ZScore, Rank, MapQuantiles


@define
class Clip(StatelessTransform, ColumnsTransform):
    upper: Optional[float] = None
    lower: Optional[float] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{
                col: df_apply[col].clip(upper=self.upper, lower=self.lower)
                for col in self.cols
            }
        )


@define
class Winsorize(ColumnsTransform):
    # assume symmetric, i.e. trim the upper and lower `limit` percent of observations
    limit: float

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


@define
class ImputeConstant(StatelessTransform, ColumnsTransform):
    value: object

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{col: df_apply[col].fillna(self.value) for col in self.cols}
        )


def weighted_means(df, cols, w_col):
    return df[cols].multiply(df[w_col], axis="index").sum() / df[w_col].sum()


@define
class DeMean(WeightedTransform, ColumnsTransform):
    """
    De-mean some columns.
    """

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            return weighted_means(df_fit, self.cols, self.w_col)
        return df_fit[self.cols].mean()

    def _apply(self, df_apply: pd.DataFrame, state: pd.DataFrame):
        means = state
        return df_apply.assign(**{c: df_apply[c] - means[c] for c in self.cols})


@define
class ImputeMean(WeightedTransform, ColumnsTransform):
    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            return weighted_means(df_fit, self.cols, self.w_col)
        return df_fit[self.cols].mean()

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        means = state
        return df_apply.assign(**{c: df_apply[c].fillna(means[c]) for c in self.cols})


# class Filter:
#     pass


@define
class Print(Identity):
    """
    An Identity transform that has the side-effect of printing a message at fit- and/or
    apply-time.
    """

    fit_msg: Optional[str] = None
    """Message to print at fit-time."""

    apply_msg: Optional[str] = None
    """Message to print at apply-time."""

    dest: Optional[TextIO | str] = None  # if str, will be opened in append mode
    """
    File object to which to print, or the name of a file to open in append mode. If
    None (default), print to stdout.
    """

    def _fit(self, df_fit: pd.DataFrame):
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.fit_msg, file=dest)
        else:
            print(self.fit_msg, file=self.dest)
        return Identity._fit(self, df_fit)

        # Idiomatic super() doesn't work because at call time self is a FitPrint
        # instance, which inherits directly from FitTransform, and not from
        # Print/Identity. Could maybe FIXME by futzing with base classes in the
        # metaprogramming that goes on in core.py
        # return super()._fit(df_fit)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.apply_msg, file=dest)
        else:
            print(self.apply_msg, file=self.dest)
        return Identity._apply(self, df_apply, state=state)
        # Same issue with super() as in _fit().
        # return super()._apply(df_apply)


@define
class LogMessage(Identity):
    """
    An Identity transform that has the side-effect of logging a message at fit- and/or
    apply-time.
    """

    fit_msg: Optional[str] = None
    """Message to log at fit-time."""

    apply_msg: Optional[str] = None
    """Message to log at apply-time."""

    logger: Optional[Logger] = None
    """Logger instance to which to log. If None (default), use transforms.LOG"""

    level: int = logging.INFO
    """Level at which to log, default INFO."""

    def _fit(self, df_fit: pd.DataFrame):
        if self.fit_msg is not None:
            logger = self.logger or LOG
            logger.log(self.level, self.fit_msg)
        return Identity._fit(self, df_fit)
        # return super()._fit(df_fit)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if self.apply_msg is not None:
            logger = self.logger or LOG
            logger.log(self.level, self.apply_msg)
        return Identity._apply(self, df_apply, state=state)
        # return super()._apply(df_apply)


# Timeseries?
# Graph-making transforms:
# Pipeline, Join, JoinAsOf (time series), IfHyperparamTrue, IfTrainingDataHasProperty,
# GroupedBy, Longitudinally (time series), CrossSectionally (time seires),
# Sequentially (time series), AcrossHyperParamGrid
