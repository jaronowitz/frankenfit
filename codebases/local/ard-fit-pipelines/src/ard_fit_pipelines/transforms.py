from __future__ import annotations
import logging
from logging import Logger

from typing import Callable, TextIO
from attrs import define, field
import pandas as pd

from . import core
from .core import (
    Transform,
    StatelessTransform,
    ColumnsTransform,
    hp,
    not_empty,
    #columns_tuple_converter
)

LOG = logging.getLogger(__name__)

class Identity(StatelessTransform):
    """
    The stateless Transform that, at apply-time, simply returns the input
    data unaltered.
    """
    def _apply(self, X_apply: pd.DataFrame, state: object = None):
        return X_apply

class DeMean(ColumnsTransform):
    """
    De-mean some columns.
    """
    def _fit(self, X_fit: pd.DataFrame) -> object:
        return X_fit[self.cols].mean()
    
    def _apply(self, X_apply: pd.DataFrame, state: object):
        means = state
        return X_apply.assign(**{
            c: X_apply[c] - means[c]
            for c in self.cols
        })

@define
class CopyColumns(StatelessTransform, ColumnsTransform):
    """
    A stateless Transform that copies values from one or more source columns
    into corresponding destination columns, either creating them or overwriting
    their contents.
    """
    dest_cols: list[str | hp] = field() # TODO: converter/validator

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
                'When copying more than one source column, '
                f'cols (len {lc}) and dest_cols (len {lv}) must have the same '
                'length.'
            )

    def _apply(self, X_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if len(self.cols) == 1:
            src_col = self.cols[0]
            return X_apply.assign(**{
                dest_col: X_apply[src_col] for dest_col in self.dest_cols
            })

        return X_apply.assign(**{
            dest_col: X_apply[src_col] for src_col, dest_col in
                zip(self.cols, self.dest_cols)
        })

# @define
# class RenameColumns(core.ColumnsTransform):
#     how: hp | Callable | dict[str | hp, str | hp]
# 
class KeepColumns(StatelessTransform, ColumnsTransform):
    def _apply(self, X_apply: pd.DataFrame, state: object=None) -> pd.DataFrame:
        return X_apply[self.cols]
    
# 
# class DropColumns(core.ColumnsTransform):
#     pass
# 
# 
# # Transform graphs
# 
# #class Pipeline(Transform):
# #    pass

# Clip, Impute, Winsorize, DeMean, ZScore, Rank, MapQuantiles
# Inliners: StatelessLambda, StatefulLambda

class DropColumns:
    pass

class Filter:
    pass

@define
class Print(Identity):
    """
    An Identity transform that has the side-effect of printing a message at fit-
    and/or apply-time.
    """
    fit_msg: str = None
    "Message to print at fit-time."

    apply_msg: str = None
    "Message to print at apply-time."

    dest: TextIO | str = None # if str, will be opened in append mode
    """File object to which to print, or the name of a file to open in append
    mode. If None (default), print to stdout."""

    def _fit(self, X_fit: pd.DataFrame):
        if type(self.dest) is str:
            with open(self.dest, 'a') as dest:
                print(self.fit_msg, dest=dest)
        else:
            print(self.fit_msg, dest=self.dest)
        return super()._fit(X_fit)

    def _apply(self, X_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if type(self.dest) is str:
            with open(self.dest, 'a') as dest:
                print(self.apply_msg, dest=dest)
        else:
            print(self.apply_msg, dest=self.dest)
        return super()._apply(X_apply)

@define
class LogMessage(Identity):
    """
    An Identity transform that has the side-effect of logging a message at fit-
    and/or apply-time.
    """
    fit_msg: str = None
    "Message to log at fit-time."

    apply_msg: str = None
    "Message to log at apply-time."

    logger: Logger = None
    "Logger instance to which to log. If None (default), use transforms.LOG"

    level: int = logging.INFO   
    "Level at which to log, default INFO."

    def _fit(self, X_fit: pd.DataFrame):
        if self.fit_msg is not None:
            logger = self.logger or LOG
            logger.log(self.level, self.fit_msg)
        return super()._fit(X_fit)

    def _apply(self, X_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if self.apply_msg is not None:
            logger = self.logger or LOG
            logger.log(self.level, self.apply_msg)
        return super()._apply(X_apply)


class Read:
    pass

class Write:
    pass

# Timeseries?
# Graph-making transforms:
# Pipeline, Join, JoinAsOf (time series), IfHyperparamTrue, IfTrainingDataHasProperty,
# GroupedBy, Longitudinally (time series), CrossSectionally (time seires),
# Sequentially (time series), AcrossHyperParamGrid