from __future__ import annotations

from typing import Callable
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

class Identity(StatelessTransform):
    """
    The stateless Transform that, at apply-time, simply returns the input
    data unaltered.
    """
    def _apply(self, X_apply: pd.DataFrame, state: object = None):
        return X_apply

class DeMean(ColumnsTransform):
    "Docstr for MyDeMean"
    
    def _fit(self, X_fit):
        return X_fit[self.cols].mean()
    
    def _apply(self, X_apply, state):
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
    dest_cols: str[str | hp] = field() # TODO: converter/validator

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
# class KeepColumns(core.ColumnsTransform):
#     pass
# 
# class DropColumns(core.ColumnsTransform):
#     pass
# 
# 
# # Transform graphs
# 
# #class Pipeline(Transform):
# #    pass