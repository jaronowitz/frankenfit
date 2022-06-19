from __future__ import annotations

from typing import Callable
from attrs import define, field
import pandas as pd

from . import core
from .core import hp, not_empty, columns_tuple_converter

class Identity(core.Transform):
    """
    The stateless Transform that, at apply-time, simply returns the input
    data unaltered.
    """
    def apply(self, X_apply: pd.DataFrame) -> pd.DataFrame:
        X_apply = super().apply(X_apply)
        return X_apply

@define
class CopyColumns(core.ColumnsTransform):
    """
    A stateless Transform that copies values from one or more source columns
    into corresponding destination columns, either creating them or overwriting
    their contents.
    """
    dest_cols: tuple[str | hp] = field(converter=columns_tuple_converter)

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

    def apply(self, X_apply: pd.DataFrame) -> pd.DataFrame:
        X_apply = super().apply(X_apply)
        if len(self.cols) == 1:
            src_col = self.cols[0]
            return X_apply.assign(**{
                dest_col: X_apply[src_col] for dest_col in self.dest_cols
            })

        return X_apply.assign(**{
            dest_col: X_apply[src_col] for src_col, dest_col in
                zip(self.cols, self.dest_cols)
        })

@define
class RenameColumns(core.ColumnsTransform):
    how: hp | Callable | dict[str | hp, str | hp]

class KeepColumns(core.ColumnsTransform):
    pass

class DropColumns(core.ColumnsTransform):
    pass


# Transform graphs

#class Pipeline(Transform):
#    pass