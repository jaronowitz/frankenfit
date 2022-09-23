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
Transforms related to reading and writing stored data.
"""

from __future__ import annotations

import logging
from attrs import define, field
import pandas as pd

from typing import Optional

from pyarrow import dataset

from .core import StatelessTransform, HP, fmt_str_field, columns_field, HPCols

from .transforms import (
    Identity,
)

_LOG = logging.getLogger(__name__)


# A DataReader is nothing more than a constant, stateless transform, duh.
@define
class DataReader(StatelessTransform):
    is_constant = True


@define
class ReadDataFrame(DataReader):
    df: pd.DataFrame

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return self.df


@define
class ReadPandasCSV(DataReader):
    filepath: str | HP = fmt_str_field()
    read_csv_args: Optional[dict] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return pd.read_csv(self.filepath, **(self.read_csv_args or {}))


@define
class WritePandasCSV(Identity):
    path: str | HP = fmt_str_field()
    index_label: str | HP = fmt_str_field()
    to_csv_kwargs: Optional[dict] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None):
        df_apply.to_csv(
            self.path, index_label=self.index_label, **(self.to_csv_kwargs or {})
        )
        return df_apply


@define
class ReadDataset(DataReader):
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
