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

from __future__ import annotations

from dask import distributed
import numpy as np
import pandas as pd
import pytest
from pydataset import data  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore


import frankenfit as ff


@pytest.fixture
def diamonds_df():
    return data("diamonds")


@pytest.fixture(scope="module")
def dask_client():
    # spin up a local cluster and client
    cluster = distributed.LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        scheduler_port=0,
        dashboard_address=":0",
    )
    client = distributed.Client(cluster)
    yield client
    # Shutting down takes too long :(
    # client.shutdown()
    # client.close()


@pytest.fixture
def dask(dask_client):
    return ff.DaskBackend(dask_client)


def test_pipeline_straight(
    diamonds_df: pd.DataFrame, dask: ff.DaskBackend, tmp_path
) -> None:
    FEATURES = ["carat", "x", "y", "z", "depth", "table"]

    def bake_features(cols):
        return (
            ff.DataFramePipeline()
            .print(fit_msg=f"Baking: {cols}")
            .winsorize(cols, limit=0.05)
            .z_score(cols)
            .impute_constant(cols, 0.0)
            .clip(cols, upper=2, lower=-2)
        )

    model = (
        ff.DataFramePipeline()[FEATURES + ["{response_col}"]]
        .copy("{response_col}", "{response_col}_train")
        .winsorize("{response_col}_train", limit=0.05)
        .pipe(["carat", "{response_col}_train"], np.log1p)
        .then(bake_features(FEATURES))
        .sk_learn(
            LinearRegression,
            # x_cols=["carat", "depth", "table"],
            x_cols=ff.HP("predictors"),
            response_col="{response_col}_train",
            hat_col="{response_col}_hat",
            class_params={"fit_intercept": True},
        )
        # transform {response_col}_hat from log-dollars back to dollars
        .copy("{response_col}_hat", "{response_col}_hat_dollars")
        .pipe("{response_col}_hat_dollars", np.expm1)
    )

    # write data to csv
    df = diamonds_df.reset_index().set_index("index")
    path = str(tmp_path / "diamonds.csv")
    ff.DataFramePipeline().write_pandas_csv(
        path,
        index_label="index",
    ).apply(df)

    pipeline = (
        ff.DataFramePipeline()
        .read_dataset(path, format="csv", index_col="index")
        .then(model)
    )
    bindings = {
        "response_col": "price",
        "bake_features": True,
        "predictors": FEATURES,
    }

    result_pandas = pipeline.apply(bindings=bindings)

    result_dask = pipeline.apply(bindings=bindings, backend=dask).result()
    assert result_pandas.equals(result_dask)

    result_dask = pipeline.fit(bindings=bindings, backend=dask).apply()
    assert result_pandas.equals(result_dask)

    result_dask = pipeline.fit(bindings=bindings).apply(backend=dask).result()
    assert result_pandas.equals(result_dask)

    result_dask = (
        pipeline.fit(bindings=bindings, backend=dask).apply(backend=dask).result()
    )
    assert result_pandas.equals(result_dask)
