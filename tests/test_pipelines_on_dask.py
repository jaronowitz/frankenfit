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

import numpy as np
import pandas as pd
import pytest
from dask import distributed
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


@pytest.mark.dask
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
        .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
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

    result_dask = dask.apply(pipeline, bindings=bindings).result()
    assert result_pandas.equals(result_dask)

    result_dask = dask.fit(pipeline, bindings=bindings).apply()
    assert result_pandas.equals(result_dask)

    result_dask = dask.apply(pipeline.fit(bindings=bindings)).result()
    assert result_pandas.equals(result_dask)

    result_dask = dask.apply(dask.fit(pipeline, bindings=bindings)).result()
    assert result_pandas.equals(result_dask)


@pytest.mark.dask
def test_parallelized_pipeline_1(
    diamonds_df: pd.DataFrame, dask: ff.DaskBackend, tmp_path
) -> None:
    from sklearn.linear_model import LinearRegression

    FEATURES = ["carat", "x", "y", "z", "depth", "table"]

    def bake_features(cols) -> ff.DataFramePipeline:
        return (
            ff.DataFramePipeline(tag="bake_features")
            .winsorize(cols, limit=0.05)
            .z_score(cols)
            .impute_constant(cols, 0.0)
            .clip(cols, upper=2, lower=-2)
        )

    # per-cut feature means
    per_cut_means = (
        ff.DataFramePipeline(tag="per_cut_means")
        .group_by_cols(["cut"])
        .then(
            ff.DataFramePipeline()[ff.HP("predictors")].stateful_lambda(
                fit_fun=lambda df: df.mean(),
                apply_fun=lambda df, mean: mean.rename(lambda c: f"cut_mean_{c}"),
            )
        )
    )

    complex_pipeline = (
        ff.DataFramePipeline()
        .select(FEATURES + ["{response_col}", "cut"])
        .copy("{response_col}", "{response_col}_train")
        .winsorize("{response_col}_train", limit=0.05)
        .pipe(["carat", "{response_col}_train"], np.log1p)
        .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
        .join(per_cut_means, how="left", on="cut")
        .sk_learn(
            LinearRegression,
            # x_cols=["carat", "depth", "table"],
            x_cols=ff.HPLambda(
                lambda bindings: bindings["predictors"]
                + [f"cut_mean_{c}" for c in bindings["predictors"]]
            ),
            response_col="{response_col}_train",
            hat_col="{response_col}_hat",
            class_params={"fit_intercept": True},
        )
        # transform {response_col}_hat from log-dollars back to dollars
        .copy("{response_col}_hat", "{response_col}_hat_dollars")
        .pipe("{response_col}_hat_dollars", np.expm1)
    )

    assert complex_pipeline.hyperparams() == {
        "response_col",
        "bake_features",
        "predictors",
    }
    bindings = {
        "response_col": "price",
        "bake_features": True,
        "predictors": ["carat", "x", "y", "z", "depth", "table"],
    }

    local_result = complex_pipeline.apply(diamonds_df, bindings=bindings)
    assert local_result.equals(
        dask.apply(complex_pipeline, diamonds_df, bindings=bindings).result()
    )

    pip = (
        ff.DataFramePipeline()
        .group_by_bindings(
            [
                {"predictors": ["carat"]},
                {"predictors": ["depth"]},
                {"predictors": ["table"]},
            ],
        )
        .then(
            ff.ReadDataFrame(diamonds_df)
            .then(complex_pipeline)
            .correlation(["{response_col}_hat_dollars"], ["{response_col}"])
        )
    )
    bindings = {"response_col": "price", "bake_features": True}
    local_result = pip.apply(bindings=bindings)
    assert local_result.equals(dask.apply(pip, bindings=bindings).result())
