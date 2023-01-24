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

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dask import distributed
from pandas.testing import assert_frame_equal
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
            .winsorize(0.05, cols)
            .z_score(cols)
            .impute_constant(0.0, cols)
            .clip(upper=2, lower=-2, cols=cols)
        )

    model = (
        ff.DataFramePipeline()[FEATURES + ["{response_col}"]]
        .copy("{response_col}", "{response_col}_train")
        .pipe(np.log1p, ["carat", "{response_col}_train"])
        .winsorize(0.05, "{response_col}_train")
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
        .pipe(np.expm1, "{response_col}_hat_dollars")
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

    result_pandas = pipeline.apply(**bindings)

    result_dask = dask.apply(pipeline, **bindings).result()
    assert_frame_equal(result_pandas, result_dask)

    result_dask = dask.fit(pipeline, **bindings).apply()
    assert_frame_equal(result_pandas, result_dask)

    # FIXME
    result_dask = dask.apply(pipeline.fit(**bindings)).result()
    assert_frame_equal(result_pandas, result_dask)

    result_dask = dask.apply(dask.fit(pipeline, **bindings)).result()
    assert_frame_equal(result_pandas, result_dask)


@pytest.mark.dask
def test_parallelized_pipeline_1(
    diamonds_df: pd.DataFrame, dask: ff.DaskBackend, tmp_path
) -> None:
    from sklearn.linear_model import LinearRegression

    FEATURES = ["carat", "x", "y", "z", "depth", "table"]

    def bake_features(cols) -> ff.DataFramePipeline:
        return (
            ff.DataFramePipeline(tag="bake_features")
            .winsorize(0.05, cols)
            .z_score(cols)
            .impute_constant(0.0, cols)
            .clip(upper=2, lower=-2, cols=cols)
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
        .winsorize(0.05, "{response_col}_train")
        .pipe(np.log1p, ["carat", "{response_col}_train"])
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
        .pipe(np.expm1, "{response_col}_hat_dollars")
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

    local_result = complex_pipeline.apply(diamonds_df, bindings)
    dask_result = dask.apply(complex_pipeline, diamonds_df, bindings).result()

    assert_frame_equal(local_result, dask_result)

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
    local_result = pip.apply(**bindings)
    dask_result = dask.apply(pip, **bindings).result()
    assert_frame_equal(local_result, dask_result)
