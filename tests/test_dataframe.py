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

from os import path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds  # type: ignore
import pytest
from pydataset import data  # type: ignore

import frankenfit as ff
import frankenfit.core as ffc
import frankenfit.dataframe as ffdf
from frankenfit.core import LocalFuture


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_then(diamonds_df: pd.DataFrame):
    t: ff.DataFramePipeline = (
        ff.ReadDataFrame(diamonds_df).then().z_score(["price"]).clip(-2, 2, ["price"])
    )
    assert isinstance(t, ff.DataFramePipeline)


def test_ColumnsTransform(diamonds_df: pd.DataFrame):
    df = diamonds_df
    # test cols behavior
    # the simplest concrete ColumnsTransform is Select
    t = ffdf.Select(["x", "y", "z"])
    assert t.apply(df).equals(df[["x", "y", "z"]])
    t = ffdf.Select("z")  # type: ignore [arg-type]
    assert t.apply(df).equals(df[["z"]])
    t = ffdf.Select(ff.HP("which_cols"))
    assert t.fit(df, which_cols=["x", "y", "z"]).apply(df).equals(df[["x", "y", "z"]])

    bindings = {"some_col": "y"}
    assert ff.HPCols(cols=["x", "y", "z"]).resolve(bindings) == ["x", "y", "z"]
    assert ff.HPCols(cols=["x", ff.HP("some_col"), "z"]).resolve(bindings) == [
        "x",
        "y",
        "z",
    ]
    assert ff.HPCols(cols=["x", "{some_col}", "z"]).resolve(bindings) == [
        "x",
        "y",
        "z",
    ]

    t = ffdf.Select(["x", ff.HP("some_col"), "z"])  # type: ignore [list-item]
    assert t.fit(df, bindings).apply(df).equals(df[["x", "y", "z"]])
    t = ffdf.Select(["x", "{some_col}", "z"])
    assert t.fit(df, bindings).apply(df).equals(df[["x", "y", "z"]])


def test_DeMean(diamonds_df: pd.DataFrame):
    cols = ["price", "x", "y", "z"]
    t = ff.DataFramePipeline().de_mean(cols)
    result = t.fit(diamonds_df).apply(diamonds_df)
    assert (result[cols].mean().abs() < 1e-10).all()

    df = pd.DataFrame(
        {
            "col1": pd.Series([1.0, np.nan, 2.0]),
            "col2": pd.Series([0.3, 0.3, 0.4]),
        }
    )
    wmean = (0.3 / 0.7) * 1 + (0.4 / 0.7) * 2
    assert (
        ff.DataFramePipeline()
        .de_mean(["col1"], w_col="col2")
        .apply(df)["col1"]
        .equals(df["col1"] - wmean)
    )


def test_CopyColumns(diamonds_df: pd.DataFrame):
    cols = ["price", "x", "y", "z"]
    df = diamonds_df[cols]
    result = ff.DataFramePipeline().copy(["price"], ["price_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    # optional list literals for lists of 1
    result = ff.DataFramePipeline().copy("price", "price_copy").apply(df)
    assert result["price_copy"].equals(df["price"])

    result = (
        ff.DataFramePipeline().copy(["price"], ["price_copy1", "price_copy2"]).apply(df)
    )
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])
    # optional list literals for lists of 1
    result = (
        ff.DataFramePipeline().copy("price", ["price_copy1", "price_copy2"]).apply(df)
    )
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])

    result = (
        ff.DataFramePipeline().copy(["price", "x"], ["price_copy", "x_copy"]).apply(df)
    )
    assert result["price_copy"].equals(df["price"])
    assert result["x_copy"].equals(df["x"])

    with pytest.raises(ValueError):
        result = (
            ff.DataFramePipeline()
            .copy(
                ["price", "x"],
                [
                    "price_copy",
                ],
            )
            .apply(df)
        )

    # with hyperparams
    bindings = {"response": "price"}
    result = (
        ff.DataFramePipeline()
        .copy(["{response}"], ["{response}_copy"])
        .apply(df, bindings)
    )
    assert result["price_copy"].equals(df["price"])

    result = (
        ff.DataFramePipeline().copy("{response}", "{response}_copy").apply(df, bindings)
    )
    assert result["price_copy"].equals(df["price"])

    result = (
        ff.DataFramePipeline()
        .copy([ff.HP("response")], "{response}_copy")
        .fit(df, bindings)
        .apply(df)
    )
    assert result["price_copy"].equals(df["price"])

    with pytest.raises(TypeError):
        # HP("response") resolves to a str, not a list of str
        _ = (
            ff.DataFramePipeline()
            .copy(ff.HP("response"), "{response}_copy")
            .apply(df, bindings)
        )
    with pytest.raises(TypeError):
        # HP("dest") resolves to a str, not a list of str
        _ = (
            ff.DataFramePipeline()
            .copy(["price"], ff.HP("dest"))
            .apply(df, dest="price_copy")
        )


def test_Select(diamonds_df: pd.DataFrame):
    kept = ["price", "x", "y", "z"]
    result = ff.DataFramePipeline().select(kept).apply(diamonds_df)
    assert result.equals(diamonds_df[kept])
    result = ff.DataFramePipeline()[kept].apply(diamonds_df)
    assert result.equals(diamonds_df[kept])


def test_Filter(diamonds_df: pd.DataFrame):
    ideal_df = (ff.DataFramePipeline().filter(lambda df: df["cut"] == "Ideal")).apply(
        diamonds_df
    )
    assert (ideal_df["cut"] == "Ideal").all()

    # taking a hyperparam
    pip = ff.DataFramePipeline().filter(lambda df, which_cut: df["cut"] == which_cut)
    assert pip.hyperparams() == {"which_cut"}
    for which_cut in ("Premium", "Good"):
        result_df = pip.apply(diamonds_df, which_cut=which_cut)
        assert (result_df["cut"] == which_cut).all()

    with pytest.raises(ff.UnresolvedHyperparameterError):
        pip.apply(diamonds_df)
    with pytest.raises(ff.UnresolvedHyperparameterError):
        pip.apply(diamonds_df, irrelevant=100)

    with pytest.raises(TypeError):
        ff.DataFramePipeline().filter(lambda df, *args: df["cut"] == "Ideal")

    # hyperparam with default value
    pip = ff.DataFramePipeline().filter(
        lambda df, which_cut="Ideal": df["cut"] == which_cut
    )
    assert pip.hyperparams() == {"which_cut"}
    result_df = pip.apply(diamonds_df)
    assert (result_df["cut"] == "Ideal").all()
    result_df = pip.apply(diamonds_df, which_cut="Good")
    assert (result_df["cut"] == "Good").all()

    with pytest.raises(TypeError):
        ff.DataFramePipeline().filter(ff.HP("foo"))  # type: ignore [arg-type]


def test_RenameColumns(diamonds_df: pd.DataFrame):
    result = ff.DataFramePipeline().rename({"price": "price_orig"}).apply(diamonds_df)
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))
    result = (
        ff.DataFramePipeline()
        .rename(lambda c: c + "_orig" if c == "price" else c)
        .apply(diamonds_df)
    )
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))

    result = (
        ff.DataFramePipeline()
        .rename(ff.HPLambda(lambda b: {b["response"]: b["response"] + "_orig"}))
        .apply(diamonds_df, response="price")
    )
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))


def test_affixes(diamonds_df: pd.DataFrame):
    df = (
        ff.DataFramePipeline().affix("pre_", "_post", cols=["price"]).apply(diamonds_df)
    )
    assert df.equals(diamonds_df.rename(columns={"price": "pre_price_post"}))
    df = ff.DataFramePipeline().affix("pre_", "_post").apply(diamonds_df)
    assert all(c.startswith("pre_") and c.endswith("_post") for c in df.columns)

    df = ff.DataFramePipeline().prefix("pre_", cols=["price"]).apply(diamonds_df)
    assert df.equals(diamonds_df.rename(columns={"price": "pre_price"}))
    df = ff.DataFramePipeline().prefix("pre_").apply(diamonds_df)
    assert all(c.startswith("pre_") for c in df.columns)

    df = ff.DataFramePipeline().suffix("_suf", cols=["price"]).apply(diamonds_df)
    assert df.equals(diamonds_df.rename(columns={"price": "price_suf"}))
    df = ff.DataFramePipeline().suffix("_suf").apply(diamonds_df)
    assert all(c.endswith("_suf") for c in df.columns)


def test_Clip(diamonds_df: pd.DataFrame):
    df = diamonds_df
    result = ff.DataFramePipeline().clip(upper=150, lower=100, cols=["price"]).apply(df)
    assert (result["price"] <= 150).all() & (result["price"] >= 100).all()
    clip_price = ff.DataFramePipeline().clip(
        upper=ff.HP("upper"), lower=None, cols=["price"]
    )
    for upper in (100, 200, 300):
        result = clip_price.apply(df, upper=upper)
        assert (result["price"] <= upper).all()
    clip_price = ff.DataFramePipeline().clip(
        lower=ff.HP("lower"), upper=None, cols=["price"]
    )
    for lower in (100, 200, 300):
        result = clip_price.apply(df, lower=lower)
        assert (result["price"] >= lower).all()


def test_ImputeConstant() -> None:
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        ff.DataFramePipeline()
        .impute_constant(0.0, ["col1"])
        .apply(df)
        .equals(df.fillna(0.0))
    )


def test_Winsorize() -> None:
    df = pd.DataFrame({"col1": [float(x) for x in range(1, 101)]})
    result = ff.DataFramePipeline().winsorize(0.2, ["col1"]).fit(df).apply(df)
    assert (result["col1"] > 20).all() and (result["col1"] < 81).all()

    # limits out of bounds
    with pytest.raises(ValueError):
        ff.DataFramePipeline().winsorize(-0.2, ["col1"]).fit(df)
    with pytest.raises(ValueError):
        ff.DataFramePipeline().winsorize(1.2, ["col1"]).fit(df)
    with pytest.raises(TypeError):
        ff.DataFramePipeline().winsorize(ff.HP("limit"), ["col1"]).fit(
            df, limit="a"  # non-float
        )


def test_ImputeMean() -> None:
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        ff.DataFramePipeline()
        .impute_mean(["col1"])
        .fit(df)
        .apply(df)
        .equals(pd.DataFrame({"col1": pd.Series([1.0, 1.5, 2.0])}))
    )
    # with weights
    df = pd.DataFrame(
        {
            "col1": pd.Series([1.0, np.nan, 2.0]),
            "col2": pd.Series([0.1, np.nan, 0.9]),
        }
    )
    assert (
        ff.DataFramePipeline()
        .impute_mean(["col1"], w_col="col2")
        .apply(df)["col1"]
        .equals(pd.Series([1.0, 0.1 * 1 + 0.9 * 2, 2.0]))
    )
    # with weights (ignoring weights of missing obs)
    df = pd.DataFrame(
        {
            "col1": pd.Series([1.0, np.nan, 2.0]),
            "col2": pd.Series([0.3, 0.3, 0.4]),
        }
    )
    assert (
        ff.DataFramePipeline()
        .impute_mean(["col1"], w_col="col2")
        .apply(df)["col1"]
        .equals(pd.Series([1.0, (0.3 / 0.7) * 1 + (0.4 / 0.7) * 2, 2.0]))
    )


def test_ZScore(diamonds_df: pd.DataFrame):
    result = (
        ff.DataFramePipeline().z_score(["price"]).fit(diamonds_df).apply(diamonds_df)
    )
    assert result["price"].equals(
        (diamonds_df["price"] - diamonds_df["price"].mean())
        / diamonds_df["price"].std()
    )

    df = pd.DataFrame(
        {
            "col1": pd.Series([1.0, np.nan, 2.0]),
            "col2": pd.Series([0.3, 0.3, 0.4]),
        }
    )
    wmean = (0.3 / 0.7) * 1 + (0.4 / 0.7) * 2
    assert (
        ff.DataFramePipeline()
        .z_score(["col1"], w_col="col2")
        .apply(df)["col1"]
        .equals((df["col1"] - wmean) / df["col1"].std())
    )


def test_Join(diamonds_df: pd.DataFrame):
    diamonds_df = diamonds_df.assign(diamond_id=diamonds_df.index)
    xyz_df = diamonds_df[["diamond_id", "x", "y", "z"]]
    cut_df = diamonds_df[["diamond_id", "cut"]]
    target = pd.merge(xyz_df, cut_df, how="left", on="diamond_id")

    t = ffdf.Join(
        ffdf.ReadDataFrame(xyz_df),
        ffdf.ReadDataFrame(cut_df),
        how="left",
        on="diamond_id",
    )
    result = t.fit().apply()
    assert result.equals(target)
    # assert result.equals(diamonds_df[["diamond_id", "x", "y", "z", "cut"]])

    p = ff.DataFramePipeline(
        transforms=[
            ffdf.Join(
                ffdf.ReadDataFrame(xyz_df),
                ffdf.ReadDataFrame(cut_df),
                how="left",
                on="diamond_id",
            )
        ]
    )
    result = p.apply()
    assert result.equals(target)

    p = (
        ff.DataFramePipeline()
        .read_data_frame(xyz_df)
        .join(
            ff.DataFramePipeline().read_data_frame(cut_df), how="left", on="diamond_id"
        )
    )
    result = p.apply()
    assert result.equals(target)

    deviances = (
        ff.DataFramePipeline()[["cut", "price"]]
        .join(
            (
                ff.DataFramePipeline()
                .group_by_cols("cut")
                .stateless_lambda(lambda df: df[["price"]].mean())
                .rename({"price": "mean_price"})
            ),
            on="cut",
            how="left",
        )
        .stateless_lambda(
            lambda df: df.assign(price_deviance=df["price"] - df["mean_price"])
        )
    )
    result = deviances.apply(diamonds_df)
    assert np.abs(result["price_deviance"].mean()) < 1e-10


def test_SKLearn(diamonds_df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression  # type: ignore

    target_preds = (
        LinearRegression(fit_intercept=True)
        .fit(diamonds_df[["carat", "depth", "table"]], diamonds_df["price"])
        .predict(diamonds_df[["carat", "depth", "table"]])
    )
    target = diamonds_df.assign(price_hat=target_preds)

    sk = ff.DataFramePipeline().sk_learn(
        LinearRegression,
        ["carat", "depth", "table"],
        "price",
        "price_hat",
        class_params={"fit_intercept": True},
    )
    result = sk.fit(diamonds_df).apply(diamonds_df)
    assert result.equals(target)

    # with sample weight
    target_preds = (
        LinearRegression(fit_intercept=True)
        .fit(
            diamonds_df[["depth", "table"]],
            diamonds_df["price"],
            sample_weight=diamonds_df["carat"],
        )
        .predict(diamonds_df[["depth", "table"]])
    )
    target = diamonds_df.assign(price_hat=target_preds)
    sk = ff.DataFramePipeline().sk_learn(
        LinearRegression,
        ["depth", "table"],
        "price",
        "price_hat",
        w_col="carat",
        class_params={"fit_intercept": True},
    )
    result = sk.fit(diamonds_df).apply(diamonds_df)
    assert result.equals(target)

    # TODO: test hyperparameterizations


def test_Statsmodels(diamonds_df: pd.DataFrame):
    from statsmodels.api import OLS  # type: ignore

    ols = OLS(diamonds_df["price"], diamonds_df[["carat", "depth", "table"]])
    target_preds = ols.fit().predict(diamonds_df[["carat", "depth", "table"]])
    target = diamonds_df.assign(price_hat=target_preds)

    sm = ff.DataFramePipeline().statsmodels(
        OLS,
        ["carat", "depth", "table"],
        "price",
        "price_hat",
    )
    result = sm.fit(diamonds_df).apply(diamonds_df)
    assert result.equals(target)

    # TODO: test w_col
    # TODO: test hyperparameterizations


def test_complex_pipeline_1(diamonds_df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression

    FEATURES = ["carat", "x", "y", "z", "depth", "table"]

    def bake_features(cols):
        return (
            ff.DataFramePipeline()
            .print(fit_msg=f"Baking: {cols}")
            .winsorize(limit=0.05, cols=cols)
            .z_score(cols)
            .impute_constant(0.0, cols)
            .clip(upper=2, lower=-2, cols=cols)
        )

    pipeline = (
        ff.DataFramePipeline()[FEATURES + ["{response_col}"]]
        .copy("{response_col}", "{response_col}_train")
        .winsorize(0.05, "{response_col}_train")
        .pipe(np.log1p, ["carat", "{response_col}_train"])
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

    assert pipeline.hyperparams() == {"bake_features", "predictors", "response_col"}

    # should be picklable without errors
    import cloudpickle  # type: ignore

    assert cloudpickle.loads(cloudpickle.dumps(pipeline)) == pipeline

    # should visualize without error
    pipeline.visualize()

    # TODO: test more stuff with this pipeline


def test_GroupBy(diamonds_df: pd.DataFrame):
    df: pd.DataFrame = diamonds_df.reset_index().drop(["index"], axis=1)
    target_s = df.groupby("cut", as_index=False, sort=False).apply(len)
    pip = ff.DataFramePipeline().stateless_lambda(len)
    assert pip.fit(df).apply(df) == len(df)

    result = ffdf.GroupByCols("cut", pip).fit(df).apply(df)
    assert result.equals(target_s)

    pip = ff.DataFramePipeline().group_by_cols("cut").stateless_lambda(len)
    assert pip.fit(df).apply(df).equals(target_s)

    # A sttaeful transform
    pip = (
        ff.DataFramePipeline().group_by_cols("cut").de_mean(["price"])[["cut", "price"]]
    )
    result = pip.apply(df)
    assert np.abs(result["price"].mean()) < 1e-10
    assert all(np.abs(result.groupby("cut")["price"].mean()) < 1e-10)

    # "cross-validated" de-meaning hah
    pip = (
        ff.DataFramePipeline()
        .group_by_cols("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
        .de_mean(["price"])[["cut", "price"]]
    )
    result = pip.apply(df)
    assert all(np.abs(result.groupby("cut")["price"].mean()) > 4)

    pip = (
        ff.DataFramePipeline()
        .group_by_cols("cut")
        .stateless_lambda(lambda df: df[["price"]].mean())
    )
    result = pip.apply(df).set_index("cut").sort_index().reset_index()
    target: pd.DataFrame = (
        df.groupby("cut")[["price"]].mean().sort_index().reset_index()
    )
    assert result.equals(target)

    pip = (
        ff.DataFramePipeline()
        .group_by_cols("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
        .de_mean("price")[["cut", "price"]]
    )
    result = pip.apply(df)
    cuts = pd.Series(df["cut"].unique(), name="cut")
    cut_means = pd.DataFrame(
        dict(cut=cuts, price=cuts.map(lambda v: df.loc[df["cut"] != v]["price"].mean()))
    )
    target = df.merge(cut_means, how="left", on="cut", suffixes=("", "_mean")).assign(
        price=lambda df: df["price"] - df["price_mean"]
    )[["cut", "price"]]
    assert result.equals(target)

    pip = ff.DataFramePipeline().group_by_cols("cut").de_mean(["price"])
    with pytest.raises(ff.UnfitGroupError):
        pip.fit(df.loc[df["cut"] != "Fair"]).apply(df)


def test_Correlation(diamonds_df: pd.DataFrame):
    target = diamonds_df[["price", "carat"]].corr()
    cm = ff.DataFramePipeline().correlation(["price"], ["carat"]).apply(diamonds_df)
    assert cm.iloc[0, 0] == target.iloc[0, 1]


def test_ReadDataFrame(diamonds_df: pd.DataFrame):
    df = diamonds_df.reset_index().drop(["index"], axis=1)
    assert ff.DataFramePipeline().read_data_frame(df).apply().equals(df)

    # another way to start a pipeline with a reader
    pip = ff.ReadDataFrame(df).then(ff.Identity())
    assert pip.apply().equals(df)
    pip = ff.ReadDataFrame(df).then().identity()
    assert pip.apply().equals(df)


def test_ReadPandasCSV(diamonds_df: pd.DataFrame, tmp_path: str):
    df = diamonds_df.reset_index().drop(["index"], axis=1)
    fp = path.join(tmp_path, "diamonds.csv")
    df.to_csv(fp)

    with pytest.warns(ffc.NonInitialConstantTransformWarning):
        ff.ReadPandasCSV(fp).apply(df)

    result = ff.DataFramePipeline().read_pandas_csv(fp, dict(index_col=0)).apply()
    assert result.equals(df)

    with pytest.warns(ffc.NonInitialConstantTransformWarning):
        pip = ff.DataFramePipeline()[["price"]].read_pandas_csv(fp)

    with pytest.warns(ffc.NonInitialConstantTransformWarning):
        pip.apply(df)

    with pytest.warns(ffc.NonInitialConstantTransformWarning):
        fit = pip.fit(df)

    with pytest.warns(ffc.NonInitialConstantTransformWarning):
        fit.apply(df)

    # by default ReadPandasCSV is pure, unless the no_cache param is given
    assert ff.ReadPandasCSV(fp).pure
    assert not ff.ReadPandasCSV(fp, no_cache=True).pure


def test_read_write_csv(diamonds_df: pd.DataFrame, tmp_path):
    df = diamonds_df.reset_index().set_index("index")
    ff.DataFramePipeline().write_pandas_csv(
        # TODO: in core, define a field type for pathlib.PosixPath's containing
        # hyperparameter format strings
        str(tmp_path / "diamonds.csv"),
        index_label="index",
    ).apply(df)

    result = (
        ff.DataFramePipeline()
        .read_pandas_csv(str(tmp_path / "diamonds.csv"), dict(index_col="index"))
        .apply()
    )
    assert result.equals(df)

    result = (
        ff.DataFramePipeline()
        .read_dataset(str(tmp_path / "diamonds.csv"), format="csv", index_col="index")
        .apply()
    )
    assert result.equals(df)


def test_read_write_dataset(diamonds_df: pd.DataFrame, tmp_path):
    df = diamonds_df.reset_index().set_index("index")
    path = str(tmp_path / "diamonds.csv")
    ff.DataFramePipeline().write_pandas_csv(
        path,
        index_label="index",
    ).apply(df)

    target = df.loc[3:6]

    result = (
        ff.DataFramePipeline()
        .read_dataset(
            path,
            format="csv",
            filter=(ds.field("index") > 2) & (ds.field("index") < 7),
            index_col="index",
        )
        .apply()
    )
    assert result.equals(target)

    bindings = {"filter": (ds.field("index") > 2) & (ds.field("index") < 7)}

    result = (
        ff.DataFramePipeline()
        .read_dataset(
            path,
            format="csv",
            filter=ff.HP("filter"),
            index_col="index",
        )
        .fit(**bindings)
        .apply()
    )
    assert result.equals(target)

    # by default ReadPandasCSV is pure, unless the no_cache param is given
    assert ff.ReadDataset(
        [path],
        ["foo"],
        format="csv",
        filter=ff.HP("filter"),
        index_col="index",
    ).pure
    assert not ff.ReadDataset(
        [path],
        ["foo"],
        format="csv",
        filter=ff.HP("filter"),
        index_col="index",
        no_cache=True,
    ).pure


def test_write_read_dataset(diamonds_df: pd.DataFrame, tmp_path):
    (ff.DataFramePipeline().write_dataset(str(tmp_path / "test.dataset"))).apply(
        diamonds_df
    )

    df = (
        ff.DataFramePipeline().read_dataset(
            str(tmp_path / "test.dataset"),
        )
    ).apply()

    assert diamonds_df.equals(df)


def test_Assign(diamonds_df: pd.DataFrame):
    do = ff.DataFramePipeline()
    other_pipeline = do[["price", "carat"]].winsorize(0.1).suffix("_win")
    backend = ff.LocalBackend()

    pipeline = do.assign(
        # multi-column assigments
        do[["price", "carat"]].de_mean().suffix("_dmn"),  # pipeline
        backend.apply(other_pipeline, diamonds_df),  # future
        # lambda is wrapped in a StatelessLambda transform
        lambda df: pd.DataFrame().assign(uppercut=df["cut"].str.upper()),
        # named column assignments: transforms with 1-column output
        price_dmn2=do["price"].de_mean(),
        price_win2=backend.apply(other_pipeline["price_win"], diamonds_df),  # future
        # lambda is wrapped in a StatelessLambda transform
        price_rank=lambda df, price_scale=1.0: price_scale
        * ((df["price"] - df["price"].min()) / (df["price"].max() - df["price"].min())),
        intercept=1.0,  # scalar
    )
    assert pipeline.hyperparams() == {"price_scale"}
    result = pipeline.on_backend(backend).apply(diamonds_df, price_scale=2.0)
    assert set(result.columns) == set(diamonds_df.columns).union(
        [
            "price_dmn",
            "carat_dmn",
            "price_win",
            "carat_win",
            "uppercut",
            "price_dmn2",
            "price_win2",
            "price_rank",
            "intercept",
        ]
    )

    fit = pipeline.on_backend(backend).fit(diamonds_df, price_scale=2.0)
    fit.materialize_state()

    with pytest.raises(ValueError):
        # too many columns for a named assignment
        do.assign(foo=do[["price", "carat"]]).apply(diamonds_df)


def test_Assign_child_state(diamonds_df: pd.DataFrame):
    do = ff.DataFramePipeline()
    pipeline = do.assign(price_dmn=do["price"].de_mean())[["price_dmn"]]

    df1 = diamonds_df.sample(1000)
    df2 = diamonds_df.sample(5000)

    fit = pipeline.fit(df1)
    own_result = fit.apply(df1)
    other_result = fit.apply(df2)

    assert own_result["price_dmn"].equals(df1["price"] - df1["price"].mean())
    # assert we subtracted the first dataframe's mean
    assert other_result["price_dmn"].equals(df2["price"] - df1["price"].mean())


def test_GroupByBindings(diamonds_df: pd.DataFrame):
    df = diamonds_df.head()
    result = (
        ff.DataFramePipeline()
        .group_by_bindings(
            [
                {"target_col": "price"},
                {"target_col": "depth"},
                {"target_col": "table"},
            ],
            as_index=True,
        )
        .select(["{target_col}"])
    ).apply(df)

    target = pd.concat(
        [
            df[["price"]].assign(target_col="price"),
            df[["depth"]].assign(target_col="depth"),
            df[["table"]].assign(target_col="table"),
        ],
        axis=0,
    ).set_index("target_col")

    assert result.equals(target)


def test_Drop(diamonds_df: pd.DataFrame):
    result = ff.DataFramePipeline().drop(["price"]).apply(diamonds_df)
    assert result.equals(diamonds_df.drop(["price"], axis=1))


def test_Pipe(diamonds_df: pd.DataFrame):
    result = (
        ff.DataFramePipeline().pipe(np.log1p, ["carat", "price"]).apply(diamonds_df)
    )
    assert (result["carat"] == np.log1p(diamonds_df["carat"])).all()
    assert (result["price"] == np.log1p(diamonds_df["price"])).all()


def test_empty_dataframe_pipeline(diamonds_df: pd.DataFrame):
    df = diamonds_df
    empty_df = pd.DataFrame()
    pip = ff.DataFramePipeline()
    fit = pip.fit()

    # data_apply is not None, backend is None: identity
    assert pip.apply(df).equals(df)
    assert fit.apply(df).equals(df)

    # data_apply is None, backend is None: empty_constructor() -> empty_df
    assert pip.apply().equals(empty_df)
    assert fit.apply().equals(empty_df)

    # data_apply is not None, backend is not None: future identity
    local = ff.LocalBackend()
    assert local.apply(pip, df).result().equals(df)
    assert local.apply(fit, df).result().equals(df)

    # data_apply is None, backend is not None: future empty_constructor() ->
    # future empty_df
    assert local.apply(pip).result().equals(empty_df)
    assert local.apply(fit).result().equals(empty_df)

    # data_apply is future not None, backend is None: identity
    assert pip.apply(LocalFuture(df)).equals(df)
    assert fit.apply(LocalFuture(df)).equals(df)

    # data_apply is future None, backend is None: future empty_constructor() ->
    # future empty_df. This is actually an ill-formed call according to
    # typechecker but we test it anyway
    # assert pip.apply(LocalFuture(None)).equals(empty_df)  # type: ignore [arg-type]
    # assert fit.apply(LocalFuture(None)).equals(empty_df)  # type: ignore [arg-type]

    # data_apply is future not None, backend is not None: future identity
    assert local.apply(pip, LocalFuture(df)).result().equals(df)
    assert local.apply(fit, LocalFuture(df)).result().equals(df)


def test_ALL_COLS(diamonds_df: pd.DataFrame):
    # Mypy has a hard time understanding that the `cols` argument is optional
    dmn_unw = ff.dataframe.DeMean()  # type: ignore [call-arg]
    dmn_w = ff.dataframe.DeMean(w_col="carat")  # type: ignore [call-arg]

    assert dmn_unw.resolve_cols(
        dmn_unw.cols, diamonds_df, ignore=dmn_unw.w_col
    ) == list(diamonds_df.columns)
    assert dmn_w.resolve_cols(dmn_w.cols, diamonds_df, ignore=dmn_w.w_col) == list(
        c for c in diamonds_df.columns if c != "carat"
    )


def test_FitTransform_in_Pipeline(diamonds_df: pd.DataFrame):
    pip = ff.DataFramePipeline().de_mean("price")
    train_df = diamonds_df.sample(10_000)
    fit_pip = pip.fit(train_df)

    assert (
        (ff.DataFramePipeline().read_data_frame(diamonds_df).then(fit_pip))
        .apply()["price"]
        .equals(diamonds_df["price"] - train_df["price"].mean())
    )

    assert isinstance(fit_pip.then(), ff.DataFramePipeline)

    assert (
        (fit_pip.then()["price"].clip(lower=-100, upper=100))
        .apply(diamonds_df)
        .equals((diamonds_df[["price"]] - train_df["price"].mean()).clip(-100, 100))
    )

    gbpip = ff.DataFramePipeline().group_by_cols("cut", group_keys=True).then(fit_pip)
    assert isinstance(gbpip.apply(diamonds_df), pd.DataFrame)
