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

from os import path
from typing import cast

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
        ff.ReadDataFrame(diamonds_df)
        .then()
        .z_score(["price"])
        .clip(["price"], lower=-2, upper=2)
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
    assert (
        t.fit(df, bindings={"which_cols": ["x", "y", "z"]})
        .apply(df)
        .equals(df[["x", "y", "z"]])
    )

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
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])
    t = ffdf.Select(["x", "{some_col}", "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])


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
        .apply(df, bindings=bindings)
    )
    assert result["price_copy"].equals(df["price"])

    result = (
        ff.DataFramePipeline()
        .copy("{response}", "{response}_copy")
        .apply(df, bindings=bindings)
    )
    assert result["price_copy"].equals(df["price"])

    result = (
        ff.DataFramePipeline()
        .copy([ff.HP("response")], "{response}_copy")
        .fit(df, bindings=bindings)
        .apply(df)
    )
    assert result["price_copy"].equals(df["price"])

    with pytest.raises(TypeError):
        # HP("response") resolves to a str, not a list of str
        _ = (
            ff.DataFramePipeline()
            .copy(ff.HP("response"), "{response}_copy")
            .fit(df, bindings=bindings)
        )
    with pytest.raises(TypeError):
        # HP("dest") resolves to a str, not a list of str
        _ = (
            ff.DataFramePipeline()
            .copy(["price"], ff.HP("dest"))
            .fit(df, bindings={"dest": "price_copy"})
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

    pip = (
        ff.DataFramePipeline()
        # with second arg for bindings
        .filter(lambda df, bindings: df["cut"] == bindings["which_cut"])
    )
    for which_cut in ("Premium", "Good"):
        result_df = pip.apply(diamonds_df, bindings={"which_cut": which_cut})
        assert (result_df["cut"] == which_cut).all()

    # TypeError if filter_fun takes wrong number of args
    with pytest.raises(TypeError):
        ff.DataFramePipeline().filter(lambda: True).apply(  # type: ignore [arg-type]
            diamonds_df
        )

    with pytest.raises(TypeError):
        ff.DataFramePipeline().filter(
            lambda a, b, c: True  # type: ignore [arg-type]
        ).apply(diamonds_df)


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
        .apply(diamonds_df, bindings={"response": "price"})
    )
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))


def test_Clip(diamonds_df: pd.DataFrame):
    df = diamonds_df
    result = ff.DataFramePipeline().clip(["price"], upper=150, lower=100).apply(df)
    assert (result["price"] <= 150).all() & (result["price"] >= 100).all()
    clip_price = ff.DataFramePipeline().clip(["price"], upper=ff.HP("upper"))
    for upper in (100, 200, 300):
        result = clip_price.apply(df, bindings={"upper": upper})
        assert (result["price"] <= upper).all()
    clip_price = ff.DataFramePipeline().clip(["price"], lower=ff.HP("lower"))
    for lower in (100, 200, 300):
        result = clip_price.apply(df, bindings={"lower": lower})
        assert (result["price"] >= lower).all()


def test_ImputeConstant() -> None:
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        ff.DataFramePipeline()
        .impute_constant(["col1"], 0.0)
        .apply(df)
        .equals(df.fillna(0.0))
    )


def test_Winsorize() -> None:
    df = pd.DataFrame({"col1": [float(x) for x in range(1, 101)]})
    result = ff.DataFramePipeline().winsorize(["col1"], 0.2).fit(df).apply(df)
    assert (result["col1"] > 20).all() and (result["col1"] < 81).all()

    # limits out of bounds
    with pytest.raises(ValueError):
        ff.DataFramePipeline().winsorize(["col1"], -0.2).fit(df)
    with pytest.raises(ValueError):
        ff.DataFramePipeline().winsorize(["col1"], 1.2).fit(df)
    with pytest.raises(TypeError):
        ff.DataFramePipeline().winsorize(["col1"], ff.HP("limit")).fit(
            df, bindings={"limit": "a"}  # non-float
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
            .winsorize(cols, limit=0.05)
            .z_score(cols)
            .impute_constant(cols, 0.0)
            .clip(cols, upper=2, lower=-2)
        )

    pipeline = (
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
        .fit(bindings=bindings)
        .apply()
    )
    assert result.equals(target)


def test_Assign(diamonds_df: pd.DataFrame):
    # kwargs style
    result = (
        ff.DataFramePipeline().assign(
            intercept=1,
            grp=lambda df: df.index % 5,
            grp_self=lambda self, df: df.index % len(self.assignments),
            grp_2=lambda self, bindings, df: df.index % bindings["k"],
        )
    ).apply(diamonds_df, bindings={"k": 3})
    assert cast(pd.DataFrame, result["intercept"] == 1).all()
    assert cast(
        pd.DataFrame,
        # pandas-stubs is too dumb to type-check this...
        result["grp"] == (diamonds_df.index % 5),  # type: ignore [operator]
    ).all()
    assert cast(
        pd.DataFrame,
        result["grp_2"] == (diamonds_df.index % 3),  # type: ignore [operator]
    ).all()

    # assignment_dict style
    result = (
        ff.DataFramePipeline().assign(
            {
                "intercept": 1,
                ff.HPFmtStr("grp_{k}"): lambda self, bindings, df: df.index
                % bindings["k"],
            },
            tag="foo",
        )
    ).apply(diamonds_df, bindings={"k": 3})
    assert cast(pd.DataFrame, result["intercept"] == 1).all()
    assert cast(
        pd.DataFrame,
        result["grp_3"] == (diamonds_df.index % 3),  # type: ignore [operator]
    ).all()

    with pytest.raises(ValueError):
        ff.DataFramePipeline().assign({"foo": 1}, bar=1)

    with pytest.raises(ValueError):
        ff.DataFramePipeline().assign({"foo": 1}, {"bar": 1})

    with pytest.raises(TypeError):
        # lambda takes too many args
        ff.DataFramePipeline().assign(foo=lambda self, bindings, df, extra: 1).apply()


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
        ff.DataFramePipeline().pipe(["carat", "price"], np.log1p).apply(diamonds_df)
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
    assert pip.apply(LocalFuture(None)).equals(empty_df)  # type: ignore [arg-type]
    assert fit.apply(LocalFuture(None)).equals(empty_df)  # type: ignore [arg-type]

    # data_apply is future not None, backend is not None: future identity
    assert local.apply(pip, LocalFuture(df)).result().equals(df)
    assert local.apply(fit, LocalFuture(df)).result().equals(df)
