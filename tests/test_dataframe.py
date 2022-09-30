import pytest
import numpy as np
from os import path
import pandas as pd
import warnings

from pydataset import data
import pyarrow.dataset as ds

import frankenfit as ff
import frankenfit.core as ffc
import frankenfit.dataframe as ffdf


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_ColumnsTransform(diamonds_df):
    df = diamonds_df
    # test cols behavior
    # the simplest concrete ColumnsTransform is Select
    t = ffdf.Select(["x", "y", "z"])
    assert t.apply(df).equals(df[["x", "y", "z"]])
    t = ffdf.Select("z")
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

    t = ffdf.Select(["x", ff.HP("some_col"), "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])
    t = ffdf.Select(["x", "{some_col}", "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])


def test_DeMean(diamonds_df):
    cols = ["price", "x", "y", "z"]
    t = ff.DataFramePipeline().de_mean(cols)
    result = t.fit(diamonds_df).apply(diamonds_df)
    assert (result[cols].mean().abs() < 1e-10).all()


def test_CopyColumns(diamonds_df):
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
        result = (
            ff.DataFramePipeline()
            .copy(ff.HP("response"), "{response}_copy")
            .fit(df, bindings=bindings)
        )


def test_Select(diamonds_df):
    kept = ["price", "x", "y", "z"]
    result = ff.DataFramePipeline().select(kept).apply(diamonds_df)
    assert result.equals(diamonds_df[kept])


def test_RenameColumns(diamonds_df):
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


def test_Clip(diamonds_df):
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


def test_ImputeConstant():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        ff.DataFramePipeline()
        .impute_constant(["col1"], 0.0)
        .apply(df)
        .equals(df.fillna(0.0))
    )


def test_Winsorize():
    df = pd.DataFrame({"col1": [float(x) for x in range(1, 101)]})
    result = ff.DataFramePipeline().winsorize(["col1"], 0.2).fit(df).apply(df)
    assert (result["col1"] > 20).all() and (result["col1"] < 81).all()


def test_ImputeMean():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        ff.DataFramePipeline()
        .impute_mean(["col1"])
        .fit(df)
        .apply(df)
        .equals(pd.DataFrame({"col1": pd.Series([1.0, 1.5, 2.0])}))
    )


def test_ZScore(diamonds_df):
    result = (
        ff.DataFramePipeline().z_score(["price"]).fit(diamonds_df).apply(diamonds_df)
    )
    assert result["price"].equals(
        (diamonds_df["price"] - diamonds_df["price"].mean())
        / diamonds_df["price"].std()
    )


def test_Join(diamonds_df):
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


def test_SKLearn(diamonds_df):
    from sklearn.linear_model import LinearRegression

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

    # TODO: test w_col
    # TODO: test hyperparameterizations


def test_Statsmodels(diamonds_df):
    from statsmodels.api import OLS

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


def test_complex_pipeline_1(diamonds_df):
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
    # TODO: test more stuff with this pipeline


def test_GroupBy(diamonds_df):
    df: pd.DataFrame = diamonds_df.reset_index().drop(["index"], axis=1)
    target = df.groupby("cut", as_index=False, sort=False).apply(len)
    pip = ff.DataFramePipeline().stateless_lambda(len)
    assert pip.fit(df).apply(df) == len(df)

    result = ffdf.GroupByCols("cut", pip).fit(df).apply(df)
    assert result.equals(target)

    pip = ff.DataFramePipeline().group_by_cols("cut").stateless_lambda(len)
    assert pip.fit(df).apply(df).equals(target)

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
    target = df.groupby("cut")[["price"]].mean().sort_index().reset_index()
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


def test_Correlation(diamonds_df):
    target = diamonds_df[["price", "carat"]].corr()
    cm = ff.DataFramePipeline().correlation(["price"], ["carat"]).apply(diamonds_df)
    assert cm.iloc[0, 0] == target.iloc[0, 1]


def test_ReadDataFrame(diamonds_df):
    df = diamonds_df.reset_index().drop(["index"], axis=1)
    assert ff.DataFramePipeline().read_data_frame(df).apply().equals(df)

    # another way to start a pipeline with a reader
    pip = ff.ReadDataFrame(df).then(ff.Identity())
    assert pip.apply().equals(df)
    pip = ff.ReadDataFrame(df).then().identity()
    assert pip.apply().equals(df)


def test_ReadPandasCSV(diamonds_df, tmp_path):
    df = diamonds_df.reset_index().drop(["index"], axis=1)
    fp = path.join(tmp_path, "diamonds.csv")
    df.to_csv(fp)

    result = ff.DataFramePipeline().read_pandas_csv(fp, dict(index_col=0)).apply()
    assert result.equals(df)

    with warnings.catch_warnings(record=True) as w:
        pip = ff.DataFramePipeline()[["price"]].read_pandas_csv(fp)
        assert len(w) == 1
        assert issubclass(w[-1].category, ffc.NonInitialConstantTransformWarning)

    with warnings.catch_warnings(record=True) as w:
        pip.fit(df)
        assert len(w) == 1
        assert issubclass(w[-1].category, ffc.NonInitialConstantTransformWarning)


def test_read_write_csv(diamonds_df, tmp_path):
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


def test_read_write_dataset(diamonds_df, tmp_path):
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


def test_Assign(diamonds_df):
    result = (
        ff.DataFramePipeline().assign(
            intercept=1,
            grp=lambda df: df.index % 5,
            grp_2=lambda self, df: df.index % self.bindings()["k"],
        )
    ).apply(diamonds_df, bindings={"k": 3})
    assert (result["intercept"] == 1).all()
    assert (result["grp"] == (diamonds_df.index % 5)).all()
    assert (result["grp_2"] == (diamonds_df.index % 3)).all()

    result = (
        ff.DataFramePipeline().assign(
            {
                "intercept": 1,
                ff.HPFmtStr("grp_{k}"): lambda self, df: df.index
                % self.bindings()["k"],
            },
            tag="foo",
        )
    ).apply(diamonds_df, bindings={"k": 3})
    assert (result["intercept"] == 1).all()
    assert (result["grp_3"] == (diamonds_df.index % 3)).all()


def test_GroupByBindings(diamonds_df):
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
