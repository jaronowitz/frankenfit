from io import StringIO

from attrs import define
import pytest
import numpy as np
from os import path
import pandas as pd
import warnings

from pydataset import data
import pyarrow.dataset as ds

import frankenfit as ff


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_Identity(diamonds_df):
    # identity should do nothing
    d1 = ff.Identity().fit(diamonds_df).apply(diamonds_df)
    assert d1.equals(diamonds_df)

    # test the special optional-fit behavior of StatelessTransform
    d2 = ff.Identity().apply(diamonds_df)
    assert d2.equals(diamonds_df)

    result = (ff.Identity() + ff.Identity()).apply(diamonds_df)
    assert result.equals(diamonds_df)


def test_Transform(diamonds_df):
    class DeMean(ff.ColumnsTransform):
        def _fit(self, df_fit):
            return df_fit[self.cols].mean()

        def _apply(self, df_apply, state):
            means = state
            return df_apply.assign(**{c: df_apply[c] - means[c] for c in self.cols})

    assert isinstance(DeMean.FitDeMean, type)
    assert DeMean.FitDeMean.__name__ == "FitDeMean"
    cols = ["price", "x", "y", "z"]
    t = DeMean(cols, tag="mytag")
    assert repr(t) == ("DeMean(tag=%r, cols=%r)" % ("mytag", cols))
    assert t.params() == ["tag", "cols"]
    fit = t.fit(diamonds_df)
    assert fit.state().equals(diamonds_df[cols].mean())
    result = fit.apply(diamonds_df)
    assert result[cols].equals(diamonds_df[cols] - diamonds_df[cols].mean())

    assert isinstance(t, ff.Transform)
    assert not isinstance(fit, ff.Transform)
    assert isinstance(fit, ff.FitTransform)

    with pytest.raises(AttributeError):

        class Bad(ff.Transform):
            # not allowed to have an attribute named "state"
            state: int = 1


def test_hyperparams(diamonds_df):
    bindings = {
        "bool_param": True,
        "int_param": 42,
        "response_col": "price",
    }
    assert ff.HP.resolve_maybe("foo", bindings) == "foo"
    assert ff.HP.resolve_maybe(21, bindings) == 21
    assert ff.HP.resolve_maybe(ff.HP("int_param"), bindings) == 42

    assert (
        ff.HP.resolve_maybe(ff.HPFmtStr("{response_col}_train"), bindings)
        == "price_train"
    )

    @define
    class TestTransform(ff.Transform):
        some_param: str

        def _fit(self, df_fit: pd.DataFrame) -> object:
            return None

        def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
            return df_apply

    t = TestTransform(some_param=ff.HP("response_col"))
    assert t.hyperparams() == {"response_col"}
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == "price"

    t = TestTransform(some_param=ff.HP("undefined_hyperparam"))
    with pytest.raises(ff.UnresolvedHyperparameterError):
        tfit = t.fit(diamonds_df, bindings=bindings)

    t = TestTransform(
        some_param=ff.HPLambda(
            lambda b: {b["response_col"]: b["response_col"] + "_orig"}
        )
    )
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == {"price": "price_orig"}


def test_ColumnsTransform(diamonds_df):
    df = diamonds_df
    # test cols behavior
    # the simplest concrete ColumnsTransform is KeepColumns
    t = ff.Select(["x", "y", "z"])
    assert t.apply(df).equals(df[["x", "y", "z"]])
    t = ff.Select("z")
    assert t.apply(df).equals(df[["z"]])
    t = ff.Select(ff.HP("which_cols"))
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

    t = ff.Select(["x", ff.HP("some_col"), "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])
    t = ff.Select(["x", "{some_col}", "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])


def test_DeMean(diamonds_df):
    cols = ["price", "x", "y", "z"]
    t = ff.DeMean(cols)
    result = t.fit(diamonds_df).apply(diamonds_df)
    assert (result[cols].mean().abs() < 1e-10).all()


def test_CopyColumns(diamonds_df):
    cols = ["price", "x", "y", "z"]
    df = diamonds_df[cols]
    result = ff.Copy(["price"], ["price_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    # optional list literals for lists of 1
    result = ff.Copy("price", "price_copy").apply(df)
    assert result["price_copy"].equals(df["price"])

    result = ff.Copy(["price"], ["price_copy1", "price_copy2"]).apply(df)
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])
    # optional list literals for lists of 1
    result = ff.Copy("price", ["price_copy1", "price_copy2"]).apply(df)
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])

    result = ff.Copy(["price", "x"], ["price_copy", "x_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    assert result["x_copy"].equals(df["x"])

    with pytest.raises(ValueError):
        result = ff.Copy(
            ["price", "x"],
            [
                "price_copy",
            ],
        ).apply(df)

    # with hyperparams
    bindings = {"response": "price"}
    result = ff.Copy(["{response}"], ["{response}_copy"]).apply(df, bindings=bindings)
    assert result["price_copy"].equals(df["price"])

    result = ff.Copy("{response}", "{response}_copy").apply(df, bindings=bindings)
    assert result["price_copy"].equals(df["price"])

    result = (
        ff.Copy([ff.HP("response")], "{response}_copy")
        .fit(None, bindings=bindings)
        .apply(df)
    )
    assert result["price_copy"].equals(df["price"])

    with pytest.raises(TypeError):
        # HP("response") resolves to a str, not a list of str
        result = ff.Copy(ff.HP("response"), "{response}_copy").fit(
            None, bindings=bindings
        )


def test_KeepColumns(diamonds_df):
    kept = ["price", "x", "y", "z"]
    result = ff.Select(kept).apply(diamonds_df)
    assert result.equals(diamonds_df[kept])


def test_RenameColumns(diamonds_df):
    result = ff.Rename({"price": "price_orig"}).apply(diamonds_df)
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))
    result = ff.Rename(lambda c: c + "_orig" if c == "price" else c).apply(diamonds_df)
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))

    result = ff.Rename(
        ff.HPLambda(lambda b: {b["response"]: b["response"] + "_orig"})
    ).apply(diamonds_df, bindings={"response": "price"})
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))


def test_StatelessLambda(diamonds_df):
    df = diamonds_df
    result = ff.StatelessLambda(
        lambda df: df.rename(columns={"price": "price_orig"})
    ).apply(df)
    assert result.equals(df.rename(columns={"price": "price_orig"}))

    result = ff.StatelessLambda(
        lambda df, bindings: df.rename(columns={bindings["response"]: "foo"})
    ).apply(df, bindings={"response": "price"})
    assert result.equals(df.rename(columns={"price": "foo"}))

    with pytest.raises(TypeError):
        ff.StatelessLambda(
            lambda df, bindings, _: df.rename(columns={bindings["response"]: "foo"})
        ).apply(df, bindings={"response": "price"})


def test_StatefulLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    result = lambda_demean.fit(df).apply(df)

    # with bindings
    lambda_demean = ff.StatefulLambda(
        fit_fun=lambda df, bindings: df[bindings["col"]].mean(),
        apply_fun=lambda df, mean, bindings: df.assign(
            **{bindings["col"]: df[bindings["col"]] - mean}
        ),
    )
    result = lambda_demean.fit(df, bindings={"col": "price"}).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))


def test_Clip(diamonds_df):
    df = diamonds_df
    result = ff.Clip(["price"], upper=150, lower=100).apply(df)
    assert (result["price"] <= 150).all() & (result["price"] >= 100).all()
    clip_price = ff.Clip(["price"], upper=ff.HP("upper"))
    for upper in (100, 200, 300):
        result = clip_price.apply(df, bindings={"upper": upper})
        assert (result["price"] <= upper).all()
    clip_price = ff.Clip(["price"], lower=ff.HP("lower"))
    for lower in (100, 200, 300):
        result = clip_price.apply(df, bindings={"lower": lower})
        assert (result["price"] >= lower).all()


def test_ImputeConstant():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert ff.ImputeConstant(["col1"], 0.0).apply(df).equals(df.fillna(0.0))


def test_Winsorize():
    df = pd.DataFrame({"col1": [float(x) for x in range(1, 101)]})
    result = ff.Winsorize(["col1"], 0.2).fit(df).apply(df)
    assert (result["col1"] > 20).all() and (result["col1"] < 81).all()


def test_ImputeMean():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        ff.ImputeMean(["col1"])
        .fit(df)
        .apply(df)
        .equals(pd.DataFrame({"col1": pd.Series([1.0, 1.5, 2.0])}))
    )


def test_ZScore(diamonds_df):
    result = ff.ZScore(["price"]).fit(diamonds_df).apply(diamonds_df)
    assert result["price"].equals(
        (diamonds_df["price"] - diamonds_df["price"].mean())
        / diamonds_df["price"].std()
    )


def test_Print(diamonds_df):
    fit_msg = "Fitting!"
    apply_msg = "Applying!"
    buf = StringIO()
    t = ff.Print(fit_msg=fit_msg, apply_msg=apply_msg, dest=buf)
    assert isinstance(t, ff.Identity)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == fit_msg + "\n" + apply_msg + "\n"
    assert df.equals(diamonds_df)

    buf = StringIO()
    t = ff.Print(fit_msg=None, apply_msg=None, dest=buf)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == ""
    assert df.equals(diamonds_df)


def test_Pipeline(diamonds_df):
    p = ff.Pipeline()
    assert len(p) == 0
    # empty pipeline equiv to identity
    assert diamonds_df.equals(p.fit(diamonds_df).apply(diamonds_df))

    # bare transform, automatically becomes list of 1
    p = ff.Pipeline(transforms=ff.Select(["x"]))
    assert len(p) == 1
    assert p.fit(diamonds_df).apply(diamonds_df).equals(diamonds_df[["x"]])

    p = ff.Pipeline(
        transforms=[
            ff.Copy(["price"], ["price_train"]),
            ff.DeMean(["price_train"]),
            ff.Select(["x", "y", "z", "price", "price_train"]),
        ]
    )
    assert len(p) == 3
    target_df = diamonds_df.assign(price_train=lambda df: df["price"]).assign(
        price_train=lambda df: df["price_train"] - df["price_train"].mean()
    )[["x", "y", "z", "price", "price_train"]]
    df = p.fit(diamonds_df).apply(diamonds_df)
    assert df.equals(target_df)

    # apply() gives same result
    df = p.apply(diamonds_df)
    assert df.equals(target_df)

    # pipeline of pipeline is coalesced
    p2 = ff.Pipeline(transforms=p)
    assert len(p2) == len(p)
    assert p2 == p
    p2 = ff.Pipeline(transforms=[p])
    assert len(p2) == len(p)
    assert p2 == p

    # TypeError for a non-Transform in the pipeline
    with pytest.raises(TypeError):
        ff.Pipeline(transforms=42)
    with pytest.raises(TypeError):
        ff.Pipeline(transforms=[ff.DeMean(["price"]), 42])


def test_IfHyperparamIsTrue(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    result = (
        ff.IfHyperparamIsTrue("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(df)  # identity
    result = (
        ff.IfHyperparamIsTrue("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": True})
        .apply(df)
    )
    assert result.equals(target_demean)
    with pytest.raises(ff.UnresolvedHyperparameterError):
        result = (
            ff.IfHyperparamIsTrue("do_it", then=lambda_demean)
            .fit(df, bindings={})
            .apply(df)
        )
    result = (
        ff.IfHyperparamIsTrue("do_it", then=lambda_demean, allow_unresolved=True)
        .fit(df, bindings={})
        .apply(df)
    )
    assert result.equals(df)  # identity

    result = (
        ff.IfHyperparamIsTrue("do_it", then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.IfHyperparamIsTrue("do_it", then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfHyperparamLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    condition = lambda bindings: bindings["x"] > 0 and bindings["y"] > 0  # noqa: E731

    result = (
        ff.IfHyperparamLambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(df)
    result = (
        ff.IfHyperparamLambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": 1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        ff.IfHyperparamLambda(condition, then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.IfHyperparamLambda(condition, then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfTrainingDataHasProperty(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    property = lambda df: len(df.columns) > 1  # noqa: E731

    result = (
        ff.IfTrainingDataHasProperty(property, then=lambda_demean)
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(df)
    result = (
        ff.IfTrainingDataHasProperty(property, then=lambda_demean).fit(df).apply(df)
    )
    assert result.equals(target_demean)
    result = (
        ff.IfTrainingDataHasProperty(
            property, then=lambda_demean, otherwise=lambda_add_ones
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.IfTrainingDataHasProperty(
            property, then=lambda_add_ones, otherwise=lambda_demean
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_demean)


def test_Pipeline_callchaining(diamonds_df):
    # call-chaining should give the same result as list of transform instances
    cols = ["carat", "x", "y", "z", "depth", "table"]
    pipeline_con = ff.Pipeline(
        transforms=[
            ff.Pipe(["carat"], np.log1p),
            ff.Winsorize(cols, limit=0.05),
            ff.ZScore(cols),
            ff.ImputeConstant(cols, 0.0),
            ff.Clip(cols, upper=2, lower=-2),
        ]
    )
    pipeline_chain = (
        ff.Pipeline()
        .pipe(["carat"], np.log1p)
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )
    assert (
        pipeline_con.fit(diamonds_df)
        .apply(diamonds_df)
        .equals(pipeline_chain.fit(diamonds_df).apply(diamonds_df))
    )


def test_Join(diamonds_df):
    diamonds_df = diamonds_df.assign(diamond_id=diamonds_df.index)
    xyz_df = diamonds_df[["diamond_id", "x", "y", "z"]]
    cut_df = diamonds_df[["diamond_id", "cut"]]
    target = pd.merge(xyz_df, cut_df, how="left", on="diamond_id")

    t = ff.Join(
        ff.ReadDataFrame(xyz_df), ff.ReadDataFrame(cut_df), how="left", on="diamond_id"
    )
    result = t.fit().apply()
    assert result.equals(target)
    # assert result.equals(diamonds_df[["diamond_id", "x", "y", "z", "cut"]])

    p = ff.Pipeline(
        transforms=[
            ff.Join(
                ff.ReadDataFrame(xyz_df),
                ff.ReadDataFrame(cut_df),
                how="left",
                on="diamond_id",
            )
        ]
    )
    result = p.apply()
    assert result.equals(target)

    p = (
        ff.Pipeline()
        .read_data_frame(xyz_df)
        .join(ff.Pipeline().read_data_frame(cut_df), how="left", on="diamond_id")
    )
    result = p.apply()
    assert result.equals(target)

    deviances = (
        ff.Pipeline()[["cut", "price"]]
        .join(
            (
                ff.Pipeline()
                .group_by("cut")
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

    sk = ff.SKLearn(
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

    sm = ff.Statsmodels(
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
            ff.Pipeline()
            .print(fit_msg=f"Baking: {cols}")
            .winsorize(cols, limit=0.05)
            .z_score(cols)
            .impute_constant(cols, 0.0)
            .clip(cols, upper=2, lower=-2)
        )

    pipeline = (
        ff.Pipeline()[FEATURES + ["{response_col}"]]
        .copy("{response_col}", "{response_col}_train")
        .winsorize("{response_col}_train", limit=0.05)
        .pipe(["carat", "{response_col}_train"], np.log1p)
        .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
        .sklearn(
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
    pip = ff.Pipeline().stateless_lambda(len)
    assert pip.fit(df).apply(df) == len(df)

    result = ff.GroupBy("cut", pip).fit(df).apply(df)
    assert result.equals(target)

    pip = ff.Pipeline().group_by("cut").stateless_lambda(len)
    assert pip.fit(df).apply(df).equals(target)

    # A sttaeful transform
    pip = ff.Pipeline().group_by("cut").de_mean(["price"])[["cut", "price"]]
    result = pip.apply(df)
    assert np.abs(result["price"].mean()) < 1e-10
    assert all(np.abs(result.groupby("cut")["price"].mean()) < 1e-10)

    # "cross-validated" de-meaning hah
    pip = (
        ff.Pipeline()
        .group_by("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
        .de_mean(["price"])[["cut", "price"]]
    )
    result = pip.apply(df)
    assert all(np.abs(result.groupby("cut")["price"].mean()) > 4)

    pip = (
        ff.Pipeline().group_by("cut").stateless_lambda(lambda df: df[["price"]].mean())
    )
    result = pip.apply(df).set_index("cut").sort_index().reset_index()
    target = df.groupby("cut")[["price"]].mean().sort_index().reset_index()
    assert result.equals(target)

    pip = (
        ff.Pipeline()
        .group_by("cut", fitting_schedule=ff.fit_group_on_all_other_groups)
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

    pip = ff.Pipeline().group_by("cut").de_mean(["price"])
    with pytest.raises(ff.UnfitGroupError):
        pip.fit(df.loc[df["cut"] != "Fair"]).apply(df)


def test_Correlation(diamonds_df):
    target = diamonds_df[["price", "carat"]].corr()
    cm = ff.Correlation(["price"], ["carat"]).apply(diamonds_df)
    assert cm.iloc[0, 0] == target.iloc[0, 1]


def test_tags(diamonds_df):
    from sklearn.linear_model import LinearRegression

    FEATURES = ["carat", "x", "y", "z", "depth", "table"]

    def bake_features(cols):
        return (
            ff.Pipeline()
            .print(fit_msg=f"Baking: {cols}")
            .winsorize(cols, limit=0.05)
            .z_score(cols)
            .impute_constant(cols, 0.0)
            .clip(cols, upper=2, lower=-2)
        )

    pipeline = (
        ff.Pipeline()[FEATURES + ["{response_col}"]]
        .copy("{response_col}", "{response_col}_train")
        .winsorize("{response_col}_train", limit=0.05)
        .pipe(["carat", "{response_col}_train"], np.log1p)
        .if_hyperparam_is_true("bake_features", bake_features(FEATURES))
        .sklearn(
            LinearRegression,
            # x_cols=["carat", "depth", "table"],
            x_cols=ff.HP("predictors"),
            response_col="{response_col}_train",
            hat_col="{response_col}_hat",
            class_params={"fit_intercept": True},
            tag="my-regression",
        )
        # transform {response_col}_hat from log-dollars back to dollars
        .copy("{response_col}_hat", "{response_col}_hat_dollars")
        .pipe("{response_col}_hat_dollars", np.expm1)
    )

    assert isinstance(pipeline.find_by_tag("my-regression"), ff.SKLearn)

    fit = pipeline.fit(
        diamonds_df,
        bindings={
            "response_col": "price",
            "bake_features": True,
            "predictors": FEATURES,
        },
    )
    assert isinstance(fit.find_by_tag("my-regression"), ff.SKLearn.FitSKLearn)
    assert isinstance(fit.find_by_tag("my-regression").state().coef_, np.ndarray)


def test_ReadDataFrame(diamonds_df):
    df = diamonds_df.reset_index().drop(["index"], axis=1)
    assert ff.ReadDataFrame(df).apply().equals(df)

    pip = ff.Pipeline().group_by("cut").de_mean("price")[["cut", "price"]]
    # here's a spiffy equivalence... at some point we could even consider making
    # fit(df), apply(df), syntactic sugar for prepending a ReadDataFrame.
    assert pip.apply(df).equals(ff.Pipeline().read_data_frame(df).then(pip).apply())


def test_ReadPandasCSV(diamonds_df, tmp_path):
    df = diamonds_df.reset_index().drop(["index"], axis=1)
    fp = path.join(tmp_path, "diamonds.csv")
    df.to_csv(fp)

    result = ff.Pipeline().read_pandas_csv(fp, dict(index_col=0)).apply()
    assert result.equals(df)

    with warnings.catch_warnings(record=True) as w:
        pip = ff.Pipeline()[["price"]].read_pandas_csv(fp)
        assert len(w) == 1
        assert issubclass(w[-1].category, ff.NonInitialConstantTransformWarning)

    with warnings.catch_warnings(record=True) as w:
        pip.fit(df)
        assert len(w) == 1
        assert issubclass(w[-1].category, ff.NonInitialConstantTransformWarning)


def test_read_write_csv(diamonds_df, tmp_path):
    df = diamonds_df.reset_index().set_index("index")
    ff.WritePandasCSV(
        # TODO: in core, define a field type for pathlib.PosixPath's containing
        # hyperparameter format strings
        str(tmp_path / "diamonds.csv"),
        index_label="index",
    ).apply(df)

    result = ff.ReadPandasCSV(
        str(tmp_path / "diamonds.csv"), dict(index_col="index")
    ).apply()
    assert result.equals(df)

    result = ff.ReadDataset(
        str(tmp_path / "diamonds.csv"), format="csv", index_col="index"
    ).apply()
    assert result.equals(df)


def test_read_write_dataset(diamonds_df, tmp_path):
    df = diamonds_df.reset_index().set_index("index")
    path = str(tmp_path / "diamonds.csv")
    ff.WritePandasCSV(
        path,
        index_label="index",
    ).apply(df)

    target = df.loc[3:6]

    result = ff.ReadDataset(
        path,
        format="csv",
        filter=(ds.field("index") > 2) & (ds.field("index") < 7),
        index_col="index",
    ).apply()
    assert result.equals(target)

    bindings = {"filter": (ds.field("index") > 2) & (ds.field("index") < 7)}

    result = (
        ff.ReadDataset(
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
    ff.Assign(foo=1, bar=2, logprice=lambda df: np.log1p(df["price"]), tag="boo")

    result = (
        ff.Pipeline().assign(
            intercept=1,
            grp=lambda df: df.index % 5,
            grp_2=lambda self, df: df.index % self.bindings()["k"],
        )
    ).apply(diamonds_df, bindings={"k": 3})
    assert (result["intercept"] == 1).all()
    assert (result["grp"] == (diamonds_df.index % 5)).all()
    assert (result["grp_2"] == (diamonds_df.index % 3)).all()

    result = (
        ff.Pipeline().assign(
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
