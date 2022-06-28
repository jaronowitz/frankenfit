from io import StringIO

from attrs import define
import pytest
import numpy as np
import pandas as pd

from pydataset import data

import ard_fit_pipelines.core as fpc
import ard_fit_pipelines.transforms as fpt


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_Identity(diamonds_df):
    # identity should do nothing
    d1 = fpt.Identity().fit(diamonds_df).apply(diamonds_df)
    assert d1.equals(diamonds_df)

    # test the special optional-fit behavior of StatelessTransform
    d2 = fpt.Identity().apply(diamonds_df)
    assert d2.equals(diamonds_df)


def test_Transform(diamonds_df):
    class DeMean(fpc.ColumnsTransform):
        def _fit(self, df_fit):
            return df_fit[self.cols].mean()

        def _apply(self, df_apply, state):
            means = state
            return df_apply.assign(**{c: df_apply[c] - means[c] for c in self.cols})

    assert isinstance(DeMean.FitDeMean, type)
    assert DeMean.FitDeMean.__name__ == "FitDeMean"
    cols = ["price", "x", "y", "z"]
    t = DeMean(cols)
    assert repr(t) == ("DeMean(cols=%r)" % cols)
    assert t.params() == ["cols"]
    fit = t.fit(diamonds_df)
    assert fit.state().equals(diamonds_df[cols].mean())
    result = fit.apply(diamonds_df)
    assert result[cols].equals(diamonds_df[cols] - diamonds_df[cols].mean())

    assert isinstance(t, fpc.Transform)
    assert not isinstance(fit, fpc.Transform)
    assert isinstance(fit, fpc.FitTransform)

    with pytest.raises(AttributeError):

        class Bad(fpc.Transform):
            # not allowed to have an attribute named "state"
            state: int = 1


def test_hyperparams(diamonds_df):
    bindings = {
        "bool_param": True,
        "int_param": 42,
        "response_col": "price",
    }
    assert fpc.HP.resolve_maybe("foo", bindings) == "foo"
    assert fpc.HP.resolve_maybe(21, bindings) == 21
    assert fpc.HP.resolve_maybe(fpc.HP("int_param"), bindings) == 42

    assert (
        fpc.HP.resolve_maybe(fpc.HPFmtStr("{response_col}_train"), bindings)
        == "price_train"
    )

    @define
    class TestTransform(fpc.Transform):
        some_param: str

        def _fit(self, df_fit: pd.DataFrame) -> object:
            return None

        def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
            return df_apply

    t = TestTransform(some_param=fpc.HP("response_col"))
    assert t.hyperparams() == {"some_param": fpc.HP("response_col")}
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == "price"

    t = TestTransform(some_param=fpc.HP("undefined_hyperparam"))
    with pytest.raises(fpc.UnresolvedHyperparameterError):
        tfit = t.fit(diamonds_df, bindings=bindings)

    t = TestTransform(
        some_param=fpc.HPLambda(
            lambda b: {b["response_col"]: b["response_col"] + "_orig"}
        )
    )
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == {"price": "price_orig"}


def test_ColumnsTransform(diamonds_df):
    df = diamonds_df
    # test cols behavior
    # the simplest concrete ColumnsTransform is KeepColumns
    t = fpt.KeepColumns(["x", "y", "z"])
    assert t.apply(df).equals(df[["x", "y", "z"]])
    t = fpt.KeepColumns("z")
    assert t.apply(df).equals(df[["z"]])
    t = fpt.KeepColumns(fpc.HP("which_cols"))
    assert (
        t.fit(df, bindings={"which_cols": ["x", "y", "z"]})
        .apply(df)
        .equals(df[["x", "y", "z"]])
    )

    bindings = {"some_col": "y"}
    assert fpc.HPCols(cols=["x", "y", "z"]).resolve(bindings) == ["x", "y", "z"]
    assert fpc.HPCols(cols=["x", fpc.HP("some_col"), "z"]).resolve(bindings) == [
        "x",
        "y",
        "z",
    ]
    assert fpc.HPCols(cols=["x", "{some_col}", "z"]).resolve(bindings) == [
        "x",
        "y",
        "z",
    ]

    t = fpt.KeepColumns(["x", fpc.HP("some_col"), "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])
    t = fpt.KeepColumns(["x", "{some_col}", "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])


def test_DeMean(diamonds_df):
    cols = ["price", "x", "y", "z"]
    t = fpt.DeMean(cols)
    result = t.fit(diamonds_df).apply(diamonds_df)
    assert (result[cols].mean().abs() < 1e-10).all()


def test_CopyColumns(diamonds_df):
    cols = ["price", "x", "y", "z"]
    df = diamonds_df[cols]
    result = fpt.CopyColumns(["price"], ["price_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    # optional list literals for lists of 1
    result = fpt.CopyColumns("price", "price_copy").apply(df)
    assert result["price_copy"].equals(df["price"])

    result = fpt.CopyColumns(["price"], ["price_copy1", "price_copy2"]).apply(df)
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])
    # optional list literals for lists of 1
    result = fpt.CopyColumns("price", ["price_copy1", "price_copy2"]).apply(df)
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])

    result = fpt.CopyColumns(["price", "x"], ["price_copy", "x_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    assert result["x_copy"].equals(df["x"])

    with pytest.raises(ValueError):
        result = fpt.CopyColumns(
            ["price", "x"],
            [
                "price_copy",
            ],
        ).apply(df)

    # with hyperparams
    bindings = {"response": "price"}
    result = fpt.CopyColumns(["{response}"], ["{response}_copy"]).apply(
        df, bindings=bindings
    )
    assert result["price_copy"].equals(df["price"])

    result = fpt.CopyColumns("{response}", "{response}_copy").apply(
        df, bindings=bindings
    )
    assert result["price_copy"].equals(df["price"])

    result = (
        fpt.CopyColumns([fpc.HP("response")], "{response}_copy")
        .fit(None, bindings=bindings)
        .apply(df)
    )
    assert result["price_copy"].equals(df["price"])

    with pytest.raises(TypeError):
        # HP("response") resolves to a str, not a list of str
        result = fpt.CopyColumns(fpc.HP("response"), "{response}_copy").fit(
            None, bindings=bindings
        )


def test_KeepColumns(diamonds_df):
    kept = ["price", "x", "y", "z"]
    result = fpt.KeepColumns(kept).apply(diamonds_df)
    assert result.equals(diamonds_df[kept])


def test_RenameColumns(diamonds_df):
    result = fpt.RenameColumns({"price": "price_orig"}).apply(diamonds_df)
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))
    result = fpt.RenameColumns(lambda c: c + "_orig" if c == "price" else c).apply(
        diamonds_df
    )
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))

    result = fpt.RenameColumns(
        fpc.HPLambda(lambda b: {b["response"]: b["response"] + "_orig"})
    ).apply(diamonds_df, bindings={"response": "price"})
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))


def test_StatelessLambda(diamonds_df):
    df = diamonds_df
    result = fpt.StatelessLambda(
        lambda df: df.rename(columns={"price": "price_orig"})
    ).apply(df)
    assert result.equals(df.rename(columns={"price": "price_orig"}))

    result = fpt.StatelessLambda(
        lambda df, bindings: df.rename(columns={bindings["response"]: "foo"})
    ).apply(df, bindings={"response": "price"})
    assert result.equals(df.rename(columns={"price": "foo"}))

    with pytest.raises(TypeError):
        fpt.StatelessLambda(
            lambda df, bindings, _: df.rename(columns={bindings["response"]: "foo"})
        ).apply(df, bindings={"response": "price"})


def test_StatefulLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = fpt.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    result = lambda_demean.fit(df).apply(df)

    # with bindings
    lambda_demean = fpt.StatefulLambda(
        fit_fun=lambda df, bindings: df[bindings["col"]].mean(),
        apply_fun=lambda df, mean, bindings: df.assign(
            **{bindings["col"]: df[bindings["col"]] - mean}
        ),
    )
    result = lambda_demean.fit(df, bindings={"col": "price"}).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))


def test_Clip(diamonds_df):
    df = diamonds_df
    result = fpt.Clip(["price"], upper=150, lower=100).apply(df)
    assert (result["price"] <= 150).all() & (result["price"] >= 100).all()
    clip_price = fpt.Clip(["price"], upper=fpc.HP("upper"))
    for upper in (100, 200, 300):
        result = clip_price.apply(df, bindings={"upper": upper})
        assert (result["price"] <= upper).all()
    clip_price = fpt.Clip(["price"], lower=fpc.HP("lower"))
    for lower in (100, 200, 300):
        result = clip_price.apply(df, bindings={"lower": lower})
        assert (result["price"] >= lower).all()


def test_ImputeConstant():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert fpt.ImputeConstant(["col1"], 0.0).apply(df).equals(df.fillna(0.0))


def test_Winsorize():
    df = pd.DataFrame({"col1": [float(x) for x in range(1, 101)]})
    result = fpt.Winsorize(["col1"], 0.2).fit(df).apply(df)
    assert (result["col1"] > 20).all() and (result["col1"] < 81).all()


def test_Print(diamonds_df):
    fit_msg = "Fitting!"
    apply_msg = "Applying!"
    buf = StringIO()
    t = fpt.Print(fit_msg=fit_msg, apply_msg=apply_msg, dest=buf)
    assert isinstance(t, fpt.Identity)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == fit_msg + "\n" + apply_msg + "\n"
    assert df.equals(diamonds_df)


def test_Pipeline(diamonds_df):
    p = fpc.Pipeline()
    assert len(p) == 0
    # empty pipeline equiv to identity
    assert diamonds_df.equals(p.fit(diamonds_df).apply(diamonds_df))

    # bare transform, automatically becomes list of 1
    p = fpc.Pipeline(fpt.KeepColumns(["x"]))
    assert len(p) == 1
    assert p.fit(diamonds_df).apply(diamonds_df).equals(diamonds_df[["x"]])

    p = fpc.Pipeline(
        [
            fpt.CopyColumns(["price"], ["price_train"]),
            fpt.DeMean(["price_train"]),
            fpt.KeepColumns(["x", "y", "z", "price", "price_train"]),
        ]
    )
    assert len(p) == 3
    target_df = diamonds_df.assign(price_train=lambda df: df["price"]).assign(
        price_train=lambda df: df["price_train"] - df["price_train"].mean()
    )[["x", "y", "z", "price", "price_train"]]
    df = p.fit(diamonds_df).apply(diamonds_df)
    assert df.equals(target_df)

    # pipeline of pipeline is coalesced
    p2 = fpc.Pipeline(p)
    assert len(p2) == len(p)
    assert p2 == p
    p2 = fpc.Pipeline([p])
    assert len(p2) == len(p)
    assert p2 == p

    # TypeError for a non-Transform in the pipeline
    with pytest.raises(TypeError):
        fpc.Pipeline(42)
    with pytest.raises(TypeError):
        fpc.Pipeline([fpt.DeMean(["price"]), 42])


def test_IfHyperparamIsTrue(diamonds_df):
    df = diamonds_df
    lambda_demean = fpt.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = fpt.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    result = (
        fpc.IfHyperparamIsTrue("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(df)  # identity
    result = (
        fpc.IfHyperparamIsTrue("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": True})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        fpc.IfHyperparamIsTrue("do_it", then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        fpc.IfHyperparamIsTrue("do_it", then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfHyperparamLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = fpt.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = fpt.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    condition = lambda bindings: bindings["x"] > 0 and bindings["y"] > 0  # noqa: E731

    result = (
        fpc.IfHyperparamLambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(df)
    result = (
        fpc.IfHyperparamLambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": 1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        fpc.IfHyperparamLambda(condition, then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        fpc.IfHyperparamLambda(condition, then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfTrainingDataHasProperty(diamonds_df):
    df = diamonds_df
    lambda_demean = fpt.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = fpt.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    property = lambda df: len(df.columns) > 1  # noqa: E731

    result = (
        fpc.IfTrainingDataHasProperty(property, then=lambda_demean)
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(df)
    result = (
        fpc.IfTrainingDataHasProperty(property, then=lambda_demean).fit(df).apply(df)
    )
    assert result.equals(target_demean)
    result = (
        fpc.IfTrainingDataHasProperty(
            property, then=lambda_demean, otherwise=lambda_add_ones
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        fpc.IfTrainingDataHasProperty(
            property, then=lambda_add_ones, otherwise=lambda_demean
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_demean)
