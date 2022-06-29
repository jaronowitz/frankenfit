from io import StringIO

from attrs import define
import pytest
import numpy as np
import pandas as pd

from pydataset import data

import ard_fit_pipelines as fp


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_Identity(diamonds_df):
    # identity should do nothing
    d1 = fp.Identity().fit(diamonds_df).apply(diamonds_df)
    assert d1.equals(diamonds_df)

    # test the special optional-fit behavior of StatelessTransform
    d2 = fp.Identity().apply(diamonds_df)
    assert d2.equals(diamonds_df)


def test_Transform(diamonds_df):
    class DeMean(fp.ColumnsTransform):
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

    assert isinstance(t, fp.Transform)
    assert not isinstance(fit, fp.Transform)
    assert isinstance(fit, fp.FitTransform)

    with pytest.raises(AttributeError):

        class Bad(fp.Transform):
            # not allowed to have an attribute named "state"
            state: int = 1


def test_hyperparams(diamonds_df):
    bindings = {
        "bool_param": True,
        "int_param": 42,
        "response_col": "price",
    }
    assert fp.HP.resolve_maybe("foo", bindings) == "foo"
    assert fp.HP.resolve_maybe(21, bindings) == 21
    assert fp.HP.resolve_maybe(fp.HP("int_param"), bindings) == 42

    assert (
        fp.HP.resolve_maybe(fp.HPFmtStr("{response_col}_train"), bindings)
        == "price_train"
    )

    @define
    class TestTransform(fp.Transform):
        some_param: str

        def _fit(self, df_fit: pd.DataFrame) -> object:
            return None

        def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
            return df_apply

    t = TestTransform(some_param=fp.HP("response_col"))
    assert t.hyperparams() == {"some_param": fp.HP("response_col")}
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == "price"

    t = TestTransform(some_param=fp.HP("undefined_hyperparam"))
    with pytest.raises(fp.UnresolvedHyperparameterError):
        tfit = t.fit(diamonds_df, bindings=bindings)

    t = TestTransform(
        some_param=fp.HPLambda(
            lambda b: {b["response_col"]: b["response_col"] + "_orig"}
        )
    )
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == {"price": "price_orig"}


def test_ColumnsTransform(diamonds_df):
    df = diamonds_df
    # test cols behavior
    # the simplest concrete ColumnsTransform is KeepColumns
    t = fp.KeepColumns(["x", "y", "z"])
    assert t.apply(df).equals(df[["x", "y", "z"]])
    t = fp.KeepColumns("z")
    assert t.apply(df).equals(df[["z"]])
    t = fp.KeepColumns(fp.HP("which_cols"))
    assert (
        t.fit(df, bindings={"which_cols": ["x", "y", "z"]})
        .apply(df)
        .equals(df[["x", "y", "z"]])
    )

    bindings = {"some_col": "y"}
    assert fp.HPCols(cols=["x", "y", "z"]).resolve(bindings) == ["x", "y", "z"]
    assert fp.HPCols(cols=["x", fp.HP("some_col"), "z"]).resolve(bindings) == [
        "x",
        "y",
        "z",
    ]
    assert fp.HPCols(cols=["x", "{some_col}", "z"]).resolve(bindings) == [
        "x",
        "y",
        "z",
    ]

    t = fp.KeepColumns(["x", fp.HP("some_col"), "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])
    t = fp.KeepColumns(["x", "{some_col}", "z"])
    assert t.fit(df, bindings=bindings).apply(df).equals(df[["x", "y", "z"]])


def test_DeMean(diamonds_df):
    cols = ["price", "x", "y", "z"]
    t = fp.DeMean(cols)
    result = t.fit(diamonds_df).apply(diamonds_df)
    assert (result[cols].mean().abs() < 1e-10).all()


def test_CopyColumns(diamonds_df):
    cols = ["price", "x", "y", "z"]
    df = diamonds_df[cols]
    result = fp.CopyColumns(["price"], ["price_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    # optional list literals for lists of 1
    result = fp.CopyColumns("price", "price_copy").apply(df)
    assert result["price_copy"].equals(df["price"])

    result = fp.CopyColumns(["price"], ["price_copy1", "price_copy2"]).apply(df)
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])
    # optional list literals for lists of 1
    result = fp.CopyColumns("price", ["price_copy1", "price_copy2"]).apply(df)
    assert result["price_copy1"].equals(df["price"])
    assert result["price_copy2"].equals(df["price"])

    result = fp.CopyColumns(["price", "x"], ["price_copy", "x_copy"]).apply(df)
    assert result["price_copy"].equals(df["price"])
    assert result["x_copy"].equals(df["x"])

    with pytest.raises(ValueError):
        result = fp.CopyColumns(
            ["price", "x"],
            [
                "price_copy",
            ],
        ).apply(df)

    # with hyperparams
    bindings = {"response": "price"}
    result = fp.CopyColumns(["{response}"], ["{response}_copy"]).apply(
        df, bindings=bindings
    )
    assert result["price_copy"].equals(df["price"])

    result = fp.CopyColumns("{response}", "{response}_copy").apply(
        df, bindings=bindings
    )
    assert result["price_copy"].equals(df["price"])

    result = (
        fp.CopyColumns([fp.HP("response")], "{response}_copy")
        .fit(None, bindings=bindings)
        .apply(df)
    )
    assert result["price_copy"].equals(df["price"])

    with pytest.raises(TypeError):
        # HP("response") resolves to a str, not a list of str
        result = fp.CopyColumns(fp.HP("response"), "{response}_copy").fit(
            None, bindings=bindings
        )


def test_KeepColumns(diamonds_df):
    kept = ["price", "x", "y", "z"]
    result = fp.KeepColumns(kept).apply(diamonds_df)
    assert result.equals(diamonds_df[kept])


def test_RenameColumns(diamonds_df):
    result = fp.RenameColumns({"price": "price_orig"}).apply(diamonds_df)
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))
    result = fp.RenameColumns(lambda c: c + "_orig" if c == "price" else c).apply(
        diamonds_df
    )
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))

    result = fp.RenameColumns(
        fp.HPLambda(lambda b: {b["response"]: b["response"] + "_orig"})
    ).apply(diamonds_df, bindings={"response": "price"})
    assert result.equals(diamonds_df.rename(columns={"price": "price_orig"}))


def test_StatelessLambda(diamonds_df):
    df = diamonds_df
    result = fp.StatelessLambda(
        lambda df: df.rename(columns={"price": "price_orig"})
    ).apply(df)
    assert result.equals(df.rename(columns={"price": "price_orig"}))

    result = fp.StatelessLambda(
        lambda df, bindings: df.rename(columns={bindings["response"]: "foo"})
    ).apply(df, bindings={"response": "price"})
    assert result.equals(df.rename(columns={"price": "foo"}))

    with pytest.raises(TypeError):
        fp.StatelessLambda(
            lambda df, bindings, _: df.rename(columns={bindings["response"]: "foo"})
        ).apply(df, bindings={"response": "price"})


def test_StatefulLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = fp.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    result = lambda_demean.fit(df).apply(df)

    # with bindings
    lambda_demean = fp.StatefulLambda(
        fit_fun=lambda df, bindings: df[bindings["col"]].mean(),
        apply_fun=lambda df, mean, bindings: df.assign(
            **{bindings["col"]: df[bindings["col"]] - mean}
        ),
    )
    result = lambda_demean.fit(df, bindings={"col": "price"}).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))


def test_Clip(diamonds_df):
    df = diamonds_df
    result = fp.Clip(["price"], upper=150, lower=100).apply(df)
    assert (result["price"] <= 150).all() & (result["price"] >= 100).all()
    clip_price = fp.Clip(["price"], upper=fp.HP("upper"))
    for upper in (100, 200, 300):
        result = clip_price.apply(df, bindings={"upper": upper})
        assert (result["price"] <= upper).all()
    clip_price = fp.Clip(["price"], lower=fp.HP("lower"))
    for lower in (100, 200, 300):
        result = clip_price.apply(df, bindings={"lower": lower})
        assert (result["price"] >= lower).all()


def test_ImputeConstant():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert fp.ImputeConstant(["col1"], 0.0).apply(df).equals(df.fillna(0.0))


def test_Winsorize():
    df = pd.DataFrame({"col1": [float(x) for x in range(1, 101)]})
    result = fp.Winsorize(["col1"], 0.2).fit(df).apply(df)
    assert (result["col1"] > 20).all() and (result["col1"] < 81).all()


def test_ImputeMean():
    df = pd.DataFrame({"col1": pd.Series([1.0, np.nan, 2.0])})
    assert (
        fp.ImputeMean(["col1"])
        .fit(df)
        .apply(df)
        .equals(pd.DataFrame({"col1": pd.Series([1.0, 1.5, 2.0])}))
    )


def test_ZScore(diamonds_df):
    result = fp.ZScore(["price"]).fit(diamonds_df).apply(diamonds_df)
    assert result["price"].equals(
        (diamonds_df["price"] - diamonds_df["price"].mean())
        / diamonds_df["price"].std()
    )


def test_Print(diamonds_df):
    fit_msg = "Fitting!"
    apply_msg = "Applying!"
    buf = StringIO()
    t = fp.Print(fit_msg=fit_msg, apply_msg=apply_msg, dest=buf)
    assert isinstance(t, fp.Identity)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == fit_msg + "\n" + apply_msg + "\n"
    assert df.equals(diamonds_df)


def test_Pipeline(diamonds_df):
    p = fp.Pipeline()
    assert len(p) == 0
    # empty pipeline equiv to identity
    assert diamonds_df.equals(p.fit(diamonds_df).apply(diamonds_df))

    # bare transform, automatically becomes list of 1
    p = fp.Pipeline(transforms=fp.KeepColumns(["x"]))
    assert len(p) == 1
    assert p.fit(diamonds_df).apply(diamonds_df).equals(diamonds_df[["x"]])

    p = fp.Pipeline(
        transforms=[
            fp.CopyColumns(["price"], ["price_train"]),
            fp.DeMean(["price_train"]),
            fp.KeepColumns(["x", "y", "z", "price", "price_train"]),
        ]
    )
    assert len(p) == 3
    target_df = diamonds_df.assign(price_train=lambda df: df["price"]).assign(
        price_train=lambda df: df["price_train"] - df["price_train"].mean()
    )[["x", "y", "z", "price", "price_train"]]
    df = p.fit(diamonds_df).apply(diamonds_df)
    assert df.equals(target_df)

    # pipeline of pipeline is coalesced
    p2 = fp.Pipeline(transforms=p)
    assert len(p2) == len(p)
    assert p2 == p
    p2 = fp.Pipeline(transforms=[p])
    assert len(p2) == len(p)
    assert p2 == p

    # TypeError for a non-Transform in the pipeline
    with pytest.raises(TypeError):
        fp.Pipeline(transforms=42)
    with pytest.raises(TypeError):
        fp.Pipeline(transforms=[fp.DeMean(["price"]), 42])


def test_IfHyperparamIsTrue(diamonds_df):
    df = diamonds_df
    lambda_demean = fp.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = fp.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    result = (
        fp.IfHyperparamIsTrue("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(df)  # identity
    result = (
        fp.IfHyperparamIsTrue("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": True})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        fp.IfHyperparamIsTrue("do_it", then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        fp.IfHyperparamIsTrue("do_it", then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfHyperparamLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = fp.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = fp.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    condition = lambda bindings: bindings["x"] > 0 and bindings["y"] > 0  # noqa: E731

    result = (
        fp.IfHyperparamLambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(df)
    result = (
        fp.IfHyperparamLambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": 1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        fp.IfHyperparamLambda(condition, then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        fp.IfHyperparamLambda(condition, then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfTrainingDataHasProperty(diamonds_df):
    df = diamonds_df
    lambda_demean = fp.StatefulLambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = fp.StatelessLambda(apply_fun=lambda df: df.assign(ones=1.0))
    target_add_ones = df.assign(ones=1.0)

    property = lambda df: len(df.columns) > 1  # noqa: E731

    result = (
        fp.IfTrainingDataHasProperty(property, then=lambda_demean)
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(df)
    result = (
        fp.IfTrainingDataHasProperty(property, then=lambda_demean).fit(df).apply(df)
    )
    assert result.equals(target_demean)
    result = (
        fp.IfTrainingDataHasProperty(
            property, then=lambda_demean, otherwise=lambda_add_ones
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        fp.IfTrainingDataHasProperty(
            property, then=lambda_add_ones, otherwise=lambda_demean
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_demean)


def test_Pipeline_callchaining(diamonds_df):
    # call-chaining should give the same result as list of transform instances
    cols = ["carat", "x", "y", "z", "depth", "table"]
    pipeline_con = fp.Pipeline(
        transforms=[
            fp.Pipe(["carat"], np.log1p),
            fp.Winsorize(cols, limit=0.05),
            fp.ZScore(cols),
            fp.ImputeConstant(cols, 0.0),
            fp.Clip(cols, upper=2, lower=-2),
        ]
    )
    pipeline_chain = (
        fp.Pipeline()
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


def test_Dataset(diamonds_df):
    ds = fp.PandasDataset(diamonds_df)
    assert ds.to_dataframe().equals(diamonds_df)
    dsc = fp.DatasetCollection({"__data__": ds})
    assert dsc.to_dataframe().equals(diamonds_df)
    dsc2 = fp.DatasetCollection({"__data__": diamonds_df})
    assert dsc2.to_dataframe().equals(diamonds_df)

    assert fp.data_to_dataframe(diamonds_df).equals(diamonds_df)
    assert fp.data_to_dataframe(ds).equals(diamonds_df)
    assert fp.data_to_dataframe(dsc).equals(diamonds_df)
    assert fp.data_to_dataframe(dsc2).equals(diamonds_df)


def test_data_selection(diamonds_df):
    ds = fp.PandasDataset(diamonds_df)
    dsc = fp.DatasetCollection({"__data__": ds})
    iddy = fp.Identity()
    diddy = fp.Pipeline("foo")  # Identity, but selects "foo" dataset

    for arg in (diamonds_df, ds, dsc):
        assert iddy.apply(arg).equals(diamonds_df)

    for arg in (diamonds_df, ds, dsc):
        # diddy is looking for 'foo', not the default '__data__'
        with pytest.raises(fp.UnknownDatasetError):
            diddy.fit(arg).apply(arg)

    dsc_with_foo = fp.DatasetCollection({"foo": ds})
    assert diddy.fit(dsc_with_foo).apply(dsc_with_foo).equals(diamonds_df)

    dsc_with_foo = fp.DatasetCollection({"foo": diamonds_df})
    assert diddy.fit(dsc_with_foo).apply(dsc_with_foo).equals(diamonds_df)


def test_data_selection_in_pipeline(diamonds_df):
    df = diamonds_df
    index_all = set(df.index)
    index_in = set(np.random.choice(df.index, size=int(len(df) / 2), replace=False))
    index_out = index_all - index_in
    len(index_all), len(index_in), len(index_out)
    df_in = df.loc[list(index_in)]
    df_out = df.loc[list(index_out)]
    dsc = fp.DatasetCollection({"in": df_in, "out": df_out})

    cols = ["carat", "x", "y", "z", "depth", "table"]
    pipeline_con = fp.Pipeline(
        transforms=[
            fp.Pipe(["carat"], np.log1p),
            fp.Winsorize(cols, limit=0.05),
            fp.ZScore(cols),
            fp.ImputeConstant(cols, 0.0),
            fp.Clip(cols, upper=2, lower=-2),
        ]
    )
    with pytest.raises(fp.UnknownDatasetError):
        # pipeline_con is just looking for the default __data__, which our dsc doesn't
        # have
        pipeline_con.fit(dsc)

    pipeline_con_in = fp.Pipeline(
        "in",
        [
            fp.Pipe(["carat"], np.log1p),
            fp.Winsorize(cols, limit=0.05),
            fp.ZScore(cols),
            fp.ImputeConstant(cols, 0.0),
            fp.Clip(cols, upper=2, lower=-2),
        ],
    )
    result_in = pipeline_con_in.fit(dsc).apply(dsc)
    assert result_in.equals(pipeline_con.fit(df_in).apply(df_in))

    pipeline_con_out = fp.Pipeline(
        "out",
        [
            fp.Pipe(["carat"], np.log1p),
            fp.Winsorize(cols, limit=0.05),
            fp.ZScore(cols),
            fp.ImputeConstant(cols, 0.0),
            fp.Clip(cols, upper=2, lower=-2),
        ],
    )
    result_out = pipeline_con_out.fit(dsc).apply(dsc)
    assert result_out.equals(pipeline_con.fit(df_out).apply(df_out))

    pipeline_chain_in = (
        fp.Pipeline("in")
        .pipe(["carat"], np.log1p)
        .winsorize(cols, limit=0.05)
        .z_score(cols)
        .impute_constant(cols, 0.0)
        .clip(cols, upper=2, lower=-2)
    )
    result_in = pipeline_chain_in.fit(dsc).apply(dsc)
    assert result_in.equals(pipeline_con.fit(df_in).apply(df_in))
