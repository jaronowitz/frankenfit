import pytest

# TODO: how to encode the fact that we have a TEST-time requirement for
# pydataset (but not build-time or runtime)
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


# def test_hyperparams(diamonds_df):
#     pass


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

    result = fpt.CopyColumns(["price"], ["price_copy1", "price_copy2"]).apply(df)
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


def test_KeepColumns(diamonds_df):
    kept = ["price", "x", "y", "z"]
    result = fpt.KeepColumns(kept).apply(diamonds_df)
    assert result.equals(diamonds_df[kept])
