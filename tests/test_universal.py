from io import StringIO

import pytest

from pydataset import data

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


def test_Print(diamonds_df):
    fit_msg = "Fitting!"
    apply_msg = "Applying!"
    buf = StringIO()
    t = ff.Pipeline().print(fit_msg=fit_msg, apply_msg=apply_msg, dest=buf)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == fit_msg + "\n" + apply_msg + "\n" + apply_msg + "\n"
    assert df.equals(diamonds_df)

    buf = StringIO()
    t = ff.Pipeline().print(fit_msg=None, apply_msg=None, dest=buf)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == ""
    assert df.equals(diamonds_df)


def test_IfHyperparamIsTrue(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.Pipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.Pipeline().stateless_lambda(
        apply_fun=lambda df: df.assign(ones=1.0)
    )
    target_add_ones = df.assign(ones=1.0)

    result = (
        ff.Pipeline()
        .if_hyperparam_is_true("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(df)  # identity
    result = (
        ff.Pipeline()
        .if_hyperparam_is_true("do_it", then=lambda_demean)
        .fit(df, bindings={"do_it": True})
        .apply(df)
    )
    assert result.equals(target_demean)
    with pytest.raises(ff.UnresolvedHyperparameterError):
        result = (
            ff.Pipeline()
            .if_hyperparam_is_true("do_it", then=lambda_demean)
            .fit(df, bindings={})
            .apply(df)
        )
    result = (
        ff.Pipeline()
        .if_hyperparam_is_true("do_it", then=lambda_demean, allow_unresolved=True)
        .fit(df, bindings={})
        .apply(df)
    )
    assert result.equals(df)  # identity

    result = (
        ff.Pipeline()
        .if_hyperparam_is_true("do_it", then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.Pipeline()
        .if_hyperparam_is_true("do_it", then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfHyperparamLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.Pipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.Pipeline().stateless_lambda(
        apply_fun=lambda df: df.assign(ones=1.0)
    )
    target_add_ones = df.assign(ones=1.0)

    condition = lambda bindings: bindings["x"] > 0 and bindings["y"] > 0  # noqa: E731

    result = (
        ff.Pipeline()
        .if_hyperparam_lambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(df)
    result = (
        ff.Pipeline()
        .if_hyperparam_lambda(condition, then=lambda_demean)
        .fit(df, bindings={"x": 1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        ff.Pipeline()
        .if_hyperparam_lambda(condition, then=lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.Pipeline()
        .if_hyperparam_lambda(condition, then=lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfFittingDataHasProperty(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.Pipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.Pipeline().stateless_lambda(
        apply_fun=lambda df: df.assign(ones=1.0)
    )
    target_add_ones = df.assign(ones=1.0)

    property = lambda df: len(df.columns) > 1  # noqa: E731

    result = (
        ff.Pipeline()
        .if_fitting_data_has_property(property, then=lambda_demean)
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(df)
    result = (
        ff.Pipeline()
        .if_fitting_data_has_property(property, then=lambda_demean)
        .fit(df)
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        ff.Pipeline()
        .if_fitting_data_has_property(
            property, then=lambda_demean, otherwise=lambda_add_ones
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.Pipeline()
        .if_fitting_data_has_property(
            property, then=lambda_add_ones, otherwise=lambda_demean
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_demean)


def test_StatelessLambda(diamonds_df):
    df = diamonds_df
    result = (
        ff.Pipeline()
        .stateless_lambda(lambda df: df.rename(columns={"price": "price_orig"}))
        .apply(df)
    )
    assert result.equals(df.rename(columns={"price": "price_orig"}))

    result = (
        ff.Pipeline()
        .stateless_lambda(
            lambda df, bindings: df.rename(columns={bindings["response"]: "foo"})
        )
        .apply(df, bindings={"response": "price"})
    )
    assert result.equals(df.rename(columns={"price": "foo"}))

    with pytest.raises(TypeError):
        ff.Pipeline().stateless_lambda(
            lambda df, bindings, _: df.rename(columns={bindings["response"]: "foo"})
        ).apply(df, bindings={"response": "price"})


def test_StatefulLambda(diamonds_df):
    df = diamonds_df
    lambda_demean = ff.Pipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    result = lambda_demean.fit(df).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))

    # with bindings
    lambda_demean = ff.Pipeline().stateful_lambda(
        fit_fun=lambda df, bindings: df[bindings["col"]].mean(),
        apply_fun=lambda df, mean, bindings: df.assign(
            **{bindings["col"]: df[bindings["col"]] - mean}
        ),
    )
    result = lambda_demean.fit(df, bindings={"col": "price"}).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))
