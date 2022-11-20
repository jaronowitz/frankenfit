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

from io import StringIO
from typing import Any

import pandas as pd
import pytest
from pydataset import data  # type: ignore

import frankenfit as ff


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_Identity(diamonds_df: pd.DataFrame):
    # identity should do nothing
    d1 = ff.Identity[pd.DataFrame]().fit(diamonds_df).apply(diamonds_df)
    assert d1.equals(diamonds_df)

    d1 = ff.Identity[pd.DataFrame]().fit().apply(diamonds_df)
    assert d1.equals(diamonds_df)

    # test the special optional-fit behavior of StatelessTransform
    d2 = ff.Identity[pd.DataFrame]().apply(diamonds_df)
    assert d2.equals(diamonds_df)

    result = (ff.Identity() + ff.Identity()).apply(diamonds_df)
    assert result.equals(diamonds_df)


def test_Print(diamonds_df: pd.DataFrame):
    fit_msg = "Fitting!"
    apply_msg = "Applying!"
    buf = StringIO()
    t = ff.UniversalPipeline().print(fit_msg=fit_msg, apply_msg=apply_msg, dest=buf)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == fit_msg + "\n" + apply_msg + "\n" + apply_msg + "\n"
    assert df.equals(diamonds_df)

    buf = StringIO()
    t = ff.UniversalPipeline().print(fit_msg=None, apply_msg=None, dest=buf)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == ""
    assert df.equals(diamonds_df)


def test_IfHyperparamIsTrue(diamonds_df: pd.DataFrame):
    df = diamonds_df
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.UniversalPipeline().stateless_lambda(
        apply_fun=lambda df: df.assign(ones=1.0)
    )
    target_add_ones = df.assign(ones=1.0)

    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(df)  # identity
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean)
        .fit(df, bindings={"do_it": True})
        .apply(df)
    )
    assert result.equals(target_demean)
    with pytest.raises(ff.core.UnresolvedHyperparameterError):
        result = (
            ff.UniversalPipeline()
            .if_hyperparam_is_true("do_it", lambda_demean)
            .fit(df, bindings={})
            .apply(df)
        )
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean, allow_unresolved=True)
        .fit(df, bindings={})
        .apply(df)
    )
    assert result.equals(df)  # identity

    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"do_it": False})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfHyperparamLambda(diamonds_df: pd.DataFrame):
    df = diamonds_df
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.UniversalPipeline().stateless_lambda(
        apply_fun=lambda df: df.assign(ones=1.0)
    )
    target_add_ones = df.assign(ones=1.0)

    condition = lambda bindings: bindings["x"] > 0 and bindings["y"] > 0  # noqa: E731

    result = (
        ff.UniversalPipeline()
        .if_hyperparam_lambda(condition, lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(df)
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_lambda(condition, lambda_demean)
        .fit(df, bindings={"x": 1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_lambda(condition, lambda_demean, otherwise=lambda_add_ones)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_lambda(condition, lambda_add_ones, otherwise=lambda_demean)
        .fit(df, bindings={"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)


def test_IfFittingDataHasProperty(diamonds_df: pd.DataFrame):
    df = diamonds_df
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    target_demean = df.assign(price=df["price"] - df["price"].mean())
    lambda_add_ones = ff.UniversalPipeline().stateless_lambda(
        apply_fun=lambda df: df.assign(ones=1.0)
    )
    target_add_ones = df.assign(ones=1.0)

    property = lambda df: len(df.columns) > 1  # noqa: E731

    result = (
        ff.UniversalPipeline()
        .if_fitting_data_has_property(property, lambda_demean)
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(df)
    result = (
        ff.UniversalPipeline()
        .if_fitting_data_has_property(property, lambda_demean)
        .fit(df)
        .apply(df)
    )
    assert result.equals(target_demean)
    result = (
        ff.UniversalPipeline()
        .if_fitting_data_has_property(
            property, lambda_demean, otherwise=lambda_add_ones
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.UniversalPipeline()
        .if_fitting_data_has_property(
            property, lambda_add_ones, otherwise=lambda_demean
        )
        .fit(df[["price"]])
        .apply(df)
    )
    assert result.equals(target_demean)


def test_StatelessLambda(diamonds_df: pd.DataFrame):
    df = diamonds_df
    result = (
        ff.UniversalPipeline()
        .stateless_lambda(lambda df: df.rename(columns={"price": "price_orig"}))
        .apply(df)
    )
    assert result.equals(df.rename(columns={"price": "price_orig"}))

    result = (
        ff.UniversalPipeline()
        .stateless_lambda(
            lambda df, bindings: df.rename(columns={bindings["response"]: "foo"})
        )
        .apply(df, bindings={"response": "price"})
    )
    assert result.equals(df.rename(columns={"price": "foo"}))

    with pytest.raises(TypeError):
        ff.UniversalPipeline().stateless_lambda(
            lambda df, bindings, _: df.rename(columns={bindings["response"]: "foo"})
        ).apply(df, bindings={"response": "price"})


def test_StatefulLambda(diamonds_df: pd.DataFrame):
    df = diamonds_df
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    result = lambda_demean.fit(df).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))

    # with bindings
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df, bindings: df[bindings["col"]].mean(),
        apply_fun=lambda df, mean, bindings: df.assign(
            **{bindings["col"]: df[bindings["col"]] - mean}
        ),
    )
    result = lambda_demean.fit(df, bindings={"col": "price"}).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))


def test_ForBindings(diamonds_df: pd.DataFrame):
    df = diamonds_df.head()
    result = (
        ff.universal.ForBindings(
            [
                {"target_col": "price"},
                {"target_col": "depth"},
                {"target_col": "table"},
            ],
            ff.dataframe.Select(["{target_col}"]),
        )
        .fit(df)
        .apply(df)
    )

    for x in result:
        assert x.result.equals(df[[x.bindings["target_col"]]])

    result = (
        ff.UniversalPipeline[Any]()
        .for_bindings(
            [
                {"target_col": "price"},
                {"target_col": "depth"},
                {"target_col": "table"},
            ]
        )
        .stateless_lambda(lambda df, bindings: df[[bindings["target_col"]]])
    ).apply(df)

    for x in result:
        assert x.result.equals(df[[x.bindings["target_col"]]])
