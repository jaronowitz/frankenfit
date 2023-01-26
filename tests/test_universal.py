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
    assert buf.getvalue() == fit_msg + "\n" + apply_msg + "\n"
    assert df.equals(diamonds_df)
    buf.close()

    buf = StringIO()
    t = ff.UniversalPipeline().print(fit_msg=None, apply_msg=None, dest=buf)
    df = t.fit(diamonds_df).apply(diamonds_df)
    assert buf.getvalue() == ""
    assert df.equals(diamonds_df)
    buf.close()


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
        .fit(df, do_it=False)
        .apply(df)
    )
    assert result.equals(df)  # identity
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean)
        .fit(df, do_it=True)
        .apply(df)
    )
    assert result.equals(target_demean)
    with pytest.raises(ff.core.UnresolvedHyperparameterError):
        result = (
            ff.UniversalPipeline()
            .if_hyperparam_is_true("do_it", lambda_demean)
            .fit(df, {})
            .apply(df)
        )
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean, allow_unresolved=True)
        .fit(df, {})
        .apply(df)
    )
    assert result.equals(df)  # identity

    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_demean, otherwise=lambda_add_ones)
        .fit(df, do_it=False)
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_is_true("do_it", lambda_add_ones, otherwise=lambda_demean)
        .fit(df, do_it=False)
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

    def condition(bindings):
        return bindings.get("x", 1) > 0 and bindings.get("y", 0) > 0

    pip = ff.UniversalPipeline().if_hyperparam_lambda(condition, lambda_demean)
    assert pip.hyperparams() == {"x", "y"}

    result = pip.fit(df, {"x": -1, "y": 1}).apply(df)
    assert result.equals(df)
    result = pip.fit(df, {"x": 1, "y": 1}).apply(df)
    assert result.equals(target_demean)

    result = (
        ff.UniversalPipeline()
        .if_hyperparam_lambda(condition, lambda_demean, otherwise=lambda_add_ones)
        .fit(df, {"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_add_ones)
    result = (
        ff.UniversalPipeline()
        .if_hyperparam_lambda(condition, lambda_add_ones, otherwise=lambda_demean)
        .fit(df, {"x": -1, "y": 1})
        .apply(df)
    )
    assert result.equals(target_demean)

    (
        ff.UniversalPipeline().if_hyperparam_lambda(condition, lambda_add_ones)
    ).visualize()
    (
        ff.UniversalPipeline().if_hyperparam_lambda(
            condition, lambda_add_ones, otherwise=lambda_demean
        )
    ).visualize()


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

    (
        ff.UniversalPipeline().if_fitting_data_has_property(property, lambda_add_ones)
    ).visualize()
    (
        ff.UniversalPipeline().if_fitting_data_has_property(
            property, lambda_add_ones, otherwise=lambda_demean
        )
    ).visualize()

    property_with_hp = lambda df, min_cols=1: len(df.columns) >= min_cols  # noqa: E731
    pip = ff.UniversalPipeline().if_fitting_data_has_property(
        property_with_hp, lambda_add_ones, otherwise=lambda_demean
    )
    assert pip.hyperparams() == {"min_cols"}
    assert pip.fit(df[["price"]]).apply(df).equals(target_add_ones)
    assert pip.fit(df[["price"]], {"min_cols": 2}).apply(df).equals(target_demean)


def test_StatelessLambda(diamonds_df: pd.DataFrame):
    df = diamonds_df
    result = (
        ff.UniversalPipeline()
        .stateless_lambda(lambda df: df.rename(columns={"price": "price_orig"}))
        .apply(df)
    )
    assert result.equals(df.rename(columns={"price": "price_orig"}))

    pip = ff.UniversalPipeline().stateless_lambda(
        lambda df, response: df.rename(columns={response: "foo"})
    )
    result = pip.apply(df, {"response": "price"})
    assert result.equals(df.rename(columns={"price": "foo"}))

    with pytest.raises(ff.UnresolvedHyperparameterError):
        pip.apply(df)

    # hyperparam with default value
    pip = ff.UniversalPipeline().stateless_lambda(
        lambda df, response="price": df.rename(columns={response: f"{response}_2"})
    )
    result = pip.apply(df)
    assert result.equals(df.rename(columns={"price": "price_2"}))
    result = pip.apply(df, {"response": "depth"})
    assert result.equals(df.rename(columns={"depth": "depth_2"}))

    with pytest.raises(TypeError):
        ff.UniversalPipeline().stateless_lambda(ff.HP("foo"))  # type: ignore [arg-type]


def test_StatefulLambda(diamonds_df: pd.DataFrame):
    df = diamonds_df
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df: df["price"].mean(),
        apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
    )
    result = lambda_demean.fit(df).apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))

    # with hyperparams
    lambda_demean = ff.UniversalPipeline().stateful_lambda(
        fit_fun=lambda df, col: df[col].mean(),
        apply_fun=lambda df, mean, col: df.assign(**{col: df[col] - mean}),
    )
    result = lambda_demean.fit(df, col="price").apply(df)
    assert result.equals(df.assign(price=df["price"] - df["price"].mean()))
    with pytest.raises(ff.UnresolvedHyperparameterError):
        lambda_demean.fit(df)

    # lambdas with too many args
    with pytest.raises(TypeError):
        ff.UniversalPipeline().stateful_lambda(
            fit_fun=ff.HP("foo"),  # type: ignore [arg-type]
            apply_fun=lambda df, mean: df.assign(price=df["price"] - mean),
        )
    with pytest.raises(TypeError):
        ff.UniversalPipeline().stateful_lambda(
            fit_fun=lambda df, col: df[col].mean(),
            apply_fun=ff.HP("foo"),  # type: ignore [arg-type]
        )


def test_ForBindings(diamonds_df: pd.DataFrame):
    df = diamonds_df.head()
    result = (
        ff.universal.ForBindings[pd.DataFrame](
            [
                {"target_col": "price"},
                {"target_col": "depth"},
                {"target_col": "table"},
            ],
            ff.dataframe.Select(["{target_col}"]),
            combine_fun=list,  # type: ignore [arg-type]
        )
        .fit(df)
        .apply(df)
    )

    for x in result:
        assert x.result.equals(df[[x.bindings["target_col"]]])  # type: ignore

    result = (
        ff.UniversalPipeline[Any]()
        .for_bindings(
            [
                {"target_col": "price"},
                {"target_col": "depth"},
                {"target_col": "table"},
            ],
            combine_fun=list,
        )
        .stateless_lambda(lambda df, target_col: df[[target_col]])
    ).apply(df)

    for x in result:
        assert x.result.equals(df[[x.bindings["target_col"]]])  # type: ignore


def test_StateOf(diamonds_df: pd.DataFrame) -> None:
    @ff.params
    class MyDeMean(ff.Transform[pd.DataFrame]):
        col: str

        def _fit(self, data_fit: pd.DataFrame) -> Any:
            return data_fit[[self.col]].mean()

        def _apply(self, data_apply: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
            return data_apply.assign(
                **{self.col: data_apply[self.col] - state[self.col]}
            )

    mean = diamonds_df[["price"]].mean()
    assert mean.equals(
        ff.universal.StateOf(MyDeMean("price"))
        .fit(diamonds_df)  # type: ignore [arg-type]
        .apply()
    )
    p = ff.UniversalPipeline[pd.DataFrame]().then(MyDeMean("price")).last_state()
    x = p.fit(diamonds_df).apply()
    assert mean.equals(x)  # type: ignore [arg-type]

    with pytest.raises(ValueError):
        ff.UniversalPipeline().last_state()
