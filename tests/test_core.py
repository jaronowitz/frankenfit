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

import inspect
from sys import version_info
from typing import Any, Callable, ClassVar, Optional, Type, TypeVar, cast

import numpy as np
import pandas as pd
import pytest
from attrs import define, field
from pydataset import data  # type: ignore

import frankenfit as ff
import frankenfit.core as core
import frankenfit.universal as universal
from frankenfit.core import Future, LocalFuture

PYVERSION = (version_info.major, version_info.minor)


@pytest.fixture
def diamonds_df() -> pd.DataFrame:
    return data("diamonds")


def test_Transform(diamonds_df: pd.DataFrame) -> None:
    @ff.params
    class DeMean(ff.Transform):
        cols: list[str]

        def _fit(self, data_fit):
            return data_fit[self.cols].mean()

        def _apply(self, data_apply, state):
            means = state
            return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})

    cols = ["price", "x", "y", "z"]
    t = DeMean(cols, tag="mytag")
    assert repr(t) == ("DeMean(tag=%r, cols=%r)" % ("mytag", cols))
    assert t.params() == ["tag", "cols"]
    fit = t.fit(diamonds_df)
    assert repr(fit) == (
        f"FitTransform(resolved_transform={t!r}, "
        "state=<class 'pandas.core.series.Series'>, "
        "bindings={})"
    )
    assert fit.bindings() == {}
    assert fit.materialize_state().state().equals(diamonds_df[cols].mean())
    result = fit.apply(diamonds_df)
    assert result[cols].equals(diamonds_df[cols] - diamonds_df[cols].mean())

    assert isinstance(t, ff.Transform)
    assert not isinstance(fit, ff.Transform)
    assert isinstance(fit, ff.FitTransform)
    with pytest.raises(TypeError):
        DeMean(cols, tag=42)  # type: ignore [arg-type]

    assert DeMean(cols) != 42
    m1 = DeMean(cols)
    m2 = DeMean(cols)
    assert m1 == m2  # tag doesn't matter for equality
    assert DeMean(cols, tag="foo") == DeMean(cols, tag="foo")

    assert m1.fit(diamonds_df) == m2.fit(diamonds_df)
    assert m1.fit(diamonds_df) == m1.fit(diamonds_df)
    assert m1.fit(diamonds_df) != 42

    class Remember(ff.Transform):
        def _fit(self, data_fit: Any) -> Any:
            return data_fit

        def _apply(self, data_apply: Any, state: Any) -> Any:
            return state

    r1 = Remember()
    assert r1.fit("x") == r1.fit("x")
    assert r1.fit("x") != r1.fit("y")
    assert r1.fit(diamonds_df) == r1.fit(diamonds_df)
    assert r1.fit(np.array([1, 2])) == r1.fit(np.array([1, 2]))
    assert m1.fit(diamonds_df) != r1.fit("x")


def test_apply_none_result() -> None:
    assert ff.Identity().fit(None).apply(None) is None
    assert ff.LocalBackend().apply(ff.Identity().fit(None), None).result() is None


def test_Transform_fit_apply_valence() -> None:
    foo_str = "foo"
    meow_bindings = {"meow": "cat"}
    local = ff.LocalBackend()

    class Fit1(ff.Transform):
        def _fit(self, data_fit: Any) -> Any:
            assert data_fit == foo_str
            return None

        def _apply(self, data_apply: Any, state: Any) -> Any:
            return data_apply

    Fit1().fit(foo_str)

    class Fit2(ff.Transform):
        def _fit(self, data_fit: Any, bindings=None) -> Any:
            assert data_fit == foo_str
            assert bindings == meow_bindings
            return None

        def _apply(self, data_apply: Any, state: Any) -> Any:
            return data_apply

    Fit2().fit(foo_str, meow_bindings)

    class Fit3(ff.Transform):
        def _fit(self, data_fit: Any, bindings=None, backend=None) -> Any:
            assert data_fit == foo_str
            assert bindings == meow_bindings
            assert backend is local
            return None

        def _apply(self, data_apply: Any, state: Any) -> Any:
            return data_apply

    # TODO:
    # ff.LocalBackend().fit(Fit3(), foo_str, meow_bindings, backend=local)
    # Fit3().fit(foo_str, meow_bindings, )

    class Fit0(ff.Transform):
        def _fit(self) -> Any:  # type: ignore [override]
            return None

        def _apply(self, data_apply: Any, state: Any) -> Any:
            return data_apply

    with pytest.raises(TypeError):
        Fit0().fit(foo_str)

    class Apply2(ff.Transform):
        def _fit(self, data_fit: Any) -> Any:
            return "woof"

        def _apply(self, data_apply: Any, state: Any) -> Any:
            assert data_apply == foo_str
            assert state == "woof"

    Apply2().fit().apply(foo_str)

    class Apply1(ff.Transform):
        def _fit(self, data_fit: Any) -> Any:
            return "woof"

        def _apply(self, data_apply: Any) -> Any:  # type: ignore [override]
            assert data_apply == foo_str

    with pytest.raises(TypeError):
        Apply1().fit(**meow_bindings).apply(foo_str)


def test_fit_apply_futures() -> None:
    # t = ff.Identity[str]()
    t = ff.universal.StatefulLambda(  # type: ignore [call-arg]
        fit_fun=lambda df: None, apply_fun=lambda df, _: df
    )
    p = t.then()
    local = ff.LocalBackend()

    def has_materialized_state(fit_transform: ff.FitTransform):
        return fit_transform.state() == fit_transform.materialize_state().state()

    # fit with no backend -> materialized state
    # fit with backend -> future state
    t_fit_1 = t.fit()
    t_fit_2 = local.fit(t)
    p_fit_1 = p.fit()
    p_fit_2 = local.fit(p)
    assert t_fit_1.state() is None
    assert not has_materialized_state(t_fit_2)
    assert has_materialized_state(p_fit_1)
    assert not has_materialized_state(p_fit_2)

    # apply with no backend -> materialized DataResult
    # apply with backend -> future DataResult
    assert t_fit_1.apply("x") == "x"
    assert t_fit_2.apply("x") == "x"
    assert isinstance(local.apply(t_fit_1, "x"), ff.Future)
    assert isinstance(local.apply(t_fit_2, "x"), ff.Future)
    assert p_fit_1.apply("x") == "x"
    assert p_fit_2.apply("x") == "x"
    assert isinstance(local.apply(p_fit_1, "x"), ff.Future)
    assert isinstance(local.apply(p_fit_2, "x"), ff.Future)


def test_then() -> None:
    t = ff.Identity[str]()
    t2 = ff.Identity[str]()
    assert isinstance(t.then(), ff.Pipeline)
    p = t.then(t2)
    assert isinstance(p, ff.Pipeline)
    assert len(p) == 2
    p = t.then([t2])
    assert isinstance(p, ff.Pipeline)
    assert len(p) == 2
    with pytest.raises(TypeError):
        t.then(42)  # type: ignore [arg-type]


def test_fit_with_bindings(diamonds_df: pd.DataFrame) -> None:
    @ff.params
    class TestTransform(ff.Transform):
        # _fit method can optionally accept a bindings arg
        def _fit(self, data_fit: object, bindings: Optional[ff.Bindings] = None):
            return bindings

        def _apply(self, data_apply, state):
            return data_apply

    t = TestTransform()
    fit_t = t.fit(diamonds_df, {"foo": 1})
    assert fit_t.state() == {"foo": 1}
    fit_t = t.fit(diamonds_df, bar=2)
    assert fit_t.state() == {"bar": 2}
    fit_t = t.fit(diamonds_df, {"foo": 1}, bar=2)
    assert fit_t.state() == {"foo": 1, "bar": 2}

    with pytest.raises(TypeError):
        ff.LocalBackend().apply(  # type: ignore [call-overload]
            fit_t, diamonds_df, foo=1
        )


@pytest.mark.skipif(PYVERSION < (3, 9), reason="Python < 3.9")
def test_Transform_signatures() -> None:
    @ff.params
    class DeMean(ff.Transform):
        """
        De-mean some columns.
        """

        cols: list[str]

        def _fit(self, data_fit: pd.DataFrame) -> pd.Series:
            return data_fit[self.cols].mean()

        def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
            means = state
            return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})

    # test the automagic
    assert (
        str(inspect.signature(DeMean))
        == "(cols: 'list[str]', *, tag: 'str' = NOTHING) -> None"
    )


def test_override_fit_apply(
    diamonds_df: pd.DataFrame, capsys: pytest.CaptureFixture
) -> None:
    class FitDeMean(ff.FitTransform["DeMean", pd.DataFrame]):
        def apply(
            self,
            data_apply: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        ) -> pd.DataFrame:
            """My apply docstr"""
            print("my overridden apply")
            return super().apply(data_apply=data_apply)

    @ff.params
    class DeMean(ff.Transform[pd.DataFrame]):
        """
        De-mean some columns.
        """

        cols: list[str]

        fit_transform_class: ClassVar[Type[ff.FitTransform]] = FitDeMean

        def _fit(self, data_fit: pd.DataFrame, bindings=None) -> pd.Series:
            return data_fit[self.cols].mean()

        def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
            means = state
            return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})

        Self = TypeVar("Self", bound="DeMean")  # noqa

        def fit(
            self: Self,
            data_fit: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
            bindings: Optional[ff.Bindings] = None,
            /,
            **kwargs,
        ) -> FitDeMean:
            """My fit docstr"""
            print("my overridden fit")
            # return cast(FitDeMean, super().fit(data_fit, bindings, backend=backend))
            return cast(FitDeMean, super().fit(data_fit, bindings, **kwargs))

    dmn = DeMean(["price"])

    fit = dmn.fit(diamonds_df)
    out, err = capsys.readouterr()
    assert "my overridden fit" in out

    _ = fit.apply(diamonds_df)
    out, err = capsys.readouterr()
    assert "my overridden apply" in out


def test_hyperparams(diamonds_df: pd.DataFrame) -> None:
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

    @ff.params
    class TestTransform(ff.Transform):
        some_param: str | ff.HP

        def _fit(self, data_fit: pd.DataFrame) -> None:
            return None

        def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
            return data_apply

    t = TestTransform(some_param=ff.HP("response_col"))
    assert t.hyperparams() == {"response_col"}
    tfit = t.fit(diamonds_df, bindings)
    assert tfit.resolved_transform().some_param == "price"

    t = TestTransform(some_param=ff.HP("undefined_hyperparam"))
    with pytest.raises(core.UnresolvedHyperparameterError):
        tfit = t.fit(diamonds_df, bindings)

    t = TestTransform(
        some_param=ff.HPLambda(
            lambda b: {b["response_col"]: b["response_col"] + "_orig"}
        )
    )
    tfit = t.fit(diamonds_df, bindings)
    assert tfit.resolved_transform().some_param == {"price": "price_orig"}

    pipeline = ff.DataFramePipeline().select(["{response_col}"])
    with pytest.raises(core.UnresolvedHyperparameterError):
        pipeline.fit(diamonds_df)


def test_Pipeline(diamonds_df: pd.DataFrame) -> None:
    p = core.Pipeline[pd.DataFrame]()
    assert len(p) == 0
    # empty pipeline equiv to identity
    assert diamonds_df.equals(p.fit(diamonds_df).apply(diamonds_df))

    # bare transform, automatically becomes list of 1
    p = core.Pipeline[pd.DataFrame](transforms=ff.Identity())
    assert len(p) == 1
    assert p.fit(diamonds_df).apply(diamonds_df).equals(diamonds_df)

    p = core.Pipeline[pd.DataFrame](
        transforms=[
            ff.Identity(),
            ff.Identity(),
            ff.Identity(),
        ]
    )
    assert len(p) == 3
    df = p.fit(diamonds_df).apply(diamonds_df)
    assert df.equals(diamonds_df)

    # apply() gives same result
    df = p.apply(diamonds_df)
    assert df.equals(diamonds_df)

    # pipeline of pipeline is coalesced
    p2 = core.Pipeline[pd.DataFrame](transforms=p)
    assert len(p2) == len(p)
    assert p2 == p
    p2 = core.Pipeline[pd.DataFrame](transforms=[p])
    assert len(p2) == len(p)
    assert p2 == p

    # TypeError for a non-Transform in the pipeline
    with pytest.raises(TypeError):
        core.Pipeline(transforms=42)
    with pytest.raises(TypeError):
        core.Pipeline(transforms=[ff.Identity(), 42])


def test_Pipeline_callchaining(diamonds_df: pd.DataFrame) -> None:
    # call-chaining should give the same result as list of transform instances
    PipelineWithMethods = core.Pipeline[pd.DataFrame].with_methods(identity=ff.Identity)
    assert (
        inspect.signature(
            PipelineWithMethods.identity  # type: ignore [attr-defined]
        ).return_annotation
        == "PipelineWithMethods"
    )
    pipeline_con = core.Pipeline[pd.DataFrame](transforms=[ff.Identity()])
    pipeline_chain = PipelineWithMethods().identity()  # type: ignore [attr-defined]
    assert (
        pipeline_con.fit(diamonds_df)
        .apply(diamonds_df)
        .equals(pipeline_chain.fit(diamonds_df).apply(diamonds_df))
    )


def test_tags(diamonds_df: pd.DataFrame) -> None:
    tagged_ident = ff.Identity[Any](tag="mytag")
    pip = core.Pipeline[Any](transforms=[ff.Identity(), tagged_ident, ff.Identity()])
    assert pip.find_by_name("Identity#mytag") is tagged_ident
    with pytest.raises(KeyError):
        pip.find_by_name("mingus dew")

    fit = pip.fit(diamonds_df)
    assert isinstance(
        cast(ff.FitTransform, fit.find_by_name("Identity#mytag")).resolved_transform(),
        ff.Identity,
    )
    with pytest.raises(KeyError):
        fit.find_by_name("mingus dew")

    ihp = universal.IfHyperparamIsTrue("my-hp", ff.Identity(), otherwise=tagged_ident)
    assert ihp.find_by_name("Identity#mytag") is tagged_ident
    ihp_fit = ihp.fit(**{"my-hp": False})
    assert isinstance(
        cast(
            ff.FitTransform, ihp_fit.find_by_name("Identity#mytag")
        ).resolved_transform(),
        ff.Identity,
    )

    ihp_fit = ihp.fit(**{"my-hp": True})
    with pytest.raises(KeyError):
        ihp_fit.find_by_name("mytag")


def test_FitTransform_materialize_state() -> None:
    def has_materialized_state(fit_transform: ff.FitTransform):
        return fit_transform.state() == fit_transform.materialize_state().state()

    tagged_ident = ff.Identity[Any](tag="mytag")
    pip = core.Pipeline[Any](transforms=[ff.Identity(), tagged_ident, ff.Identity()])
    fit = ff.LocalBackend().fit(pip)
    assert not has_materialized_state(fit)

    # with pytest.raises(ValueError):
    #     fit.find_by_name("Identity#mytag")

    fit_mat = fit.materialize_state()
    assert has_materialized_state(fit_mat)
    assert isinstance(
        cast(
            ff.FitTransform, fit_mat.find_by_name("Identity#mytag")
        ).resolved_transform(),
        ff.Identity,
    )

    # there should be nothing to materialize
    fit = pip.fit()
    assert fit.materialize_state() == fit


def test_simple_visualize() -> None:
    p = ff.Identity[str]().then()
    # for now, just ensure no exceptions
    p.visualize()


def test_empty_Pipeline() -> None:
    pip = ff.Pipeline[str]()
    fit = pip.fit()

    # data_apply is not None, backend is None: identity
    assert pip.apply("foo") == "foo"
    assert fit.apply("foo") == "foo"

    # data_apply is None, backend is None: empty_constructor() -> None
    assert pip.apply() is None
    assert fit.apply() is None

    # data_apply is not None, backend is not None: future identity
    local = ff.LocalBackend()
    assert local.apply(pip, "foo").result() == "foo"
    assert local.apply(fit, "foo").result() == "foo"

    # data_apply is None, backend is not None: future empty_constructor() -> future None
    assert local.apply(pip).result() is None
    assert local.apply(fit).result() is None

    # data_apply is future not None, backend is None: identity
    assert pip.apply(LocalFuture("foo")) == "foo"
    assert fit.apply(LocalFuture("foo")) == "foo"

    # data_apply is future None, backend is None: future empty_constructor() -> future
    # None. We don't actually like to allow Future[None]s in the type checker but we
    # test anyway
    assert pip.apply(LocalFuture(None)) is None  # type: ignore [arg-type]
    assert fit.apply(LocalFuture(None)) is None  # type: ignore [arg-type]

    # data_apply is future not None, backend is not None: future identity
    assert local.apply(pip, LocalFuture("foo")).result() == "foo"
    assert local.apply(fit, LocalFuture("foo")).result() == "foo"

    # data_apply is future None, backend is not None: future empty_constructor() ->
    # future None
    assert (
        local.apply(pip, LocalFuture(None)).result() is None  # type: ignore [arg-type]
    )
    assert (
        local.apply(fit, LocalFuture(None)).result() is None  # type: ignore [arg-type]
    )


def test_pipeline_then() -> None:
    pip = ff.Pipeline[str](transforms=[ff.Identity[str]()])
    pip2 = ff.Pipeline[str](transforms=[ff.Identity[str](), ff.Identity[str]()])
    pip3 = ff.Pipeline[str](
        transforms=[ff.Identity[str](), ff.Identity[str](), ff.Identity[str]()]
    )

    assert pip.then() == pip
    assert (pip + None) == pip

    assert pip.then(pip) == pip2
    assert pip + pip == pip2

    assert pip.then(ff.Identity[str]()) == pip2
    assert pip + ff.Identity[str]() == pip2

    assert pip.then([ff.Identity[str]()]) == pip2
    assert pip + [ff.Identity[str]()] == pip2

    assert pip.then(pip2) == pip3
    assert pip + pip2 == pip3

    assert pip.then(pip2.transforms) == pip3
    assert pip + pip2.transforms == pip3

    with pytest.raises(TypeError):
        pip.then("meow")  # type: ignore [arg-type]

    with pytest.raises(TypeError):
        pip + "meow"  # type: ignore [operator]


def test_pipeline_backends(diamonds_df: pd.DataFrame) -> None:
    @define
    class TracingBackend(ff.LocalBackend):
        key_counts: dict = field(factory=dict)

        def submit(
            self, key_prefix: str, function: Callable, *function_args, **function_kwargs
        ) -> core.LocalFuture[Any]:
            key = ".".join(self.trace + (key_prefix,))
            self.key_counts[key] = self.key_counts.get(key, 0) + 1
            return super().submit(
                key_prefix, function, *function_args, **function_kwargs
            )

    pip = (
        ff.DataFramePipeline(tag="Outer")
        .for_bindings(
            [{"foo": x} for x in range(3)],
            lambda _: pd.DataFrame(),
            tag="1",
        )
        .then(
            ff.DataFramePipeline(tag="Inner").stateless_lambda(
                lambda df, foo: df.assign(foo=foo), tag="1"
            )
        )
    )

    tb1_counts: dict[str, int] = {}
    tb1 = TracingBackend(key_counts=tb1_counts)
    fit = tb1.fit(pip, diamonds_df)

    # Was for_bindings able to parallelize correctly?
    assert sum(v for k, v in tb1.key_counts.items() if k.endswith("._fit")) == 3

    tb2_counts: dict[str, int] = {}
    tb2 = TracingBackend(key_counts=tb2_counts)
    tb2.apply(fit, diamonds_df)
    assert sum(v for k, v in tb2.key_counts.items() if k.endswith("._apply")) == 3

    tb3_counts: dict[str, int] = {}
    tb3 = TracingBackend(key_counts=tb3_counts)
    tb3.apply(pip, diamonds_df)
    assert sum(v for k, v in tb3.key_counts.items() if k.endswith("._fit")) == 3
    assert sum(v for k, v in tb3.key_counts.items() if k.endswith("._apply")) == 3


def test_IfPipelineIsFitting(diamonds_df: pd.DataFrame):
    @define
    class TracingBackend(ff.LocalBackend):
        key_counts: dict = field(factory=dict)

        def submit(
            self, key_prefix: str, function: Callable, *function_args, **function_kwargs
        ) -> core.LocalFuture[Any]:
            key = ".".join(self.trace + (key_prefix,))
            self.key_counts[key] = self.key_counts.get(key, 0) + 1
            return super().submit(
                key_prefix, function, *function_args, **function_kwargs
            )

    pip = ff.Pipeline[pd.DataFrame](
        transforms=[
            ff.Identity(),
            ff.core.IfPipelineIsFitting(ff.Identity()),
            ff.Identity(),
        ]
    )

    tb1 = TracingBackend()
    fit = tb1.fit(pip, diamonds_df)
    assert sum(v for k, v in tb1.key_counts.items() if k.endswith("._fit")) == 3

    tb2 = TracingBackend()
    tb2.apply(fit, diamonds_df)
    assert sum(v for k, v in tb2.key_counts.items() if k.endswith("._apply")) == 2

    tb3 = TracingBackend()
    tb3.apply(pip, diamonds_df)
    assert sum(v for k, v in tb3.key_counts.items() if k.endswith("._apply")) == 3

    # Again with callchain syntax
    pip = (
        ff.UniversalPipeline[pd.DataFrame]()
        .identity()
        .if_fitting(ff.Identity())
        .identity()
    )

    tb1 = TracingBackend()
    fit = tb1.fit(pip, diamonds_df)
    assert sum(v for k, v in tb1.key_counts.items() if k.endswith("._fit")) == 3

    tb2 = TracingBackend()
    tb2.apply(fit, diamonds_df)
    assert sum(v for k, v in tb2.key_counts.items() if k.endswith("._apply")) == 2

    tb3 = TracingBackend()
    tb3.apply(pip, diamonds_df)
    assert sum(v for k, v in tb3.key_counts.items() if k.endswith("._apply")) == 3


def test_pipeline_with_FitTransform(diamonds_df: pd.DataFrame):
    @ff.params
    class DeMean(ff.Transform[pd.DataFrame]):
        """
        De-mean some columns.
        """

        cols: list[str]

        def _fit(self, data_fit: pd.DataFrame, bindings=None) -> pd.Series:
            return data_fit[self.cols].mean()

        def _apply(self, data_apply: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
            means = state
            return data_apply.assign(**{c: data_apply[c] - means[c] for c in self.cols})

    df1 = diamonds_df.sample(10_000)
    fit = DeMean(["price"]).fit(df1)

    target = diamonds_df.assign(price=diamonds_df["price"] - df1["price"].mean())
    pip = ff.ReadDataFrame(diamonds_df).then(fit)
    assert pip.apply().equals(target)

    pip = ff.DataFramePipeline().read_data_frame(diamonds_df).then(fit)
    assert pip.apply().equals(target)

    pip_fit = pip.fit()
    pip_fit.then(ff.universal.Identity()).apply().equals(target)

    dmn = ff.DataFramePipeline(tag="dmn").de_mean("price", tag="dmn_price")

    pip = (
        ff.DataFramePipeline()
        .filter(lambda df: df["carat"] > 1)
        .then(dmn.fit(df1))
        .select("price", tag="select_price")
    )

    assert isinstance(pip.find_by_name("DeMean#dmn_price"), ff.FitTransform)
    assert isinstance(pip.find_by_name("Select#select_price"), ff.Transform)

    # don't crash!
    pip.visualize()


def test_ApplyFitTransform(diamonds_df: pd.DataFrame):
    dmn = ff.dataframe.DeMean(["price"])
    df1 = diamonds_df.sample(10_000)
    fit = dmn.fit(df1)

    pip = ff.Pipeline[pd.DataFrame](transforms=[core.ApplyFitTransform(fit)])
    target = diamonds_df.assign(price=diamonds_df["price"] - df1["price"].mean())
    assert pip.apply(diamonds_df).equals(target)
