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
    assert isinstance(t.then(), ff.BasePipeline)
    p = t.then(t2)
    assert isinstance(p, ff.BasePipeline)
    assert len(p) == 2
    p = t.then([t2])
    assert isinstance(p, ff.BasePipeline)
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
    class FitDeMean(ff.FitTransform["DeMean", pd.DataFrame, pd.DataFrame]):
        def apply(
            self,
            data_apply: Optional[pd.DataFrame | Future[pd.DataFrame]] = None,
        ) -> pd.DataFrame:
            """My apply docstr"""
            print("my overridden apply")
            return super().apply(data_apply=data_apply)

    @ff.params
    class DeMean(ff.Transform[pd.DataFrame, pd.DataFrame]):
        """
        De-mean some columns.
        """

        cols: list[str]

        FitTransformClass: ClassVar[Type[ff.FitTransform]] = FitDeMean

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
    p = core.BasePipeline[pd.DataFrame]()
    assert len(p) == 0
    # empty pipeline equiv to identity
    assert diamonds_df.equals(p.fit(diamonds_df).apply(diamonds_df))

    # bare transform, automatically becomes list of 1
    p = core.BasePipeline[pd.DataFrame](transforms=ff.Identity())
    assert len(p) == 1
    assert p.fit(diamonds_df).apply(diamonds_df).equals(diamonds_df)

    p = core.BasePipeline[pd.DataFrame](
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
    p2 = core.BasePipeline[pd.DataFrame](transforms=p)
    assert len(p2) == len(p)
    assert p2 == p
    p2 = core.BasePipeline[pd.DataFrame](transforms=[p])
    assert len(p2) == len(p)
    assert p2 == p

    # TypeError for a non-Transform in the pipeline
    with pytest.raises(TypeError):
        core.BasePipeline(transforms=42)
    with pytest.raises(TypeError):
        core.BasePipeline(transforms=[ff.Identity(), 42])


def test_Pipeline_callchaining(diamonds_df: pd.DataFrame) -> None:
    # call-chaining should give the same result as list of transform instances
    PipelineWithMethods = core.BasePipeline[pd.DataFrame].with_methods(
        identity=ff.Identity
    )
    assert (
        inspect.signature(
            PipelineWithMethods.identity  # type: ignore [attr-defined]
        ).return_annotation
        == "BasePipelineWithMethods"
    )
    pipeline_con = core.BasePipeline[pd.DataFrame](transforms=[ff.Identity()])
    pipeline_chain = PipelineWithMethods().identity()  # type: ignore [attr-defined]
    assert (
        pipeline_con.fit(diamonds_df)
        .apply(diamonds_df)
        .equals(pipeline_chain.fit(diamonds_df).apply(diamonds_df))
    )


def test_tags(diamonds_df: pd.DataFrame) -> None:
    tagged_ident = ff.Identity[Any](tag="mytag")
    pip = core.BasePipeline[Any](
        transforms=[ff.Identity(), tagged_ident, ff.Identity()]
    )
    assert pip.find_by_name("Identity#mytag") is tagged_ident
    with pytest.raises(KeyError):
        pip.find_by_name("mingus dew")

    fit = pip.fit(diamonds_df)
    assert isinstance(
        fit.find_by_name("Identity#mytag").resolved_transform(), ff.Identity
    )
    with pytest.raises(KeyError):
        fit.find_by_name("mingus dew")

    ihp = universal.IfHyperparamIsTrue("my-hp", ff.Identity(), otherwise=tagged_ident)
    assert ihp.find_by_name("Identity#mytag") is tagged_ident
    ihp_fit = ihp.fit(**{"my-hp": False})
    assert isinstance(
        ihp_fit.find_by_name("Identity#mytag").resolved_transform(), ff.Identity
    )

    ihp_fit = ihp.fit(**{"my-hp": True})
    with pytest.raises(KeyError):
        ihp_fit.find_by_name("mytag")


def test_FitTransform_materialize_state() -> None:
    def has_materialized_state(fit_transform: ff.FitTransform):
        return fit_transform.state() == fit_transform.materialize_state().state()

    tagged_ident = ff.Identity[Any](tag="mytag")
    pip = core.BasePipeline[Any](
        transforms=[ff.Identity(), tagged_ident, ff.Identity()]
    )
    fit = ff.LocalBackend().fit(pip)
    assert not has_materialized_state(fit)
    with pytest.raises(ValueError):
        fit.find_by_name("Identity#mytag")

    fit_mat = fit.materialize_state()
    assert has_materialized_state(fit_mat)
    assert isinstance(
        fit_mat.find_by_name("Identity#mytag").resolved_transform(), ff.Identity
    )

    # there should be nothing to materialize
    fit = pip.fit()
    assert fit.materialize_state() == fit


def test_simple_visualize() -> None:
    p = ff.Identity[str]().then()
    # for now, just ensure no exceptions
    p.visualize()


def test_empty_BasePipeline() -> None:
    pip = ff.BasePipeline[str]()
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
    pip = ff.BasePipeline[str](transforms=[ff.Identity[str]()])
    pip2 = ff.BasePipeline[str](transforms=[ff.Identity[str](), ff.Identity[str]()])
    pip3 = ff.BasePipeline[str](
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
    assert tb1.key_counts == {
        (
            "DataFramePipeline#Outer.ForBindings#1"
            ".DataFramePipeline#Inner.StatelessLambda#1._fit"
        ): 3
    }

    tb2_counts: dict[str, int] = {}
    tb2 = TracingBackend(key_counts=tb2_counts)
    tb2.apply(fit, diamonds_df)
    assert tb2_counts == {
        (
            "DataFramePipeline#Outer.ForBindings#1"
            ".DataFramePipeline#Inner.StatelessLambda#1._apply"
        ): 3,
        "DataFramePipeline#Outer.ForBindings#1._combine_results": 1,
    }
    # tb1 wasn't touched:
    assert tb1_counts == {
        (
            "DataFramePipeline#Outer.ForBindings#1"
            ".DataFramePipeline#Inner.StatelessLambda#1._fit"
        ): 3
    }

    tb3_counts: dict[str, int] = {}
    tb3 = TracingBackend(key_counts=tb3_counts)
    tb3.apply(pip, diamonds_df)
    assert tb3_counts == {
        (
            "DataFramePipeline#Outer.ForBindings#1"
            ".DataFramePipeline#Inner.StatelessLambda#1._fit"
        ): 3,
        (
            "DataFramePipeline#Outer.ForBindings#1"
            ".DataFramePipeline#Inner.StatelessLambda#1._apply"
        ): 3,
        "DataFramePipeline#Outer.ForBindings#1._combine_results": 1,
    }
