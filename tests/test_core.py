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
from abc import abstractmethod
from typing import Any, ClassVar, Optional, Type, TypeVar

import pandas as pd
import pytest
from pydataset import data  # type: ignore

import frankenfit as ff
import frankenfit.core as core


@pytest.fixture
def diamonds_df() -> pd.DataFrame:
    return data("diamonds")


def test_Transform(diamonds_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        # should be abstract
        ff.Transform()  # type: ignore

    @ff.transform
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
    assert fit.state().equals(diamonds_df[cols].mean())
    result = fit.apply(diamonds_df)
    assert result[cols].equals(diamonds_df[cols] - diamonds_df[cols].mean())

    assert isinstance(t, ff.Transform)
    assert not isinstance(fit, ff.Transform)
    assert isinstance(fit, ff.FitTransform)


def test_fit_with_bindings(diamonds_df: pd.DataFrame) -> None:
    @ff.transform
    class TestTransform(ff.Transform):
        # _fit method can optionally accept a bindings arg
        def _fit(self, data_fit: object, bindings: Optional[ff.Bindings] = None):
            return bindings

        def _apply(self, data_apply, state):
            return data_apply

    t = TestTransform()
    fit_t = t.fit(diamonds_df, bindings={"foo": 1})
    assert fit_t.state() == {"foo": 1}


def test_Transform_signatures() -> None:
    @ff.transform
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
    # assert (
    #     str(inspect.signature(DeMean.fit))
    #     == "(self, data_fit: pandas.core.frame.DataFrame = None, "
    #     "bindings: 'Optional[Bindings]' = None, "
    #     "backend: 'Optional[Backend]' = None) "
    #     "-> 'test_Transform_signatures.<locals>.DeMean.FitDeMean'"
    # )
    # assert (
    #     str(inspect.signature(DeMean.FitDeMean))
    #     == "(resolved_transform: 'DeMean', state: pandas.core.series.Series, "
    #     "bindings: 'Optional[Bindings]' = None)"
    # )
    # assert (
    #     str(inspect.signature(DeMean.FitDeMean.state))
    #     == "(self) -> pandas.core.series.Series"
    # )
    # assert (
    #     str(inspect.signature(DeMean.FitDeMean.apply))
    #     == "(self, data_apply: pandas.core.frame.DataFrame = None, "
    #     "backend: 'Optional[Backend]' = None) -> pandas.core.frame.DataFrame"
    # )


def test_override_fit_apply(
    diamonds_df: pd.DataFrame, capsys: pytest.CaptureFixture
) -> None:
    class FitDeMean(ff.FitTransform["DeMean", pd.DataFrame, pd.DataFrame]):
        @abstractmethod
        def apply(
            self,
            data_apply: Optional[pd.DataFrame] = None,
            backend: Optional[ff.Backend] = None,
        ) -> pd.DataFrame:
            """My apply docstr"""
            print("my overridden apply")
            return super().apply(data_apply=data_apply, backend=backend)

    @ff.transform
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
            data_fit: Optional[pd.DataFrame] = None,
            bindings: Optional[ff.Bindings] = None,
            backend: Optional[ff.Backend] = None,
        ) -> ff.FitTransform[Self, pd.DataFrame, pd.DataFrame]:
            """My fit docstr"""
            print("my overridden fit")
            return super().fit(data_fit, bindings, backend)

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

    @ff.transform
    class TestTransform(ff.Transform):
        some_param: str | ff.HP

        def _fit(self, data_fit: pd.DataFrame) -> None:
            return None

        def _apply(self, data_apply: pd.DataFrame, state: None) -> pd.DataFrame:
            return data_apply

    t = TestTransform(some_param=ff.HP("response_col"))
    assert t.hyperparams() == {"response_col"}
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.resolved_transform().some_param == "price"

    t = TestTransform(some_param=ff.HP("undefined_hyperparam"))
    with pytest.raises(core.UnresolvedHyperparameterError):
        tfit = t.fit(diamonds_df, bindings=bindings)

    t = TestTransform(
        some_param=ff.HPLambda(
            lambda b: {b["response_col"]: b["response_col"] + "_orig"}
        )
    )
    tfit = t.fit(diamonds_df, bindings=bindings)
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
    assert pip.find_by_tag("mytag") is tagged_ident

    fit = pip.fit(diamonds_df)
    assert isinstance(fit.find_by_tag("mytag").resolved_transform(), ff.Identity)
