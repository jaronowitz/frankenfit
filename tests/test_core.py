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

import pytest
import pandas as pd

from pydataset import data

import frankenfit as ff
import frankenfit.core as core


@pytest.fixture
def diamonds_df():
    return data("diamonds")


def test_Transform(diamonds_df):
    @ff.transform
    class DeMean(ff.Transform):
        cols: list[str]

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

    @ff.transform
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
    with pytest.raises(core.UnresolvedHyperparameterError):
        tfit = t.fit(diamonds_df, bindings=bindings)

    t = TestTransform(
        some_param=ff.HPLambda(
            lambda b: {b["response_col"]: b["response_col"] + "_orig"}
        )
    )
    tfit = t.fit(diamonds_df, bindings=bindings)
    assert tfit.some_param == {"price": "price_orig"}

    pipeline = ff.DataFramePipeline().select(["{response_col}"])
    with pytest.raises(core.UnresolvedHyperparameterError):
        pipeline.fit(diamonds_df)


def test_Pipeline(diamonds_df):
    p = ff.core.ObjectPipeline()
    assert len(p) == 0
    # empty pipeline equiv to identity
    assert diamonds_df.equals(p.fit(diamonds_df).apply(diamonds_df))

    # bare transform, automatically becomes list of 1
    p = core.ObjectPipeline(transforms=ff.Identity())
    assert len(p) == 1
    assert p.fit(diamonds_df).apply(diamonds_df).equals(diamonds_df)

    p = core.ObjectPipeline(
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
    p2 = core.ObjectPipeline(transforms=p)
    assert len(p2) == len(p)
    assert p2 == p
    p2 = core.ObjectPipeline(transforms=[p])
    assert len(p2) == len(p)
    assert p2 == p

    # TypeError for a non-Transform in the pipeline
    with pytest.raises(TypeError):
        core.ObjectPipeline(transforms=42)
    with pytest.raises(TypeError):
        core.ObjectPipeline(transforms=[ff.Identity(), 42])


def test_Pipeline_callchaining(diamonds_df):
    # call-chaining should give the same result as list of transform instances
    PipelineWithMethods = core.ObjectPipeline.with_methods(identity=ff.Identity)
    pipeline_con = core.ObjectPipeline(transforms=[ff.Identity()])
    pipeline_chain = PipelineWithMethods().identity()
    assert (
        pipeline_con.fit(diamonds_df)
        .apply(diamonds_df)
        .equals(pipeline_chain.fit(diamonds_df).apply(diamonds_df))
    )


def test_tags(diamonds_df):
    tagged_ident = ff.Identity(tag="mytag")
    pip = core.ObjectPipeline(transforms=[ff.Identity(), tagged_ident, ff.Identity()])
    assert pip.find_by_tag("mytag") is tagged_ident

    fit = pip.fit(diamonds_df)
    assert isinstance(fit.find_by_tag("mytag"), ff.Identity.FitIdentity)
