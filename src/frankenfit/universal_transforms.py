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

"""
Frankenfit's built-in library of generically useful Transforms that make no
assumptions about the type or shape of the data to which they are applied.
"""

from __future__ import annotations
import inspect

import logging
from attrs import define
import pandas as pd

from typing import Callable, Optional, TextIO, TypeVar

from . import core as ffc
from .core import Transform, StatelessTransform, BasePipeline

_LOG = logging.getLogger(__name__)

U = TypeVar("U", bound="UniversalTransform")


@define
class UniversalTransform(Transform):
    def then(
        self: UniversalTransform, other: Transform | list[Transform]
    ) -> "Pipeline":
        result = super().then(other)
        return Pipeline(tag=self.tag, transforms=result.transforms)


class Identity(StatelessTransform, UniversalTransform):
    """
    The stateless Transform that, at apply-time, simply returns the input data
    unaltered.
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None):
        return df_apply


@define
class IfHyperparamIsTrue(UniversalTransform):
    name: str
    then: Transform
    otherwise: Optional[Transform] = None
    allow_unresolved: Optional[bool] = False

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()
        if (not self.allow_unresolved) and self.name not in bindings:
            raise ffc.UnresolvedHyperparameterError(
                f"IfHyperparamIsTrue: no binding for {self.name!r} but "
                "allow_unresolved is False"
            )
        if bindings.get(self.name):
            return self.then.fit(df_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        result.add(self.name)
        return result

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@define
class IfHyperparamLambda(UniversalTransform):
    fun: Callable  # dict[str, object] -> bool
    then: Transform
    otherwise: Optional[Transform] = None

    def _fit(self, df_fit: pd.DataFrame) -> object:
        bindings = self.bindings()
        if self.fun(bindings):
            return self.then.fit(df_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        # find out what bindings our lambda function queries
        sd = ffc.SentinelDict()
        self.fun(sd)
        result |= sd.keys_checked or set()
        return result

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@define
class IfTrainingDataHasProperty(UniversalTransform):
    fun: Callable  # df -> bool
    then: Transform
    otherwise: Optional[Transform] = None

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.fun(df_fit):
            return self.then.fit(df_fit, bindings=self.bindings())
        elif self.otherwise is not None:
            return self.otherwise.fit(df_fit, bindings=self.bindings())
        return None  # act like Identity

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if state is not None:
            return state.apply(df_apply)
        return df_apply  # act like Identity

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@define
class GroupByBindings(UniversalTransform):
    bindings_sequence: iter[dict]
    transform: ffc.Transform
    include_binding: bool = True

    def _fit(self, df_fit: pd.DataFrame) -> object:
        # TODO: parallelize
        # bindings from above
        base_bindings = self.bindings()
        fits = {}
        for bindings in self.bindings_sequence:
            frozen_bindings = tuple(bindings.items())
            fits[frozen_bindings] = self.transform.fit(
                df_fit, bindings=base_bindings | bindings
            )
        return fits

    def _apply(self, df_apply: pd.DataFrame, state: dict) -> pd.DataFrame:
        # TODO: parallelize
        dfs = []
        for frozen_bindings, fit in state:
            bindings_df = fit.apply(df_apply).assign(**dict(frozen_bindings))
            dfs.append(bindings_df)
            return pd.concat(dfs, axis=0)


@define
class StatelessLambda(StatelessTransform, UniversalTransform):
    apply_fun: Callable  # df[, bindings] -> df

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 1:
            return self.apply_fun(df_apply)
        elif len(sig) == 2:
            return self.apply_fun(df_apply, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")


@define
class StatefulLambda(UniversalTransform):
    fit_fun: Callable  # df[, bindings] -> state
    apply_fun: Callable  # df, state[, bindings] -> df

    def _fit(self, df_fit: pd.DataFrame) -> object:
        sig = inspect.signature(self.fit_fun).parameters
        if len(sig) == 1:
            return self.fit_fun(df_fit)
        elif len(sig) == 2:
            return self.fit_fun(df_fit, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 2:
            return self.apply_fun(df_apply, state)
        elif len(sig) == 3:
            return self.apply_fun(df_apply, state, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 2 or 3 parameters, found {len(sig)}")


@define
class Print(Identity):
    """
    An identity transform that has the side-effect of printing a message at fit- and/or
    apply-time.

    :param fit_msg: Message to print at fit-time.
    :param apply_msg: Message to print at apply-time.
    :param dest: File object to which to print, or the name of a file to open in append
        mode. If ``None`` (default), print to stdout.
    """

    fit_msg: Optional[str] = None
    apply_msg: Optional[str] = None
    dest: Optional[TextIO | str] = None  # if str, will be opened in append mode

    def _fit(self, df_fit: pd.DataFrame):
        if self.fit_msg is None:
            return
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.fit_msg, file=dest)
        else:
            print(self.fit_msg, file=self.dest)

        # Idiomatic super() doesn't work because at call time self is a FitPrint
        # instance, which inherits directly from FitTransform, and not from
        # Print/Identity. Could maybe FIXME by futzing with base classes in the
        # metaprogramming that goes on in core.py
        # return super()._fit(df_fit)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if self.apply_msg is None:
            return df_apply
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.apply_msg, file=dest)
        else:
            print(self.apply_msg, file=self.dest)
        return df_apply
        # Same issue with super() as in _fit().
        # return super()._apply(df_apply)


@define
class LogMessage(Identity):
    """
    An identity transform that has the side-effect of logging a message at fit- and/or
    apply-time. The message string(s) must be fully known at construction-time.

    :param fit_msg: Message to log at fit-time.
    :param apply_msg: Message to log at apply-time.
    :param logger: Logger instance to which to log. If None (default), use
        ``logging.getLogger("frankenfit.transforms")``
    :param level: Level at which to log, default ``INFO``.
    """

    fit_msg: Optional[str] = None
    apply_msg: Optional[str] = None
    logger: Optional[logging.Logger] = None
    level: int = logging.INFO

    def _fit(self, df_fit: pd.DataFrame):
        if self.fit_msg is not None:
            logger = self.logger or _LOG
            logger.log(self.level, self.fit_msg)
        return Identity._fit(self, df_fit)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if self.apply_msg is not None:
            logger = self.logger or _LOG
            logger.log(self.level, self.apply_msg)
        return Identity._apply(self, df_apply, state=state)


class Pipeline(
    BasePipeline.with_methods(
        identity=Identity,
        if_hyperparam_is_true=IfHyperparamIsTrue,
        if_hyperparam_lambda=IfHyperparamLambda,
        if_training_data_has_property=IfTrainingDataHasProperty,
        stateless_lambda=StatelessLambda,
        stateful_lambda=StatefulLambda,
        print=Print,
        log_message=LogMessage,
    )
):
    pass


# UniversalTransform.pipeline_type = Pipeline
