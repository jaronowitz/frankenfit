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
from typing import Any, Callable, Generic, Iterable, Optional, TextIO, TypeVar, cast

from attrs import define

from .params import (
    HP,
    transform,
    UnresolvedHyperparameterError,
)
from .backend import Backend
from .core import (
    BasePipeline,
    Bindings,
    DataInOut,
    FitTransform,
    Grouper,
    P_co,
    SentinelDict,
    StatelessTransform,
    Transform,
    callchain,
)

_LOG = logging.getLogger(__name__)

U = TypeVar("U", bound="UniversalTransform")
T = TypeVar("T")
Obj = TypeVar("Obj")


@transform
class UniversalTransform(Transform):
    def then(
        self, other: Optional[Transform | list[Transform]] = None
    ) -> "UniversalPipelineInterface":
        result = super().then(other)
        return UniversalPipeline(transforms=result.transforms)


class Identity(Generic[T], StatelessTransform[T, T], UniversalTransform):
    """
    The stateless Transform that, at apply-time, simply returns the input data
    unaltered.
    """

    def _apply(self, data_apply: T, state: None) -> T:
        return data_apply

    Self = TypeVar("Self", bound="Identity")

    def fit(
        self: Self,
        data_fit: Optional[T] = None,
        bindings: Optional[Bindings] = None,
        backend: Optional[Backend] = None,
    ) -> FitTransform[Self, T, T]:
        return super().fit(data_fit, bindings, backend)

    def apply(
        self,
        data_apply: Optional[T] = None,
        bindings: Optional[Bindings] = None,
        backend: Optional[Backend] = None,
    ) -> T:
        return super().apply(data_apply, bindings, backend)


@transform
class IfHyperparamIsTrue(UniversalTransform):
    name: str
    then_transform: Transform
    otherwise: Optional[Transform] = None
    allow_unresolved: Optional[bool] = False

    def _fit(self, data_fit: object, bindings: Optional[Bindings] = None) -> Any:
        bindings = bindings or {}
        if (not self.allow_unresolved) and self.name not in bindings:
            raise UnresolvedHyperparameterError(
                f"IfHyperparamIsTrue: no binding for {self.name!r} but "
                "allow_unresolved is False"
            )
        if bindings.get(self.name):
            return self.then_transform.fit(data_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(data_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, data_apply: Any, state: Any) -> Any:
        if state is not None:
            return state.apply(data_apply)
        return data_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        result.add(self.name)
        return result

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@transform
class IfHyperparamLambda(UniversalTransform):
    fun: Callable  # dict[str, object] -> bool
    then_transform: Transform
    otherwise: Optional[Transform] = None

    def _fit(self, data_fit: Any, bindings: Optional[Bindings] = None) -> Any:
        bindings = bindings or {}
        if self.fun(bindings):
            return self.then_transform.fit(data_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(data_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, data_apply: Any, state: Any) -> Any:
        if state is not None:
            return state.apply(data_apply)
        return data_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        # find out what bindings our lambda function queries
        sd = SentinelDict()
        self.fun(sd)
        result |= sd.keys_checked or set()
        return result

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@transform
class IfFittingDataHasProperty(UniversalTransform):
    fun: Callable  # df -> bool
    then_transform: Transform
    otherwise: Optional[Transform] = None

    def _fit(self, data_fit: Any, bindings: Optional[Bindings] = None) -> Any:
        bindings = bindings or {}
        if self.fun(data_fit):
            return self.then_transform.fit(data_fit, bindings=bindings)
        elif self.otherwise is not None:
            return self.otherwise.fit(data_fit, bindings=bindings)
        return None  # act like Identity

    def _apply(self, data_apply: Any, state: Any) -> Any:
        if state is not None:
            return state.apply(data_apply)
        return data_apply  # act like Identity

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@transform
class ForBindings(UniversalTransform):
    bindings_sequence: Iterable[Bindings]
    transform: Transform

    # TODO: consider: a required function parameter for combining the
    # ApplyResults, rather than just returning the raw list of ApplyResults from
    # _apply(). Currently nothing stops us from breaking the decalred DataInOut
    # type of an enclosing pipeline.

    @define
    class FitResult:
        bindings: Bindings
        fit: FitTransform

    @define
    class ApplyResult:
        bindings: Bindings
        result: object  # TODO: make this generic in the result type

    def _fit(
        self, data_fit: Any, base_bindings: Optional[Bindings] = None
    ) -> list[ForBindings.FitResult]:
        # TODO: parallelize
        base_bindings = base_bindings or {}
        fits = []
        for bindings in self.bindings_sequence:
            fits.append(
                ForBindings.FitResult(
                    bindings,
                    self.transform.fit(data_fit, bindings=base_bindings | bindings),
                )
            )
        return fits

    def _apply(
        self, data_apply: Any, state: list[ForBindings.FitResult]
    ) -> list[ForBindings.ApplyResult]:
        # TODO: parallelize
        results = []
        for fit_result in state:
            results.append(
                ForBindings.ApplyResult(
                    fit_result.bindings, fit_result.fit.apply(data_apply)
                )
            )
        return results


@transform
class StatelessLambda(UniversalTransform, StatelessTransform):
    apply_fun: Callable  # df[, bindings] -> df

    def _apply(
        self, data_apply: Any, state: None, bindings: Optional[Bindings] = None
    ) -> Any:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 1:
            return self.apply_fun(data_apply)
        elif len(sig) == 2:
            return self.apply_fun(data_apply, bindings or {})
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")


@transform
class StatefulLambda(UniversalTransform):
    fit_fun: Callable  # df[, bindings] -> state
    apply_fun: Callable  # df, state[, bindings] -> df

    def _fit(self, data_fit: Any, bindings: Optional[Bindings] = None) -> Any:
        sig = inspect.signature(self.fit_fun).parameters
        if len(sig) == 1:
            return self.fit_fun(data_fit)
        elif len(sig) == 2:
            return self.fit_fun(data_fit, bindings or {})
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")

    def _apply(
        self, data_apply: Any, state: Any, bindings: Optional[Bindings] = None
    ) -> Any:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 2:
            return self.apply_fun(data_apply, state)
        elif len(sig) == 3:
            return self.apply_fun(data_apply, state, bindings or {})
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 2 or 3 parameters, found {len(sig)}")


@transform
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

    def _fit(self, data_fit: T) -> None:
        if self.fit_msg is None:
            return
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.fit_msg, file=dest)
        else:
            print(self.fit_msg, file=self.dest)

        return super()._fit(data_fit)

    def _apply(self, data_apply: T, state: None) -> T:
        if self.apply_msg is None:
            return data_apply
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.apply_msg, file=dest)
        else:
            print(self.apply_msg, file=self.dest)

        return super()._apply(data_apply, state)


@transform
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

    def _fit(self, data_fit: Any) -> None:
        if self.fit_msg is not None:
            logger = self.logger or _LOG
            logger.log(self.level, self.fit_msg)
        return super()._fit(data_fit)

    def _apply(self, data_apply: T, state: None) -> T:
        if self.apply_msg is not None:
            logger = self.logger or _LOG
            logger.log(self.level, self.apply_msg)
        return super()._apply(data_apply, state)


class UniversalCallChain(Generic[P_co]):
    @callchain(Identity)
    def identity(  # type: ignore [empty-body]
        self, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append an :class:`Identity` transform to this pipeline.
        """

    @callchain(IfHyperparamIsTrue)
    def if_hyperparam_is_true(  # type: ignore [empty-body]
        self,
        name: str | HP,
        then_transform: Transform | HP,
        otherwise: Transform | HP | None = None,
        allow_unresolved: bool | HP = False,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`IfHyperparamIsTrue` transform to this pipeline.
        """

    @callchain(IfHyperparamLambda)
    def if_hyperparam_lambda(  # type: ignore [empty-body]
        self,
        fun: Callable | HP,
        then_transform: Transform | HP,
        otherwise: Transform | HP | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`IfHyperparamLambda` transform to this pipeline.
        """

    @callchain(IfFittingDataHasProperty)
    def if_fitting_data_has_property(  # type: ignore [empty-body]
        self,
        fun: Callable | HP,
        then_transform: Transform | HP,
        otherwise: Transform | HP | None = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append an :class:`IfFittingDataHasProperty` transform to this pipeline.
        """

    @callchain(StatelessLambda)
    def stateless_lambda(  # type: ignore [empty-body]
        self, apply_fun: Callable | HP, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`StatelessLambda` transform to this pipeline.
        """

    @callchain(StatefulLambda)
    def stateful_lambda(  # type: ignore [empty-body]
        self,
        fit_fun: Callable | HP,
        apply_fun: Callable | HP,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`StatefulLambda` transform to this pipeline.
        """

    @callchain(Print)
    def print(  # type: ignore [empty-body]
        self,
        fit_msg: Optional[str | HP] = None,
        apply_msg: Optional[str | HP] = None,
        dest: Optional[TextIO | str | HP] = None,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`Print` transform to this pipeline.
        """

    @callchain(LogMessage)
    def log_message(  # type: ignore [empty-body]
        self,
        fit_msg: Optional[str | HP] = None,
        apply_msg: Optional[str | HP] = None,
        logger: Optional[logging.Logger] = None,
        level: int | HP = logging.INFO,
        *,
        tag: Optional[str] = None,
    ) -> P_co:
        """
        Append a :class:`LogMessage` transform to this pipeline.
        """


class UniversalGrouper(Generic[P_co], Grouper[P_co], UniversalCallChain[P_co]):
    ...


G_co = TypeVar("G_co", bound=UniversalGrouper, covariant=True)


class UniversalPipelineInterface(
    Generic[DataInOut, G_co, P_co], UniversalCallChain[P_co], BasePipeline[DataInOut]
):
    # Self = TypeVar("Self", bound="UniversalPipelineInterface")

    _Grouper: type[UniversalGrouper[P_co]] = UniversalGrouper[P_co]

    def for_bindings(self, bindings_sequence: Iterable[Bindings]) -> G_co:
        """
        Consume the next transform ``T`` in the call-chain by appending
        ``ForBindings(bindings_sequence=..., transform=T)`` to this pipeline.
        """
        grouper = type(self)._Grouper(
            self,
            ForBindings,
            "transform",
            bindings_sequence=bindings_sequence,
        )
        return cast(G_co, grouper)


class UniversalPipeline(
    Generic[DataInOut],
    UniversalPipelineInterface[
        DataInOut, UniversalGrouper["UniversalPipeline"], "UniversalPipeline"
    ],
):
    ...
