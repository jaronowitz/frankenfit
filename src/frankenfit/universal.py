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
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    cast,
)

from attrs import define, field

from .params import (
    HP,
    params,
    UnresolvedHyperparameterError,
)
from .core import (
    BasePipeline,
    Bindings,
    DataIn,
    DataResult,
    DataInOut,
    FitTransform,
    Future,
    Grouper,
    LocalBackend,
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


@params
class UniversalTransform(Generic[DataIn, DataResult], Transform[DataIn, DataResult]):
    def then(
        self, other: Optional[Transform | list[Transform]] = None
    ) -> "UniversalPipeline":
        result = super().then(other)
        return UniversalPipeline(transforms=result.transforms)


class Identity(Generic[T], StatelessTransform[T, T], UniversalTransform[T, T]):
    """
    The stateless Transform that, at apply-time, simply returns the input data
    unaltered.
    """

    def _apply(self, data_apply: T, state: None) -> T:
        return data_apply

    _Self = TypeVar("_Self", bound="Identity")

    # The overrides below are just to present a more specific type signature

    def fit(
        self: _Self,
        data_fit: Optional[T | Future[T]] = None,
        bindings: Optional[Bindings] = None,
    ) -> FitTransform[_Self, T, T]:
        return super().fit(data_fit, bindings)

    def apply(
        self,
        data_apply: Optional[T | Future[T]] = None,
        bindings: Optional[Bindings] = None,
    ) -> T:
        return super().apply(data_apply, bindings)


@params
class IfHyperparamIsTrue(UniversalTransform):
    name: str
    then_transform: Transform
    otherwise: Optional[Transform] = None
    allow_unresolved: bool = False

    def _fit(
        self, data_fit: Any, bindings: Optional[Bindings] = None
    ) -> FitTransform | None:
        bindings = bindings or {}
        if (not self.allow_unresolved) and self.name not in bindings:
            raise UnresolvedHyperparameterError(
                f"IfHyperparamIsTrue: no binding for {self.name!r} but "
                "allow_unresolved is False"
            )
        local = LocalBackend()
        if bindings.get(self.name):
            return local.fit(
                self.then_transform.on_backend(self.backend), data_fit, bindings
            ).materialize_state()
        elif self.otherwise is not None:
            return local.fit(
                self.otherwise.on_backend(self.backend), data_fit, bindings
            ).materialize_state()
        return None  # act like Identity

    def _apply(self, data_apply: Any, state: FitTransform | None) -> Any:
        if state is not None:
            local = LocalBackend()
            return local.apply(state.on_backend(self.backend), data_apply).result()
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


@params
class IfHyperparamLambda(UniversalTransform):
    fun: Callable  # dict[str, object] -> bool
    then_transform: Transform
    otherwise: Optional[Transform] = None

    def _fit(
        self, data_fit: Any, bindings: Optional[Bindings] = None
    ) -> FitTransform | None:
        bindings = bindings or {}
        local = LocalBackend()
        if self.fun(bindings):
            return local.fit(
                self.then_transform.on_backend(self.backend), data_fit, bindings
            ).materialize_state()
        elif self.otherwise is not None:
            return local.fit(
                self.otherwise.on_backend(self.backend), data_fit, bindings
            ).materialize_state()
        return None  # act like Identity

    def _apply(self, data_apply: Any, state: FitTransform | None) -> Any:
        if state is not None:
            local = LocalBackend()
            return local.apply(state.on_backend(self.backend), data_apply).result()
        return data_apply  # act like Identity

    def hyperparams(self) -> set[str]:
        # Note we don't use UserLambdaHyperparams because the lambda receives the WHOLE
        # bindings dict
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


@params(auto_attribs=False)
class IfFittingDataHasProperty(UniversalTransform):
    fun: Callable = field()  # df -> bool
    then_transform: Transform = field()
    # TODO: we should really use UserLambdaHyperparams

    fun_bindings: Bindings
    fun_hyperparams: UserLambdaHyperparams

    otherwise: Optional[Transform] = field(default=None)

    _Self = TypeVar("_Self", bound="IfFittingDataHasProperty")

    def __attrs_post_init__(self):
        if isinstance(self.fun, HP):
            raise TypeError(
                f"IfFittingDataHasProperty.fun must not be a hyperparameter; got:"
                f"{self.fun!r}. Instead consider supplying a function that "
                f"requests hyperparameter bindings in its parameter signature."
            )
        self.fun_bindings = {}
        self.fun_hyperparams = UserLambdaHyperparams.from_function_sig(self.fun, 1)

    def hyperparams(self) -> set[str]:
        return super().hyperparams().union(self.fun_hyperparams.required_or_optional())

    def resolve(self: _Self, bindings: Optional[Bindings] = None) -> _Self:
        # override _resolve_hyperparams() to collect hyperparam bindings at fit-time
        resolved_self = super().resolve(bindings)
        resolved_self.fun_bindings = self.fun_hyperparams.collect_bindings(
            bindings or {}
        )
        return resolved_self

    def _fit(
        self, data_fit: Any, bindings: Optional[Bindings] = None
    ) -> FitTransform | None:
        bindings = bindings or {}
        local = LocalBackend()
        if self.fun(data_fit, **self.fun_bindings):
            return local.fit(
                self.then_transform.on_backend(self.backend), data_fit, bindings
            ).materialize_state()
        elif self.otherwise is not None:
            return local.fit(
                self.otherwise.on_backend(self.backend), data_fit, bindings
            ).materialize_state()
        return None  # act like Identity

    def _apply(self, data_apply: Any, state: FitTransform | None) -> Any:
        if state is not None:
            local = LocalBackend()
            return local.apply(state.on_backend(self.backend), data_apply).result()
        return data_apply  # act like Identity

    def _visualize(self, digraph, bg_fg: tuple[str, str]):
        entries, exits = super()._visualize(digraph, bg_fg)
        if self.otherwise is None:
            exits = exits + [(self.tag, "otherwise")]
        return entries, exits


@params
class ForBindings(Generic[DataIn, DataResult], UniversalTransform[DataIn, DataResult]):
    bindings_sequence: Sequence[Bindings]
    transform: Transform
    combine_fun: Callable[[Sequence[ForBindings.ApplyResult]], DataResult]

    @define
    class FitResult:
        bindings: Bindings
        fit: FitTransform

    @define
    class ApplyResult:
        bindings: Bindings
        result: Any  # TODO: make this generic in DataResult type?

    def _fit(
        self, data_fit: Any, base_bindings: Optional[Bindings] = None
    ) -> list[ForBindings.FitResult]:
        assert self.backend is not None
        base_bindings = base_bindings or {}
        fits: list[ForBindings.FitResult] = []
        with self.backend.submitting_from_transform() as backend:
            if len(self.bindings_sequence) > 0:
                data_fit = backend.maybe_put(data_fit)
            for bindings in self.bindings_sequence:
                fits.append(
                    ForBindings.FitResult(
                        bindings,
                        # submit in parallel on backend
                        backend.fit(
                            self.transform.on_backend(backend),
                            data_fit,
                            bindings={**base_bindings, **bindings},
                        ),
                    )
                )
            # materialize all states. this is where we wait for all the parallel _fit
            # tasks to complete
            for fit_result in fits:
                fit_result.fit = fit_result.fit.materialize_state()
        return fits

    def _apply(self, data_apply: Any, state: list[ForBindings.FitResult]) -> DataResult:
        assert self.backend is not None
        with self.backend.submitting_from_transform() as backend:
            if len(self.bindings_sequence) > 0:
                data_apply = backend.maybe_put(data_apply)
            results: list[ForBindings.ApplyResult] = []
            for fit_result in state:
                results.append(
                    ForBindings.ApplyResult(
                        fit_result.bindings,
                        # submit in parallel on backend
                        backend.apply(fit_result.fit.on_backend(backend), data_apply),
                    )
                )
            # materialize all results before sending to combine_fun. This is where we
            # wait for all the parallel _apply tasks to finish.
            for apply_result in results:
                apply_result.result = cast(
                    Future[DataResult], apply_result.result
                ).result()
        return self.combine_fun(results)


PROHIBITED_USER_LAMBDA_PARAMETER_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
)


@define
class UserLambdaHyperparams:
    required: set[str]
    optional: set[str]

    _Self = TypeVar("_Self", bound="UserLambdaHyperparams")

    @classmethod
    def from_function_sig(cls: type[_Self], fun: Callable, n_data_args: int) -> _Self:
        required: set[str] = set()
        optional: set[str] = set()
        fun_params = inspect.signature(fun).parameters
        if len(fun_params) <= n_data_args:
            return cls(required=required, optional=optional)

        for name, info in list(fun_params.items())[n_data_args:]:
            if info.kind in PROHIBITED_USER_LAMBDA_PARAMETER_KINDS:
                raise TypeError(
                    f"Filter: user lambda function's signature must allow requested "
                    f"hyperparameters to be supplied non-variadically by name at "
                    f"call-time but parameter {name!r} has kind {info.kind}. Full "
                    f"signature: {inspect.signature(fun)}"
                )
            if info.default is inspect._empty:
                required.add(name)
            else:
                optional.add(name)

        return cls(required=required, optional=optional)

    def required_or_optional(self):
        return self.required.union(self.optional)

    def collect_bindings(self, bindings: Bindings) -> Bindings:
        result: Bindings = {}
        missing: set[str] = set()
        for hp in self.required:
            try:
                result[hp] = bindings[hp]
            except KeyError:
                missing.add(hp)
                continue
        if missing:
            raise UnresolvedHyperparameterError(
                f"Requested the values of hyperparameters {self.required}, but the "
                f"following hyperparameters were not resolved at fit-time: {missing}. "
                f"Bindings were: {bindings}"
            )
        for hp in self.optional:
            try:
                result[hp] = bindings[hp]
            except KeyError:
                continue
        return result


@params(auto_attribs=False)
class StatelessLambda(UniversalTransform, StatelessTransform):
    apply_fun: Callable = field()  # df[, bindings] -> df

    apply_fun_bindings: Bindings
    apply_fun_hyperparams: UserLambdaHyperparams

    _Self = TypeVar("_Self", bound="StatelessLambda")

    def __attrs_post_init__(self):
        if isinstance(self.apply_fun, HP):
            raise TypeError(
                f"StatelessLambda.apply_fun must not be a hyperparameter; got:"
                f"{self.apply_fun!r}. Instead consider supplying a function that "
                f"requests hyperparameter bindings in its parameter signature."
            )
        self.apply_fun_bindings = {}
        self.apply_fun_hyperparams = UserLambdaHyperparams.from_function_sig(
            self.apply_fun, 1
        )

    def hyperparams(self) -> set[str]:
        return (
            super()
            .hyperparams()
            .union(self.apply_fun_hyperparams.required_or_optional())
        )

    def resolve(self: _Self, bindings: Optional[Bindings] = None) -> _Self:
        # override _resolve_hyperparams() to collect hyperparam bindings at fit-time
        # so we don't need bindings arg to _apply.
        resolved_self = super().resolve(bindings)
        resolved_self.apply_fun_bindings = self.apply_fun_hyperparams.collect_bindings(
            bindings or {}
        )
        return resolved_self

    def _apply(self, data_apply: Any, state: None) -> Any:
        fun_params = inspect.signature(self.apply_fun).parameters
        positional_args = (data_apply,) if len(fun_params) > 0 else tuple()
        return self.apply_fun(*positional_args, **self.apply_fun_bindings)


@params(auto_attribs=False)
class StatefulLambda(UniversalTransform):
    fit_fun: Callable = field()  # df[, bindings] -> state
    apply_fun: Callable = field()  # df, state[, bindings] -> df

    fit_fun_bindings: Bindings
    fit_fun_hyperparams: UserLambdaHyperparams
    apply_fun_bindings: Bindings
    apply_fun_hyperparams: UserLambdaHyperparams

    _Self = TypeVar("_Self", bound="StatefulLambda")

    def __attrs_post_init__(self):
        if isinstance(self.fit_fun, HP):
            raise TypeError(
                f"StatefulLambda.fit_fun must not be a hyperparameter; got:"
                f"{self.fit_fun!r}. Instead consider supplying a function that "
                f"requests hyperparameter bindings in its parameter signature."
            )
        if isinstance(self.apply_fun, HP):
            raise TypeError(
                f"StatefulLambda.apply_fun must not be a hyperparameter; got:"
                f"{self.apply_fun!r}. Instead consider supplying a function that "
                f"requests hyperparameter bindings in its parameter signature."
            )
        self.fit_fun_bindings = {}
        self.fit_fun_hyperparams = UserLambdaHyperparams.from_function_sig(
            self.fit_fun, 1
        )
        self.apply_fun_bindings = {}
        self.apply_fun_hyperparams = UserLambdaHyperparams.from_function_sig(
            self.apply_fun, 2
        )

    def hyperparams(self) -> set[str]:
        return (
            super()
            .hyperparams()
            .union(self.fit_fun_hyperparams.required_or_optional())
            .union(self.apply_fun_hyperparams.required_or_optional())
        )

    def resolve(self: _Self, bindings: Optional[Bindings] = None) -> _Self:
        # override _resolve_hyperparams() to collect hyperparam bindings at fit-time
        # so we don't need bindings arg to _apply.
        resolved_self = super().resolve(bindings)
        resolved_self.fit_fun_bindings = self.fit_fun_hyperparams.collect_bindings(
            bindings or {}
        )
        resolved_self.apply_fun_bindings = self.apply_fun_hyperparams.collect_bindings(
            bindings or {}
        )
        return resolved_self

    def _fit(self, data_fit: Any) -> Any:
        fun_params = inspect.signature(self.fit_fun).parameters
        positional_args = (data_fit,) if len(fun_params) > 0 else tuple()
        return self.fit_fun(*positional_args, **self.fit_fun_bindings)

    def _apply(self, data_apply: Any, state: Any) -> Any:
        fun_params = inspect.signature(self.apply_fun).parameters
        positional_args: tuple = tuple()
        if len(fun_params) > 0:
            positional_args += (data_apply,)
        if len(fun_params) > 1:
            positional_args += (state,)
        return self.apply_fun(*positional_args, **self.apply_fun_bindings)


@params
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


@params
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
    logger_name: Optional[str] = None
    level: int = logging.INFO

    def _fit(self, data_fit: Any) -> None:
        if self.fit_msg is not None:
            if self.logger_name is not None:
                logger = logging.getLogger(self.logger_name)
            else:
                logger = _LOG
            logger.log(self.level, self.fit_msg)
        return super()._fit(data_fit)

    def _apply(self, data_apply: T, state: None) -> T:
        if self.apply_msg is not None:
            if self.logger_name is not None:
                logger = logging.getLogger(self.logger_name)
            else:
                logger = _LOG
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
        self, apply_fun: Callable, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`StatelessLambda` transform to this pipeline.
        """

    @callchain(StatefulLambda)
    def stateful_lambda(  # type: ignore [empty-body]
        self,
        fit_fun: Callable,
        apply_fun: Callable,
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

    def for_bindings(
        self,
        bindings_sequence: Iterable[Bindings],
        combine_fun: Callable[[Sequence[ForBindings.ApplyResult]], DataInOut],
    ) -> G_co:
        """
        Consume the next transform ``T`` in the call-chain by appending
        ``ForBindings(bindings_sequence=..., combine_fun=..., transform=T)`` to
        this pipeline.
        """
        grouper = type(self)._Grouper(
            self,
            ForBindings,
            "transform",
            bindings_sequence=bindings_sequence,
            combine_fun=combine_fun,
        )
        return cast(G_co, grouper)


class UniversalPipeline(
    Generic[DataInOut],
    UniversalPipelineInterface[
        DataInOut, UniversalGrouper["UniversalPipeline"], "UniversalPipeline"
    ],
):
    ...
