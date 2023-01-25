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
    ClassVar,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TextIO,
    Type,
    TypeVar,
    cast,
)

from attrs import NOTHING, define, field

from .core import (
    Bindings,
    DataType,
    FitTransform,
    Future,
    Grouper,
    P_co,
    Pipeline,
    PipelineMember,
    R,
    R_co,
    SentinelDict,
    StatelessTransform,
    Transform,
    callchain,
)
from .params import HP, UnresolvedHyperparameterError, params

_LOG = logging.getLogger(__name__)

U = TypeVar("U", bound="UniversalTransform")
T = TypeVar("T")


class FitUniversalTransform(Generic[R_co, DataType], FitTransform[R_co, DataType]):
    # The point of this is just to specialize the type signature to help IDEs
    def then(
        self,
        other: PipelineMember | Sequence[PipelineMember] | None = None,
    ) -> "UniversalPipeline":
        result = super().then(other)
        return UniversalPipeline(transforms=result.transforms)


@params
class UniversalTransform(Generic[DataType], Transform[DataType]):
    fit_transform_class: ClassVar[Type[FitTransform]] = FitUniversalTransform

    # The point of this is just to specialize the type signature to help IDEs
    def then(
        self,
        other: PipelineMember | Sequence[PipelineMember] | None = None,
    ) -> "UniversalPipeline":
        result = super().then(other)
        return UniversalPipeline(transforms=result.transforms)

    def fit(
        self: R,
        data_fit: Optional[DataType | Future[DataType]] = None,
        bindings: Optional[Bindings] = None,
        /,
        **kwargs,
    ) -> FitUniversalTransform[R, DataType]:
        return cast(
            FitUniversalTransform[R, DataType],
            super().fit(data_fit, bindings, **kwargs),
        )


class Identity(Generic[T], StatelessTransform[T], UniversalTransform[T]):
    """
    The stateless Transform that, at apply-time, simply returns the input data
    unaltered.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`
    """

    def _apply(self, data_apply: T, state: None) -> T:
        return data_apply

    _Self = TypeVar("_Self", bound="Identity")

    # The overrides below are just to present a more specific type signature

    def fit(
        self: _Self,
        data_fit: Optional[T | Future[T]] = None,
        bindings: Optional[Bindings] = None,
        /,
        **kwargs,
    ) -> FitUniversalTransform[_Self, T]:
        return super().fit(data_fit, bindings)

    def apply(
        self,
        data_apply: Optional[T | Future[T]] = None,
        bindings: Optional[Bindings] = None,
        /,
        **kwargs,
    ) -> T:
        return super().apply(data_apply, bindings)


@params
class StateOf(Generic[DataType], Transform[DataType]):
    transform: PipelineMember

    def _submit_fit(
        self,
        data_fit: Optional[DataType | Future[DataType]] = None,
        bindings: Optional[Bindings] = None,
    ) -> Any:
        if isinstance(self.transform, FitTransform):
            return self.transform
        assert isinstance(self.transform, Transform)
        with self.parallel_backend() as backend:
            return backend.fit(self.transform, data_fit, bindings)

    def _apply(self, data_apply: DataType, state: Any) -> DataType:
        state = cast(FitTransform[Any, DataType], state)
        return state.materialize_state().state()


@params
class BranchTransform(UniversalTransform):
    def _submit_apply(
        self,
        data_apply: Optional[Any | Future[Any]] = None,
        state: FitTransform | None = None,
    ) -> Future[Any] | None:
        with self.parallel_backend() as backend:
            if state is None:
                # behave like identity if the condition was false and there is no
                # otherwise transform
                return backend.maybe_put(data_apply)

            return backend.apply(state, data_apply)

    def _visualize(self, digraph, bg_fg: tuple[str, str]) -> tuple[list, list]:
        entries, exits = super()._visualize(digraph, bg_fg)
        if getattr(self, "otherwise", None) is None:
            exits = exits + [(self.name, "otherwise")]
        return entries, exits

    def _materialize_state(self, state: FitTransform | None):
        return state.materialize_state() if state is not None else state


@params
class IfHyperparamIsTrue(BranchTransform):
    hp_name: str
    then_transform: Transform
    otherwise: Optional[Transform] = None
    allow_unresolved: bool = False

    def _submit_fit(
        self,
        data_fit: Any | Future[Any] | None = None,
        bindings: Optional[Bindings] = None,
    ) -> FitTransform | None:
        bindings = bindings or {}
        if (not self.allow_unresolved) and self.hp_name not in bindings:
            raise UnresolvedHyperparameterError(
                f"IfHyperparamIsTrue: no binding for {self.hp_name!r} but "
                "allow_unresolved is False"
            )
        with self.parallel_backend() as backend:
            if bindings.get(self.hp_name):
                return backend.push_trace("then").fit(
                    self.then_transform, data_fit, bindings
                )
            elif self.otherwise is not None:
                return backend.push_trace("otherwise").fit(
                    self.otherwise, data_fit, bindings
                )
            return None

    def hyperparams(self) -> set[str]:
        result = super().hyperparams()
        result.add(self.hp_name)
        return result


@params
class IfHyperparamLambda(BranchTransform):
    fun: Callable  # dict[str, object] -> bool
    then_transform: Transform
    otherwise: Optional[Transform] = None

    def _submit_fit(
        self,
        data_fit: Any | Future[Any] | None = None,
        bindings: Optional[Bindings] = None,
    ) -> FitTransform | None:
        bindings = bindings or {}
        with self.parallel_backend() as backend:
            if self.fun(bindings):
                return backend.push_trace("then").fit(
                    self.then_transform, data_fit, bindings
                )
            elif self.otherwise is not None:
                return backend.push_trace("otherwise").fit(
                    self.otherwise, data_fit, bindings
                )
            return None

    def hyperparams(self) -> set[str]:
        # Note we don't use UserLambdaHyperparams because the lambda receives the WHOLE
        # bindings dict
        result = super().hyperparams()
        # find out what bindings our lambda function queries
        sd = SentinelDict()
        self.fun(sd)
        result |= sd.keys_checked or set()
        return result


@params(auto_attribs=False)
class IfFittingDataHasProperty(BranchTransform):
    test_transform: StatelessLambda

    fun: Callable = field()  # df -> bool
    then_transform: Transform = field()
    otherwise: Optional[Transform] = field(default=None)

    _Self = TypeVar("_Self", bound="IfFittingDataHasProperty")

    def __attrs_post_init__(self):
        self.test_transform = StatelessLambda(self.fun)

    def hyperparams(self) -> set[str]:
        return self.test_transform.hyperparams()

    def _submit_fit(
        self,
        data_fit: Any | Future[Any] | None = None,
        bindings: Optional[Bindings] = None,
    ) -> FitTransform | None:
        bindings = bindings or {}
        with self.parallel_backend() as backend:
            test_result: bool = self.test_transform.on_backend(
                backend.push_trace("test")
            ).apply(data_fit, bindings)
            if test_result:
                return backend.push_trace("then").fit(
                    self.then_transform, data_fit, bindings
                )
            elif self.otherwise is not None:
                return backend.push_trace("otherwise").fit(
                    self.otherwise, data_fit, bindings
                )
            else:
                return None


@params
class ForBindings(Generic[DataType], UniversalTransform[DataType]):
    bindings_sequence: Sequence[Bindings]
    transform: Transform
    combine_fun: Callable[[Sequence[ForBindings.ApplyResult]], DataType]

    @define
    class FitResult:
        bindings: Bindings
        fit: FitTransform

    @define
    class ApplyResult:
        bindings: Bindings
        result: Any  # TODO: make this generic in DataType type?

    def _submit_fit(
        self,
        data_fit: DataType | Future[DataType] | None = None,
        bindings: Optional[Bindings] = None,
    ) -> list[ForBindings.FitResult]:
        base_bindings = bindings or {}
        fits: list[ForBindings.FitResult] = []
        with self.parallel_backend() as backend:
            if len(self.bindings_sequence) > 0:
                data_fit = backend.maybe_put(data_fit)
            for i, bindings in enumerate(self.bindings_sequence):
                fits.append(
                    ForBindings.FitResult(
                        bindings,
                        # submit in parallel on backend
                        backend.push_trace(f"[{i}]").fit(
                            self.transform,
                            data_fit,
                            {**base_bindings, **bindings},
                        ),
                    )
                )
            # materialize all states. this is where we wait for all the parallel _fit
            # tasks to complete
            # for fit_result in fits:
            #     fit_result.fit = fit_result.fit.materialize_state()
        return fits

    def _combine_results(self, bindings_seq, *results) -> DataType:
        assert len(bindings_seq) == len(results)
        return self.combine_fun(
            [
                ForBindings.ApplyResult(bindings, result)
                for bindings, result in zip(bindings_seq, results)
            ]
        )

    def _submit_apply(
        self,
        data_apply: Optional[DataType | Future[DataType]] = None,
        state: list[ForBindings.FitResult] | None = None,
    ) -> Future[DataType]:
        assert state is not None
        with self.parallel_backend() as backend:
            if len(self.bindings_sequence) > 0:
                data_apply = backend.maybe_put(data_apply)
            bindings = []
            results: list[Future[DataType]] = []
            for i, fit_result in enumerate(state):
                bindings.append(fit_result.bindings)
                results.append(
                    backend.push_trace(f"[{i}]").apply(fit_result.fit, data_apply),
                )
            return backend.submit(
                "_combine_results",
                self._combine_results,
                bindings,
                *results,
            )


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
class StatelessLambda(
    Generic[DataType],
    UniversalTransform[DataType],
    StatelessTransform[DataType],
):
    """
    Generically wrap a user-supplied function as a :class:`StatelessTransform`. At
    apply-time, the input data is passed unaltered to ``apply_fun``, the result of which
    is then the result of the Transform.

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    apply_fun: Callable
        The function through which to pass the apply-time data. ``apply_fun`` will be
        called with the apply-time data as its first argument. Any additional arguments
        in ``apply_fun``'s signature are treated as the names of hyperparameters, whose
        values will also be supplied as named arguments with ``apply_fun`` is called.
        See XXX for an overview of referencing hyperparameters from user-supplied
        callable parameters.

    Examples
    --------
    ::

        df = diamonds_df
        result = (
            ff.UniversalPipeline()
            .stateless_lambda(lambda df: df.rename(columns={"price": "price_orig"}))
            .apply(df)
        )
        assert result.equals(df.rename(columns={"price": "price_orig"}))

    With ``apply_fun`` referencing a hyperparameter named "response"::

        pip = ff.UniversalPipeline().stateless_lambda(
            lambda df, response: df.rename(columns={response: "foo"})
        )
        result = pip.apply(df, {"response": "price"})
        assert result.equals(df.rename(columns={"price": "foo"}))
    """

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

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    fit_msg: str, optional
        Message to print at fit-time.
    apply_msg: str, optional
        Message to print at apply-time.
    dest: TextIO | str, optional
        File object to which to print, or the name of a file to open in append mode. If
        ``None`` (default), print to stdout.
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

    🏳️ :class:`Stateless <frankenfit.StatelessTransform>`

    Parameters
    ----------
    fit_msg: str, optional
        Message to log at fit-time.

    apply_msg: str, optional
        Message to log at apply-time.

    logger_name: str, optional
        Logger instance to which to log. If ``None`` (default), use
        ``logging.getLogger("frankenfit.universal")``

    level: int, optional
        Level at which to log, default ``INFO``.
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
        Append an :class:`~frankenfit.universal.Identity` transform to this pipeline.
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
        Append an :class:`~frankenfit.universal.IfHyperparamIsTrue` transform to this
        pipeline.
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
        Append an :class:`~frankenfit.universal.IfHyperparamLambda` transform to this
        pipeline.
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
        Append an :class:`~frankenfit.universal.IfFittingDataHasProperty` transform to
        this pipeline.
        """

    @callchain(StatelessLambda)
    def stateless_lambda(  # type: ignore [empty-body]
        self, apply_fun: Callable, *, tag: Optional[str] = None
    ) -> P_co:
        """
        Append a :class:`~frankenfit.universal.StatelessLambda` transform to this
        pipeline.
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
        Append a :class:`~frankenfit.universal.StatefulLambda` transform to this
        pipeline.
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
        Append a :class:`~frankenfit.universal.Print` transform to this pipeline.
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
        Append a :class:`~frankenfit.universal.LogMessage` transform to this pipeline.
        """


class UniversalGrouper(Generic[P_co], Grouper[P_co], UniversalCallChain[P_co]):
    ...


G_co = TypeVar("G_co", bound=UniversalGrouper, covariant=True)
SelfUPI = TypeVar("SelfUPI", bound="UniversalPipelineInterface")


class UniversalPipelineInterface(
    Generic[DataType, G_co, P_co], UniversalCallChain[P_co], Pipeline[DataType]
):

    _Grouper: type[UniversalGrouper[P_co]] = UniversalGrouper[P_co]

    def for_bindings(
        self,
        bindings_sequence: Iterable[Bindings],
        combine_fun: Callable[[Sequence[ForBindings.ApplyResult]], DataType],
        *,
        tag: str | None = None,
    ) -> G_co:
        """
        Consume the next transform ``t`` in the call-chain by appending
        :class:`ForBindings(bindings_sequence=..., combine_fun=..., transform=t)
        <frankenfit.universal.ForBindings>` to this pipeline.

        Examples
        --------
        ::

            from pydataset import data
            df = data("diamonds")
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
                assert x.result.equals(df[[x.bindings["target_col"]]])
        """
        grouper = type(self)._Grouper(
            self,
            ForBindings,
            "transform",
            bindings_sequence=bindings_sequence,
            combine_fun=combine_fun,
            tag=tag if tag is not None else NOTHING,
        )
        return cast(G_co, grouper)

    def last_state(self: SelfUPI) -> SelfUPI:
        """
        Replace the last transform ``t`` in the pipeline with
        :class:`StateOf(t) <frankenfit.universal.StateOf>`.

        Raises
        ------
        ValueError
            If the pipeline is empty.
        """
        if not len(self.transforms):
            raise ValueError("last_state: undefined on empty pipeline")

        self_copy = self + []
        self_copy.transforms.append(StateOf(self_copy.transforms.pop()))
        return self_copy


class UniversalPipeline(
    Generic[DataType],
    UniversalPipelineInterface[
        DataType, UniversalGrouper["UniversalPipeline"], "UniversalPipeline"
    ],
):
    fit_transform_class: ClassVar[Type[FitTransform]] = FitUniversalTransform

    def fit(
        self: R,
        data_fit: Optional[DataType | Future[DataType]] = None,
        bindings: Optional[Bindings] = None,
        /,
        **kwargs,
    ) -> FitUniversalTransform[R, DataType]:
        return cast(
            FitUniversalTransform[R, DataType],
            super().fit(data_fit, bindings, **kwargs),
        )
