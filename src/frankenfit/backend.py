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
Classes used by the core module (and some Transform subclasses) to abstract over
computational backends: in-process pandas, dask-distributed, and maybe someday
ray.
"""

from __future__ import annotations
from contextlib import contextmanager

import logging
from typing import Any, ClassVar, Iterator, cast, Callable, Optional, TypeVar, Generic
import uuid
import warnings

from attrs import define, field

try:
    from dask import distributed
    from dask.base import tokenize
except ImportError:  # pragma: no cover
    distributed = None  # type: ignore [assignment]

from .core import Future, Backend

_LOG = logging.getLogger(__name__)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
D = TypeVar("D", bound="DaskBackend")


def _convert_to_address(obj: str | None | distributed.Client):
    if distributed is None:  # pragma: no cover
        warnings.warn(
            "Creating a DaskBackend but dask.distributed is not installed. Try "
            'installing frankenfit with the "dask" extra; that is:  `pip install '
            "frankenfit[dask]`."
        )
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, distributed.Client):
        return obj.scheduler.address
    raise TypeError(f"Don't know how to create DaskBackend from {type(obj)}: {obj!r}")


@define
class DaskFuture(Generic[T_co], Future[T_co]):
    dask_future: distributed.Future

    def result(self) -> T_co:
        return cast(T_co, self.dask_future.result())

    def unwrap(self) -> distributed.Future:
        return self.dask_future

    @staticmethod
    def unwrap_or_result(obj):
        if isinstance(obj, DaskFuture):
            return obj.unwrap()
        if isinstance(obj, Future):
            # A future from some other backend, so we need to materialize it.
            # this will probably emit a warning about putting a large object
            # into the scheduler
            return obj.result()
        return obj


@define
class DaskBackend(Backend):
    FutureType: ClassVar[type[Future]] = DaskFuture

    address: Optional[str] = field(converter=_convert_to_address, default=None)
    transform_names: tuple[str, ...] = tuple()

    def put(self, data: T) -> DaskFuture[T]:
        from dask import distributed

        client: distributed.Client = distributed.get_client(self.address)
        _LOG.debug("%r: scattering data of type %s", self, type(data))
        return DaskFuture(client.scatter(data))

    def submit(
        self,
        key_prefix: str,
        function: Callable,
        *function_args,
        pure: bool = True,
        **function_kwargs,
    ) -> DaskFuture[Any]:
        # attempt import so that we fail with a sensible exception if distributed is not
        # installed:
        from dask import distributed

        args = tuple(DaskFuture.unwrap_or_result(a) for a in function_args)
        kwargs = {k: DaskFuture.unwrap_or_result(v) for k, v in function_kwargs.items()}
        if pure:
            token = tokenize(function, function_kwargs, *function_args)
        else:
            token = str(uuid.uuid4())
        key = " / ".join(self.transform_names + (key_prefix,)) + "-" + token
        # hmm, there could be a problem here with collision between function
        # kwargs and submit kwargs, but this is inherent to distributed's API
        # design :/. In general I suppose callers should prefer to provide
        # everything as positoinal arguments.
        client: distributed.Client = distributed.get_client(self.address)
        _LOG.debug("%r: submitting task %r to %r", self, key, client)
        fut = client.submit(function, *args, key=key, **kwargs)
        return DaskFuture(fut)

    @contextmanager
    def submitting_from_transform(self: D, name: str = "") -> Iterator[D]:
        client: distributed.Client = distributed.get_client(self.address)
        try:
            worker = distributed.get_worker()
        except (AttributeError, ValueError):
            # we're not actually running on a dask.ditributed worker
            worker = None

        self_copy = type(self)(
            address=self.address, transform_names=self.transform_names + (name,)
        )

        if (
            worker is None
            or client.scheduler.address != worker.client.scheduler.address
        ):
            # Either we're not running on a worker, OR (weirdly) we're running on a
            # worker in a different cluster than this backend object's.
            yield self_copy
            return

        # See:
        # https://distributed.dask.org/en/stable/_modules/distributed/worker_client.html
        from distributed.metrics import time
        from distributed.worker import thread_state
        from distributed.threadpoolexecutor import rejoin, secede
        from distributed.worker_state_machine import SecedeEvent

        _LOG.debug("%r.submitting_from_transform(): worker seceding", self)
        duration = time() - thread_state.start_time
        secede()  # have this thread secede from the thread pool
        worker.loop.add_callback(
            worker.handle_stimulus,
            SecedeEvent(
                key=thread_state.key,
                compute_duration=duration,
                stimulus_id=f"worker-client-secede-{time()}",
            ),
        )
        try:
            yield self_copy
        finally:
            _LOG.debug("%r.submitting_from_transform(): worker rejoining", self)
            rejoin()
