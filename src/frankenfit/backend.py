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
Classes used by the core module (and some Transform subclasses) to abstract over
computational backends: in-process pandas, dask-distributed, and maybe someday
ray.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar, cast

from attrs import define, field

try:
    from dask import distributed
    from dask.base import tokenize
except ImportError:  # pragma: no cover
    distributed = None  # type: ignore [assignment]

from .core import Backend, Future

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

    def __eq__(self, other):
        if type(self) is not type(other):  # pragma: no cover
            return False
        return self.dask_future.key == other.dask_future.key

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

    def belongs_to(self, backend: Backend) -> bool:
        if not isinstance(backend, DaskBackend):
            return False
        from dask import distributed

        client: distributed.Client = distributed.get_client(backend.addr)
        return self.unwrap().client.scheduler.address == client.scheduler.address

    def __dask_tokenize__(self):
        return tokenize(self.dask_future)


@define
class DaskBackend(Backend):
    addr: Optional[str] = field(converter=_convert_to_address, default=None)
    trace: tuple[str, ...] = tuple()

    def put(self, data: T) -> DaskFuture[T]:
        from dask import distributed

        client: distributed.Client = distributed.get_client(self.addr)
        _LOG.debug("%r: scattering data of type %s", self, type(data))
        # regarding hash=False, see:
        # https://github.com/dask/distributed/issues/4612
        # https://github.com/dask/distributed/issues/3703
        # regarding [data]: we wrap the data in a list to prevent scatter() from
        # mangling list or dict input
        return DaskFuture(client.scatter([data], hash=False)[0])

    def submit(
        self,
        key_prefix: str,
        function: Callable,
        *function_args,
        pure: bool = True,
        **function_kwargs,
    ) -> DaskFuture[Any]:
        args = tuple(DaskFuture.unwrap_or_result(a) for a in function_args)
        kwargs = {k: DaskFuture.unwrap_or_result(v) for k, v in function_kwargs.items()}
        if pure:
            token = tokenize(function, function_kwargs, *function_args)
        else:
            token = str(uuid.uuid4())
        key = ".".join(self.trace + (key_prefix,)) + "-" + token
        # attempt import so that we fail with a sensible exception if distributed is not
        # installed:
        from dask import distributed

        # hmm, there could be a problem here with collision between function
        # kwargs and submit kwargs, but this is inherent to distributed's API
        # design :/. In general I suppose callers should prefer to provide
        # everything as positoinal arguments.
        client: distributed.Client = distributed.get_client(self.addr)
        _LOG.debug(
            "%r: submitting task %r to %r%s",
            self,
            key,
            client,
            " (pure)" if pure else "",
        )
        fut = client.submit(function, *args, key=key, **kwargs)
        return DaskFuture(fut)

    @contextmanager
    def submitting_from_transform(self: D, name: str = "") -> Iterator[D]:
        client: distributed.Client = distributed.get_client(self.addr)
        try:
            worker = distributed.get_worker()
        except (AttributeError, ValueError):
            # we're not actually running on a dask.ditributed worker
            worker = None

        self_copy = type(self)(addr=self.addr, trace=self.trace + (name,))

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
        if True:  # pragma: no cover
            # note that we actually DO test this section (test_pipelines_on_dask), it's
            # just that this code inherently runs on a dask worker, which is in another
            # process, where its execution cannot be detected by coverage.
            from distributed.metrics import time
            from distributed.threadpoolexecutor import rejoin, secede
            from distributed.worker import thread_state
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
