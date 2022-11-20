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

from abc import ABC, abstractmethod
from typing import Callable, Optional

from attrs import define, field
from dask import distributed
from dask.base import tokenize


class Backend(ABC):
    @abstractmethod
    def submit(
        self,
        key_prefix: str,
        function: Callable,
        *function_args,
        **function_kwargs,
    ):
        raise NotImplementedError


@define
class DummyFuture:
    obj: object

    def result(self):
        return self.obj


@define
class DummyBackend(Backend):
    def submit(
        self,
        key_prefix: str,
        function: Callable,
        *function_args,
        **function_kwargs,
    ):
        # "materialize" any DummyFuture args
        function_args = tuple(
            (a.result() if isinstance(a, DummyFuture) else a) for a in function_args
        )
        function_kwargs = {
            k: (v.result() if isinstance(v, DummyFuture) else v)
            for k, v in function_kwargs.items()
        }
        result = function(*function_args, **function_kwargs)
        return DummyFuture(result)


def _convert_to_address(obj: str | None | distributed.Client):
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, distributed.Client):
        return obj.scheduler.address
    raise TypeError(f"Don't know how to create DaskBackend from {type(obj)}: {obj!r}")


@define
class DaskBackend(Backend):
    address: Optional[str] = field(converter=_convert_to_address, default=None)

    def submit(
        self,
        key_prefix: str,
        function: Callable,
        *function_args,
        **function_kwargs,
    ):
        client = distributed.get_client(self.address)
        # TODO: should we do anything about impure functions? i.e., data readers
        key = key_prefix + "-" + tokenize(function, function_kwargs, *function_args)
        # hmm, there could be a problem here with collision between function
        # kwargs and submit kwargs, but this is inherent to distributed's API
        # design :/. In general I suppose callers should prefer to provide
        # everything as positoinal arguments.
        fut = client.submit(function, *function_args, key=key, **function_kwargs)
        return fut