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

import pytest
from dask import distributed

import frankenfit as ff
from frankenfit.backend import DummyFuture


def test_DummyBackend():
    def foo(x):
        return f"foo({x})"

    backend = ff.DummyBackend()

    dummy_fut = backend.submit("key-foo", foo, 42)
    assert dummy_fut.result() == "foo(42)"

    # future arg gets materialized
    dummy_fut = backend.submit("key-foo", foo, DummyFuture(24))
    assert dummy_fut.result() == "foo(24)"


def test_DaskBackend():
    def foo(x):
        return f"foo({x})"

    def forty_two():
        return 42

    backend = ff.DaskBackend()

    # should fail with no dask client having yet been created
    with pytest.raises(ValueError):
        backend.submit("key-foo", foo, 42)

    # spin up a local cluster and client
    client = distributed.Client(dashboard_address=None)
    backend = ff.DaskBackend(client)

    fut = backend.submit("key-foo", foo, 42)
    assert fut.result() == "foo(42)"

    fut_42 = backend.submit("forty_two", forty_two)
    fut = backend.submit("key-foo", foo, fut_42)
    assert fut.result() == "foo(42)"

    # should find global client, per distributed.get_client()
    backend = ff.DaskBackend()
    fut = backend.submit("key-foo", foo, 42)
    assert fut.result() == "foo(42)"

    # string address
    backend = ff.DaskBackend(client.scheduler.address)
    fut = backend.submit("key-foo", foo, 42)
    assert fut.result() == "foo(42)"

    with pytest.raises(TypeError):
        ff.DaskBackend(42.0)

    # Dummy future should be materialized seamlessly
    dummy_fut = ff.DummyBackend().submit("forty_two", forty_two)
    fut = backend.submit("key-foo", foo, dummy_fut)
    assert fut.result() == "foo(42)"
