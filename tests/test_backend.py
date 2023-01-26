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

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dask import distributed

import frankenfit as ff
from frankenfit.core import LocalFuture


@pytest.fixture(scope="module")
def dask_client():
    # spin up a local cluster and client
    client = distributed.Client(dashboard_address=":0", scheduler_port=0)
    yield client
    # client.shutdown()
    # client.close()


def test_DummyBackend():
    def foo(x):
        return f"foo({x})"

    backend = ff.LocalBackend()

    dummy_fut = backend.submit("key-foo", foo, 42)
    assert dummy_fut.result() == "foo(42)"

    # future arg gets materialized
    dummy_fut = backend.submit("key-foo", foo, LocalFuture(24))
    assert dummy_fut.result() == "foo(24)"


def test_compare_futures():
    local = ff.LocalBackend()
    fut_other = local.put("other")
    for obj in (42, "foo"):
        fut = local.put(obj)
        assert fut == fut
        assert fut == local.put(obj)
        assert fut != fut_other
        assert fut.belongs_to(local)


@pytest.mark.dask
def test_DaskBackend(dask_client):
    def foo(x):
        return f"foo({x})"

    def forty_two():
        return 42

    # spin up a local cluster and client
    backend = ff.DaskBackend(dask_client)

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
    backend = ff.DaskBackend(dask_client.scheduler.address)
    fut = backend.submit("key-foo", foo, 42)
    assert fut.result() == "foo(42)"

    with pytest.raises(TypeError):
        ff.DaskBackend(42.0)

    dask_fut = backend.put(42)
    fut = backend.submit("key-foo", foo, dask_fut)
    assert fut.result() == "foo(42)"

    # Dummy future should be materialized seamlessly
    dummy_fut = ff.LocalBackend().submit("forty_two", forty_two)
    fut = backend.submit("key-foo", foo, dummy_fut)
    assert fut.result() == "foo(42)"

    assert fut != dummy_fut
    assert fut == fut
    assert fut.belongs_to(backend)
    assert not fut.belongs_to(ff.LocalBackend())

    with backend.submitting_from_transform("foo") as b:
        assert "foo" in b.trace
        assert b.submit("key-foo", foo, 420).result() == "foo(420)"

    # make sure dask doesn't mangle list and dict args
    assert backend.put(["foo", "bar"]).result() == ["foo", "bar"]
    assert backend.put({"foo": "bar"}).result() == {"foo": "bar"}
    assert backend.put({"foo": "bibble"}).result() == {"foo": "bibble"}


@pytest.mark.dask
def test_dask_purity(dask_client):
    def random_df():
        return pd.DataFrame(
            {
                "x": np.random.normal(size=100),
                "y": np.random.normal(size=100),
            }
        )

    dask = ff.DaskBackend(dask_client)

    # the function inherently is impure
    assert not random_df().equals(random_df)

    # but by default the backend will treat it as pure
    fut1 = dask.submit("random_df", random_df)
    fut2 = dask.submit("random_df", random_df)
    assert fut1.result().equals(fut2.result())

    # but we can use the pure=False kwarg to let it know what's up
    fut1 = dask.submit("random_df", random_df, pure=False)
    fut2 = dask.submit("random_df", random_df, pure=False)
    assert not fut1.result().equals(fut2.result())

    # so all Transforms have a `pure` attribute which says whether their _fit/_apply
    # functions should be submitted to backends as pure or not. Default is True.
    class RandomTransformPure(ff.ConstantTransform):
        def _apply(self, df_apply, state=None):
            return pd.DataFrame(
                {
                    "x": np.random.normal(size=100),
                    "y": np.random.normal(size=100),
                }
            )

    class RandomTransformImpure(ff.ConstantTransform):
        pure = False

        def _apply(self, df_apply, state=None):
            return pd.DataFrame(
                {
                    "x": np.random.normal(size=100),
                    "y": np.random.normal(size=100),
                }
            )

    rtp = RandomTransformPure()
    fut1 = dask.apply(rtp)
    fut2 = dask.apply(rtp)
    assert fut1.result().equals(fut2.result())

    rti = RandomTransformImpure()
    fut1 = dask.apply(rti)
    fut2 = dask.apply(rti)
    assert not fut1.result().equals(fut2.result())
