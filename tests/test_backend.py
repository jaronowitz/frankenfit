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

    with backend.submitting_from_transform("foo") as b:
        assert "foo" in b.transform_names
        assert b.submit("key-foo", foo, 420).result() == "foo(420)"


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
