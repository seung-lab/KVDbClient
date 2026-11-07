# pylint: disable=missing-docstring
import numpy as np
import pytest

pytestmark = pytest.mark.parametrize("backend", ["hbase"], indirect=True)


def test_renewing_an_unheld_root_lock_fails(client):
    assert client.renew_lock(np.uint64(100), np.uint64(1)) is False
