# pylint: disable=missing-docstring
import numpy as np
import pytest

from kvdbclient import exceptions
from kvdbclient.attributes import TableMeta

pytestmark = pytest.mark.parametrize("backend", ["hbase"], indirect=True)


def test_renewing_an_unheld_root_lock_fails(client):
    assert client.renew_lock(np.uint64(100), np.uint64(1)) is False


def test_a_client_that_opens_no_forks_refuses_a_fork_table(client):
    client.write([client.mutate_row(TableMeta.key, {TableMeta.Fork: "{}"})])
    with pytest.raises(exceptions.ForkTableError, match="is a fork"):
        client.read_table_meta()
