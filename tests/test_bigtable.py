# pylint: disable=missing-docstring
from datetime import timedelta

import numpy as np
import pytest
from google.cloud.bigtable.column_family import GCRuleIntersection
from google.cloud.bigtable.column_family import MaxAgeGCRule
from google.cloud.bigtable.column_family import MaxVersionsGCRule

from kvdbclient.base import ColumnFamilyConfig

from helpers import CHILD
from helpers import node_ids
from helpers import raise_backend_error
from helpers import write_log
from helpers import write_node

pytestmark = pytest.mark.parametrize("backend", ["bigtable"], indirect=True)


def test_renewing_an_unheld_root_lock_takes_it(client):
    assert client.renew_lock(np.uint64(100), np.uint64(1)) is True
    assert client.renew_lock(np.uint64(100), np.uint64(2)) is False


def test_a_user_filter_keeps_that_user_s_ops(client):
    write_log(client, client.create_operation_id(), user="alice")
    write_log(client, client.create_operation_id(), user="bob")
    assert set(client.read_log_entries(user_id="alice")) == {1}


def test_an_empty_column_filter_raises(client):
    write_node(client, 100, {CHILD: node_ids(10)})
    with pytest.raises(ValueError, match="Empty column filter"):
        client.read_nodes(node_ids=[np.uint64(100)], properties=[])


def test_a_family_with_both_limits_collects_by_both(client_no_table):
    both = ColumnFamilyConfig("5", max_versions=1, max_age=timedelta(days=1))
    families = [ColumnFamilyConfig("0"), both]
    client_no_table.create_table({}, "1.0", column_families=families)
    rule = client_no_table._admin_table.list_column_families()["5"].gc_rule
    expected = [MaxVersionsGCRule(1), MaxAgeGCRule(timedelta(days=1))]
    assert rule == GCRuleIntersection(expected)


def test_close_survives_a_data_client_that_fails_to_close(client, monkeypatch):
    monkeypatch.setattr(client._data_client, "close", raise_backend_error)
    client.close()
