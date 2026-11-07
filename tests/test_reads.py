# pylint: disable=missing-docstring, redefined-outer-name
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient import exceptions
from kvdbclient.serializers import serialize_uint64

from helpers import CHILD
from helpers import node_ids
from helpers import write_node
from helpers import write_nodes

PARENT = attributes.Hierarchy.Parent
NODE = np.uint64(100)
TABLE_ROWS = {attributes.TableMeta.key, attributes.TableVersion.key}
_HOUR = timedelta(hours=1)
_READS = {
    "ids": lambda client, **kw: client.read_nodes(
        node_ids=[NODE], properties=CHILD, **kw
    ),
    "range": lambda client, **kw: client.read_nodes(
        start_id=NODE, end_id=NODE, end_id_inclusive=True, properties=CHILD, **kw
    ),
}


@pytest.fixture
def node(client):
    write_node(client, NODE, {CHILD: node_ids(10), PARENT: np.uint64(42)})
    return client


class TestReadNode:
    def test_one_column_reads_its_cells(self, node):
        cells = node.read_node(NODE, properties=CHILD)
        np.testing.assert_array_equal(cells[0].value, [10])

    @pytest.mark.parametrize(
        "properties", [[CHILD, PARENT], None], ids=["listed", "all"]
    )
    def test_several_columns_read_by_column(self, node, properties):
        assert set(node.read_node(NODE, properties=properties)) == {CHILD, PARENT}

    def test_an_absent_node_reads_empty(self, client):
        assert client.read_node(np.uint64(999999)) == {}


class TestReadNodes:
    def test_by_ids(self, client):
        write_nodes(client, [100, 200, 300])
        data = client.read_nodes(node_ids=node_ids(100, 200, 300), properties=CHILD)
        assert set(data) == {100, 200, 300}

    @pytest.mark.parametrize(
        "inclusive, expected", [(False, {100, 200}), (True, {100, 200, 300})]
    )
    def test_by_range_up_to_its_end(self, client, inclusive, expected):
        write_nodes(client, [100, 200, 300])
        data = client.read_nodes(
            start_id=np.uint64(100),
            end_id=np.uint64(300),
            end_id_inclusive=inclusive,
            properties=CHILD,
        )
        assert set(data) == expected

    def test_needs_ids_or_a_range(self, client):
        with pytest.raises(exceptions.PreconditionError):
            client.read_nodes(properties=CHILD)

    def test_attr_keys_false_keys_columns_by_bytes(self, node):
        data = node.read_nodes(node_ids=[NODE], properties=[CHILD], attr_keys=False)
        keys = list(data[NODE])
        assert keys and all(isinstance(key, bytes) for key in keys)

    def test_fake_edge_rows_read_by_their_own_key(self, client):
        row_key = serialize_uint64(NODE, fake_edges=True)
        client.write([client.mutate_row(row_key, {CHILD: node_ids(10)})])
        assert len(client.read_node(NODE, fake_edges=True, properties=CHILD)) == 1


@pytest.mark.parametrize("read", _READS.values(), ids=_READS.keys())
@pytest.mark.parametrize(
    "window, found",
    [
        ({"start_time": -_HOUR}, True),
        ({"start_time": _HOUR}, False),
        ({"end_time": _HOUR}, True),
        ({"end_time": -_HOUR}, False),
    ],
    ids=["start_past", "start_future", "end_future", "end_past"],
)
def test_a_time_window_keeps_only_cells_inside_it(node, read, window, found):
    now = datetime.now(timezone.utc)
    data = read(node, **{bound: now + offset for bound, offset in window.items()})
    assert (NODE in data) is found


class TestReadAllRows:
    def test_a_fresh_table_holds_its_meta_and_version(self, client):
        assert set(client.read_all_rows().rows) == TABLE_ROWS

    def test_holds_written_rows(self, client):
        ids = write_nodes(client, [100, 200, 300])
        keys = {serialize_uint64(i) for i in ids}
        assert set(client.read_all_rows().rows) == keys | TABLE_ROWS


def test_a_read_spanning_several_requests_returns_every_row(client_small_batch):
    ids = write_nodes(client_small_batch, range(7_000_000, 7_000_100))
    data = client_small_batch.read_nodes(node_ids=ids, properties=CHILD)
    np.testing.assert_array_equal([data[i][0].value[0] for i in ids], ids)
