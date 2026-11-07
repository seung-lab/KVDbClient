# pylint: disable=missing-docstring
from datetime import datetime, timezone

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient import exceptions
from kvdbclient.serializers import serialize_uint64

from helpers import CHILD
from helpers import node_ids
from helpers import write_node
from helpers import write_nodes

NODE = np.uint64(100)
EARLY = datetime(2024, 1, 1, tzinfo=timezone.utc)
LATE = datetime(2024, 1, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "column, value",
    [
        (CHILD, node_ids(10, 20, 30)),
        (attributes.Connectivity.CrossChunkEdge[3], node_ids([1, 2], [3, 4])),
        (attributes.Connectivity.AtomicCrossChunkEdge[0], node_ids([100, 200], [3, 4])),
        (attributes.Hierarchy.Parent, np.uint64(42)),
        (attributes.OperationLogs.UserID, "alice"),
        (attributes.TableMeta.Meta, {"complex": [1, 2, 3]}),
    ],
    ids=["array", "edges", "compressed_edges", "scalar", "string", "pickle"],
)
def test_a_written_value_reads_back(client, column, value):
    write_node(client, NODE, {column: value})
    np.testing.assert_equal(client.read_node(NODE, properties=column)[0].value, value)


class TestWrite:
    def test_one_write_holds_many_rows(self, client):
        ids = write_nodes(client, range(200, 205))
        data = client.read_nodes(node_ids=ids, properties=CHILD)
        np.testing.assert_array_equal([data[i][0].value[0] for i in ids], ids)

    def test_nothing_to_write_is_a_no_op(self, client):
        client.write([])

    def test_lands_while_the_op_holds_its_roots(self, client):
        root, op = np.uint64(600), np.uint64(1)
        client.lock_root(root, op)
        row = client.mutate_row(serialize_uint64(NODE), {CHILD: node_ids(1)})
        client.write([row], root_ids=[root], operation_id=op)
        np.testing.assert_array_equal(
            client.read_node(NODE, properties=CHILD)[0].value, [1]
        )

    def test_refused_while_another_op_holds_a_root(self, client):
        root = np.uint64(500)
        client.lock_root(root, np.uint64(1))
        row = client.mutate_row(serialize_uint64(NODE), {CHILD: node_ids(1)})
        with pytest.raises(exceptions.LockingError):
            client.write([row], root_ids=[root], operation_id=np.uint64(999))
        assert client.read_node(NODE) == {}


class TestDelete:
    def test_delete_row_removes_it(self, client):
        row_key = write_node(client, NODE, {CHILD: node_ids(10)})
        client.delete_row(row_key)
        assert client.read_node(NODE) == {}

    def test_deleting_an_absent_row_is_a_no_op(self, client):
        client.delete_row(b"nonexistent")

    def test_delete_cells_removes_only_the_listed_rows(self, client):
        gone, kept = write_nodes(client, [100, 200])
        client.delete_cells([], row_keys_to_delete=[serialize_uint64(gone)])
        assert set(client.read_nodes(node_ids=[gone, kept], properties=CHILD)) == {kept}

    def test_delete_cells_removes_only_the_listed_version(self, client):
        write_node(client, NODE, {CHILD: node_ids(1)}, EARLY)
        write_node(client, NODE, {CHILD: node_ids(2)}, LATE)
        client.delete_cells([(serialize_uint64(NODE), CHILD, [LATE])])
        np.testing.assert_array_equal(
            client.read_node(NODE, properties=CHILD)[0].value, [1]
        )
