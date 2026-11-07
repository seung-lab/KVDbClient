"""Tests for kvdbclient.attributes"""

import numpy as np
import pytest

from kvdbclient.attributes import (
    _Attribute,
    Concurrency,
    Connectivity,
    Hierarchy,
    OperationLogs,
    from_key,
)
from kvdbclient import basetypes

from helpers import node_ids


@pytest.mark.parametrize(
    "column, value",
    [
        (Hierarchy.Child, node_ids(1, 2, 3)),
        (OperationLogs.UserID, "test_user"),
        (Connectivity.CrossChunkEdge, node_ids([1, 2], [3, 4])),
    ],
    ids=["array", "string", "edge_array"],
)
def test_a_column_value_roundtrips(column, value):
    np.testing.assert_equal(column.deserialize(column.serialize(value)), value)


class TestAttribute:
    def test_basetype(self):
        assert Hierarchy.Child.basetype == basetypes.NODE_ID.type
        assert OperationLogs.UserID.basetype == str

    def test_index(self):
        assert Connectivity.CrossChunkEdge[5].index == 5

    def test_family_id(self):
        assert Hierarchy.Child.family_id == "0"
        assert Concurrency.Counter.family_id == "1"
        assert OperationLogs.UserID.family_id == "2"


class TestAttributeArray:
    def test_getitem(self):
        attr = Connectivity.AtomicCrossChunkEdge[3]
        assert isinstance(attr, _Attribute)
        assert attr.key == b"atomic_cross_edges_3"

    def test_pattern(self):
        assert Connectivity.CrossChunkEdge.pattern == b"cross_edges_%d"


class TestFromKey:
    def test_valid_key(self):
        assert from_key("0", b"children") is Hierarchy.Child

    def test_invalid_key_raises(self):
        with pytest.raises(KeyError, match="Unknown key"):
            from_key("99", b"nonexistent")


class TestOperationLogs:
    def test_all_returns_list(self):
        result = OperationLogs.all()
        assert isinstance(result, list)
        # Representative sample — no brittle count. Any addition to the
        # attribute set shouldn't break this test.
        for attr in (
            OperationLogs.OperationID,
            OperationLogs.UserID,
            OperationLogs.RootID,
            OperationLogs.AddedEdge,
            OperationLogs.Status,
            OperationLogs.OperationTimeStamp,
            OperationLogs.L2ChunkLockScope,
        ):
            assert attr in result

    def test_status_codes(self):
        assert OperationLogs.StatusCodes.SUCCESS.value == 0
        assert OperationLogs.StatusCodes.CREATED.value == 1
        assert OperationLogs.StatusCodes.EXCEPTION.value == 2
        assert OperationLogs.StatusCodes.WRITE_STARTED.value == 3
        assert OperationLogs.StatusCodes.WRITE_FAILED.value == 4
