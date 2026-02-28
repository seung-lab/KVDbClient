# pylint: disable=missing-docstring, redefined-outer-name

from datetime import datetime, timedelta, timezone
import time

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient import basetypes
from kvdbclient import exceptions
from kvdbclient.base import Cell
from kvdbclient.serializers import serialize_uint64


# ── Helpers ───────────────────────────────────────────────────────────────


def write_node(client, node_id, val_dict, time_stamp=None):
    row_key = serialize_uint64(np.uint64(node_id))
    entry = client.mutate_row(row_key, val_dict, time_stamp=time_stamp)
    client.write([entry])
    return row_key


# ── Cell equality ─────────────────────────────────────────────────────────


class TestCellEquality:
    def test_equal_cells(self):
        now = datetime.now(timezone.utc)
        assert Cell(value=b"a", timestamp=now) == Cell(value=b"a", timestamp=now)

    def test_unequal_value(self):
        now = datetime.now(timezone.utc)
        assert Cell(value=b"a", timestamp=now) != Cell(value=b"b", timestamp=now)

    def test_unequal_timestamp(self):
        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        assert Cell(value=b"a", timestamp=t1) != Cell(value=b"a", timestamp=t2)

    def test_hash_equal(self):
        now = datetime.now(timezone.utc)
        c1 = Cell(value=b"a", timestamp=now)
        c2 = Cell(value=b"a", timestamp=now)
        assert hash(c1) == hash(c2)

    def test_not_cell_returns_not_implemented(self):
        now = datetime.now(timezone.utc)
        c = Cell(value=b"a", timestamp=now)
        assert c.__eq__("string") is NotImplemented


# ── lock_roots_indefinitely (base.py) ─────────────────────────────────────


class TestLockRootsIndefinitely:
    def test_all_succeed(self, bt_client):
        roots = [np.uint64(100), np.uint64(200), np.uint64(300)]
        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}

        success, locked, failed = bt_client.lock_roots_indefinitely(roots, op, future_d)
        assert success is True
        assert len(locked) == 3
        assert len(failed) == 0

    def test_partial_failure_unlocks(self, bt_client):
        roots = [np.uint64(100), np.uint64(200)]
        # Pre-lock root 200 indefinitely with a different operation
        bt_client.lock_root_indefinitely(np.uint64(200), np.uint64(999))

        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, locked, failed = bt_client.lock_roots_indefinitely(roots, op, future_d)
        assert success is False
        assert np.uint64(200) in failed

        # Root 100 should have been cleaned up (can be re-locked)
        assert bt_client.lock_root_indefinitely(np.uint64(100), np.uint64(2)) is True


# ── get_consolidated_lock_timestamp (base.py) ─────────────────────────────


class TestConsolidatedLockTimestamp:
    def test_returns_datetime(self, bt_client):
        roots = [np.uint64(100), np.uint64(200)]
        ops = [np.uint64(1), np.uint64(2)]
        for r, o in zip(roots, ops):
            bt_client.lock_root(r, o)

        ts = bt_client.get_consolidated_lock_timestamp(roots, ops)
        assert isinstance(ts, datetime)

    def test_empty_returns_none(self, bt_client):
        ts = bt_client.get_consolidated_lock_timestamp([], [])
        assert ts is None

    def test_wrong_op_returns_none(self, bt_client):
        bt_client.lock_root(np.uint64(100), np.uint64(1))
        ts = bt_client.get_consolidated_lock_timestamp(
            [np.uint64(100)], [np.uint64(999)]
        )
        assert ts is None


# ── Time-filtered reads ──────────────────────────────────────────────────


class TestTimeFilteredReads:
    def test_read_with_start_time(self, bt_client):
        now = datetime.now(timezone.utc)
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_node(
            np.uint64(100),
            properties=attributes.Hierarchy.Child,
            start_time=now - timedelta(seconds=10),
        )
        assert len(data) > 0

    def test_read_with_future_start_time_returns_empty(self, bt_client):
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_node(
            np.uint64(100),
            properties=attributes.Hierarchy.Child,
            start_time=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert len(data) == 0

    def test_read_with_end_time(self, bt_client):
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_node(
            np.uint64(100),
            properties=attributes.Hierarchy.Child,
            end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert len(data) > 0


# ── User-filtered reads ──────────────────────────────────────────────────


class TestUserFilteredReads:
    def test_read_log_entries_by_user(self, bt_client):
        for i, user in enumerate(["alice", "bob"]):
            write_node(bt_client, np.uint64(i + 1), {
                attributes.OperationLogs.UserID: user,
                attributes.OperationLogs.RootID: np.array([100 * (i + 1)], dtype=basetypes.NODE_ID),
                attributes.OperationLogs.OperationTimeStamp: datetime.now(timezone.utc),
            })
        # Create max op ID so read_log_entries can determine range
        bt_client.create_operation_id()
        bt_client.create_operation_id()
        bt_client.create_operation_id()

        logs = bt_client.read_log_entries(user_id="alice")
        for op_id, record in logs.items():
            assert record[attributes.OperationLogs.UserID] == "alice"


# ── Delete cells with timestamps ─────────────────────────────────────────


class TestDeleteCellsTimestamp:
    def test_delete_specific_cell_version(self, bt_client):
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        assert len(data) > 0
        ts = data[0].timestamp

        bt_client.delete_cells([
            (serialize_uint64(np.uint64(100)), attributes.Hierarchy.Child, [ts]),
        ])
        data = bt_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        assert len(data) == 0


# ── Read edge cases ──────────────────────────────────────────────────────


class TestReadNodesEdgeCases:
    def test_end_id_inclusive(self, bt_client):
        nid = np.uint64(100)
        write_node(bt_client, nid, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_nodes(
            start_id=nid,
            end_id=nid,
            end_id_inclusive=True,
            properties=attributes.Hierarchy.Child,
        )
        assert nid in data

    def test_fake_edges(self, bt_client):
        nid = np.uint64(100)
        row_key = serialize_uint64(nid, fake_edges=True)
        entry = bt_client.mutate_row(
            row_key,
            {attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID)},
        )
        bt_client.write([entry])
        data = bt_client.read_node(nid, fake_edges=True, properties=attributes.Hierarchy.Child)
        assert len(data) > 0


class TestEmptyColumnFilter:
    def test_raises_value_error(self, bt_client):
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        with pytest.raises(ValueError, match="Empty column filter"):
            bt_client.read_nodes(
                node_ids=[np.uint64(100)],
                properties=[],
            )


# ── get_compatible_timestamp ─────────────────────────────────────────────


class TestGetCompatibleTimestamp:
    def test_returns_datetime(self, bt_client):
        now = datetime.now(timezone.utc)
        result = bt_client.get_compatible_timestamp(now)
        assert isinstance(result, datetime)

    def test_round_up(self, bt_client):
        # Timestamp with sub-millisecond precision
        ts = datetime(2024, 6, 15, 12, 0, 0, 500, tzinfo=timezone.utc)
        result_down = bt_client.get_compatible_timestamp(ts, round_up=False)
        result_up = bt_client.get_compatible_timestamp(ts, round_up=True)
        assert result_up >= result_down


# ── read_log_entries by range ────────────────────────────────────────────


class TestReadLogEntriesByRange:
    def test_read_all_logs(self, bt_client):
        # Create some operation IDs first
        op1 = bt_client.create_operation_id()
        op2 = bt_client.create_operation_id()
        for op_id in [op1, op2]:
            write_node(bt_client, op_id, {
                attributes.OperationLogs.UserID: "tester",
                attributes.OperationLogs.RootID: np.array([100], dtype=basetypes.NODE_ID),
                attributes.OperationLogs.OperationTimeStamp: datetime.now(timezone.utc),
            })
        logs = bt_client.read_log_entries()
        assert len(logs) >= 2


# ── Sharded reads ────────────────────────────────────────────────────────


class TestShardedRead:
    def test_read_exceeding_max_row_key_count(self, bt_client_small_batch):
        """MAX_ROW_KEY_COUNT=50, write 100 rows to trigger sharded path."""
        client = bt_client_small_batch
        count = 100
        base = 7_000_000
        entries = []
        node_ids = []
        for i in range(count):
            nid = np.uint64(base + i)
            node_ids.append(nid)
            entries.append(client.mutate_row(
                serialize_uint64(nid),
                {attributes.Hierarchy.Child: np.array([i], dtype=basetypes.NODE_ID)},
            ))
        client.write(entries)

        data = client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == count
