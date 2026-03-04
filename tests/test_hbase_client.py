# pylint: disable=missing-docstring, redefined-outer-name

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient import basetypes
from kvdbclient import exceptions
from kvdbclient.serializers import serialize_uint64


# ── Helpers ───────────────────────────────────────────────────────────────


def write_node(client, node_id, val_dict, time_stamp=None):
    row_key = serialize_uint64(np.uint64(node_id))
    entry = client.mutate_row(row_key, val_dict, time_stamp=time_stamp)
    client.write([entry])
    return row_key


# ── Table Setup ──────────────────────────────────────────────────────────


class TestHBaseCreateTable:
    def test_creates_table_and_sets_meta(self, hbase_client_no_table):
        client = hbase_client_no_table
        meta = {"chunk_size": 512}
        client.create_table(meta, "1.0")
        assert client.read_table_version() == "1.0"
        assert client.read_table_meta() == meta

    def test_raises_on_duplicate(self, hbase_client):
        with pytest.raises(ValueError, match="already exists"):
            hbase_client.create_table({}, "2.0")

    def test_custom_column_families(self, hbase_client_no_table):
        from kvdbclient.base import ColumnFamilyConfig
        families = [
            ColumnFamilyConfig("0"),
            ColumnFamilyConfig("1", max_versions=3),
            ColumnFamilyConfig("2", max_age=timedelta(days=30)),
        ]
        client = hbase_client_no_table
        client.create_table({"custom": True}, "1.0", column_families=families)
        assert client.read_table_version() == "1.0"


class TestHBaseCreateColumnFamily:
    def test_create_extra_family(self, hbase_client):
        hbase_client.create_column_family("5")

    def test_create_with_config(self, hbase_client):
        from kvdbclient.base import ColumnFamilyConfig
        hbase_client.create_column_family(ColumnFamilyConfig("6", max_versions=2))


# ── Table Meta ───────────────────────────────────────────────────────────


class TestHBaseTableMeta:
    def test_read_table_meta(self, hbase_client):
        assert hbase_client.read_table_meta() == {"test": True}

    def test_update_table_meta(self, hbase_client):
        hbase_client.update_table_meta({"new_field": 42})
        assert hbase_client.read_table_meta() == {"new_field": 42}

    def test_read_table_version(self, hbase_client):
        assert hbase_client.read_table_version() == "0.0.1"

    def test_add_table_version_overwrite(self, hbase_client):
        hbase_client.add_table_version("2.0", overwrite=True)
        assert hbase_client.read_table_version() == "2.0"


# ── Write + Read ──────────────────────────────────────────────────────────


class TestHBaseWriteRead:
    def test_single_row(self, hbase_client):
        arr = np.array([10, 20, 30], dtype=basetypes.NODE_ID)
        write_node(hbase_client, 100, {attributes.Hierarchy.Child: arr})
        data = hbase_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        np.testing.assert_array_equal(data[0].value, arr)

    def test_bulk_write(self, hbase_client):
        entries = []
        for i in range(5):
            node_id = np.uint64(200 + i)
            arr = np.array([i], dtype=basetypes.NODE_ID)
            entry = hbase_client.mutate_row(
                serialize_uint64(node_id),
                {attributes.Hierarchy.Child: arr},
            )
            entries.append(entry)
        hbase_client.write(entries)
        for i in range(5):
            data = hbase_client.read_node(
                np.uint64(200 + i), properties=attributes.Hierarchy.Child
            )
            np.testing.assert_array_equal(data[0].value, np.array([i], dtype=basetypes.NODE_ID))

    def test_empty_write_noop(self, hbase_client):
        hbase_client.write([])

    def test_write_with_lock_renewal_success(self, hbase_client):
        root_id = np.uint64(600)
        op_id = np.uint64(1)
        assert hbase_client.lock_root(root_id, op_id) is True

        entry = hbase_client.mutate_row(
            serialize_uint64(np.uint64(601)),
            {attributes.Hierarchy.Child: np.array([1], dtype=basetypes.NODE_ID)},
        )
        hbase_client.write([entry], root_ids=[root_id], operation_id=op_id)
        data = hbase_client.read_node(np.uint64(601), properties=attributes.Hierarchy.Child)
        assert len(data) > 0

    def test_write_with_lock_renewal_failure(self, hbase_client):
        root_id = np.uint64(500)
        hbase_client.lock_root(root_id, np.uint64(1))
        entry = hbase_client.mutate_row(
            serialize_uint64(np.uint64(501)),
            {attributes.Hierarchy.Child: np.array([1], dtype=basetypes.NODE_ID)},
        )
        with pytest.raises(exceptions.LockingError):
            hbase_client.write([entry], root_ids=[root_id], operation_id=np.uint64(999))


class TestHBaseReadNode:
    def test_single_property(self, hbase_client):
        arr = np.array([10, 20], dtype=basetypes.NODE_ID)
        write_node(hbase_client, 100, {attributes.Hierarchy.Child: arr})
        data = hbase_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        np.testing.assert_array_equal(data[0].value, arr)

    def test_multiple_properties(self, hbase_client):
        child_arr = np.array([10], dtype=basetypes.NODE_ID)
        parent_val = np.uint64(42)
        write_node(hbase_client, 100, {
            attributes.Hierarchy.Child: child_arr,
            attributes.Hierarchy.Parent: parent_val,
        })
        data = hbase_client.read_node(
            np.uint64(100),
            properties=[attributes.Hierarchy.Child, attributes.Hierarchy.Parent],
        )
        assert attributes.Hierarchy.Child in data
        assert attributes.Hierarchy.Parent in data

    def test_nonexistent_returns_empty(self, hbase_client):
        data = hbase_client.read_node(np.uint64(999999))
        assert data == {}


class TestHBaseReadNodes:
    def test_by_ids(self, hbase_client):
        for nid in [100, 200, 300]:
            write_node(hbase_client, nid, {
                attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID),
            })
        data = hbase_client.read_nodes(
            node_ids=[np.uint64(100), np.uint64(200), np.uint64(300)],
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == 3

    def test_by_range(self, hbase_client):
        for nid in [100, 200, 300]:
            write_node(hbase_client, nid, {
                attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID),
            })
        data = hbase_client.read_nodes(
            start_id=np.uint64(100),
            end_id=np.uint64(301),
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) >= 3

    def test_missing_params_raises(self, hbase_client):
        with pytest.raises(exceptions.PreconditionError):
            hbase_client.read_nodes(properties=attributes.Hierarchy.Child)


# ── Read All Rows ─────────────────────────────────────────────────────────


class TestHBaseReadAllRows:
    def test_returns_written_rows(self, hbase_client):
        for nid in [100, 200, 300]:
            write_node(hbase_client, nid, {
                attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID),
            })
        result = hbase_client.read_all_rows()
        assert len(result.rows) >= 5  # 3 nodes + meta + version


# ── Delete ────────────────────────────────────────────────────────────────


class TestHBaseDelete:
    def test_delete_row(self, hbase_client):
        row_key = write_node(hbase_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = hbase_client.read_node(np.uint64(100))
        assert len(data) > 0

        hbase_client.delete_row(row_key)
        data = hbase_client.read_node(np.uint64(100))
        assert data == {}


# ── Locking ───────────────────────────────────────────────────────────────


class TestHBaseLocking:
    def test_acquire(self, hbase_client):
        assert hbase_client.lock_root(np.uint64(100), np.uint64(1)) is True

    def test_double_lock_fails(self, hbase_client):
        assert hbase_client.lock_root(np.uint64(100), np.uint64(1)) is True
        assert hbase_client.lock_root(np.uint64(100), np.uint64(2)) is False

    def test_unlock_allows_relock(self, hbase_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert hbase_client.lock_root(root, op1) is True
        hbase_client.unlock_root(root, op1)
        assert hbase_client.lock_root(root, op2) is True

    def test_lock_indefinitely(self, hbase_client):
        assert hbase_client.lock_root_indefinitely(np.uint64(100), np.uint64(1)) is True
        assert hbase_client.lock_root_indefinitely(np.uint64(100), np.uint64(2)) is False

    def test_unlock_indefinitely(self, hbase_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert hbase_client.lock_root_indefinitely(root, op1) is True
        hbase_client.unlock_indefinitely_locked_root(root, op1)
        assert hbase_client.lock_root_indefinitely(root, op2) is True

    def test_renew_lock(self, hbase_client):
        root, op = np.uint64(100), np.uint64(1)
        hbase_client.lock_root(root, op)
        assert hbase_client.renew_lock(root, op) is True

    def test_renew_lock_wrong_op(self, hbase_client):
        root = np.uint64(100)
        hbase_client.lock_root(root, np.uint64(1))
        assert hbase_client.renew_lock(root, np.uint64(2)) is False

    def test_different_roots_independent(self, hbase_client):
        assert hbase_client.lock_root(np.uint64(100), np.uint64(1)) is True
        assert hbase_client.lock_root(np.uint64(200), np.uint64(2)) is True

    def test_wrong_op_unlock_does_not_release(self, hbase_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert hbase_client.lock_root(root, op1) is True
        hbase_client.unlock_root(root, op2)  # wrong op_id
        assert hbase_client.lock_root(root, op2) is False  # still locked

    def test_indefinite_wrong_op_unlock_does_not_release(self, hbase_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert hbase_client.lock_root_indefinitely(root, op1) is True
        hbase_client.unlock_indefinitely_locked_root(root, op2)
        assert hbase_client.lock_root_indefinitely(root, op2) is False

    def test_renew_not_locked_fails(self, hbase_client):
        # HBase check_and_put requires existing value match, so renew on
        # a non-existent lock returns False (differs from BigTable)
        assert hbase_client.renew_lock(np.uint64(100), np.uint64(1)) is False

    def test_lock_root_blocked_by_indefinite_lock(self, hbase_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert hbase_client.lock_root_indefinitely(root, op1) is True
        assert hbase_client.lock_root(root, op2) is False


class TestHBaseLockRoots:
    def test_all_succeed(self, hbase_client):
        roots = [np.uint64(100), np.uint64(200)]
        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, locked_ids = hbase_client.lock_roots(roots, op, future_d)
        assert success is True
        assert len(locked_ids) == 2

    def test_partial_failure_unlocks_all(self, hbase_client):
        roots = [np.uint64(100), np.uint64(200)]
        hbase_client.lock_root(np.uint64(200), np.uint64(999))

        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, _ = hbase_client.lock_roots(roots, op, future_d)
        assert success is False

        # root 100 should have been unlocked (can be re-locked)
        assert hbase_client.lock_root(np.uint64(100), np.uint64(2)) is True


class TestHBaseLockRootsIndefinitely:
    def test_all_succeed(self, hbase_client):
        roots = [np.uint64(100), np.uint64(200), np.uint64(300)]
        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, locked, failed = hbase_client.lock_roots_indefinitely(roots, op, future_d)
        assert success is True
        assert len(locked) == 3
        assert len(failed) == 0

    def test_partial_failure_unlocks(self, hbase_client):
        roots = [np.uint64(100), np.uint64(200)]
        hbase_client.lock_root_indefinitely(np.uint64(200), np.uint64(999))

        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, locked, failed = hbase_client.lock_roots_indefinitely(roots, op, future_d)
        assert success is False
        assert np.uint64(200) in failed

        assert hbase_client.lock_root_indefinitely(np.uint64(100), np.uint64(2)) is True


class TestHBaseRenewLocks:
    def test_all_succeed(self, hbase_client):
        roots = [np.uint64(100), np.uint64(200)]
        op = np.uint64(1)
        for r in roots:
            hbase_client.lock_root(r, op)
        assert hbase_client.renew_locks(roots, op) is True

    def test_one_fails(self, hbase_client):
        hbase_client.lock_root(np.uint64(100), np.uint64(1))
        hbase_client.lock_root(np.uint64(200), np.uint64(2))
        assert hbase_client.renew_locks([np.uint64(100), np.uint64(200)], np.uint64(1)) is False


class TestHBaseGetLockTimestamp:
    def test_returns_datetime_when_locked(self, hbase_client):
        root, op = np.uint64(100), np.uint64(1)
        hbase_client.lock_root(root, op)
        ts = hbase_client.get_lock_timestamp(root, op)
        assert isinstance(ts, (datetime, int, float))

    def test_no_lock_returns_none(self, hbase_client):
        assert hbase_client.get_lock_timestamp(np.uint64(100), np.uint64(1)) is None

    def test_wrong_op_returns_none(self, hbase_client):
        hbase_client.lock_root(np.uint64(100), np.uint64(1))
        assert hbase_client.get_lock_timestamp(np.uint64(100), np.uint64(2)) is None


class TestHBaseConsolidatedLockTimestamp:
    def test_returns_datetime(self, hbase_client):
        roots = [np.uint64(100), np.uint64(200)]
        ops = [np.uint64(1), np.uint64(2)]
        for r, o in zip(roots, ops):
            hbase_client.lock_root(r, o)
        ts = hbase_client.get_consolidated_lock_timestamp(roots, ops)
        assert isinstance(ts, datetime)

    def test_empty_returns_none(self, hbase_client):
        ts = hbase_client.get_consolidated_lock_timestamp([], [])
        assert ts is None

    def test_wrong_op_returns_none(self, hbase_client):
        hbase_client.lock_root(np.uint64(100), np.uint64(1))
        ts = hbase_client.get_consolidated_lock_timestamp(
            [np.uint64(100)], [np.uint64(999)]
        )
        assert ts is None


class TestHBaseLockExpiry:
    def test_expired_lock_allows_relock(self, hbase_client_short_expiry):
        client = hbase_client_short_expiry
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert client.lock_root(root, op1) is True
        time.sleep(1.5)
        assert client.lock_root(root, op2) is True

    def test_non_expired_lock_blocks(self, hbase_client_short_expiry):
        client = hbase_client_short_expiry
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert client.lock_root(root, op1) is True
        assert client.lock_root(root, op2) is False


# ── ID Generation ─────────────────────────────────────────────────────────


class TestHBaseNodeIds:
    def test_create_unique(self, hbase_client):
        chunk_id = np.uint64(1 << 32)
        id1 = hbase_client.create_node_id(chunk_id)
        id2 = hbase_client.create_node_id(chunk_id)
        assert id1 != id2

    def test_create_batch(self, hbase_client):
        chunk_id = np.uint64(1 << 32)
        ids = hbase_client.create_node_ids(chunk_id, 5)
        assert len(ids) == 5
        assert len(np.unique(ids)) == 5

    def test_set_max_node_id_fresh(self, hbase_client):
        """On a fresh counter, set_max_node_id sets the counter to the segment ID."""
        chunk_id = np.uint64(1 << 32)
        node_id = chunk_id | np.uint64(10)
        hbase_client.set_max_node_id(chunk_id, node_id)
        max_id = hbase_client.get_max_node_id(chunk_id)
        assert max_id == node_id

    def test_set_max_node_id_then_create_no_collision(self, hbase_client):
        """After set_max_node_id, create_node_id should return IDs above the set max."""
        chunk_id = np.uint64(1 << 32)
        node_id = chunk_id | np.uint64(10)
        hbase_client.set_max_node_id(chunk_id, node_id)
        new_id = hbase_client.create_node_id(chunk_id)
        segment_id = int(np.uint64(new_id) ^ np.uint64(chunk_id))
        assert segment_id > 10

    def test_set_max_node_id_is_additive(self, hbase_client):
        """Calling set_max_node_id twice increments cumulatively (not idempotent)."""
        chunk_id = np.uint64(1 << 32)
        node_id = chunk_id | np.uint64(5)
        hbase_client.set_max_node_id(chunk_id, node_id)
        hbase_client.set_max_node_id(chunk_id, node_id)
        max_id = hbase_client.get_max_node_id(chunk_id)
        # Counter was incremented by 5 twice -> segment_id is 10
        assert int(np.uint64(max_id) ^ np.uint64(chunk_id)) == 10

    def test_set_max_node_id_after_create(self, hbase_client):
        """set_max_node_id after create_node_ids advances counter further."""
        chunk_id = np.uint64(1 << 32)
        hbase_client.create_node_ids(chunk_id, 3)  # counter at 3
        node_id = chunk_id | np.uint64(7)
        hbase_client.set_max_node_id(chunk_id, node_id)  # increments by 7 -> counter at 10
        max_id = hbase_client.get_max_node_id(chunk_id)
        assert int(np.uint64(max_id) ^ np.uint64(chunk_id)) == 10


class TestHBaseOperationIds:
    def test_create_unique(self, hbase_client):
        id1 = hbase_client.create_operation_id()
        id2 = hbase_client.create_operation_id()
        assert id1 != id2

    def test_get_max_initial(self, hbase_client):
        max_id = hbase_client.get_max_operation_id()
        assert max_id == 0


# ── Data Type Roundtrips ──────────────────────────────────────────────────


class TestHBaseDataTypeRoundtrips:
    def test_numpy_array(self, hbase_client):
        arr = np.array([10, 20, 30], dtype=basetypes.NODE_ID)
        write_node(hbase_client, 100, {attributes.Hierarchy.Child: arr})
        data = hbase_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        np.testing.assert_array_equal(data[0].value, arr)

    def test_scalar(self, hbase_client):
        write_node(hbase_client, 100, {attributes.Hierarchy.Parent: np.uint64(42)})
        data = hbase_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Parent)
        assert data[0].value == np.uint64(42)

    def test_string(self, hbase_client):
        write_node(hbase_client, 100, {attributes.OperationLogs.UserID: "alice"})
        data = hbase_client.read_node(np.uint64(100), properties=attributes.OperationLogs.UserID)
        assert data[0].value == "alice"

    def test_pickle_meta(self, hbase_client):
        meta = {"complex": [1, 2, 3]}
        write_node(hbase_client, 100, {attributes.TableMeta.Meta: meta})
        data = hbase_client.read_node(np.uint64(100), properties=attributes.TableMeta.Meta)
        assert data[0].value == meta
