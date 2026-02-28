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


class TestCreateTable:
    def test_creates_table_and_sets_meta(self, bt_client_no_table):
        client = bt_client_no_table
        meta = {"chunk_size": 512}
        client.create_table(meta, "1.0")
        assert client.read_table_version() == "1.0"
        assert client.read_table_meta() == meta

    def test_raises_on_duplicate(self, bt_client):
        with pytest.raises(ValueError, match="already exists"):
            bt_client.create_table({}, "2.0")

    def test_custom_column_families(self, bt_client_no_table):
        from kvdbclient.base import ColumnFamilyConfig
        families = [
            ColumnFamilyConfig("0"),
            ColumnFamilyConfig("1", max_versions=3),
            ColumnFamilyConfig("2", max_age=timedelta(days=30)),
        ]
        client = bt_client_no_table
        client.create_table({"custom": True}, "1.0", column_families=families)
        assert client.read_table_version() == "1.0"
        assert client.read_table_meta() == {"custom": True}


class TestCreateColumnFamily:
    def test_create_extra_family(self, bt_client):
        bt_client.create_column_family("5")

    def test_create_with_config(self, bt_client):
        from kvdbclient.base import ColumnFamilyConfig
        bt_client.create_column_family(ColumnFamilyConfig("6", max_versions=2))


# ── Table Meta ───────────────────────────────────────────────────────────


class TestTableMeta:
    def test_read_table_meta(self, bt_client):
        assert bt_client.read_table_meta() == {"test": True}

    def test_update_table_meta(self, bt_client):
        bt_client.update_table_meta({"new_field": 42})
        assert bt_client.read_table_meta() == {"new_field": 42}

    def test_update_table_meta_overwrite(self, bt_client):
        bt_client.update_table_meta({"replaced": True}, overwrite=True)
        assert bt_client.read_table_meta() == {"replaced": True}

    def test_read_table_version(self, bt_client):
        assert bt_client.read_table_version() == "0.0.1"

    def test_add_table_version_overwrite(self, bt_client):
        bt_client.add_table_version("2.0", overwrite=True)
        assert bt_client.read_table_version() == "2.0"

    def test_add_table_version_duplicate_raises(self, bt_client):
        with pytest.raises(AssertionError):
            bt_client.add_table_version("1.0", overwrite=False)


# ── Write + Read ──────────────────────────────────────────────────────────


class TestWriteRead:
    def test_single_row(self, bt_client):
        arr = np.array([10, 20, 30], dtype=basetypes.NODE_ID)
        write_node(bt_client, 100, {attributes.Hierarchy.Child: arr})
        data = bt_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        np.testing.assert_array_equal(data[0].value, arr)

    def test_bulk_write(self, bt_client):
        entries = []
        for i in range(5):
            node_id = np.uint64(200 + i)
            arr = np.array([i], dtype=basetypes.NODE_ID)
            entry = bt_client.mutate_row(
                serialize_uint64(node_id),
                {attributes.Hierarchy.Child: arr},
            )
            entries.append(entry)
        bt_client.write(entries)

        for i in range(5):
            data = bt_client.read_node(
                np.uint64(200 + i), properties=attributes.Hierarchy.Child
            )
            np.testing.assert_array_equal(data[0].value, np.array([i], dtype=basetypes.NODE_ID))

    def test_empty_write_noop(self, bt_client):
        bt_client.write([])

    def test_write_with_lock_renewal_failure(self, bt_client):
        root_id = np.uint64(500)
        # Lock the root with a different operation so renewal with op 999 fails
        bt_client.lock_root(root_id, np.uint64(1))
        entry = bt_client.mutate_row(
            serialize_uint64(np.uint64(501)),
            {attributes.Hierarchy.Child: np.array([1], dtype=basetypes.NODE_ID)},
        )
        with pytest.raises(exceptions.LockingError):
            bt_client.write([entry], root_ids=[root_id], operation_id=np.uint64(999))

    def test_write_with_lock_renewal_success(self, bt_client):
        root_id = np.uint64(600)
        op_id = np.uint64(1)
        assert bt_client.lock_root(root_id, op_id) is True

        entry = bt_client.mutate_row(
            serialize_uint64(np.uint64(601)),
            {attributes.Hierarchy.Child: np.array([1], dtype=basetypes.NODE_ID)},
        )
        bt_client.write([entry], root_ids=[root_id], operation_id=op_id)

        data = bt_client.read_node(np.uint64(601), properties=attributes.Hierarchy.Child)
        assert len(data) > 0


class TestReadNode:
    def test_single_property(self, bt_client):
        arr = np.array([10, 20], dtype=basetypes.NODE_ID)
        write_node(bt_client, 100, {attributes.Hierarchy.Child: arr})
        data = bt_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        np.testing.assert_array_equal(data[0].value, arr)

    def test_multiple_properties(self, bt_client):
        child_arr = np.array([10], dtype=basetypes.NODE_ID)
        parent_val = np.uint64(42)
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: child_arr,
            attributes.Hierarchy.Parent: parent_val,
        })
        data = bt_client.read_node(
            np.uint64(100),
            properties=[attributes.Hierarchy.Child, attributes.Hierarchy.Parent],
        )
        assert attributes.Hierarchy.Child in data
        assert attributes.Hierarchy.Parent in data

    def test_all_properties(self, bt_client):
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
            attributes.Hierarchy.Parent: np.uint64(42),
        })
        data = bt_client.read_node(np.uint64(100))
        assert attributes.Hierarchy.Child in data
        assert attributes.Hierarchy.Parent in data

    def test_nonexistent_returns_empty(self, bt_client):
        data = bt_client.read_node(np.uint64(999999))
        assert data == {}


class TestReadNodes:
    def test_by_ids(self, bt_client):
        for nid in [100, 200, 300]:
            write_node(bt_client, nid, {
                attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID),
            })
        data = bt_client.read_nodes(
            node_ids=[np.uint64(100), np.uint64(200), np.uint64(300)],
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == 3

    def test_by_range(self, bt_client):
        for nid in [100, 200, 300]:
            write_node(bt_client, nid, {
                attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID),
            })
        data = bt_client.read_nodes(
            start_id=np.uint64(100),
            end_id=np.uint64(301),
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) >= 3

    def test_missing_params_raises(self, bt_client):
        with pytest.raises(exceptions.PreconditionError):
            bt_client.read_nodes(properties=attributes.Hierarchy.Child)

    def test_attr_keys_false(self, bt_client):
        write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_nodes(
            node_ids=[np.uint64(100)],
            properties=[attributes.Hierarchy.Child],
            attr_keys=False,
        )
        node_data = data[np.uint64(100)]
        keys = list(node_data.keys())
        assert all(isinstance(k, bytes) for k in keys)


class TestReadAllRows:
    def test_fresh_table_has_meta_rows(self, bt_client):
        result = bt_client.read_all_rows()
        assert len(result.rows) >= 2  # meta + version rows

    def test_returns_written_rows(self, bt_client):
        for nid in [100, 200, 300]:
            write_node(bt_client, nid, {
                attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID),
            })
        result = bt_client.read_all_rows()
        assert len(result.rows) >= 5  # 3 nodes + meta + version


# ── Delete ────────────────────────────────────────────────────────────────


class TestDelete:
    def test_delete_row(self, bt_client):
        row_key = write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        data = bt_client.read_node(np.uint64(100))
        assert len(data) > 0

        bt_client.delete_row(row_key)
        data = bt_client.read_node(np.uint64(100))
        assert data == {}

    def test_delete_row_nonexistent_is_noop(self, bt_client):
        bt_client.delete_row(b"nonexistent")

    def test_delete_cells_with_row_keys(self, bt_client):
        key1 = write_node(bt_client, 100, {
            attributes.Hierarchy.Child: np.array([10], dtype=basetypes.NODE_ID),
        })
        key2 = write_node(bt_client, 200, {
            attributes.Hierarchy.Child: np.array([20], dtype=basetypes.NODE_ID),
        })
        bt_client.delete_cells([], row_keys_to_delete=[key1])

        assert bt_client.read_node(np.uint64(100)) == {}
        assert len(bt_client.read_node(np.uint64(200))) > 0

    def test_delete_meta(self, bt_client):
        bt_client._delete_meta()
        with pytest.raises(KeyError):
            bt_client.read_table_meta()


# ── Locking ───────────────────────────────────────────────────────────────


class TestLockRoot:
    def test_acquire(self, bt_client):
        assert bt_client.lock_root(np.uint64(100), np.uint64(1)) is True

    def test_double_lock_fails(self, bt_client):
        assert bt_client.lock_root(np.uint64(100), np.uint64(1)) is True
        assert bt_client.lock_root(np.uint64(100), np.uint64(2)) is False

    def test_different_roots_independent(self, bt_client):
        assert bt_client.lock_root(np.uint64(100), np.uint64(1)) is True
        assert bt_client.lock_root(np.uint64(200), np.uint64(2)) is True


class TestLockRootIndefinitely:
    def test_acquire(self, bt_client):
        assert bt_client.lock_root_indefinitely(np.uint64(100), np.uint64(1)) is True

    def test_double_lock_fails(self, bt_client):
        assert bt_client.lock_root_indefinitely(np.uint64(100), np.uint64(1)) is True
        assert bt_client.lock_root_indefinitely(np.uint64(100), np.uint64(2)) is False


class TestUnlockRoot:
    def test_unlock_allows_relock(self, bt_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert bt_client.lock_root(root, op1) is True
        bt_client.unlock_root(root, op1)
        assert bt_client.lock_root(root, op2) is True

    def test_wrong_op_id_does_not_release(self, bt_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert bt_client.lock_root(root, op1) is True
        bt_client.unlock_root(root, op2)  # wrong op_id
        assert bt_client.lock_root(root, op2) is False  # still locked


class TestUnlockIndefinitelyLockedRoot:
    def test_unlock_allows_relock(self, bt_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert bt_client.lock_root_indefinitely(root, op1) is True
        bt_client.unlock_indefinitely_locked_root(root, op1)
        assert bt_client.lock_root_indefinitely(root, op2) is True

    def test_wrong_op_id_does_not_release(self, bt_client):
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert bt_client.lock_root_indefinitely(root, op1) is True
        bt_client.unlock_indefinitely_locked_root(root, op2)
        assert bt_client.lock_root_indefinitely(root, op2) is False


class TestRenewLock:
    def test_correct_op_id(self, bt_client):
        root, op = np.uint64(100), np.uint64(1)
        bt_client.lock_root(root, op)
        assert bt_client.renew_lock(root, op) is True

    def test_wrong_op_id(self, bt_client):
        root = np.uint64(100)
        bt_client.lock_root(root, np.uint64(1))
        assert bt_client.renew_lock(root, np.uint64(2)) is False

    def test_not_locked_acquires(self, bt_client):
        # renew_lock on a non-existent row effectively acquires the lock
        assert bt_client.renew_lock(np.uint64(100), np.uint64(1)) is True
        # Prove it's now locked: a different op_id cannot renew
        assert bt_client.renew_lock(np.uint64(100), np.uint64(2)) is False


class TestLockRoots:
    def test_all_succeed(self, bt_client):
        roots = [np.uint64(100), np.uint64(200)]
        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, locked_ids = bt_client.lock_roots(roots, op, future_d)
        assert success is True
        assert len(locked_ids) == 2

    def test_partial_failure_unlocks_all(self, bt_client):
        roots = [np.uint64(100), np.uint64(200)]
        # Pre-lock root 200 with a different operation
        bt_client.lock_root(np.uint64(200), np.uint64(999))

        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}
        success, _ = bt_client.lock_roots(roots, op, future_d)
        assert success is False

        # root 100 should have been unlocked (can be re-locked)
        assert bt_client.lock_root(np.uint64(100), np.uint64(2)) is True


class TestRenewLocks:
    def test_all_succeed(self, bt_client):
        roots = [np.uint64(100), np.uint64(200)]
        op = np.uint64(1)
        for r in roots:
            bt_client.lock_root(r, op)
        assert bt_client.renew_locks(roots, op) is True

    def test_one_fails(self, bt_client):
        bt_client.lock_root(np.uint64(100), np.uint64(1))
        bt_client.lock_root(np.uint64(200), np.uint64(2))
        assert bt_client.renew_locks([np.uint64(100), np.uint64(200)], np.uint64(1)) is False


# ── ID Generation ─────────────────────────────────────────────────────────


class TestNodeIds:
    def test_create_node_id_unique(self, bt_client):
        chunk_id = np.uint64(1 << 32)
        id1 = bt_client.create_node_id(chunk_id)
        id2 = bt_client.create_node_id(chunk_id)
        assert id1 != id2

    def test_create_node_ids_count(self, bt_client):
        chunk_id = np.uint64(1 << 32)
        ids = bt_client.create_node_ids(chunk_id, 5)
        assert len(ids) == 5
        assert len(np.unique(ids)) == 5

    def test_get_max_node_id(self, bt_client):
        chunk_id = np.uint64(1 << 32)
        bt_client.create_node_ids(chunk_id, 3)
        max_id = bt_client.get_max_node_id(chunk_id)
        assert max_id > chunk_id


class TestOperationIds:
    def test_create_unique(self, bt_client):
        id1 = bt_client.create_operation_id()
        id2 = bt_client.create_operation_id()
        assert id1 != id2

    def test_get_max_operation_id_initial(self, bt_client):
        max_id = bt_client.get_max_operation_id()
        assert max_id == 0

    def test_get_max_operation_id_after_creates(self, bt_client):
        for _ in range(3):
            bt_client.create_operation_id()
        assert bt_client.get_max_operation_id() >= 3


# ── Operation Logging ─────────────────────────────────────────────────────


class TestLogEntries:
    def test_read_log_entry_empty(self, bt_client):
        log, ts = bt_client.read_log_entry(np.uint64(99999))
        assert log == {}
        assert ts is None

    def test_write_and_read_log_entry(self, bt_client):
        op_id = np.uint64(1)
        root_ids = np.array([100, 200], dtype=basetypes.NODE_ID)
        now = datetime.now(timezone.utc)

        write_node(bt_client, op_id, {
            attributes.OperationLogs.UserID: "alice",
            attributes.OperationLogs.RootID: root_ids,
            attributes.OperationLogs.Status: attributes.OperationLogs.StatusCodes.SUCCESS,
            attributes.OperationLogs.OperationTimeStamp: now,
        })

        log, ts = bt_client.read_log_entry(op_id)
        assert log[attributes.OperationLogs.UserID] == "alice"
        np.testing.assert_array_equal(log[attributes.OperationLogs.RootID], root_ids)

    def test_read_log_entries_by_ids(self, bt_client):
        for op_id in [1, 2]:
            write_node(bt_client, np.uint64(op_id), {
                attributes.OperationLogs.UserID: f"user{op_id}",
                attributes.OperationLogs.RootID: np.array([op_id * 100], dtype=basetypes.NODE_ID),
                attributes.OperationLogs.OperationTimeStamp: datetime.now(timezone.utc),
            })
        logs = bt_client.read_log_entries(
            operation_ids=[np.uint64(1), np.uint64(2)]
        )
        assert len(logs) == 2

    def test_read_log_entries_empty(self, bt_client):
        result = bt_client.read_log_entries(
            operation_ids=[np.uint64(99999)]
        )
        assert result == {}


# ── Data Type Roundtrips ──────────────────────────────────────────────────


class TestDataTypeRoundtrips:
    def test_numpy_array_children(self, bt_client):
        arr = np.array([10, 20, 30], dtype=basetypes.NODE_ID)
        write_node(bt_client, 100, {attributes.Hierarchy.Child: arr})
        data = bt_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Child)
        np.testing.assert_array_equal(data[0].value, arr)

    def test_2d_array_cross_chunk_edges(self, bt_client):
        arr = np.array([[1, 2], [3, 4]], dtype=basetypes.NODE_ID)
        write_node(bt_client, 100, {attributes.Connectivity.CrossChunkEdge[3]: arr})
        data = bt_client.read_node(
            np.uint64(100), properties=attributes.Connectivity.CrossChunkEdge[3]
        )
        np.testing.assert_array_equal(data[0].value, arr)

    def test_scalar_parent(self, bt_client):
        write_node(bt_client, 100, {attributes.Hierarchy.Parent: np.uint64(42)})
        data = bt_client.read_node(np.uint64(100), properties=attributes.Hierarchy.Parent)
        assert data[0].value == np.uint64(42)

    def test_string_user_id(self, bt_client):
        write_node(bt_client, 100, {attributes.OperationLogs.UserID: "alice"})
        data = bt_client.read_node(np.uint64(100), properties=attributes.OperationLogs.UserID)
        assert data[0].value == "alice"

    def test_pickle_meta(self, bt_client):
        meta = {"complex": [1, 2, 3]}
        write_node(bt_client, 100, {attributes.TableMeta.Meta: meta})
        data = bt_client.read_node(np.uint64(100), properties=attributes.TableMeta.Meta)
        assert data[0].value == meta

    def test_compressed_array(self, bt_client):
        arr = np.array([[100, 200], [300, 400]], dtype=basetypes.NODE_ID)
        write_node(bt_client, 100, {attributes.Connectivity.AtomicCrossChunkEdge[0]: arr})
        data = bt_client.read_node(
            np.uint64(100), properties=attributes.Connectivity.AtomicCrossChunkEdge[0]
        )
        np.testing.assert_array_equal(data[0].value, arr)


# ── Lock Timestamps ───────────────────────────────────────────────────────


class TestGetLockTimestamp:
    def test_returns_datetime_when_locked(self, bt_client):
        root, op = np.uint64(100), np.uint64(1)
        bt_client.lock_root(root, op)
        ts = bt_client.get_lock_timestamp(root, op)
        assert isinstance(ts, (datetime, int, float))

    def test_no_lock_returns_none(self, bt_client):
        assert bt_client.get_lock_timestamp(np.uint64(100), np.uint64(1)) is None

    def test_wrong_op_returns_none(self, bt_client):
        bt_client.lock_root(np.uint64(100), np.uint64(1))
        assert bt_client.get_lock_timestamp(np.uint64(100), np.uint64(2)) is None


# ── Lock Expiry ──────────────────────────────────────────────────────────


class TestLockExpiry:
    def test_expired_lock_allows_relock(self, bt_client_short_expiry):
        client = bt_client_short_expiry
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert client.lock_root(root, op1) is True
        time.sleep(1.5)
        assert client.lock_root(root, op2) is True

    def test_non_expired_lock_blocks(self, bt_client_short_expiry):
        client = bt_client_short_expiry
        root, op1, op2 = np.uint64(100), np.uint64(1), np.uint64(2)
        assert client.lock_root(root, op1) is True
        assert client.lock_root(root, op2) is False
