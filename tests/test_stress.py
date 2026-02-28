# pylint: disable=missing-docstring, redefined-outer-name

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient import basetypes
from kvdbclient.serializers import serialize_uint64

STRESS_ROW_COUNT = int(os.environ.get("STRESS_ROW_COUNT", "10000"))
STRESS_LOCK_COUNT = int(os.environ.get("STRESS_LOCK_COUNT", "1000"))


# ── Helpers ───────────────────────────────────────────────────────────────


def _write_bulk(client, base_id, count):
    """Write `count` rows starting at base_id and return the node IDs."""
    entries = []
    node_ids = np.arange(base_id, base_id + count, dtype=basetypes.NODE_ID)
    for nid in node_ids:
        entry = client.mutate_row(
            serialize_uint64(nid),
            {attributes.Hierarchy.Child: np.array([nid], dtype=basetypes.NODE_ID)},
        )
        entries.append(entry)
    client.write(entries)
    return node_ids


# ── Stress Write ──────────────────────────────────────────────────────────


class TestStressWrite:
    def test_bulk_write(self, bt_client):
        count = STRESS_ROW_COUNT
        base = 1_000_000
        node_ids = _write_bulk(bt_client, base, count)
        data = bt_client.read_nodes(
            start_id=np.uint64(base),
            end_id=np.uint64(base + count),
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) >= count

    def test_bulk_write_read_roundtrip(self, bt_client):
        count = STRESS_ROW_COUNT
        base = 2_000_000
        node_ids = _write_bulk(bt_client, base, count)

        data = bt_client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == count
        expected = node_ids
        actual = np.array(
            [data[nid][0].value[0] for nid in node_ids], dtype=basetypes.NODE_ID
        )
        np.testing.assert_array_equal(actual, expected)


# ── Stress Read ───────────────────────────────────────────────────────────


class TestStressRead:
    def test_read_by_ids(self, bt_client):
        count = STRESS_ROW_COUNT
        base = 3_000_000
        node_ids = _write_bulk(bt_client, base, count)
        data = bt_client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == count

    def test_read_range(self, bt_client):
        count = STRESS_ROW_COUNT
        base = 4_000_000
        _write_bulk(bt_client, base, count)
        data = bt_client.read_nodes(
            start_id=np.uint64(base),
            end_id=np.uint64(base + count),
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == count


# ── Stress IDs ────────────────────────────────────────────────────────────


class TestStressIDs:
    def test_create_node_ids(self, bt_client):
        chunk_id = np.uint64(1 << 32)
        ids = bt_client.create_node_ids(chunk_id, STRESS_ROW_COUNT)
        assert len(ids) == STRESS_ROW_COUNT
        assert len(np.unique(ids)) == STRESS_ROW_COUNT


# ── Stress Locking ────────────────────────────────────────────────────────


class TestStressLocking:
    def test_lock_unlock_roots(self, bt_client):
        count = STRESS_LOCK_COUNT
        roots = [np.uint64(5_000_000 + i) for i in range(count)]
        op = np.uint64(1)
        future_d = {r: np.array([], dtype=np.uint64) for r in roots}

        success, locked_ids = bt_client.lock_roots(roots, op, future_d)
        assert success is True
        assert len(locked_ids) == count

        # Unlock them all in parallel
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(bt_client.unlock_root, r, op) for r in roots]
            for f in as_completed(futures):
                f.result()

        # Verify they can be re-locked
        op2 = np.uint64(2)
        future_d2 = {r: np.array([], dtype=np.uint64) for r in roots}
        success2, _ = bt_client.lock_roots(roots, op2, future_d2)
        assert success2 is True


# ── Sharded Read Stress ──────────────────────────────────────────────────


class TestShardedReadStress:
    def test_sharded_read_exceeding_max_row_key_count(self, bt_client_small_batch):
        """Write 100 rows, read with MAX_ROW_KEY_COUNT=50 to trigger sharding."""
        client = bt_client_small_batch
        count = 100
        base = 6_000_000
        node_ids = _write_bulk(client, base, count)

        data = client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Child,
        )
        assert len(data) == count
