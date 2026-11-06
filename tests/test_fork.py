"""A fork reads its table merged with its source's, capped at `TF`, and writes its own."""

import ast
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import kvdbclient.fork
from kvdbclient import attributes
from kvdbclient.bigtable.client import Client
from kvdbclient.bigtable.fork import ForkClient
from kvdbclient.exceptions import ForkTableError
from kvdbclient.fork.marker import ForkMarker, Hop, MarkerState
from kvdbclient.serializers import serialize_uint64

PARENT = attributes.Hierarchy.Parent
BASE = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
T1, TF1, T2, TF2, T3 = (BASE + timedelta(seconds=s) for s in (0, 10, 20, 30, 40))
ROW = serialize_uint64(np.uint64(7))
OTHER = serialize_uint64(np.uint64(8))
ROOT = np.uint64(11)
# A chunk id with only high bits set, so `id ^ CHUNK` is the segment id.
CHUNK = np.uint64(5) << np.uint64(50)


def _name():
    return f"test_{uuid.uuid4().hex[:12]}"


def _put(client, key, value, at):
    client.write([client.mutate_row(key, {PARENT: np.uint64(value)}, time_stamp=at)])


def _values(cells):
    return [int(cell.value) for cell in cells]


def _marker(*hops):
    return ForkMarker(
        chain=hops,
        state=MarkerState.READY,
        generation="g",
        source_max_op_id=0,
        requested_at=hops[0].tf,
    )


def _open(table_id, config):
    client = ForkClient(table_id, config)
    client.read_table_meta()
    return client


def _fork(config, *hops):
    return ForkClient.create_fork_table(
        _name(), _marker(*hops), {"test": True}, "0.0.1", config=config
    )


def _rows(client):
    return client.read_all_rows().rows


@pytest.fixture()
def source(bt_config):
    """A source table holding row 7 at T1 and T2, and row 8 at T2 only."""
    client = Client(table_id=_name(), config=bt_config)
    client.create_table(meta={"test": True}, version="0.0.1")
    _put(client, ROW, 1, T1)
    _put(client, ROW, 2, T2)
    _put(client, OTHER, 3, T2)
    return client


@pytest.fixture()
def fork(source, bt_config):
    """A fork of `source` at TF1, between its two writes."""
    return _fork(bt_config, Hop(source._table_id, TF1))


@pytest.fixture()
def second(fork, source, bt_config):
    """A fork of `fork` at TF2, so its chain runs through `fork` to `source`."""
    return _fork(bt_config, Hop(fork._table_id, TF2), Hop(source._table_id, TF1))


def test_a_fork_reads_its_source_up_to_tf_only(fork):
    assert _values(fork._read_byte_row(ROW, columns=PARENT)) == [1]
    assert fork._read_byte_row(OTHER, columns=PARENT) == []


def test_a_fork_row_merges_newest_first_and_wins_a_tie(fork):
    _put(fork, ROW, 5, T3)
    _put(fork, ROW, 9, T1)
    assert _values(fork._read_byte_row(ROW, columns=PARENT)) == [5, 9]


def test_ranges_and_row_keys_merge_alike(fork):
    _put(fork, OTHER, 4, T3)
    rows = fork._read_byte_rows(row_keys=[ROW, OTHER], columns=PARENT)
    ranged = fork._read_byte_rows(
        start_key=ROW, end_key=OTHER, end_key_inclusive=True, columns=PARENT
    )
    for found in (rows, ranged):
        assert _values(found[ROW]) == [1] and _values(found[OTHER]) == [4]


def test_a_window_past_tf_reads_the_fork_alone(fork):
    _put(fork, ROW, 5, T3)
    assert _values(fork._read_byte_row(ROW, columns=PARENT, start_time=T2)) == [5]


def test_an_end_before_tf_caps_the_source_further(fork):
    assert (
        fork._read_byte_row(ROW, columns=PARENT, end_time=T1 - timedelta(seconds=1))
        == []
    )


def test_the_plain_client_refuses_a_fork(fork, bt_config):
    with pytest.raises(ForkTableError):
        Client(table_id=fork._table_id, config=bt_config).read_table_meta()


def test_the_bigtable_client_opens_a_fork_and_a_plain_table(fork, source, bt_config):
    opened = _open(fork._table_id, bt_config)
    assert opened.marker == fork.marker and opened.read_table_meta() == {"test": True}
    assert _values(opened._read_byte_row(ROW, columns=PARENT)) == [1]
    plain = _open(source._table_id, bt_config)
    assert plain.marker is None
    assert _values(plain._read_byte_row(ROW, columns=PARENT)) == [2, 1]


def test_a_fork_reads_only_its_own_meta_and_version(source, bt_config):
    """Forked after the source wrote both rows, the fork still answers with its own."""
    marker = _marker(Hop(source._table_id, datetime.now(timezone.utc)))
    fork = ForkClient.create_fork_table(
        _name(), marker, {"fork": True}, "9.9.9", config=bt_config
    )
    assert fork.read_table_version() == "9.9.9"
    assert fork.read_table_meta() == {"fork": True}


WRITES = {
    "rows": lambda fork: _put(fork, ROW, 5, T3),
    "root lock": lambda fork: fork.lock_root(ROOT, np.uint64(2)),
    "id counter": lambda fork: fork.create_node_ids(CHUNK, 1),
    "operation id": lambda fork: fork.create_operation_id(),
    "meta": lambda fork: fork.update_table_meta({"test": False}, overwrite=True),
}


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_fork_writes_only_its_own_table(fork, source, bt_config, write):
    before = _rows(source)
    write(_open(fork._table_id, bt_config))
    assert _rows(source) == before


def test_every_table_upstream_refuses_writes(second):
    for upstream in (second._source, second._source._source):
        with pytest.raises(ForkTableError, match="upstream"):
            _put(upstream, ROW, 5, T3)


def test_a_fork_of_a_fork_reads_the_source_capped_at_the_earlier_tf(
    fork, second, bt_config
):
    assert _values(second._read_byte_row(ROW, columns=PARENT)) == [1]
    _put(fork, ROW, 6, T2)
    assert _values(second._read_byte_row(ROW, columns=PARENT)) == [6, 1]
    reopened = _open(second._table_id, bt_config)
    assert _values(reopened._read_byte_row(ROW, columns=PARENT)) == [6, 1]


def test_a_fork_mints_node_ids_above_its_sources(source, fork):
    source.create_node_ids(CHUNK, 5)
    assert (fork.create_node_ids(CHUNK, 2) ^ CHUNK).tolist() == [6, 7]


def test_a_chunk_seeds_above_its_legacy_row_and_every_shard(source, fork):
    source._get_ids_range(serialize_uint64(CHUNK, counter=True), 5000)
    source._get_root_segment_ids_range(CHUNK, size=40, counter=7)
    highest = 40 * 256 + 7
    roots = fork.create_node_ids(CHUNK, 3, root_chunk=True) ^ CHUNK
    legacy = fork.create_node_ids(CHUNK, 1) ^ CHUNK
    assert roots.min() > highest and legacy.min() > highest


def test_a_fork_of_a_fork_seeds_above_every_upstream_table(source, fork, second):
    source.create_node_ids(CHUNK, 5)
    fork.create_node_ids(CHUNK, 3)
    # The source counts past the fork after the fork's TF; the seed still clears it.
    source.create_node_ids(CHUNK, 4)
    assert (second.create_node_ids(CHUNK, 1) ^ CHUNK).tolist() == [10]


def test_op_ids_start_above_the_sources_at_fork_creation(source, bt_config):
    for _ in range(3):
        source.create_operation_id()
    fork = _fork(bt_config, Hop(source._table_id, TF1))
    assert fork.get_max_operation_id() == 3
    assert fork.create_operation_id() == 4


def test_renewing_a_lock_this_op_never_took_is_refused(fork):
    assert not fork.renew_lock(ROOT, np.uint64(1))


def test_a_lock_this_op_holds_renews(fork):
    fork.lock_root(ROOT, np.uint64(1))
    assert fork.renew_lock(ROOT, np.uint64(1))


def test_another_ops_lock_does_not_renew(fork):
    fork.lock_root(ROOT, np.uint64(1))
    assert not fork.renew_lock(ROOT, np.uint64(2))


def _lock_upstream(source, at):
    lock = {attributes.Concurrency.IndefiniteLock: np.uint64(9)}
    source.write([source.mutate_row(serialize_uint64(ROOT), lock, time_stamp=at)])


LOCKS = {
    "root lock": lambda fork: fork.lock_root(ROOT, np.uint64(1)),
    "indefinite": lambda fork: fork.lock_root_indefinitely(ROOT, np.uint64(1)),
    "row key": lambda fork: fork.lock_by_row_key_with_indefinite(
        serialize_uint64(ROOT), np.uint64(1)
    ),
}


@pytest.mark.parametrize("lock", LOCKS.values(), ids=LOCKS.keys())
def test_an_upstream_op_holding_a_root_at_tf_refuses_fork_locks(source, fork, lock):
    _lock_upstream(source, T1)
    assert not lock(fork)


@pytest.mark.parametrize("lock", LOCKS.values(), ids=LOCKS.keys())
def test_an_upstream_lock_taken_after_tf_does_not_reach_the_fork(source, fork, lock):
    _lock_upstream(source, T2)
    assert lock(fork)


def test_a_fork_whose_creation_never_finished_does_not_open(source, bt_config):
    """`CREATING` marks a half-built fork, so it is refused rather than read."""
    marker = replace(_marker(Hop(source._table_id, TF1)), state=MarkerState.CREATING)
    table_id = _name()
    with pytest.raises(ForkTableError, match="creating"):
        ForkClient.create_fork_table(
            table_id, marker, {"test": True}, "0.0.1", config=bt_config
        )
    with pytest.raises(ForkTableError, match="creating"):
        _open(table_id, bt_config)


def test_closing_a_fork_stops_the_source_pool_at_every_hop(second):
    second.close()
    for client in (second, second._source):
        with pytest.raises(RuntimeError, match="shutdown"):
            client._pool.submit(int)


def test_a_plain_table_opened_for_forks_still_updates_its_meta(source, bt_config):
    _open(source._table_id, bt_config).update_table_meta(
        {"test": False}, overwrite=True
    )
    assert _open(source._table_id, bt_config).read_table_meta() == {"test": False}


def test_the_fork_package_imports_no_backend():
    """Backend mechanics live in each backend's package; the fork package serves all."""
    imported = set()
    for path in Path(kvdbclient.fork.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
    backends = {"bigtable", "hbase", "google", "requests"}
    assert not [module for module in imported if backends & set(module.split("."))]
