"""A fork reads its table merged with its source's, capped at `TF`, and writes nothing."""

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient.bigtable.client import Client
from kvdbclient.exceptions import ForkTableError
from kvdbclient.fork.client import ForkClient, create_fork_table, open_client
from kvdbclient.fork.marker import ForkMarker, Hop, MarkerState
from kvdbclient.serializers import serialize_uint64

PARENT = attributes.Hierarchy.Parent
BASE = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
T1, TF1, T2, TF2, T3 = (BASE + timedelta(seconds=s) for s in (0, 10, 20, 30, 40))
ROW = serialize_uint64(np.uint64(7))
OTHER = serialize_uint64(np.uint64(8))


def _name():
    return f"test_{uuid.uuid4().hex[:12]}"


def _put(client, key, value, at):
    client.write([client.mutate_row(key, {PARENT: np.uint64(value)}, time_stamp=at)])


def _plain(fork):
    """A plain client on a fork's table, standing in for the edits forks refuse."""
    return Client(table_id=fork._table_id, config=fork._config)


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
    marker = _marker(Hop(source._table_id, TF1))
    return create_fork_table(_name(), marker, {"test": True}, "0.0.1", config=bt_config)


def test_a_fork_reads_its_source_up_to_tf_only(fork):
    assert _values(fork._read_byte_row(ROW, columns=PARENT)) == [1]
    assert fork._read_byte_row(OTHER, columns=PARENT) == []


def test_a_fork_row_merges_newest_first_and_wins_a_tie(fork):
    _put(_plain(fork), ROW, 5, T3)
    _put(_plain(fork), ROW, 9, T1)
    assert _values(fork._read_byte_row(ROW, columns=PARENT)) == [5, 9]


def test_ranges_and_row_keys_merge_alike(fork):
    _put(_plain(fork), OTHER, 4, T3)
    rows = fork._read_byte_rows(row_keys=[ROW, OTHER], columns=PARENT)
    ranged = fork._read_byte_rows(
        start_key=ROW, end_key=OTHER, end_key_inclusive=True, columns=PARENT
    )
    for found in (rows, ranged):
        assert _values(found[ROW]) == [1] and _values(found[OTHER]) == [4]


def test_a_window_past_tf_reads_the_fork_alone(fork):
    _put(_plain(fork), ROW, 5, T3)
    assert _values(fork._read_byte_row(ROW, columns=PARENT, start_time=T2)) == [5]


def test_an_end_before_tf_caps_the_source_further(fork):
    assert (
        fork._read_byte_row(ROW, columns=PARENT, end_time=T1 - timedelta(seconds=1))
        == []
    )


def test_the_plain_client_refuses_a_fork(fork, bt_config):
    with pytest.raises(ForkTableError):
        Client(table_id=fork._table_id, config=bt_config).read_table_meta()


def test_open_client_reads_through_a_fork_and_not_a_plain_table(
    fork, source, bt_config
):
    opened = open_client(fork._table_id, config=bt_config)
    assert opened.marker == fork.marker and opened.read_table_meta() == {"test": True}
    assert _values(opened._read_byte_row(ROW, columns=PARENT)) == [1]
    plain = open_client(source._table_id, config=bt_config)
    assert isinstance(plain, ForkClient) and plain.marker is None
    assert _values(plain._read_byte_row(ROW, columns=PARENT)) == [2, 1]


def test_a_fork_reads_only_its_own_meta_and_version(source, bt_config):
    """Forked after the source wrote both rows, the fork still answers with its own."""
    marker = _marker(Hop(source._table_id, datetime.now(timezone.utc)))
    fork = create_fork_table(_name(), marker, {"fork": True}, "9.9.9", config=bt_config)
    assert fork.read_table_version() == "9.9.9"
    assert fork.read_table_meta() == {"fork": True}


WRITES = {
    "rows": lambda fork: _put(fork, ROW, 5, T3),
    "root lock": lambda fork: fork.lock_root(np.uint64(1), np.uint64(2)),
    "id counter": lambda fork: fork.create_node_ids(np.uint64(1), 1),
    "operation id": lambda fork: fork.create_operation_id(),
    "meta": lambda fork: fork.update_table_meta({"test": False}, overwrite=True),
}


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_fork_refuses_every_write_and_keeps_reading(fork, bt_config, write):
    with pytest.raises(ForkTableError, match="read-only"):
        write(open_client(fork._table_id, config=bt_config))
    assert _values(fork._read_byte_row(ROW, columns=PARENT)) == [1]
    assert open_client(fork._table_id, config=bt_config).read_table_meta() == {
        "test": True
    }


def test_a_fork_of_a_fork_reads_the_source_capped_at_the_earlier_tf(
    fork, source, bt_config
):
    marker = _marker(Hop(fork._table_id, TF2), Hop(source._table_id, TF1))
    second = create_fork_table(
        _name(), marker, {"test": True}, "0.0.1", config=bt_config
    )
    assert _values(second._read_byte_row(ROW, columns=PARENT)) == [1]
    _put(_plain(fork), ROW, 6, T2)
    assert _values(second._read_byte_row(ROW, columns=PARENT)) == [6, 1]
    reopened = open_client(second._table_id, config=bt_config)
    assert _values(reopened._read_byte_row(ROW, columns=PARENT)) == [6, 1]
