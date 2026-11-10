# pylint: disable=missing-docstring, redefined-outer-name
"""A fork writes only its own table, and mints ids above every table it reads."""

import numpy as np
import pytest

from kvdbclient.exceptions import ForkTableError
from kvdbclient.fork.marker import Hop
from kvdbclient.serializers import serialize_uint64

from helpers import NODE
from helpers import T3
from helpers import TF1
from helpers import create_fork
from helpers import fork_marker
from helpers import open_fork
from helpers import write_parent

# Only Bigtable opens forks until HBase adds its fork hooks.
pytestmark = pytest.mark.parametrize("backend", ["bigtable"], indirect=True)

ROOT = np.uint64(11)
# A chunk id with only high bits set, so `id ^ CHUNK` is the segment id.
CHUNK = np.uint64(5) << np.uint64(50)
WRITES = {
    "rows": lambda fork: write_parent(fork, NODE, 5, T3),
    "root lock": lambda fork: fork.lock_root(ROOT, np.uint64(2)),
    "id counter": lambda fork: fork.create_node_ids(CHUNK, 1),
    "operation id": lambda fork: fork.create_operation_id(),
    "meta": lambda fork: fork.update_table_meta({"test": False}, overwrite=True),
}


def _rows(client):
    return client.read_all_rows().rows


def _crash(*_, **__):
    raise RuntimeError("crashed")


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_a_fork_writes_only_its_own_table(backend, source, fork, write):
    before = _rows(source)
    write(open_fork(backend, fork._table_id))
    assert _rows(source) == before


def test_every_table_upstream_refuses_writes(second):
    for upstream in (second._source, second._source._source):
        with pytest.raises(ForkTableError, match="upstream"):
            write_parent(upstream, NODE, 5, T3)


def test_a_fork_mints_node_ids_above_its_source(source, fork):
    source.create_node_ids(CHUNK, 5)
    assert (fork.create_node_ids(CHUNK, 2) ^ CHUNK).tolist() == [6, 7]


def test_a_chunk_seeds_above_its_unsharded_row_and_every_shard(source, fork):
    source._get_ids_range(serialize_uint64(CHUNK, counter=True), 5000)
    source._get_root_segment_ids_range(CHUNK, size=40, counter=7)
    highest = 40 * 256 + 7
    roots = fork.create_node_ids(CHUNK, 3, root_chunk=True) ^ CHUNK
    unsharded = fork.create_node_ids(CHUNK, 1) ^ CHUNK
    assert roots.min() > highest and unsharded.min() > highest


def test_a_fork_of_a_fork_seeds_above_every_upstream_table(source, fork, second):
    source.create_node_ids(CHUNK, 5)
    fork.create_node_ids(CHUNK, 3)
    # The source counts past the fork after the fork's TF; the seed still clears it.
    source.create_node_ids(CHUNK, 4)
    assert (second.create_node_ids(CHUNK, 1) ^ CHUNK).tolist() == [10]


def test_op_ids_start_above_the_source_s_at_fork_creation(backend, source):
    for _ in range(3):
        source.create_operation_id()
    fork = create_fork(backend, fork_marker(Hop(source._table_id, TF1)))
    assert (fork.get_max_operation_id(), fork.create_operation_id()) == (3, 4)


def test_a_meta_overwrite_cut_short_keeps_the_marker(backend, fork, monkeypatch):
    """A crash between dropping the old meta and writing the new must not unfork it."""
    monkeypatch.setattr(fork, "write", _crash)
    with pytest.raises(RuntimeError):
        fork.update_table_meta({"test": False}, overwrite=True)
    assert open_fork(backend, fork._table_id).marker == fork.marker
