# pylint: disable=missing-docstring, redefined-outer-name
"""A fork reads its own table merged with its source's, each source capped at its `TF`."""

from datetime import datetime, timezone

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient.exceptions import ForkTableError
from kvdbclient.fork.marker import Hop, MarkerState
from kvdbclient.serializers import serialize_uint64

from helpers import NODE
from helpers import OTHER
from helpers import PARENT
from helpers import T1
from helpers import T2
from helpers import T3
from helpers import TF1
from helpers import create_fork
from helpers import fork_marker
from helpers import new_table_id
from helpers import node_ids
from helpers import open_fork
from helpers import write_parent

# Only Bigtable opens forks until HBase adds its fork hooks.
pytestmark = pytest.mark.parametrize("backend", ["bigtable"], indirect=True)


def _parents(cells) -> list[int]:
    return [int(cell.value) for cell in cells]


def test_a_fork_reads_its_source_up_to_tf_only(fork):
    assert _parents(fork.read_node(NODE, properties=PARENT)) == [1]
    assert fork.read_node(OTHER, properties=PARENT) == []


def test_a_fork_merges_cells_newest_first_and_its_own_win_a_tie(fork):
    write_parent(fork, NODE, 5, T3)
    write_parent(fork, NODE, 9, T1)
    assert _parents(fork.read_node(NODE)[PARENT]) == [5, 9]


READS = {
    "ids": lambda client: client.read_nodes(
        node_ids=node_ids(*range(1, 7)), properties=PARENT
    ),
    "range": lambda client: client.read_nodes(
        start_id=np.uint64(1),
        end_id=np.uint64(6),
        end_id_inclusive=True,
        properties=PARENT,
    ),
}


@pytest.mark.parametrize("read", READS.values(), ids=READS.keys())
def test_a_fork_merges_many_rows_in_key_order(source, fork, read):
    for node in (2, 4, 6):
        write_parent(source, node, node, T1)
    for node in (1, 3, 5):
        write_parent(fork, node, node, T3)
    rows = read(fork)
    assert [int(node) for node in rows] == [1, 2, 3, 4, 5, 6]
    assert [_parents(cells) for cells in rows.values()] == [[n] for n in range(1, 7)]


def test_a_window_starting_past_tf_reads_the_fork_alone(fork):
    write_parent(fork, NODE, 5, T3)
    assert _parents(fork.read_node(NODE, properties=PARENT, start_time=T2)) == [5]


@pytest.mark.parametrize("inclusive, parents", [(False, []), (True, [1])])
def test_an_end_before_tf_caps_the_source_read(fork, inclusive, parents):
    cells = fork.read_node(
        NODE, properties=PARENT, end_time=T1, end_time_inclusive=inclusive
    )
    assert _parents(cells) == parents


def test_a_fork_of_a_fork_reads_each_table_up_to_its_own_tf(backend, fork, second):
    write_parent(fork, NODE, 6, T2)
    write_parent(fork, NODE, 8, T3)
    reopened = open_fork(backend, second._table_id)
    assert _parents(second.read_node(NODE, properties=PARENT)) == [6, 1]
    assert _parents(reopened.read_node(NODE, properties=PARENT)) == [6, 1]


def test_a_fork_answers_with_its_own_meta_and_version(backend, source):
    """Forked after the source wrote both rows, the fork still reads only its own."""
    marker = fork_marker(Hop(source._table_id, datetime.now(timezone.utc)))
    fork = create_fork(backend, marker, {"fork": True}, "9.9.9")
    assert (fork.read_table_meta(), fork.read_table_version()) == (
        {"fork": True},
        "9.9.9",
    )


def test_the_fork_client_opens_a_fork_and_a_plain_table(backend, source, fork):
    opened, plain = open_fork(backend, fork._table_id), open_fork(
        backend, source._table_id
    )
    assert (opened.marker, plain.marker) == (fork.marker, None)
    assert _parents(opened.read_node(NODE, properties=PARENT)) == [1]
    assert _parents(plain.read_node(NODE, properties=PARENT)) == [2, 1]


def test_a_fork_keeps_its_marker_through_a_meta_overwrite(backend, fork):
    fork.update_table_meta({"test": False}, overwrite=True)
    reopened = open_fork(backend, fork._table_id)
    assert (reopened.marker, reopened.table_meta) == (fork.marker, {"test": False})


def test_the_fork_client_overwrites_a_plain_tables_meta(backend, source):
    open_fork(backend, source._table_id).update_table_meta(
        {"test": False}, overwrite=True
    )
    assert open_fork(backend, source._table_id).read_table_meta() == {"test": False}


def test_a_fork_refuses_to_read_all_rows(fork):
    with pytest.raises(ForkTableError, match="read_all_rows"):
        fork.read_all_rows()


def test_the_fork_client_reads_all_rows_of_a_plain_table(backend, source):
    rows = open_fork(backend, source._table_id).read_all_rows().rows
    keys = {serialize_uint64(np.uint64(node)) for node in (NODE, OTHER)}
    assert set(rows) == {attributes.TableMeta.key, attributes.TableVersion.key, *keys}


def test_a_fork_whose_creation_never_finished_does_not_open(backend, source):
    """`CREATING` marks a half-built fork, so it is refused rather than read."""
    table_id = new_table_id()
    marker = fork_marker(Hop(source._table_id, TF1), state=MarkerState.CREATING)
    with pytest.raises(ForkTableError, match="creating"):
        create_fork(backend, marker, table_id=table_id)
    with pytest.raises(ForkTableError, match="creating"):
        open_fork(backend, table_id)


def test_closing_a_fork_stops_the_source_pool_at_every_hop(second):
    second.close()
    for client in (second, second._source):
        with pytest.raises(RuntimeError, match="shutdown"):
            client._pool.submit(int)
