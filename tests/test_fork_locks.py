# pylint: disable=missing-docstring, redefined-outer-name
"""A fork refuses a lock an op it inherits holds indefinitely upstream."""

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient.serializers import serialize_uint64

from helpers import T1
from helpers import T2
from helpers import T3

# Only Bigtable opens forks until HBase adds its fork hooks.
pytestmark = pytest.mark.parametrize("backend", ["bigtable"], indirect=True)

ROOT, OP, HOLDER = np.uint64(100), np.uint64(1), np.uint64(9)
LOCKS = {
    "root": lambda client: client.lock_root(ROOT, OP),
    "indefinite": lambda client: client.lock_root_indefinitely(ROOT, OP),
    "row key": lambda client: client.lock_by_row_key_with_indefinite(
        serialize_uint64(ROOT), OP
    ),
    "row key indefinite": lambda client: client.lock_by_row_key_indefinitely(
        serialize_uint64(ROOT), OP
    ),
}


def _log(table, at) -> None:
    """Op `HOLDER` logs CREATED in `table` at `at`, as an edit does at its op time."""
    status = {
        attributes.OperationLogs.Status: attributes.OperationLogs.StatusCodes.CREATED.value
    }
    table.write([table.mutate_row(serialize_uint64(HOLDER), status, time_stamp=at)])


def _lock(table, at) -> None:
    """Op `HOLDER` locks `ROOT` indefinitely in `table` at `at`, after its compute."""
    lock = {attributes.Concurrency.IndefiniteLock: HOLDER}
    table.write([table.mutate_row(serialize_uint64(ROOT), lock, time_stamp=at)])


@pytest.mark.parametrize("lock", LOCKS.values(), ids=LOCKS.keys())
def test_an_inherited_op_holding_a_root_refuses_fork_locks(source, fork, lock):
    _log(source, T1)
    _lock(source, T1)
    assert lock(fork) is False


@pytest.mark.parametrize("lock", LOCKS.values(), ids=LOCKS.keys())
def test_an_inherited_op_locking_after_tf_still_refuses_fork_locks(source, fork, lock):
    """An edit locks indefinitely after compute, so it can lock past its fork's `TF`."""
    _log(source, T1)
    _lock(source, T2)
    assert lock(fork) is False


def test_an_op_inherited_through_a_parent_fork_refuses_a_child_fork_lock(fork, second):
    _log(fork, T1)
    _lock(fork, T3)
    assert second.lock_root(ROOT, OP) is False


def test_an_op_inherited_from_the_first_table_refuses_a_fork_of_a_fork(source, second):
    _log(source, T1)
    _lock(source, T3)
    assert second.lock_root(ROOT, OP) is False


@pytest.mark.parametrize("lock", LOCKS.values(), ids=LOCKS.keys())
def test_an_op_begun_after_tf_refuses_no_fork_lock(source, fork, lock):
    _log(source, T2)
    _lock(source, T2)
    assert lock(fork) is True


def test_an_op_id_reused_in_another_table_is_not_inherited(source, fork, second):
    """The source's op began past `TF1`; the fork's own op of that id, by `TF2`."""
    _log(fork, T2)
    _log(source, T3)
    _lock(source, T3)
    assert second.lock_root(ROOT, OP) is True
