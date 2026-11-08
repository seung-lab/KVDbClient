# pylint: disable=missing-docstring
from datetime import datetime, timedelta, timezone

import pytest

from kvdbclient import attributes
from kvdbclient import safe_time

from helpers import write_log

LOGS = attributes.OperationLogs
LONGEST_EDIT = timedelta(minutes=21)
#: Old enough that every edit before it has logged CREATED, young enough to be searched.
SETTLED = timedelta(minutes=2)


def _now():
    # Bigtable keeps millisecond timestamps, so test times stay on whole seconds.
    return datetime.now(timezone.utc).replace(microsecond=0)


def _edit(client, op_time, status=LOGS.StatusCodes.CREATED):
    """An edit logged at `op_time` under an id the op counter issued; its id."""
    op_id = client.create_operation_id()
    write_log(client, op_id, op_time=op_time, status=status)
    return op_id


def _safe(client, requested, wait=False):
    return safe_time.safe_time(client, requested, LONGEST_EDIT, wait=wait)


def _frozen(monkeypatch, now) -> list[float]:
    """Holds safe_time's clock at `now`, each sleep moving it on; the sleeps, in order."""
    clock, sleeps = [now], []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(safe_time, "get_valid_timestamp", lambda _: clock[0])
    monkeypatch.setattr(safe_time.time, "sleep", sleep)
    return sleeps


@pytest.mark.parametrize(
    "status",
    [
        LOGS.StatusCodes.SUCCESS,
        LOGS.StatusCodes.EXCEPTION,
        LOGS.StatusCodes.WRITE_FAILED,
    ],
    ids=["success", "exception", "write_failed"],
)
def test_with_every_edit_finished_the_request_stands(client, status):
    requested = _now() - SETTLED
    _edit(client, requested - timedelta(minutes=1), status)
    assert _safe(client, requested) == requested


@pytest.mark.parametrize(
    "status",
    [LOGS.StatusCodes.CREATED, LOGS.StatusCodes.WRITE_STARTED, None],
    ids=["created", "write_started", "no_status"],
)
def test_it_steps_a_tick_below_an_unfinished_edit(client, status):
    requested = _now() - SETTLED
    began = requested - timedelta(minutes=1)
    _edit(client, began, status)
    assert _safe(client, requested) == began - safe_time.TICK


@pytest.mark.parametrize(
    "offset",
    [timedelta(minutes=1), -LONGEST_EDIT - timedelta(minutes=1)],
    ids=["after_the_request", "older_than_the_longest_edit"],
)
def test_an_edit_that_cannot_write_below_the_request_is_ignored(client, offset):
    requested = _now() - timedelta(minutes=5)
    _edit(client, requested + offset)
    assert _safe(client, requested) == requested


def test_a_request_older_than_the_longest_edit_reads_no_op_log(client, monkeypatch):
    requested = _now() - LONGEST_EDIT - timedelta(minutes=1)
    _edit(client, requested - timedelta(minutes=1))
    monkeypatch.setattr(client, "read_log_entries", lambda **_: pytest.fail("read"))
    assert _safe(client, requested) == requested


def test_the_walk_runs_from_the_newest_id_past_later_edits_and_id_gaps(
    client, monkeypatch
):
    monkeypatch.setattr(safe_time, "PAGE", 2)
    requested = _now() - timedelta(minutes=5)
    began = requested - timedelta(minutes=1)
    _edit(client, began)
    for seconds in (10, 20):
        client.create_operation_id()
        client.create_operation_id()
        _edit(client, requested + timedelta(seconds=seconds), LOGS.StatusCodes.SUCCESS)
    assert _safe(client, requested) == began - safe_time.TICK


def test_no_log_below_an_edit_older_than_the_window_is_read(client, monkeypatch):
    monkeypatch.setattr(safe_time, "PAGE", 1)
    requested = _now() - SETTLED
    _edit(client, requested - timedelta(minutes=1))
    old = requested - LONGEST_EDIT - timedelta(minutes=5)
    _edit(client, old, LOGS.StatusCodes.SUCCESS)
    assert _safe(client, requested) == requested


def test_a_request_newer_than_the_settle_steps_back_to_it(client, monkeypatch):
    now = _now()
    _frozen(monkeypatch, now)
    assert _safe(client, now) == now - safe_time.SETTLE


def test_waiting_lets_a_new_request_settle_first(client, monkeypatch):
    now = _now()
    sleeps = _frozen(monkeypatch, now)
    assert _safe(client, now, wait=True) == now
    assert sleeps == [safe_time.SETTLE.total_seconds()]


def test_waiting_lets_a_young_edit_finish_first(client, monkeypatch):
    requested = _now() - SETTLED
    began = requested - timedelta(minutes=1)
    op_id = _edit(client, began)
    sleeps = []

    def finish(seconds):
        sleeps.append(seconds)
        write_log(client, op_id, op_time=began, status=LOGS.StatusCodes.SUCCESS)

    monkeypatch.setattr(safe_time.time, "sleep", finish)
    assert _safe(client, requested, wait=True) == requested
    assert sleeps == [safe_time.POLL.total_seconds()]
