"""The newest fork point at or before a request below which no edit still writes."""

import time
import typing
from datetime import datetime, timedelta

import numpy as np

from . import attributes
from . import basetypes
from .utils import TICK
from .utils import get_valid_timestamp

_LOGS = attributes.OperationLogs
#: Statuses after which an edit writes nothing; any other, or none, may still write.
FINISHED = frozenset(
    {
        _LOGS.StatusCodes.SUCCESS.value,
        _LOGS.StatusCodes.EXCEPTION.value,
        _LOGS.StatusCodes.WRITE_FAILED.value,
    }
)
#: How many op ids one read of the op log covers.
PAGE = 1000
#: How long a waiting search sleeps between reads.
POLL = timedelta(seconds=10)
#: The longest an edit takes from its op id to its CREATED log; until then it is unseen.
SETTLE = timedelta(minutes=1)


class Edit(typing.NamedTuple):
    """One logged edit: its op time and newest status."""

    op_time: datetime
    status: typing.Optional[int]

    @property
    def unfinished(self) -> bool:
        return self.status not in FINISHED


def _read(client, low: int, high: int) -> list[Edit]:
    """The edits logged under op ids `low` to `high`, in id order."""
    ids = np.arange(low, high + 1, dtype=basetypes.OPERATION_ID)
    fields = [_LOGS.Status, _LOGS.OperationTimeStamp, _LOGS.RootID]
    logs = client.read_log_entries(operation_ids=ids, properties=fields)
    return [
        Edit(log["timestamp"], log.get(_LOGS.Status)) for _, log in sorted(logs.items())
    ]


def _still_writing(client, settled: datetime, floor: datetime) -> list[datetime]:
    """Op times from `floor` to `settled` of edits not yet finished.

    Edits older than `floor` have finished or died. Op logs are read back from the
    newest id, and an edit older than `floor` by `SETTLE` ends the walk: every lower id
    took its op id earlier, so it began before `floor`.
    """
    if settled < floor:
        return []
    high = int(client.get_max_operation_id())
    found = []
    while high > 0:
        low = max(1, high - PAGE + 1)
        edits = _read(client, low, high)
        found += [
            e.op_time for e in edits if e.unfinished and floor <= e.op_time <= settled
        ]
        if any(e.op_time < floor - SETTLE for e in edits):
            break
        high = low - 1
    return found


def safe_time(
    client, requested: datetime, longest_edit: timedelta, wait: bool = False
) -> datetime:
    """The newest point at or before `requested` below which no edit still writes.

    An edit is unseen until it logs CREATED, so a request newer than `SETTLE` steps
    back to that age, or with `wait` waits until it is that old. Edits older than
    `longest_edit` have finished or died; the point steps a tick below each younger
    one still unfinished, and with `wait` lets those finish first.
    """
    while True:
        now = get_valid_timestamp(None)
        if wait and requested > now - SETTLE:
            time.sleep((requested + SETTLE - now).total_seconds())
            continue
        settled = min(requested, now - SETTLE)
        writing = _still_writing(client, settled, now - longest_edit)
        if not (wait and writing):
            return min([settled, *(t - TICK for t in writing)])
        time.sleep(POLL.total_seconds())
