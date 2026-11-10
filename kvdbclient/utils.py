"""Timestamp and time utilities for storage backends."""

import datetime
from typing import NamedTuple

import pytz

# Storage keeps time to the millisecond, so one tick is the next instant it tells apart.
TICK = datetime.timedelta(milliseconds=1)


def get_max_time():
    """Returns the (almost) max time in datetime.datetime."""
    return datetime.datetime(9999, 12, 31, 23, 59, 59, 0)


def get_min_time():
    """Returns the min time in datetime.datetime."""
    return datetime.datetime.strptime("01/01/00 00:00", "%d/%m/%y %H:%M")


def get_valid_timestamp(timestamp):
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = pytz.UTC.localize(timestamp)
    # Comply to resolution of BigTables TimeRange
    return get_compatible_time_stamp(timestamp, round_up=False)


def get_compatible_time_stamp(
    time_stamp: datetime.datetime, round_up: bool = False
) -> datetime.datetime:
    """Round a datetime down to the millisecond, the resolution storage keeps.

    Rounding up lands strictly after `time_stamp`, so the result serves as an
    exclusive upper bound that keeps the input's millisecond.
    """
    micro_s_gap = datetime.timedelta(microseconds=time_stamp.microsecond % 1000)
    if round_up:
        return time_stamp + (TICK - micro_s_gap)
    return time_stamp - micro_s_gap


class TimeRange(NamedTuple):
    """A read's time bounds, each rounded to the millisecond, or None where open."""

    start: datetime.datetime | None
    end: datetime.datetime | None


def get_compatible_time_range(start, end, end_inclusive: bool) -> TimeRange:
    """`start` rounded down, and `end` rounded up when inclusive; None stays open."""
    return TimeRange(
        None if start is None else get_compatible_time_stamp(start),
        None if end is None else get_compatible_time_stamp(end, round_up=end_inclusive),
    )
