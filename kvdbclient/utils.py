"""Timestamp and time utilities for storage backends."""

import datetime

import pytz


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
    """Round a datetime to millisecond precision.

    Google BigTable restricts timestamp accuracy to milliseconds.
    By default, timestamps are rounded down.
    """
    micro_s_gap = datetime.timedelta(microseconds=time_stamp.microsecond % 1000)
    if micro_s_gap == 0:
        return time_stamp
    if round_up:
        time_stamp += datetime.timedelta(microseconds=1000) - micro_s_gap
    else:
        time_stamp -= micro_s_gap
    return time_stamp
