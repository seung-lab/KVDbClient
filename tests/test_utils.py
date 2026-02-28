"""Tests for kvdbclient.utils"""

import datetime

import pytz
import pytest

from kvdbclient.utils import (
    get_max_time,
    get_min_time,
    get_valid_timestamp,
    get_compatible_time_stamp,
)


class TestTimeFunctions:
    def test_get_max_time(self):
        t = get_max_time()
        assert isinstance(t, datetime.datetime)
        assert t.year == 9999

    def test_get_min_time(self):
        t = get_min_time()
        assert isinstance(t, datetime.datetime)
        assert t.year == 2000


class TestGetValidTimestamp:
    def test_none_returns_utc_now(self):
        before = datetime.datetime.now(datetime.timezone.utc)
        result = get_valid_timestamp(None)
        after = datetime.datetime.now(datetime.timezone.utc)
        assert result.tzinfo is not None
        # get_valid_timestamp rounds down to millisecond precision,
        # so result may be slightly before `before`
        tolerance = datetime.timedelta(milliseconds=1)
        assert before - tolerance <= result <= after

    def test_naive_gets_localized(self):
        naive = datetime.datetime(2023, 6, 15, 12, 0, 0)
        result = get_valid_timestamp(naive)
        assert result.tzinfo is not None

    def test_aware_passthrough(self):
        aware = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=pytz.UTC)
        result = get_valid_timestamp(aware)
        assert result.tzinfo is not None


class TestCompatibleTimestamp:
    def test_round_down(self):
        ts = datetime.datetime(2023, 6, 15, 12, 0, 0, 1500)
        result = get_compatible_time_stamp(ts, round_up=False)
        assert result.microsecond % 1000 == 0
        assert result.microsecond == 1000

    def test_round_up(self):
        ts = datetime.datetime(2023, 6, 15, 12, 0, 0, 1500)
        result = get_compatible_time_stamp(ts, round_up=True)
        assert result.microsecond % 1000 == 0
        assert result.microsecond == 2000

    def test_exact_no_change(self):
        ts = datetime.datetime(2023, 6, 15, 12, 0, 0, 3000)
        result = get_compatible_time_stamp(ts)
        assert result == ts
