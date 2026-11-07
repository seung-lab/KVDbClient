# pylint: disable=missing-docstring
from datetime import datetime, timezone

import pytest

from kvdbclient.base import Cell
from kvdbclient.base import LegacyRow
from kvdbclient.base import LegacyRows

T1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T2 = datetime(2025, 1, 1, tzinfo=timezone.utc)


class TestCell:
    def test_equal_by_value_and_timestamp(self):
        assert Cell(b"a", T1) == Cell(b"a", T1)
        assert hash(Cell(b"a", T1)) == hash(Cell(b"a", T1))

    @pytest.mark.parametrize(
        "other",
        [Cell(b"b", T1), Cell(b"a", T2), b"a"],
        ids=["value", "timestamp", "type"],
    )
    def test_differs_by_value_timestamp_or_type(self, other):
        assert Cell(b"a", T1) != other


class TestLegacyRow:
    def test_groups_cells_by_family_and_qualifier(self):
        row = LegacyRow([("0", b"q", Cell(b"a", T1)), ("0", b"q", Cell(b"b", T2))])
        assert row.cells == {"0": {b"q": [Cell(b"a", T1), Cell(b"b", T2)]}}

    def test_rows_with_the_same_cells_are_equal(self):
        cells = [("0", b"q", Cell(b"a", T1))]
        assert LegacyRow(cells) == LegacyRow(list(cells))

    def test_equal_by_replaces_the_cells_in_comparison(self):
        cells = [("0", b"q", Cell(b"a", T1))]
        assert LegacyRow(cells, equal_by="x") != LegacyRow(cells, equal_by="y")

    def test_never_equals_another_type(self):
        assert LegacyRow([]) != "row"

    def test_equal_rows_stay_distinct_in_a_set(self):
        assert len({LegacyRow([]), LegacyRow([])}) == 2


def test_consume_all_leaves_the_rows_in_place():
    rows = LegacyRows({b"k": LegacyRow([])})
    rows.consume_all()
    assert list(rows.rows) == [b"k"]
