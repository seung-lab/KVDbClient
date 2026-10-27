# pylint: disable=missing-docstring, redefined-outer-name

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from kvdbclient import attributes
from kvdbclient import basetypes

from helpers import write_node


# ── is_latest_roots ───────────────────────────────────────────────────────

CREATED = datetime(2024, 1, 1, tzinfo=timezone.utc)
EDITED = CREATED + timedelta(days=2)
OLD_ROOT, NEW_ROOT, NO_ROW = 100, 200, 300


@pytest.fixture(params=["bt_client", "bt_client_small_batch", "hbase_client"])
def edited_client(request):
    """`OLD_ROOT` exists from `CREATED` and is replaced by `NEW_ROOT` at `EDITED`."""
    client = request.getfixturevalue(request.param)
    children = np.array([1], dtype=basetypes.NODE_ID)
    write_node(client, OLD_ROOT, {attributes.Hierarchy.Child: children}, CREATED)
    write_node(client, NEW_ROOT, {attributes.Hierarchy.Child: children}, EDITED)
    write_node(
        client,
        OLD_ROOT,
        {attributes.Hierarchy.NewParent: np.array([NEW_ROOT], dtype=basetypes.NODE_ID)},
        EDITED,
    )
    return client


class TestIsLatestRoots:
    @pytest.mark.parametrize(
        "time_stamp, expected",
        [
            (CREATED + timedelta(days=1), [True, False, False]),
            (EDITED, [True, False, False]),
            (EDITED + timedelta(milliseconds=1), [False, True, False]),
            (None, [False, True, False]),
        ],
        ids=["before_edit", "at_edit", "after_edit", "now"],
    )
    def test_matches_pcg(self, edited_client, time_stamp, expected):
        latest = edited_client.root_ext.is_latest_roots(
            [OLD_ROOT, NEW_ROOT, NO_ROW], time_stamp=time_stamp
        )
        np.testing.assert_array_equal(latest, expected)

    def test_childless_row_is_not_latest(self, edited_client):
        childless = 400
        write_node(
            edited_client, childless, {attributes.Hierarchy.Parent: np.uint64(OLD_ROOT)}, CREATED
        )
        latest = edited_client.root_ext.is_latest_roots([childless])
        np.testing.assert_array_equal(latest, [False])

    def test_keeps_input_order_and_duplicates(self, edited_client):
        # 60 ids sorting below the roots push them into the second chunk of a 50-key read.
        absent = list(range(10, 70))
        latest = edited_client.root_ext.is_latest_roots(
            [NEW_ROOT, OLD_ROOT, *absent, OLD_ROOT, NEW_ROOT]
        )
        np.testing.assert_array_equal(latest, [True, False, *[False] * len(absent), False, True])

    def test_empty(self, edited_client):
        latest = edited_client.root_ext.is_latest_roots([])
        assert latest.dtype == bool
        assert latest.shape == (0,)
