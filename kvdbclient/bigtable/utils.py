from typing import Dict
from typing import Union
from typing import Iterable
from typing import Optional
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import numpy as np
from google.cloud.bigtable.data.row_filters import RowFilter
from google.cloud.bigtable.data.row_filters import PassAllFilter
from google.cloud.bigtable.data.row_filters import BlockAllFilter
from google.cloud.bigtable.data.row_filters import RowFilterChain
from google.cloud.bigtable.data.row_filters import RowFilterUnion
from google.cloud.bigtable.data.row_filters import ValueRangeFilter
from google.cloud.bigtable.data.row_filters import CellsRowLimitFilter
from google.cloud.bigtable.data.row_filters import CellsColumnLimitFilter
from google.cloud.bigtable.data.row_filters import ColumnRangeFilter
from google.cloud.bigtable.data.row_filters import TimestampRangeFilter
from google.cloud.bigtable.data.row_filters import ConditionalRowFilter
from google.cloud.bigtable.data.row_filters import ColumnQualifierRegexFilter

from .. import attributes
from ..base import Cell as BaseCell
from ..utils import get_compatible_time_range, get_compatible_time_stamp


def _exact_column_filter(col: attributes._Attribute) -> ColumnRangeFilter:
    """ColumnRangeFilter matching a single exact column qualifier."""
    return ColumnRangeFilter(
        family_id=col.family_id,
        start_qualifier=col.key,
        end_qualifier=col.key,
        inclusive_start=True,
        inclusive_end=True,
    )


def _exact_value_filter(value: bytes) -> ValueRangeFilter:
    """ValueRangeFilter matching a single exact value."""
    return ValueRangeFilter(
        start_value=value,
        end_value=value,
        inclusive_start=True,
        inclusive_end=True,
    )


def row_to_column_dict(row) -> Dict[attributes._Attribute, list]:
    """Convert a BigtableDataClient Row to {attribute: [BaseCell, ...]} dict."""
    column_dict: Dict[attributes._Attribute, list] = {}
    for cell in row.cells:
        attr = attributes.from_key(cell.family, cell.qualifier)
        ts = datetime.fromtimestamp(cell.timestamp_micros / 1_000_000, tz=timezone.utc)
        column_dict.setdefault(attr, []).append(
            BaseCell(value=cell.value, timestamp=ts)
        )
    return column_dict


def _get_column_filter(
    columns: Union[Iterable[attributes._Attribute], attributes._Attribute] = None,
) -> RowFilter:
    """Generates a RowFilter that accepts the specified columns"""
    if isinstance(columns, attributes._Attribute):
        return _exact_column_filter(columns)
    elif len(columns) == 1:
        return _exact_column_filter(columns[0])
    return RowFilterUnion([_exact_column_filter(col) for col in columns])


def _get_user_filter(user_id: str):
    """generates a ColumnRegEx Filter which filters user ids"""
    condition = RowFilterChain(
        [
            ColumnQualifierRegexFilter(attributes.OperationLogs.UserID.key),
            _exact_value_filter(str.encode(user_id)),
            CellsRowLimitFilter(1),
        ]
    )

    conditional_filter = ConditionalRowFilter(
        predicate_filter=condition,
        true_filter=PassAllFilter(True),
        false_filter=BlockAllFilter(True),
    )
    return conditional_filter


def _get_time_range_filter(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = True,
) -> RowFilter:
    """A TimestampRangeFilter over the bounds, rounded to Bigtable's millisecond."""
    bounds = get_compatible_time_range(start_time, end_time, end_inclusive)
    return TimestampRangeFilter(start=bounds.start, end=bounds.end)


def get_time_range_and_column_filter(
    columns: Optional[
        Union[Iterable[attributes._Attribute], attributes._Attribute]
    ] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = False,
    user_id: Optional[str] = None,
) -> RowFilter:
    time_filter = _get_time_range_filter(
        start_time=start_time, end_time=end_time, end_inclusive=end_inclusive
    )
    filters = [time_filter]
    if columns is not None:
        if len(columns) == 0:
            raise ValueError(
                f"Empty column filter {columns} is ambiguous. Pass `None` if no column filter should be applied."
            )
        column_filter = _get_column_filter(columns)
        filters = [column_filter, time_filter]
    if user_id is not None:
        user_filter = _get_user_filter(user_id=user_id)
        filters.append(user_filter)
    if len(filters) > 1:
        return RowFilterChain(filters)
    return filters[0]


def _unexpired(lock_expiry: timedelta) -> TimestampRangeFilter:
    """Keeps lock cells written within `lock_expiry`, to Bigtable's millisecond."""
    cutoff = get_compatible_time_stamp(datetime.now(timezone.utc) - lock_expiry)
    return TimestampRangeFilter(start=cutoff)


def _held_by(lock_column, operation_id: np.uint64) -> list[RowFilter]:
    """Filters keeping `lock_column`'s newest cell only while it holds `operation_id`."""
    return [
        _exact_column_filter(lock_column),
        CellsColumnLimitFilter(1),
        _exact_value_filter(lock_column.serialize(operation_id)),
    ]


def _new_parent_filter() -> ColumnRangeFilter:
    """Matches a superseded root's NewParent cell."""
    return _exact_column_filter(attributes.Hierarchy.NewParent)


def _refused_or_superseded(predicate: RowFilter) -> ConditionalRowFilter:
    """Matches when `predicate` does, and otherwise when the root has a NewParent."""
    return ConditionalRowFilter(
        predicate_filter=predicate,
        true_filter=PassAllFilter(True),
        false_filter=_new_parent_filter(),
    )


def get_row_key_lock_filter(lock_column, lock_expiry) -> RowFilterChain:
    """Acquire-side filter for a temporal CAS lock on an arbitrary row.

    Matches any unexpired cell in the lock column: a match means the lock is held, so
    `check_and_mutate_row` must not apply the acquire mutation.
    """
    return RowFilterChain([_unexpired(lock_expiry), _exact_column_filter(lock_column)])


def get_root_lock_filter(
    lock_column, lock_expiry, indefinite_lock_column
) -> ConditionalRowFilter:
    # Hierarchy-aware: also fails if an indefinite lock exists, and
    # returns the NewParent column when no lock is held — so a superseded
    # root (has NewParent) cannot be acquired even though its lock cell
    # is free.
    return _refused_or_superseded(
        get_row_key_lock_filter_with_indefinite(
            lock_column, lock_expiry, indefinite_lock_column
        )
    )


def get_row_key_lock_filter_with_indefinite(
    lock_column, lock_expiry, indefinite_lock_column
) -> RowFilterUnion:
    """Acquire-side filter for a temporal row-key lock extended to fail
    if an indefinite lock cell exists on the same row.

    `get_root_lock_filter` adds the NewParent hierarchy check on top (row-key
    locks have no lineage). Used for locks that need the same crash-safety
    root locks have: a worker that dies holding the indefinite column leaves
    that cell set, and a subsequent temporal acquire must see it and refuse
    rather than racing into partial state.
    """
    return RowFilterUnion(
        [
            _exact_column_filter(indefinite_lock_column),
            get_row_key_lock_filter(lock_column, lock_expiry),
        ]
    )


def get_indefinite_root_lock_filter(lock_column) -> ConditionalRowFilter:
    return _refused_or_superseded(get_has_column_filter(lock_column))


def get_has_column_filter(column: attributes._Attribute) -> ColumnRangeFilter:
    """Matches a row holding `column`: a held indefinite lock, or a seeded counter."""
    return _exact_column_filter(column)


def get_held_lock_filter(
    lock_column: attributes._Attribute, operation_id: np.uint64
) -> RowFilterChain:
    """'Is this lock still held by operation_id?' for any row.

    A match means the newest cell holds `operation_id`: a renew sets it again, and
    an indefinite unlock deletes it.
    """
    return RowFilterChain(_held_by(lock_column, operation_id))


def get_renew_lock_filter(
    lock_column: attributes._Attribute, operation_id: np.uint64
) -> ConditionalRowFilter:
    """Yields cells only while `operation_id` holds the root's newest lock.

    A root with a NewParent yields none. Renewals set the lock on a match, so a row
    nobody locked, such as a fork's inherited root, never renews.
    """
    return ConditionalRowFilter(
        predicate_filter=_new_parent_filter(),
        true_filter=BlockAllFilter(True),
        false_filter=get_held_lock_filter(lock_column, operation_id),
    )


def get_unexpired_held_lock_filter(
    lock_column, lock_expiry, operation_id
) -> RowFilterChain:
    """Matches `lock_column`'s unexpired cell holding `operation_id`, on any row."""
    return RowFilterChain(
        [_unexpired(lock_expiry), *_held_by(lock_column, operation_id)]
    )
