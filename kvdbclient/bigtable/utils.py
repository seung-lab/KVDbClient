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
from google.cloud.bigtable.data.row_filters import ColumnRangeFilter
from google.cloud.bigtable.data.row_filters import TimestampRangeFilter
from google.cloud.bigtable.data.row_filters import ConditionalRowFilter
from google.cloud.bigtable.data.row_filters import ColumnQualifierRegexFilter

from .. import attributes
from ..base import Cell as BaseCell
from ..utils import get_compatible_time_stamp as get_google_compatible_time_stamp


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
        column_dict.setdefault(attr, []).append(BaseCell(value=cell.value, timestamp=ts))
    return column_dict


def _get_column_filter(
    columns: Union[Iterable[attributes._Attribute], attributes._Attribute] = None
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
    """Generates a TimeStampRangeFilter which is inclusive for start and (optionally) end."""
    # Comply to resolution of BigTables TimeRange
    if start_time is not None:
        start_time = get_google_compatible_time_stamp(start_time, round_up=False)
    if end_time is not None:
        end_time = get_google_compatible_time_stamp(end_time, round_up=end_inclusive)
    return TimestampRangeFilter(start=start_time, end=end_time)


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


def get_row_key_lock_filter(lock_column, lock_expiry) -> RowFilterChain:
    """Acquire-side filter for a temporal CAS lock on an arbitrary row.

    Matches any non-expired cell in the lock column. A match means the
    lock is held, so `check_and_mutate_row` should NOT apply the acquire
    mutation; an empty result means the row is free.

    Used directly by `lock_by_row_key` and composed into
    `get_root_lock_filter` (root locks layer extra hierarchy checks on
    top of this same temporal match).
    """
    time_cutoff = datetime.now(timezone.utc) - lock_expiry
    # Comply to resolution of BigTables TimeRange
    time_cutoff -= timedelta(microseconds=time_cutoff.microsecond % 1000)
    return RowFilterChain([
        TimestampRangeFilter(start=time_cutoff),
        _exact_column_filter(lock_column),
    ])


def get_root_lock_filter(
    lock_column, lock_expiry, indefinite_lock_column
) -> ConditionalRowFilter:
    # Hierarchy-aware: also fails if an indefinite lock exists, and
    # returns the NewParent column when no lock is held — so a superseded
    # root (has NewParent) cannot be acquired even though its lock cell
    # is free.
    return ConditionalRowFilter(
        predicate_filter=RowFilterUnion([
            _exact_column_filter(indefinite_lock_column),
            get_row_key_lock_filter(lock_column, lock_expiry),
        ]),
        true_filter=PassAllFilter(True),
        false_filter=_exact_column_filter(attributes.Hierarchy.NewParent),
    )


def get_row_key_lock_filter_with_indefinite(
    lock_column, lock_expiry, indefinite_lock_column
) -> RowFilterUnion:
    """Acquire-side filter for a temporal row-key lock extended to fail
    if an indefinite lock cell exists on the same row.

    Counterpart to `get_root_lock_filter` but without the NewParent
    hierarchy check (row-key locks have no lineage). Used for locks that
    need the same crash-safety root locks have: a worker that dies
    holding the indefinite column leaves that cell set, and a subsequent
    temporal acquire must see it and refuse rather than racing into
    partial state.
    """
    return RowFilterUnion([
        _exact_column_filter(indefinite_lock_column),
        get_row_key_lock_filter(lock_column, lock_expiry),
    ])


def get_indefinite_root_lock_filter(lock_column) -> ConditionalRowFilter:
    return ConditionalRowFilter(
        predicate_filter=_exact_column_filter(lock_column),
        true_filter=PassAllFilter(True),
        false_filter=_exact_column_filter(attributes.Hierarchy.NewParent),
    )


def get_indefinite_row_key_lock_filter(lock_column) -> ColumnRangeFilter:
    """Acquire-side filter for an indefinite row-key lock.

    Row-key variant of `get_indefinite_root_lock_filter` without the
    NewParent hierarchy check. A match (indefinite cell exists) means
    the lock is held → acquire refuses.
    """
    return _exact_column_filter(lock_column)


def get_row_key_renew_lock_filter(
    lock_column: attributes._Attribute, operation_id: np.uint64
) -> RowFilterChain:
    """'Is this lock still held by operation_id?' for generic row keys.

    A match means our lock cell still exists → caller should renew (run
    SetCell in `true_case_mutations`). No NewParent handling — arbitrary
    rows have no hierarchy relationship.
    """
    return RowFilterChain([
        _exact_column_filter(lock_column),
        _exact_value_filter(lock_column.serialize(operation_id)),
    ])


def get_renew_lock_filter(
    lock_column: attributes._Attribute, operation_id: np.uint64
) -> ConditionalRowFilter:
    # Root renew is inverted vs the row-key version: renew only when
    # (my lock exists) AND (no NewParent). Layers a NewParent guard on
    # top of the generic "is my lock still here" check, and flips the
    # expected case so `check_and_mutate_row` runs SetCell as
    # `false_case_mutations`.
    return ConditionalRowFilter(
        predicate_filter=get_row_key_renew_lock_filter(lock_column, operation_id),
        true_filter=_exact_column_filter(attributes.Hierarchy.NewParent),
        false_filter=PassAllFilter(True),
    )


def get_unlock_root_filter(lock_column, lock_expiry, operation_id) -> RowFilterChain:
    time_cutoff = datetime.now(timezone.utc) - lock_expiry
    # Comply to resolution of BigTables TimeRange
    time_cutoff -= timedelta(microseconds=time_cutoff.microsecond % 1000)
    time_filter = TimestampRangeFilter(start=time_cutoff)

    return RowFilterChain([
        time_filter,
        _exact_column_filter(lock_column),
        _exact_value_filter(lock_column.serialize(operation_id)),
    ])


def get_indefinite_unlock_root_filter(lock_column, operation_id) -> RowFilterChain:
    return RowFilterChain([
        _exact_column_filter(lock_column),
        _exact_value_filter(lock_column.serialize(operation_id)),
    ])
