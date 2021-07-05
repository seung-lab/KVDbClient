from typing import Dict
from typing import Union
from typing import Iterable
from typing import Optional
from datetime import datetime
from datetime import timedelta

import numpy as np
from google.cloud.bigtable.row_data import PartialRowData
from google.cloud.bigtable.row_filters import RowFilter

from . import attributes


def _from_key(family_id: str, key: bytes):
    try:
        return attributes.Attribute._attributes[(family_id, key)]
    except KeyError:
        # FIXME: Look if the key matches a columnarray pattern and
        #        remove loop initialization in _AttributeArray.__init__()
        raise KeyError(f"Unknown key {family_id}:{key.decode()}")


def partial_row_data_to_column_dict(
    partial_row_data: PartialRowData,
) -> Dict[attributes.Attribute, PartialRowData]:
    new_column_dict = {}
    for family_id, column_dict in partial_row_data._cells.items():
        for column_key, column_values in column_dict.items():
            column = _from_key(family_id, column_key)
            new_column_dict[column] = column_values
    return new_column_dict


def _get_google_compatible_time_stamp(
    time_stamp: datetime, round_up: bool = False
) -> datetime:
    """
    Makes a datetime time stamp compatible with googles' services.
    Google restricts the accuracy of time stamps to milliseconds. Hence, the
    microseconds are cut of. By default, time stamps are rounded to the lower
    number.
    """
    micro_s_gap = timedelta(microseconds=time_stamp.microsecond % 1000)
    if micro_s_gap == 0:
        return time_stamp
    if round_up:
        time_stamp += timedelta(microseconds=1000) - micro_s_gap
    else:
        time_stamp -= micro_s_gap
    return time_stamp


def _get_column_filter(
    columns: Union[Iterable[attributes.Attribute], attributes.Attribute] = None
) -> RowFilter:
    """Generates a RowFilter that accepts the specified columns."""
    from google.cloud.bigtable.row_filters import RowFilterUnion
    from google.cloud.bigtable.row_filters import ColumnRangeFilter

    if isinstance(columns, attributes.Attribute):
        return ColumnRangeFilter(
            columns.family_id, start_column=columns.key, end_column=columns.key
        )
    elif len(columns) == 1:
        return ColumnRangeFilter(
            columns[0].family_id, start_column=columns[0].key, end_column=columns[0].key
        )
    return RowFilterUnion(
        [
            ColumnRangeFilter(col.family_id, start_column=col.key, end_column=col.key)
            for col in columns
        ]
    )


def _get_time_range_filter(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = True,
) -> RowFilter:
    """Generates a TimeStampRangeFilter which is inclusive for start and (optionally) end."""
    from google.cloud.bigtable.row_filters import TimestampRange
    from google.cloud.bigtable.row_filters import TimestampRangeFilter

    # Comply to resolution of BigTables TimeRange
    if start_time is not None:
        start_time = _get_google_compatible_time_stamp(start_time, round_up=False)
    if end_time is not None:
        end_time = _get_google_compatible_time_stamp(end_time, round_up=end_inclusive)
    return TimestampRangeFilter(TimestampRange(start=start_time, end=end_time))


def get_time_range_and_column_filter(
    columns: Optional[
        Union[Iterable[attributes.Attribute], attributes.Attribute]
    ] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    end_inclusive: bool = False,
) -> RowFilter:
    from google.cloud.bigtable.row_filters import RowFilterChain

    time_filter = _get_time_range_filter(
        start_time=start_time, end_time=end_time, end_inclusive=end_inclusive
    )

    if columns is not None:
        column_filter = _get_column_filter(columns)
        return RowFilterChain([column_filter, time_filter])
    else:
        return time_filter
