"""HBase REST API utility functions."""

import base64
from datetime import datetime, timedelta, timezone

from .. import attributes
from ..base import Cell


def get_hbase_compatible_time_stamp(time_stamp: datetime, round_up: bool = False) -> datetime:
    """Round datetime to millisecond precision.

    When ``round_up=True`` the result is always *strictly after* the input so
    that it can be used as an exclusive upper bound (HBase ``endTime`` /
    ``ts.to``) that includes the original millisecond.
    """
    micro_s_gap = timedelta(microseconds=time_stamp.microsecond % 1000)
    if micro_s_gap == timedelta(0):
        if round_up:
            return time_stamp + timedelta(milliseconds=1)
        return time_stamp
    if round_up:
        time_stamp += timedelta(microseconds=1000) - micro_s_gap
    else:
        time_stamp -= micro_s_gap
    return time_stamp


def datetime_to_hbase_ts(dt: datetime) -> int:
    """Convert datetime to HBase millisecond timestamp."""
    if dt is None:
        return None
    return int(dt.timestamp() * 1000)


def hbase_ts_to_datetime(ts_ms: int) -> datetime:
    """Convert HBase millisecond timestamp to timezone-aware datetime."""
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def encode_value(data: bytes) -> str:
    """Base64-encode bytes for HBase REST API."""
    return base64.b64encode(data).decode("ascii")


def decode_value(b64_str: str) -> bytes:
    """Base64-decode an HBase REST API value."""
    return base64.b64decode(b64_str)


def build_column_spec(column: attributes._Attribute) -> str:
    """Convert an _Attribute to an HBase column specifier 'family:qualifier'."""
    return f"{column.family_id}:{column.key.decode()}"


def build_column_specs(columns) -> list:
    """Convert _Attribute or list of _Attributes to HBase column specifiers."""
    if columns is None:
        return []
    if isinstance(columns, attributes._Attribute):
        return [build_column_spec(columns)]
    return [build_column_spec(c) for c in columns]


def parse_cell_response(row_data: dict, single_column=False):
    """Convert HBase REST JSON row response to {_Attribute: [Cell]} dict.

    HBase REST returns rows in this JSON format:
    {
      "Row": [
        {
          "key": "<base64>",
          "Cell": [
            {"column": "<base64 family:qualifier>", "$": "<base64 value>", "timestamp": 123}
          ]
        }
      ]
    }

    Returns a dict: {row_key_bytes: {_Attribute: [Cell, ...]}}
    If single_column is True, returns {row_key_bytes: [Cell, ...]} for the single column.
    """
    result = {}
    rows = row_data.get("Row", [])
    for row in rows:
        row_key = decode_value(row["key"])
        column_dict = {}
        for cell in row.get("Cell", []):
            col_b64 = cell["column"]
            col_decoded = decode_value(col_b64).decode("utf-8")
            family_id, col_key = col_decoded.split(":", 1)
            col_key_bytes = col_key.encode("utf-8")
            try:
                attr = attributes.from_key(family_id, col_key_bytes)
            except KeyError:
                continue
            value = decode_value(cell["$"])
            timestamp = hbase_ts_to_datetime(cell.get("timestamp"))
            c = Cell(value=value, timestamp=timestamp)
            column_dict.setdefault(attr, []).append(c)

        if single_column and column_dict:
            # Return just the cell list for the single column
            result[row_key] = next(iter(column_dict.values()))
        else:
            result[row_key] = column_dict
    return result


def build_cellset_json(mutations) -> dict:
    """Build a CellSet JSON body from a list of HBaseMutation objects for PUT.

    Returns JSON structure:
    {
      "Row": [
        {
          "key": "<base64>",
          "Cell": [
            {"column": "<base64>", "$": "<base64>", "timestamp": <ms>}
          ]
        }
      ]
    }
    """
    rows = []
    for mutation in mutations:
        cells = []
        for col_spec, (value, timestamp) in mutation.cells.items():
            cell = {
                "column": encode_value(col_spec.encode("utf-8") if isinstance(col_spec, str) else col_spec),
                "$": encode_value(value),
            }
            if timestamp is not None:
                cell["timestamp"] = datetime_to_hbase_ts(timestamp)
            cells.append(cell)
        rows.append({
            "key": encode_value(mutation.row_key),
            "Cell": cells,
        })
    return {"Row": rows}
