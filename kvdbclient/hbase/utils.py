"""HBase REST API utility functions."""

import base64
from datetime import datetime, timezone
from typing import NamedTuple

from .. import attributes
from ..base import Cell


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


class RestCell(NamedTuple):
    """One decoded REST cell: its family, its qualifier and the Cell it holds."""

    family: str
    qualifier: bytes
    cell: Cell


def parse_cells(cells: list[dict[str, object]]) -> list[RestCell]:
    """Each REST cell's base64 column, value and millisecond time, decoded."""
    parsed = []
    for one in cells:
        family, qualifier = decode_value(one["column"]).decode("utf-8").split(":", 1)
        timestamp = hbase_ts_to_datetime(one.get("timestamp"))
        cell = Cell(value=decode_value(one["$"]), timestamp=timestamp)
        parsed.append(RestCell(family, qualifier.encode("utf-8"), cell))
    return parsed


def parse_cell_response(row_data: dict, single_column=False):
    """Convert HBase REST JSON row response to {_Attribute: [Cell]} dict.

    HBase REST returns rows in this JSON format:
    {
      "Row": [
        {
          "key": "<base64>",
          "Cell": [
            {"column": "<b64 family:qualifier>", "$": "<b64 value>", "timestamp": 123}
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
        for parsed in parse_cells(row.get("Cell", [])):
            try:
                attr = attributes.from_key(parsed.family, parsed.qualifier)
            except KeyError:
                continue
            column_dict.setdefault(attr, []).append(parsed.cell)

        if single_column and column_dict:
            # Return just the cell list for the single column
            result[row_key] = next(iter(column_dict.values()))
        else:
            result[row_key] = column_dict
    return result


def build_cell(
    col_spec: str | bytes, value: bytes, timestamp: datetime | None = None
) -> dict[str, object]:
    """One CellSet cell: column and value in base64, its time in ms when given."""
    column = col_spec.encode("utf-8") if isinstance(col_spec, str) else col_spec
    cell = {"column": encode_value(column), "$": encode_value(value)}
    if timestamp is not None:
        cell["timestamp"] = datetime_to_hbase_ts(timestamp)
    return cell


def build_row(row_key: bytes, cells: list[dict[str, object]]) -> dict[str, object]:
    """One CellSet row: its key in base64, and its cells."""
    return {"key": encode_value(row_key), "Cell": cells}


def build_cellset(rows: list[dict[str, object]]) -> dict[str, object]:
    """The CellSet body every HBase REST write, increment and check sends.

    {"Row": [{"key": b64, "Cell": [{"column": b64, "$": b64, "timestamp": ms}]}]}
    """
    return {"Row": rows}


def build_cellset_json(mutations) -> dict[str, object]:
    """The CellSet body writing each HBaseMutation in `mutations`, for PUT."""
    return build_cellset(
        [
            build_row(
                mutation.row_key,
                [
                    build_cell(col_spec, value, timestamp)
                    for col_spec, (value, timestamp) in mutation.cells.items()
                ],
            )
            for mutation in mutations
        ]
    )
