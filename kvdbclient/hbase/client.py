# pylint: disable=invalid-name, missing-docstring, line-too-long, too-many-arguments

import logging
import typing
import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from . import HBaseConfig
from . import get_client_info
from . import utils as hbase_utils
from ..base import (
    ClientWithIDGen,
    ColumnFamilyConfig,
    LegacyRow,
    LegacyRows,
    OperationLogger,
    column_view,
    deserialize_cells,
    log_lock_holders,
)
from .. import attributes
from .. import exceptions
from ..serializers import serialize_uint64
from ..utils import get_compatible_time_range
from ..utils import get_valid_timestamp


class HBaseMutation:
    """Pending row mutation for HBase, parallel to bigtable.row.DirectRow."""

    __slots__ = ("row_key", "cells")

    def __init__(self, row_key: bytes):
        self.row_key = row_key
        self.cells = {}  # {str col_spec: (bytes value, datetime|None timestamp)}

    def set_cell(self, column: attributes._Attribute, value: bytes, timestamp=None):
        self.cells[hbase_utils.build_column_spec(column)] = (value, timestamp)


def _schema_entry(column_family: ColumnFamilyConfig) -> dict[str, str]:
    """One column family's REST schema entry, with its versions and TTL when set."""
    entry = {"name": column_family.family_id}
    if column_family.max_versions is not None:
        entry["VERSIONS"] = str(column_family.max_versions)
    if column_family.max_age is not None:
        entry["TTL"] = str(int(column_family.max_age.total_seconds()))
    return entry


class Client(ClientWithIDGen, OperationLogger):
    backend_name = "hbase"
    config_class = HBaseConfig

    @staticmethod
    def default_client_info():
        return get_client_info()

    def __init__(
        self,
        table_id: str,
        config: HBaseConfig = HBaseConfig(),
        table_meta=None,
        lock_expiry: timedelta = timedelta(minutes=3),
    ):
        self._base_url = config.BASE_URL.rstrip("/")
        self._table_id = table_id
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=32)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        self._init_common(
            logger_name=f"hbase/{table_id}",
            table_meta=table_meta,
            lock_expiry=lock_expiry,
            max_row_key_count=config.MAX_ROW_KEY_COUNT,
        )

    def close(self):
        self._session.close()

    def _table_url(self, suffix: str = "") -> str:
        return f"{self._base_url}/{self._table_id}{suffix}"

    def _row_url(self, row_key: bytes, *parts) -> str:
        """The REST URL of `row_key`, or of the column and time `parts` under it."""
        path = "/".join((hbase_utils.encode_value(row_key), *map(str, parts)))
        return self._table_url(f"/{path}")

    def _delete(self, row_key: bytes, *parts) -> None:
        """DELETE `row_key`, or the column or cell that `parts` name under it."""
        self._session.delete(self._row_url(row_key, *parts))

    def _put_schema(self, column_schemas: list[dict[str, str]]) -> None:
        schema = {"name": self._table_id, "ColumnSchema": column_schemas}
        resp = self._session.put(self._table_url("/schema"), json=schema)
        resp.raise_for_status()

    def _table_exists(self) -> bool:
        return self._session.get(self._table_url("/schema")).status_code == 200

    def _create_storage(self, column_families) -> None:
        self._put_schema([_schema_entry(family) for family in column_families])

    def create_column_family(self, family_id, gc_rule=None):
        if isinstance(family_id, ColumnFamilyConfig):
            column_family = family_id
        else:
            column_family = ColumnFamilyConfig(
                family_id,
                max_versions=getattr(gc_rule, "max_num_versions", None),
                max_age=getattr(gc_rule, "max_age", None),
            )
        self._put_schema([_schema_entry(column_family)])

    def mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> HBaseMutation:
        mutation = HBaseMutation(row_key)
        for column, value in val_dict.items():
            mutation.set_cell(column, column.serialize(value), timestamp=time_stamp)
        return mutation

    def _write_rows(self, rows, slow_retry=True, block_size=2000):
        rows_list = list(rows)
        for i in range(0, len(rows_list), block_size):
            block = rows_list[i : i + block_size]
            body = hbase_utils.build_cellset_json(block)
            self._put_rows_with_retry(body, slow_retry=slow_retry)

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=15),
        reraise=True,
    )
    def _put_rows_with_retry(self, body: dict, slow_retry: bool = True):
        resp = self._session.put(
            self._table_url("/false-row-key"),
            json=body,
            timeout=60,
        )
        resp.raise_for_status()

    def _put_if_absent(self, row_key: bytes, column, value) -> bool:
        return self._check_and_put(
            row_key,
            column,
            check_value=None,
            put_value=column.serialize(value),
            put_timestamp=get_valid_timestamp(None),
        )

    def _acquire_lock_cell(
        self, row_key: bytes, operation_id: np.uint64, lock_column
    ) -> bool:
        """Set the given lock cell iff it's absent. Returns True if acquired.

        On failure due to a stale (expired) cell, deletes it and retries —
        HBase REST check-and-put has no server-side TTL, so expiry is
        enforced client-side against `self._lock_expiry`.
        """
        if self._put_if_absent(row_key, lock_column, operation_id):
            return True
        cells = self._read_byte_row(row_key, columns=lock_column)
        if cells:
            lock_ts = cells[0].timestamp
            if lock_ts and datetime.now(timezone.utc) - lock_ts > self._lock_expiry:
                self._delete_column(row_key, lock_column)
                return self._acquire_lock_cell(row_key, operation_id, lock_column)
            if self.logger.isEnabledFor(logging.DEBUG):
                log_lock_holders(self.logger, "Locked", cells)
        return False

    def _release_lock_cell(
        self, row_key: bytes, operation_id: np.uint64, lock_column
    ) -> bool:
        """Deletes `lock_column`'s cell if it holds `operation_id`."""
        col_spec = hbase_utils.build_column_spec(lock_column)
        cell = hbase_utils.build_cell(col_spec, lock_column.serialize(operation_id))
        return self._check(row_key, col_spec, "delete", [cell])

    def _lock_root(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        return self._lock_by_row_key_with_indefinite(
            serialize_uint64(root_id), operation_id
        )

    def _lock_root_indefinitely(
        self, root_id: np.uint64, operation_id: np.uint64
    ) -> bool:
        row_key = serialize_uint64(root_id)
        if self._lock_by_row_key_indefinitely(row_key, operation_id):
            return True
        if self.logger.isEnabledFor(logging.DEBUG):
            lock_column = attributes.Concurrency.IndefiniteLock
            cells = self._read_byte_row(row_key, columns=lock_column)
            if cells:
                log_lock_holders(self.logger, "Indefinitely locked", cells)
        return False

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        return self.renew_lock_by_row_key(serialize_uint64(root_id), operation_id)

    def lock_by_row_key(self, row_key: bytes, operation_id: np.uint64) -> bool:
        return self._acquire_lock_cell(
            row_key, operation_id, attributes.Concurrency.Lock
        )

    def _lock_by_row_key_with_indefinite(
        self, row_key: bytes, operation_id: np.uint64
    ) -> bool:
        """Temporal lock on a row key that also refuses if indefinite is held.

        HBase REST check-and-put checks one column, so the temporal lock is
        taken first and rolled back if the indefinite column holds a cell.
        """
        lock_column = attributes.Concurrency.Lock
        if not self._acquire_lock_cell(row_key, operation_id, lock_column):
            return False
        indefinite_lock_column = attributes.Concurrency.IndefiniteLock
        if self._read_byte_row(row_key, columns=indefinite_lock_column):
            self._delete_column(row_key, lock_column)
            return False
        return True

    def unlock_by_row_key(self, row_key: bytes, operation_id: np.uint64):
        return self._release_lock_cell(
            row_key, operation_id, attributes.Concurrency.Lock
        )

    def unlock_indefinitely_locked_by_row_key(
        self, row_key: bytes, operation_id: np.uint64
    ):
        """Release the indefinite lock on any row, roots too, holding `operation_id`."""
        return self._release_lock_cell(
            row_key, operation_id, attributes.Concurrency.IndefiniteLock
        )

    def renew_lock_by_row_key(self, row_key: bytes, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        held = lock_column.serialize(operation_id)
        return self._check_and_put(
            row_key, lock_column, check_value=held, put_value=held
        )

    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        col_spec = hbase_utils.build_column_spec(attributes.Concurrency.Counter)
        increment = hbase_utils.build_cell(col_spec, struct.pack(">q", size))
        resp = self._row_request(
            self._session.post,
            key,
            col_spec,
            [increment],
            headers={"Accept": "application/octet-stream"},
        )
        resp.raise_for_status()
        high = struct.unpack(">q", resp.content)[0]
        return np.uint64(high + 1 - size), np.uint64(high)

    def _scan(self, spec: dict[str, object], on_page) -> None:
        """Feeds each page of a scanner opened with `spec` to `on_page`.

        The scanner is deleted however the scan ends.
        """
        resp = self._session.put(self._table_url("/scanner"), json=spec)
        resp.raise_for_status()
        location = resp.headers.get("Location", "")
        try:
            while True:
                resp = self._session.get(
                    location, headers={"Accept": "application/json"}
                )
                if resp.status_code == 204:
                    break
                resp.raise_for_status()
                on_page(resp.json())
        finally:
            self._session.delete(location)

    def read_all_rows(self):
        rows = []
        self._scan({"batch": 1000}, lambda page: rows.extend(page.get("Row", [])))
        return LegacyRows(
            {
                hbase_utils.decode_value(row["key"]): LegacyRow(
                    hbase_utils.parse_cells(row.get("Cell", [])), equal_by=row
                )
                for row in rows
            }
        )

    def _read_byte_rows(
        self,
        start_key=None,
        end_key=None,
        end_key_inclusive=False,
        row_keys=None,
        columns=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
        user_id=None,
    ):
        single_column = isinstance(columns, attributes._Attribute)
        col_specs = hbase_utils.build_column_specs(columns)

        if row_keys is not None:
            row_keys = list(row_keys)
            rows = {}
            for i in range(0, len(row_keys), self._max_row_key_count):
                batch = row_keys[i : i + self._max_row_key_count]
                rows.update(
                    self._fetch_rows_by_keys(
                        batch,
                        col_specs,
                        start_time,
                        end_time,
                        end_time_inclusive,
                        single_column,
                    )
                )
        elif start_key is not None and end_key is not None:
            rows = self._fetch_rows_by_range(
                start_key,
                end_key,
                end_key_inclusive,
                col_specs,
                start_time,
                end_time,
                end_time_inclusive,
                single_column,
            )
        else:
            raise exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or both, a start row and an end row."
            )

        for column_dict in rows.values():
            deserialize_cells({columns: column_dict} if single_column else column_dict)
        return rows

    def _read_byte_row(
        self,
        row_key,
        columns=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        row = self._read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return row.get(row_key, column_view({}, columns))

    def _fetch_rows_by_keys(
        self,
        row_keys,
        col_specs,
        start_time,
        end_time,
        end_time_inclusive,
        single_column=False,
    ):
        if not row_keys:
            return {}
        sorted_keys = sorted(row_keys)
        all_rows = self._fetch_rows_by_range(
            start_key=sorted_keys[0],
            end_key=sorted_keys[-1],
            end_key_inclusive=True,
            col_specs=col_specs,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
            single_column=single_column,
        )
        key_set = set(row_keys)
        return {k: v for k, v in all_rows.items() if k in key_set}

    def _fetch_rows_by_range(
        self,
        start_key,
        end_key,
        end_key_inclusive,
        col_specs,
        start_time,
        end_time,
        end_time_inclusive,
        single_column=False,
    ):
        scanner_spec = {
            "startRow": hbase_utils.encode_value(start_key),
            "batch": self._max_row_key_count,
        }

        if end_key_inclusive:
            scanner_spec["endRow"] = hbase_utils.encode_value(end_key + b"\x00")
        else:
            scanner_spec["endRow"] = hbase_utils.encode_value(end_key)

        if col_specs:
            scanner_spec["column"] = col_specs

        bounds = get_compatible_time_range(start_time, end_time, end_time_inclusive)
        if bounds.start is not None:
            scanner_spec["startTime"] = hbase_utils.datetime_to_hbase_ts(bounds.start)
        if bounds.end is not None:
            scanner_spec["endTime"] = hbase_utils.datetime_to_hbase_ts(bounds.end)

        result = {}
        self._scan(
            scanner_spec,
            lambda page: result.update(
                hbase_utils.parse_cell_response(page, single_column=single_column)
            ),
        )
        return result

    def _delete_meta(self):
        self._delete(attributes.TableMeta.key)

    def delete_cells(self, mutations, row_keys_to_delete=None):
        for row_key, column, timestamps in mutations:
            col_spec = hbase_utils.build_column_spec(column)
            for ts in timestamps:
                self._delete(row_key, col_spec, hbase_utils.datetime_to_hbase_ts(ts))
        for key in row_keys_to_delete or []:
            self._delete(key)

    def delete_row(self, row_key):
        self._delete(row_key)

    def _check_and_put(
        self,
        row_key: bytes,
        column: attributes._Attribute,
        check_value: bytes | None,
        put_value: bytes,
        put_timestamp: datetime | None = None,
    ) -> bool:
        """Puts `put_value` in `column` if its cell holds `check_value`.

        A `check_value` of None requires the cell to be absent.
        """
        col_spec = hbase_utils.build_column_spec(column)
        cells = [hbase_utils.build_cell(col_spec, put_value, put_timestamp)]
        if check_value is not None:
            cells.insert(0, hbase_utils.build_cell(col_spec, check_value))
        return self._check(row_key, col_spec, "put", cells)

    def _check(self, row_key: bytes, col_spec: str, action: str, cells) -> bool:
        """Check-and-`action` of `cells` on `row_key`; True when HBase applied it."""
        resp = self._row_request(
            self._session.put, row_key, col_spec, cells, f"?check={action}"
        )
        return resp.status_code == 200

    def _row_request(self, send, row_key: bytes, col_spec: str, cells, query="", **kw):
        """Sends `cells` as one row's CellSet to `col_spec` of `row_key` by `send`."""
        body = hbase_utils.build_cellset([hbase_utils.build_row(row_key, cells)])
        return send(self._row_url(row_key, col_spec) + query, json=body, **kw)

    def _delete_column(self, row_key: bytes, column: attributes._Attribute) -> None:
        self._delete(row_key, hbase_utils.build_column_spec(column))
