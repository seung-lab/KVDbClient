# pylint: disable=invalid-name, missing-docstring, line-too-long, too-many-arguments

import base64
import logging
import typing
import struct
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from . import HBaseConfig
from . import utils as hbase_utils
from ..base import Cell, ClientWithIDGen, ColumnFamilyConfig, DEFAULT_COLUMN_FAMILIES, OperationLogger
from .. import attributes
from .. import exceptions
from ..serializers import serialize_uint64
from ..utils import get_valid_timestamp


class _PartialRowAdapter:
    """Adapts an HBase REST JSON row to mimic bigtable's ``PartialRowData``.

    Provides ``.cells`` as ``{family_str: {qualifier_bytes: [Cell, ...]}}``.
    """

    __slots__ = ("cells", "_raw")

    def __init__(self, raw_row):
        families = defaultdict(lambda: defaultdict(list))
        for cell in raw_row.get("Cell", []):
            col_decoded = base64.b64decode(cell["column"]).decode("utf-8")
            fam, qual = col_decoded.split(":", 1)
            val = base64.b64decode(cell["$"])
            ts = hbase_utils.hbase_ts_to_datetime(cell.get("timestamp"))
            families[fam][qual.encode("utf-8")].append(Cell(value=val, timestamp=ts))
        self.cells = dict(families)
        self._raw = raw_row

    def __eq__(self, other):
        if not isinstance(other, _PartialRowAdapter):
            return NotImplemented
        return self._raw == other._raw

    def __hash__(self):
        return id(self)


class _ConsumableRows:
    """Thin wrapper so ``read_all_rows()`` is compatible with bigtable's
    ``PartialRowsData`` interface (``res.consume_all()``,
    ``key in res.rows``, ``res.rows[key].cells[family][qualifier]``)."""

    __slots__ = ("rows",)

    def __init__(self, raw_rows):
        self.rows = {}
        for row in raw_rows:
            rk = base64.b64decode(row["key"])
            self.rows[rk] = _PartialRowAdapter(row)

    def consume_all(self):
        pass


class HBaseMutation:
    """Pending row mutation for HBase, parallel to bigtable.row.DirectRow."""

    __slots__ = ("row_key", "cells")

    def __init__(self, row_key: bytes):
        self.row_key = row_key
        self.cells = {}  # {str col_spec: (bytes value, datetime|None timestamp)}

    def set_cell(self, family_id: str, column: bytes, value: bytes, timestamp=None):
        col_spec = f"{family_id}:{column.decode()}"
        self.cells[col_spec] = (value, timestamp)


class Client(ClientWithIDGen, OperationLogger):
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
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        self._init_common(
            logger_name=f"hbase/{table_id}",
            table_meta=table_meta,
            lock_expiry=lock_expiry,
            max_row_key_count=config.MAX_ROW_KEY_COUNT,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path}"

    def _table_url(self, suffix: str = "") -> str:
        return self._url(f"{self._table_id}{suffix}")

    def _create_column_families(self, column_families=None):
        if column_families is None:
            column_families = DEFAULT_COLUMN_FAMILIES
        column_schemas = []
        for cf in column_families:
            cs = {"name": cf.family_id}
            if cf.max_versions is not None:
                cs["VERSIONS"] = str(cf.max_versions)
            if cf.max_age is not None:
                cs["TTL"] = str(int(cf.max_age.total_seconds()))
            column_schemas.append(cs)
        schema = {
            "name": self._table_id,
            "ColumnSchema": column_schemas,
        }
        resp = self._session.put(self._table_url("/schema"), json=schema)
        resp.raise_for_status()

    # ── Admin ────────────────────────────────────────────────────────────

    def create_table(self, meta, version: str, column_families=None) -> None:
        resp = self._session.get(self._table_url("/schema"))
        if resp.status_code == 200:
            raise ValueError(f"{self._table_id} already exists.")
        self._create_column_families(column_families)
        self.add_table_version(version)
        self.update_table_meta(meta)

    def create_column_family(self, family_id, gc_rule=None):
        if isinstance(family_id, ColumnFamilyConfig):
            cf = family_id
            schema_entry = {"name": cf.family_id}
            if cf.max_versions is not None:
                schema_entry["VERSIONS"] = str(cf.max_versions)
            if cf.max_age is not None:
                schema_entry["TTL"] = str(int(cf.max_age.total_seconds()))
            schema = {"name": self._table_id, "ColumnSchema": [schema_entry]}
        else:
            schema = {"name": self._table_id, "ColumnSchema": [{"name": family_id}]}
            if gc_rule is not None:
                cf_entry = schema["ColumnSchema"][0]
                if hasattr(gc_rule, "max_num_versions"):
                    cf_entry["VERSIONS"] = str(gc_rule.max_num_versions)
                if hasattr(gc_rule, "max_age"):
                    cf_entry["TTL"] = str(int(gc_rule.max_age.total_seconds()))
        resp = self._session.put(self._table_url("/schema"), json=schema)
        resp.raise_for_status()

    # ── Write ────────────────────────────────────────────────────────────

    def mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> HBaseMutation:
        mutation = HBaseMutation(row_key)
        for column, value in val_dict.items():
            mutation.set_cell(
                column.family_id,
                column.key,
                column.serialize(value),
                timestamp=time_stamp,
            )
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

    # ── Locking ──────────────────────────────────────────────────────────

    def lock_root(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        lock_expiry = self._lock_expiry
        row_key = serialize_uint64(root_id)

        lock_value = serialize_uint64(operation_id)
        timestamp = get_valid_timestamp(None)
        success = self._check_and_put(
            row_key=row_key,
            check_family=lock_column.family_id,
            check_qualifier=lock_column.key,
            check_value=None,
            put_family=lock_column.family_id,
            put_qualifier=lock_column.key,
            put_value=lock_value,
            put_timestamp=timestamp,
        )

        if success:
            # Verify no indefinite lock (mirrors BigTable's RowFilterUnion check).
            # HBase REST checkAndPut only supports single-column checks, so we
            # acquire then verify, rolling back if an indefinite lock exists.
            indefinite_lock_column = attributes.Concurrency.IndefiniteLock
            if self._read_byte_row(row_key, columns=indefinite_lock_column):
                self._delete_cell(row_key, lock_column.family_id, lock_column.key)
                return False
            return True

        cells = self._read_byte_row(row_key, columns=lock_column)
        if cells:
            lock_ts = cells[0].timestamp
            if lock_ts and datetime.now(timezone.utc) - lock_ts > lock_expiry:
                self._delete_cell(row_key, lock_column.family_id, lock_column.key)
                return self.lock_root(root_id, operation_id)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Locked operation ids: {[c.value for c in cells]}")
        return False

    def lock_root_indefinitely(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.IndefiniteLock
        row_key = serialize_uint64(root_id)

        lock_value = serialize_uint64(operation_id)
        timestamp = get_valid_timestamp(None)
        success = self._check_and_put(
            row_key=row_key,
            check_family=lock_column.family_id,
            check_qualifier=lock_column.key,
            check_value=None,
            put_family=lock_column.family_id,
            put_qualifier=lock_column.key,
            put_value=lock_value,
            put_timestamp=timestamp,
        )
        if not success:
            if self.logger.isEnabledFor(logging.DEBUG):
                cells = self._read_byte_row(row_key, columns=lock_column)
                if cells:
                    self.logger.debug(f"Indefinitely locked operation ids: {[c.value for c in cells]}")
            return False
        return True

    def unlock_root(self, root_id: np.uint64, operation_id: np.uint64):
        lock_column = attributes.Concurrency.Lock
        row_key = serialize_uint64(root_id)
        lock_value = serialize_uint64(operation_id)
        return self._check_and_delete(
            row_key=row_key,
            check_family=lock_column.family_id,
            check_qualifier=lock_column.key,
            check_value=lock_value,
        )

    def unlock_indefinitely_locked_root(self, root_id: np.uint64, operation_id: np.uint64):
        lock_column = attributes.Concurrency.IndefiniteLock
        row_key = serialize_uint64(root_id)
        lock_value = serialize_uint64(operation_id)
        return self._check_and_delete(
            row_key=row_key,
            check_family=lock_column.family_id,
            check_qualifier=lock_column.key,
            check_value=lock_value,
        )

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        row_key = serialize_uint64(root_id)
        lock_value = serialize_uint64(operation_id)

        return self._check_and_put(
            row_key=row_key,
            check_family=lock_column.family_id,
            check_qualifier=lock_column.key,
            check_value=lock_value,
            put_family=lock_column.family_id,
            put_qualifier=lock_column.key,
            put_value=lock_column.serialize(operation_id),
        )

    # ── Timestamp ────────────────────────────────────────────────────────

    def get_compatible_timestamp(self, time_stamp: datetime, round_up: bool = False) -> datetime:
        return hbase_utils.get_hbase_compatible_time_stamp(time_stamp, round_up=round_up)

    # ── IDs ──────────────────────────────────────────────────────────────

    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        column = attributes.Concurrency.Counter
        col_spec = f"{column.family_id}:{column.key.decode()}"
        key_b64 = hbase_utils.encode_value(key)

        url = self._table_url(f"/{key_b64}/{col_spec}")
        increment_body = {
            "Row": [{
                "key": hbase_utils.encode_value(key),
                "Cell": [{
                    "column": hbase_utils.encode_value(col_spec.encode("utf-8")),
                    "$": hbase_utils.encode_value(struct.pack(">q", size)),
                }],
            }],
        }
        resp = self._session.post(
            url,
            json=increment_body,
            headers={"Accept": "application/octet-stream"},
        )
        resp.raise_for_status()
        high = struct.unpack(">q", resp.content)[0]
        return np.uint64(high + 1 - size), np.uint64(high)

    # ── Read ─────────────────────────────────────────────────────────────

    def read_all_rows(self):
        scanner_url = self._table_url("/scanner")
        resp = self._session.put(scanner_url, json={"batch": 1000})
        resp.raise_for_status()
        scanner_location = resp.headers.get("Location", "")
        rows = []
        while True:
            resp = self._session.get(scanner_location, headers={"Accept": "application/json"})
            if resp.status_code == 204:
                break
            resp.raise_for_status()
            row_data = resp.json()
            for row in row_data.get("Row", []):
                rows.append(row)
        self._session.delete(scanner_location)
        return _ConsumableRows(rows)

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
                rows.update(self._fetch_rows_by_keys(batch, col_specs, start_time, end_time, end_time_inclusive, single_column))
        elif start_key is not None and end_key is not None:
            rows = self._fetch_rows_by_range(
                start_key, end_key, end_key_inclusive, col_specs, start_time, end_time, end_time_inclusive, single_column
            )
        else:
            raise exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or both, a start row and an end row."
            )

        # Deserialize cell values
        for row_key, column_dict in rows.items():
            if single_column:
                for cell in column_dict:
                    col = columns
                    cell.value = col.deserialize(cell.value)
            else:
                for column, cell_entries in column_dict.items():
                    for cell_entry in cell_entries:
                        cell_entry.value = column.deserialize(cell_entry.value)
        return rows

    def _read_byte_row(self, row_key, columns=None, start_time=None, end_time=None, end_time_inclusive=False):
        single_column = isinstance(columns, attributes._Attribute)
        row = self._read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return row.get(row_key, [] if single_column else {})

    def _fetch_rows_by_keys(self, row_keys, col_specs, start_time, end_time, end_time_inclusive, single_column=False):
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

    def _fetch_rows_by_range(self, start_key, end_key, end_key_inclusive, col_specs, start_time, end_time, end_time_inclusive, single_column=False):
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

        if start_time:
            ts = hbase_utils.datetime_to_hbase_ts(
                hbase_utils.get_hbase_compatible_time_stamp(start_time)
            )
            scanner_spec["startTime"] = ts
        if end_time:
            et = hbase_utils.get_hbase_compatible_time_stamp(end_time, round_up=end_time_inclusive)
            scanner_spec["endTime"] = hbase_utils.datetime_to_hbase_ts(et)

        scanner_url = self._table_url("/scanner")
        resp = self._session.put(scanner_url, json=scanner_spec)
        resp.raise_for_status()
        scanner_location = resp.headers.get("Location", "")

        result = {}
        try:
            while True:
                resp = self._session.get(scanner_location, headers={"Accept": "application/json"})
                if resp.status_code == 204:
                    break
                resp.raise_for_status()
                parsed = hbase_utils.parse_cell_response(resp.json(), single_column=single_column)
                result.update(parsed)
        finally:
            self._session.delete(scanner_location)
        return result

    # ── Delete ───────────────────────────────────────────────────────────

    def _delete_meta(self):
        key_b64 = hbase_utils.encode_value(attributes.TableMeta.key)
        self._session.delete(self._table_url(f"/{key_b64}"))

    def delete_cells(self, mutations, row_keys_to_delete=None):
        for row_key, column, timestamps in mutations:
            col_spec = f"{column.family_id}:{column.key.decode()}"
            for ts in timestamps:
                ts_ms = hbase_utils.datetime_to_hbase_ts(ts)
                key_b64 = hbase_utils.encode_value(row_key)
                url = self._table_url(f"/{key_b64}/{col_spec}/{ts_ms}")
                self._session.delete(url)
        for key in (row_keys_to_delete or []):
            key_b64 = hbase_utils.encode_value(key)
            self._session.delete(self._table_url(f"/{key_b64}"))

    def delete_row(self, row_key):
        key_b64 = hbase_utils.encode_value(row_key)
        self._session.delete(self._table_url(f"/{key_b64}"))

    # ── Private Lock Helpers ─────────────────────────────────────────────

    def _check_and_put(
        self,
        row_key: bytes,
        check_family: str,
        check_qualifier: bytes,
        check_value,
        put_family: str,
        put_qualifier: bytes,
        put_value: bytes,
        put_timestamp: datetime = None,
    ) -> bool:
        key_b64 = hbase_utils.encode_value(row_key)
        col_spec = f"{check_family}:{check_qualifier.decode()}"
        url = self._table_url(f"/{key_b64}/{col_spec}?check=put")

        put_col_spec = f"{put_family}:{put_qualifier.decode()}"
        cell = {
            "column": hbase_utils.encode_value(put_col_spec.encode("utf-8")),
            "$": hbase_utils.encode_value(put_value),
        }
        if put_timestamp:
            cell["timestamp"] = hbase_utils.datetime_to_hbase_ts(put_timestamp)

        body = {
            "Row": [{
                "key": hbase_utils.encode_value(row_key),
                "Cell": [cell],
            }],
        }

        if check_value is not None:
            body["Row"][0]["Cell"].insert(0, {
                "column": hbase_utils.encode_value(col_spec.encode("utf-8")),
                "$": hbase_utils.encode_value(
                    check_value if isinstance(check_value, bytes) else serialize_uint64(check_value)
                ),
            })

        resp = self._session.put(url, json=body)
        return resp.status_code == 200

    def _check_and_delete(
        self,
        row_key: bytes,
        check_family: str,
        check_qualifier: bytes,
        check_value: bytes,
    ) -> bool:
        key_b64 = hbase_utils.encode_value(row_key)
        col_spec = f"{check_family}:{check_qualifier.decode()}"
        url = self._table_url(f"/{key_b64}/{col_spec}?check=delete")

        body = {
            "Row": [{
                "key": hbase_utils.encode_value(row_key),
                "Cell": [{
                    "column": hbase_utils.encode_value(col_spec.encode("utf-8")),
                    "$": hbase_utils.encode_value(check_value),
                }],
            }],
        }
        resp = self._session.put(url, json=body)
        return resp.status_code == 200

    def _delete_cell(self, row_key: bytes, family_id: str, qualifier: bytes):
        key_b64 = hbase_utils.encode_value(row_key)
        col_spec = f"{family_id}:{qualifier.decode()}"
        self._session.delete(self._table_url(f"/{key_b64}/{col_spec}"))
