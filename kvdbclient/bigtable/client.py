# pylint: disable=invalid-name, missing-docstring, line-too-long, logging-fstring-interpolation, too-many-arguments

import typing
import logging
import weakref
from collections import defaultdict
from datetime import datetime
from datetime import timedelta

import numpy as np
from google.cloud import bigtable
from google.cloud.bigtable.data import BigtableDataClient
from google.cloud.bigtable.data import ReadRowsQuery
from google.cloud.bigtable.data import RowRange
from google.cloud.bigtable.data.mutations import SetCell
from google.cloud.bigtable.data.mutations import DeleteRangeFromColumn
from google.cloud.bigtable.data.mutations import DeleteAllFromRow
from google.cloud.bigtable.data.mutations import RowMutationEntry
from google.cloud.bigtable.data.read_modify_write_rules import IncrementRule
from google.cloud.bigtable.data.exceptions import MutationsExceptionGroup
from google.cloud.bigtable.column_family import GCRuleIntersection
from google.cloud.bigtable.column_family import MaxAgeGCRule
from google.cloud.bigtable.column_family import MaxVersionsGCRule
from google.cloud.bigtable.data.row_filters import RowFilter
from google.api_core.exceptions import DeadlineExceeded
from google.api_core.exceptions import ServiceUnavailable

from . import utils
from . import BigTableConfig
from ..base import Cell
from ..base import ClientWithIDGen
from ..base import ColumnFamilyConfig
from ..base import DEFAULT_COLUMN_FAMILIES
from ..base import OperationLogger
from .. import attributes
from .. import exceptions
from ..serializers import serialize_uint64
from ..utils import get_valid_timestamp


def _datetime_to_micros(dt: datetime) -> int:
    """Convert datetime to microseconds for bigtable mutations."""
    return int(dt.timestamp() * 1_000_000)


class _CompatibleRow:
    """Wraps a new-API Row to mimic legacy PartialRowData.

    Provides ``.cells`` as ``{family_str: {qualifier_bytes: [Cell, ...]}}``.
    """

    __slots__ = ("cells",)

    def __init__(self, row):
        families = defaultdict(lambda: defaultdict(list))
        for cell in row.cells:
            qual = cell.qualifier if isinstance(cell.qualifier, bytes) else cell.qualifier.encode()
            families[cell.family][qual].append(
                Cell(value=cell.value, timestamp=cell.timestamp_micros)
            )
        self.cells = dict(families)

    def __eq__(self, other):
        if not isinstance(other, _CompatibleRow):
            return NotImplemented
        return self.cells == other.cells

    def __hash__(self):
        return id(self)


class _ReadAllRowsResult:
    """Wrapper for read_all_rows result, compatible with legacy RowIterator API."""

    def __init__(self, rows_list):
        self.rows = {row.row_key: _CompatibleRow(row) for row in rows_list}

    def consume_all(self):
        pass


class Client(ClientWithIDGen, OperationLogger):
    def __init__(
        self,
        table_id: str,
        config: BigTableConfig = BigTableConfig(),
        table_meta=None,
        lock_expiry: timedelta = timedelta(minutes=3),
    ):
        # Store config for lazy admin client creation
        self._config = config
        self._table_id = table_id
        self.__admin_table = None

        # Data client (new — all read/write/lock operations)
        data_kwargs = dict(project=config.PROJECT)
        if config.CREDENTIALS:
            data_kwargs["credentials"] = config.CREDENTIALS
        self._data_client = BigtableDataClient(**data_kwargs)
        self._table = self._data_client.get_table(
            config.INSTANCE,
            table_id,
            # Read rows
            default_read_rows_operation_timeout=600.0,
            default_read_rows_attempt_timeout=20.0,
            # Mutate rows (batch)
            default_mutate_rows_retryable_errors=(
                DeadlineExceeded, ServiceUnavailable,
            ),
            default_mutate_rows_operation_timeout=lock_expiry.total_seconds(),
            default_mutate_rows_attempt_timeout=60.0,
            # Single-row ops (lock/unlock/increment)
            default_operation_timeout=60.0,
            default_attempt_timeout=20.0,
            default_retryable_errors=(DeadlineExceeded, ServiceUnavailable),
        )

        self._init_common(
            logger_name=f"{config.PROJECT}/{config.INSTANCE}/{table_id}",
            table_meta=table_meta,
            lock_expiry=lock_expiry,
            max_row_key_count=config.MAX_ROW_KEY_COUNT,
        )
        self._finalizer = weakref.finalize(
            self, Client._shutdown, self._data_client
        )

    @staticmethod
    def _shutdown(data_client):
        try:
            data_client.close()
        except Exception:
            pass

    # ── Admin ────────────────────────────────────────────────────────────

    @property
    def _admin_table(self):
        if self.__admin_table is None:
            admin_kwargs = dict(project=self._config.PROJECT, admin=self._config.ADMIN)
            if self._config.CREDENTIALS:
                admin_kwargs["credentials"] = self._config.CREDENTIALS
            admin_client = bigtable.Client(**admin_kwargs)
            instance = admin_client.instance(self._config.INSTANCE)
            self.__admin_table = instance.table(self._table_id)
        return self.__admin_table

    @staticmethod
    def _build_gc_rule(cf):
        rules = []
        if cf.max_versions is not None:
            rules.append(MaxVersionsGCRule(cf.max_versions))
        if cf.max_age is not None:
            rules.append(MaxAgeGCRule(cf.max_age))
        if not rules:
            return None
        if len(rules) == 1:
            return rules[0]
        return GCRuleIntersection(rules)

    def _create_column_families(self, column_families=None):
        if column_families is None:
            column_families = DEFAULT_COLUMN_FAMILIES
        for cf in column_families:
            gc_rule = self._build_gc_rule(cf)
            f = self._admin_table.column_family(cf.family_id, gc_rule=gc_rule)
            f.create()

    def create_table(self, meta, version: str, column_families=None) -> None:
        if self._admin_table.exists():
            raise ValueError(f"{self._admin_table.table_id} already exists.")
        self._admin_table.create()
        self._create_column_families(column_families)
        self.add_table_version(version)
        self.update_table_meta(meta)

    def create_column_family(self, family_id, gc_rule=None):
        if isinstance(family_id, ColumnFamilyConfig):
            gc_rule = self._build_gc_rule(family_id)
            family_id = family_id.family_id
        f = self._admin_table.column_family(family_id, gc_rule=gc_rule)
        f.create()

    # ── Write ────────────────────────────────────────────────────────────

    def mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> RowMutationEntry:
        ts_micros = _datetime_to_micros(time_stamp) if time_stamp else None
        mutations = []
        for column, value in val_dict.items():
            mutations.append(
                SetCell(
                    family=column.family_id,
                    qualifier=column.key,
                    new_value=column.serialize(value),
                    timestamp_micros=ts_micros,
                )
            )
        return RowMutationEntry(row_key, mutations)

    def _write_rows(self, rows, slow_retry=True, block_size=10000):
        rows_list = list(rows)
        if not rows_list:
            return

        timeout = self._lock_expiry.total_seconds() if slow_retry else 30.0
        try:
            if len(rows_list) == 1:
                entry = rows_list[0]
                self._table.mutate_row(
                    entry.row_key, list(entry.mutations), operation_timeout=timeout,
                )
            else:
                with self._table.mutations_batcher(
                    batch_operation_timeout=timeout,
                    batch_attempt_timeout=min(timeout, 20.0),
                ) as batcher:
                    for entry in rows_list:
                        batcher.append(entry)
        except MutationsExceptionGroup as exc:
            raise exceptions.KVDBClientError(
                f"Bulk write failed: {exc}"
            ) from exc

    # ── Locking ──────────────────────────────────────────────────────────

    def lock_root(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        indefinite_lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_root_lock_filter(
            lock_column, self._lock_expiry, indefinite_lock_column
        )
        lock_acquired = not self._table.check_and_mutate_row(
            serialize_uint64(root_id),
            predicate=filter_,
            false_case_mutations=SetCell(
                family=lock_column.family_id,
                qualifier=lock_column.key,
                new_value=serialize_uint64(operation_id),
                timestamp_micros=_datetime_to_micros(get_valid_timestamp(None)),
            ),
            operation_timeout=30.0,
        )
        if not lock_acquired and self.logger.isEnabledFor(logging.DEBUG):
            row = self._read_byte_row(serialize_uint64(root_id), columns=lock_column)
            l_operation_ids = [cell.value for cell in row]
            self.logger.debug(f"Locked operation ids: {l_operation_ids}")
        return lock_acquired

    def lock_root_indefinitely(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_indefinite_root_lock_filter(lock_column)
        lock_acquired = not self._table.check_and_mutate_row(
            serialize_uint64(root_id),
            predicate=filter_,
            false_case_mutations=SetCell(
                family=lock_column.family_id,
                qualifier=lock_column.key,
                new_value=serialize_uint64(operation_id),
                timestamp_micros=_datetime_to_micros(get_valid_timestamp(None)),
            ),
            operation_timeout=30.0,
        )
        if not lock_acquired and self.logger.isEnabledFor(logging.DEBUG):
            row = self._read_byte_row(serialize_uint64(root_id), columns=lock_column)
            l_operation_ids = [cell.value for cell in row]
            self.logger.debug(f"Indefinitely locked operation ids: {l_operation_ids}")
        return lock_acquired

    def unlock_root(self, root_id: np.uint64, operation_id: np.uint64):
        lock_column = attributes.Concurrency.Lock
        filter_ = utils.get_unlock_root_filter(
            lock_column, self._lock_expiry, operation_id
        )
        return self._table.check_and_mutate_row(
            serialize_uint64(root_id),
            predicate=filter_,
            true_case_mutations=DeleteRangeFromColumn(
                family=lock_column.family_id,
                qualifier=lock_column.key,
            ),
            operation_timeout=30.0,
        )

    def unlock_indefinitely_locked_root(self, root_id: np.uint64, operation_id: np.uint64):
        lock_column = attributes.Concurrency.IndefiniteLock
        return self._table.check_and_mutate_row(
            serialize_uint64(root_id),
            predicate=utils.get_indefinite_unlock_root_filter(lock_column, operation_id),
            true_case_mutations=DeleteRangeFromColumn(
                family=lock_column.family_id,
                qualifier=lock_column.key,
            ),
            operation_timeout=30.0,
        )

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        return not self._table.check_and_mutate_row(
            serialize_uint64(root_id),
            predicate=utils.get_renew_lock_filter(lock_column, operation_id),
            false_case_mutations=SetCell(
                family=lock_column.family_id,
                qualifier=lock_column.key,
                new_value=lock_column.serialize(operation_id),
                timestamp_micros=_datetime_to_micros(get_valid_timestamp(None)),
            ),
            operation_timeout=30.0,
        )

    # ── Timestamp ────────────────────────────────────────────────────────

    def get_compatible_timestamp(self, time_stamp: datetime, round_up: bool = False) -> datetime:
        return utils.get_google_compatible_time_stamp(time_stamp, round_up=round_up)

    # ── IDs ──────────────────────────────────────────────────────────────

    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        column = attributes.Concurrency.Counter
        result_row = self._table.read_modify_write_row(
            key,
            IncrementRule(column.family_id, column.key, increment_amount=size),
        )
        cells = result_row.get_cells(column.family_id, column.key)
        high = column.deserialize(cells[0].value)
        return high + np.uint64(1) - size, high

    # ── Read ─────────────────────────────────────────────────────────────

    @staticmethod
    def _deserialize_column_dict(column_dict):
        for column, cell_entries in column_dict.items():
            for i, cell_entry in enumerate(cell_entries):
                column_dict[column][i] = Cell(
                    value=column.deserialize(cell_entry.value),
                    timestamp=cell_entry.timestamp,
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
        row_ranges = None
        if row_keys is not None:
            pass
        elif start_key is not None and end_key is not None:
            row_ranges = [RowRange(
                start_key=start_key,
                end_key=end_key,
                start_is_inclusive=True,
                end_is_inclusive=end_key_inclusive,
            )]
        else:
            raise exceptions.PreconditionError(
                "Need to either provide a valid set of rows, or"
                " both, a start row and an end row."
            )

        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
            user_id=user_id,
        )

        rows = self._read(
            row_keys=row_keys,
            row_ranges=row_ranges,
            row_filter=filter_,
        )

        for row_key, column_dict in rows.items():
            self._deserialize_column_dict(column_dict)
            if isinstance(columns, attributes._Attribute):
                rows[row_key] = column_dict.get(columns, [])
        return rows

    def _read_byte_row(self, row_key, columns=None, start_time=None, end_time=None, end_time_inclusive=False):
        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
        )
        row = self._table.read_row(row_key, row_filter=filter_)
        if row is None:
            return [] if isinstance(columns, attributes._Attribute) else {}
        column_dict = utils.row_to_column_dict(row)
        self._deserialize_column_dict(column_dict)
        if isinstance(columns, attributes._Attribute):
            return column_dict.get(columns, [])
        return column_dict

    def _read(
        self,
        row_keys=None,
        row_ranges=None,
        row_filter: typing.Optional[RowFilter] = None,
    ):
        query = ReadRowsQuery(
            row_keys=row_keys or [],
            row_ranges=row_ranges or [],
            row_filter=row_filter,
        )
        if not query.row_keys and not query.row_ranges:
            return {}
        if row_keys and len(row_keys) > self._max_row_key_count:
            combined = {}
            for i in range(0, len(row_keys), self._max_row_key_count):
                chunk_query = ReadRowsQuery(
                    row_keys=row_keys[i:i + self._max_row_key_count],
                    row_filter=row_filter,
                )
                for row in self._table.read_rows(chunk_query):
                    combined[row.row_key] = utils.row_to_column_dict(row)
            return combined
        rows = self._table.read_rows(query)
        return {row.row_key: utils.row_to_column_dict(row) for row in rows}

    def read_all_rows(self):
        return _ReadAllRowsResult(self._table.read_rows_stream(ReadRowsQuery()))

    # ── Delete ───────────────────────────────────────────────────────────

    def _delete_meta(self):
        self._table.mutate_row(attributes.TableMeta.key, DeleteAllFromRow())

    def delete_cells(self, mutations, row_keys_to_delete=None):
        entries = []
        for row_key, column, timestamps in mutations:
            row_mutations = []
            for ts in timestamps:
                start_us = _datetime_to_micros(ts)
                end_us = start_us + 1000  # 1ms (bigtable granularity)
                row_mutations.append(DeleteRangeFromColumn(
                    family=column.family_id,
                    qualifier=column.key,
                    start_timestamp_micros=start_us,
                    end_timestamp_micros=end_us,
                ))
            entries.append(RowMutationEntry(row_key, row_mutations))
        for key in (row_keys_to_delete or []):
            entries.append(RowMutationEntry(key, DeleteAllFromRow()))
        if entries:
            self._table.bulk_mutate_rows(entries)

    def delete_row(self, row_key):
        self._table.mutate_row(row_key, DeleteAllFromRow())
