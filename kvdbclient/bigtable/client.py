# pylint: disable=invalid-name, missing-docstring, line-too-long
# pylint: disable=logging-fstring-interpolation, too-many-arguments

import atexit
import typing
import logging
import weakref
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
from . import get_client_info
from ..base import Cell
from ..base import ClientWithIDGen
from ..base import ColumnFamilyConfig
from ..base import LegacyRow
from ..base import LegacyRows
from ..base import OperationLogger
from ..base import column_view
from ..base import deserialize_cells
from ..base import log_lock_holders
from .. import attributes
from .. import exceptions
from ..serializers import serialize_uint64
from ..utils import get_valid_timestamp

CHECK_AND_MUTATE_TIMEOUT_S = 30.0


def _datetime_to_micros(dt: datetime) -> int:
    """Convert datetime to microseconds for bigtable mutations."""
    return int(dt.timestamp() * 1_000_000)


def _set_cell(column: attributes._Attribute, value, timestamp_micros) -> SetCell:
    """SetCell of `value` in `column`, at `timestamp_micros`, or server time for None."""
    return SetCell(
        family=column.family_id,
        qualifier=column.key,
        new_value=column.serialize(value),
        timestamp_micros=timestamp_micros,
    )


def _set_lock_mutation(
    lock_column: attributes._Attribute, operation_id: np.uint64
) -> SetCell:
    """SetCell of `operation_id` in `lock_column`, stamped now to the millisecond."""
    return _set_cell(
        lock_column, operation_id, _datetime_to_micros(get_valid_timestamp(None))
    )


def _delete_range(
    column: attributes._Attribute, start_micros=None, end_micros=None
) -> DeleteRangeFromColumn:
    """Deletes `column`'s cells from `start_micros` until `end_micros`; None is open."""
    return DeleteRangeFromColumn(
        family=column.family_id,
        qualifier=column.key,
        start_timestamp_micros=start_micros,
        end_timestamp_micros=end_micros,
    )


def _client_kwargs(config: BigTableConfig, **extra) -> dict[str, object]:
    """A client's keywords for `config`'s project, with its credentials when set."""
    kwargs = dict(project=config.PROJECT, **extra)
    if config.CREDENTIALS:
        kwargs["credentials"] = config.CREDENTIALS
    return kwargs


def _close_quietly(data_client) -> None:
    """Closes `data_client`, ignoring one already closed or broken."""
    try:
        data_client.close()
    except Exception:
        pass


def _gc_rule(column_family: ColumnFamilyConfig):
    """`column_family`'s garbage-collection rule, or None when it keeps every cell."""
    rules = []
    if column_family.max_versions is not None:
        rules.append(MaxVersionsGCRule(column_family.max_versions))
    if column_family.max_age is not None:
        rules.append(MaxAgeGCRule(column_family.max_age))
    if not rules:
        return None
    if len(rules) == 1:
        return rules[0]
    return GCRuleIntersection(rules)


class Client(ClientWithIDGen, OperationLogger):
    config_class = BigTableConfig

    @staticmethod
    def default_client_info():
        return get_client_info(admin=True, read_only=False)

    def __init__(
        self,
        table_id: str,
        config: BigTableConfig = BigTableConfig(),
        table_meta=None,
        lock_expiry: timedelta = timedelta(minutes=3),
        data_client: typing.Optional[BigtableDataClient] = None,
    ):
        # Store config for lazy admin client creation
        self._config = config
        self._table_id = table_id
        self.__admin_table = None

        # A data client passed in is shared, and closed only by the client that made it.
        owns_data_client = data_client is None
        if owns_data_client:
            data_client = BigtableDataClient(**_client_kwargs(config))
        self._data_client = data_client
        self._table = self._data_client.get_table(
            config.INSTANCE,
            table_id,
            # Read rows
            default_read_rows_operation_timeout=600.0,
            default_read_rows_attempt_timeout=20.0,
            # Mutate rows (batch)
            default_mutate_rows_retryable_errors=(
                DeadlineExceeded,
                ServiceUnavailable,
            ),
            default_mutate_rows_operation_timeout=lock_expiry.total_seconds(),
            default_mutate_rows_attempt_timeout=60.0,
            # Single-row ops; lock compare-and-sets pass CHECK_AND_MUTATE_TIMEOUT_S
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
        self._finalizer = None
        if owns_data_client:
            self._finalizer = weakref.finalize(self, _close_quietly, self._data_client)
            atexit.register(self._finalizer)

    def close(self):
        if self._finalizer is not None:
            self._finalizer()

    @property
    def _admin_table(self):
        if self.__admin_table is None:
            kwargs = _client_kwargs(self._config, admin=self._config.ADMIN)
            instance = bigtable.Client(**kwargs).instance(self._config.INSTANCE)
            self.__admin_table = instance.table(self._table_id)
        return self.__admin_table

    def _table_exists(self) -> bool:
        return self._admin_table.exists()

    def _create_storage(self, column_families) -> None:
        self._admin_table.create()
        for column_family in column_families:
            self.create_column_family(column_family)

    def create_column_family(self, family_id, gc_rule=None):
        if isinstance(family_id, ColumnFamilyConfig):
            gc_rule = _gc_rule(family_id)
            family_id = family_id.family_id
        f = self._admin_table.column_family(family_id, gc_rule=gc_rule)
        f.create()

    def mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes._Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> RowMutationEntry:
        ts_micros = _datetime_to_micros(time_stamp) if time_stamp else None
        mutations = [
            _set_cell(column, value, ts_micros) for column, value in val_dict.items()
        ]
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
                    entry.row_key,
                    list(entry.mutations),
                    operation_timeout=timeout,
                )
            else:
                with self._table.mutations_batcher(
                    batch_operation_timeout=timeout,
                    batch_attempt_timeout=min(timeout, 20.0),
                ) as batcher:
                    for entry in rows_list:
                        batcher.append(entry)
        except MutationsExceptionGroup as exc:
            raise exceptions.KVDBClientError(f"Bulk write failed: {exc}") from exc

    def _check_and_mutate(
        self, row_key: bytes, predicate: RowFilter, **case_mutations
    ) -> bool:
        """Mutates `row_key` by whether `predicate` matches; True when it matched.

        Takes the table's `true_case_mutations` and `false_case_mutations`
        keywords, under the one compare-and-set timeout.
        """
        return self._table.check_and_mutate_row(
            row_key,
            predicate=predicate,
            operation_timeout=CHECK_AND_MUTATE_TIMEOUT_S,
            **case_mutations,
        )

    def _set_lock_unless(
        self,
        row_key: bytes,
        predicate: RowFilter,
        lock_column: attributes._Attribute,
        operation_id: np.uint64,
    ) -> bool:
        """Writes the lock cell unless `predicate` matches; True when it wrote."""
        mutation = _set_lock_mutation(lock_column, operation_id)
        return not self._check_and_mutate(
            row_key, predicate, false_case_mutations=mutation
        )

    def _delete_lock_if(
        self, row_key: bytes, predicate: RowFilter, lock_column: attributes._Attribute
    ) -> bool:
        """Deletes `lock_column` when `predicate` matches.

        Returns the table's result as is: whether `predicate` matched.
        """
        mutation = _delete_range(lock_column)
        return self._check_and_mutate(row_key, predicate, true_case_mutations=mutation)

    def _lock_root_unless(
        self,
        root_id: np.uint64,
        predicate: RowFilter,
        lock_column: attributes._Attribute,
        operation_id: np.uint64,
        log_prefix: str,
    ) -> bool:
        """Locks `lock_column` on `root_id` unless `predicate` matches.

        True when locked. A refusal logs `log_prefix` and the holding op ids, at debug.
        """
        row_key = serialize_uint64(root_id)
        acquired = self._set_lock_unless(row_key, predicate, lock_column, operation_id)
        if not acquired and self.logger.isEnabledFor(logging.DEBUG):
            row = self._read_byte_row(row_key, columns=lock_column)
            log_lock_holders(self.logger, log_prefix, row)
        return acquired

    def lock_root(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        indefinite_lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_root_lock_filter(
            lock_column, self._lock_expiry, indefinite_lock_column
        )
        return self._lock_root_unless(
            root_id, filter_, lock_column, operation_id, "Locked"
        )

    def lock_root_indefinitely(
        self, root_id: np.uint64, operation_id: np.uint64
    ) -> bool:
        lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_indefinite_root_lock_filter(lock_column)
        return self._lock_root_unless(
            root_id, filter_, lock_column, operation_id, "Indefinitely locked"
        )

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        # Root-specific: ConditionalRowFilter inverts the usual renew
        # semantics, so a root renew sets the cell unless the filter matches
        # (see `get_renew_lock_filter`); `renew_lock_by_row_key` sets it when
        # its filter matches.
        lock_column = attributes.Concurrency.Lock
        row_key = serialize_uint64(root_id)
        filter_ = utils.get_renew_lock_filter(lock_column, operation_id)
        return self._set_lock_unless(row_key, filter_, lock_column, operation_id)

    def lock_by_row_key(self, row_key: bytes, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        filter_ = utils.get_row_key_lock_filter(lock_column, self._lock_expiry)
        return self._set_lock_unless(row_key, filter_, lock_column, operation_id)

    def lock_by_row_key_with_indefinite(
        self, row_key: bytes, operation_id: np.uint64
    ) -> bool:
        """Temporal lock on a row key that also refuses if indefinite is held.

        `lock_root`'s filter adds the NewParent check on top of this one. Locks to be
        upgraded to indefinite need it: without it, an acquire could succeed on a row
        whose indefinite cell a crashed worker left set.
        """
        lock_column = attributes.Concurrency.Lock
        indefinite_lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_row_key_lock_filter_with_indefinite(
            lock_column, self._lock_expiry, indefinite_lock_column
        )
        return self._set_lock_unless(row_key, filter_, lock_column, operation_id)

    def lock_by_row_key_indefinitely(
        self, row_key: bytes, operation_id: np.uint64
    ) -> bool:
        """Indefinite-column CAS on a row key. No expiry semantics — a
        cell here persists until explicitly released (or a worker
        recovery path clears it). Counterpart to `lock_root_indefinitely`
        for hierarchy-free rows.
        """
        lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_indefinite_row_key_lock_filter(lock_column)
        return self._set_lock_unless(row_key, filter_, lock_column, operation_id)

    def unlock_by_row_key(self, row_key: bytes, operation_id: np.uint64):
        lock_column = attributes.Concurrency.Lock
        filter_ = utils.get_unexpired_held_lock_filter(
            lock_column, self._lock_expiry, operation_id
        )
        return self._delete_lock_if(row_key, filter_, lock_column)

    def unlock_indefinitely_locked_by_row_key(
        self, row_key: bytes, operation_id: np.uint64
    ):
        """Release the indefinite lock on any row, roots included, holding `operation_id`.

        Deleting only a matching cell means a lock stolen after a crash cannot be released
        under its new holder.
        """
        lock_column = attributes.Concurrency.IndefiniteLock
        filter_ = utils.get_held_lock_filter(lock_column, operation_id)
        return self._delete_lock_if(row_key, filter_, lock_column)

    def renew_lock_by_row_key(self, row_key: bytes, operation_id: np.uint64) -> bool:
        lock_column = attributes.Concurrency.Lock
        filter_ = utils.get_held_lock_filter(lock_column, operation_id)
        mutation = _set_lock_mutation(lock_column, operation_id)
        return bool(
            self._check_and_mutate(row_key, filter_, true_case_mutations=mutation)
        )

    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        column = attributes.Concurrency.Counter
        result_row = self._table.read_modify_write_row(
            key,
            IncrementRule(column.family_id, column.key, increment_amount=size),
        )
        cells = result_row.get_cells(column.family_id, column.key)
        high = column.deserialize(cells[0].value)
        return high + np.uint64(1) - size, high

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
            row_ranges = [
                RowRange(
                    start_key=start_key,
                    end_key=end_key,
                    start_is_inclusive=True,
                    end_is_inclusive=end_key_inclusive,
                )
            ]
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
            deserialize_cells(column_dict)
            rows[row_key] = column_view(column_dict, columns)
        return rows

    def _read_byte_row(
        self,
        row_key,
        columns=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
        )
        row = self._table.read_row(row_key, row_filter=filter_)
        if row is None:
            return column_view({}, columns)
        column_dict = utils.row_to_column_dict(row)
        deserialize_cells(column_dict)
        return column_view(column_dict, columns)

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
        queries = [query]
        if row_keys and len(row_keys) > self._max_row_key_count:
            step = self._max_row_key_count
            queries = [
                ReadRowsQuery(row_keys=row_keys[i : i + step], row_filter=row_filter)
                for i in range(0, len(row_keys), step)
            ]
        return {
            row.row_key: utils.row_to_column_dict(row)
            for chunk in queries
            for row in self._table.read_rows(chunk)
        }

    def read_all_rows(self):
        return LegacyRows(
            {
                row.row_key: LegacyRow(
                    (
                        cell.family,
                        (
                            cell.qualifier
                            if isinstance(cell.qualifier, bytes)
                            else cell.qualifier.encode()
                        ),
                        Cell(value=cell.value, timestamp=cell.timestamp_micros),
                    )
                    for cell in row.cells
                )
                for row in self._table.read_rows_stream(ReadRowsQuery())
            }
        )

    def _delete_meta(self):
        self._table.mutate_row(attributes.TableMeta.key, DeleteAllFromRow())

    def delete_cells(self, mutations, row_keys_to_delete=None):
        entries = []
        for row_key, column, timestamps in mutations:
            row_mutations = []
            for ts in timestamps:
                start_us = _datetime_to_micros(ts)
                end_us = start_us + 1000  # 1ms (bigtable granularity)
                row_mutations.append(_delete_range(column, start_us, end_us))
            entries.append(RowMutationEntry(row_key, row_mutations))
        for key in row_keys_to_delete or []:
            entries.append(RowMutationEntry(key, DeleteAllFromRow()))
        if entries:
            self._table.bulk_mutate_rows(entries)

    def delete_row(self, row_key):
        self._table.mutate_row(row_key, DeleteAllFromRow())
