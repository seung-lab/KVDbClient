import sys
import time
import typing
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from . import attributes
from . import exceptions
from . import basetypes
from .serializers import pad_node_id
from .serializers import serialize_key
from .serializers import serialize_uint64
from .serializers import serialize_uint64_batch
from .serializers import deserialize_uint64
from .extensions import RootExtension
from .registry import register_backend
from .utils import get_compatible_time_stamp


@dataclass(frozen=True)
class ColumnFamilyConfig:
    """Backend-agnostic column family definition."""

    family_id: str
    max_versions: typing.Optional[int] = None
    max_age: typing.Optional[timedelta] = None


DEFAULT_COLUMN_FAMILIES = [
    ColumnFamilyConfig("0"),
    ColumnFamilyConfig("1", max_versions=1),
    ColumnFamilyConfig("2"),
    ColumnFamilyConfig("3", max_age=timedelta(days=365)),
    ColumnFamilyConfig("4"),
]

# A root chunk counts its ids over this many rows, so concurrent edits rarely share one.
COUNTER_SHARDS = 2**8


def counter_key(chunk_id) -> bytes:
    """The row counting a chunk's ids, unsharded."""
    return serialize_uint64(chunk_id, counter=True)


def counter_shard_key(chunk_id, shard) -> bytes:
    """The row holding one shard of a chunk's sharded id counter."""
    return serialize_key(f"i{pad_node_id(chunk_id)}_{shard}")


def shard_segment_id(value, shard):
    """The segment id that counter `value` in shard row `shard` stands for."""
    return value * np.uint64(COUNTER_SHARDS) + shard


class CounterRow(typing.NamedTuple):
    """The chunk a counter row counts for, and its shard; None for the unsharded row."""

    chunk_id: np.uint64
    shard: typing.Optional[int]


def counter_row(key: bytes) -> CounterRow:
    """What `counter_key` or `counter_shard_key` built `key` from."""
    chunk, _, shard = key[1:].partition(b"_")
    return CounterRow(np.uint64(int(chunk)), int(shard) if shard else None)


class Cell:
    """Backend-agnostic cell representation.

    Compatible with google.cloud.bigtable.row_data.Cell interface.
    Used by non-BigTable backends; BigTable continues using its native Cell.
    """

    __slots__ = ("value", "timestamp")

    def __init__(self, value, timestamp):
        self.value = value
        self.timestamp = timestamp

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return NotImplemented
        return self.value == other.value and self.timestamp == other.timestamp

    def __hash__(self):
        return hash((self.value, self.timestamp))

    def __repr__(self):
        return f"Cell(value={self.value!r}, timestamp={self.timestamp!r})"


def deserialize_cells(column_dict: dict[attributes._Attribute, list]) -> None:
    """Deserializes, in place, each cell value in `column_dict` by its column."""
    for column, cells in column_dict.items():
        for cell in cells:
            cell.value = column.deserialize(cell.value)


def column_view(column_dict: dict[attributes._Attribute, list], columns):
    """`column_dict`, or only its cells in `columns` when that names one column."""
    if isinstance(columns, attributes._Attribute):
        return column_dict.get(columns, [])
    return column_dict


class LegacyRow:
    """A row as legacy readers see it, `.cells` as {family: {qualifier: [Cell, ...]}}.

    Rows compare by `equal_by` when given, else by their cells, and hash by identity.
    """

    __slots__ = ("cells", "_equal_by")

    def __init__(self, cells: typing.Iterable[tuple[str, bytes, Cell]], equal_by=None):
        families = defaultdict(lambda: defaultdict(list))
        for family, qualifier, cell in cells:
            families[family][qualifier].append(cell)
        self.cells = dict(families)
        self._equal_by = self.cells if equal_by is None else equal_by

    def __eq__(self, other):
        if not isinstance(other, LegacyRow):
            return NotImplemented
        return self._equal_by == other._equal_by

    def __hash__(self):
        return id(self)


class LegacyRows:
    """Rows by key, with a legacy `consume_all` that has nothing left to read."""

    __slots__ = ("rows",)

    def __init__(self, rows: dict[bytes, LegacyRow]):
        self.rows = rows

    def consume_all(self):
        pass


# Calls in flight at once while an op takes, renews, releases or dates its root locks.
LOCK_WORKERS = 8


class _Outcome(typing.NamedTuple):
    """One call of a fanned-out lock method: its arguments, and its result or error."""

    call: typing.Tuple[typing.Any, ...]
    result: typing.Any
    error: typing.Optional[Exception]

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.result)


def _with_future_roots(root_ids, lookup) -> np.ndarray:
    """`root_ids`, read once, each root with future roots from `lookup` replaced by them.

    A root that `lookup` cannot find raises from that lookup.
    """
    expanded = []
    for root_id in root_ids:
        future = lookup(root_id)
        if not future.size:
            expanded.append(root_id)
        else:
            expanded.extend(future)
    return np.unique(expanded)


_LOCK_FAILED = "Failed to lock root {root}: {error}"
_UNLOCK_FAILED = "Failed to unlock root: {error}"


def _fan_out(method, calls) -> list[_Outcome]:
    """Each `method(*call)` on up to `LOCK_WORKERS` threads, in completion order."""
    outcomes = []
    if not calls:
        return outcomes
    with ThreadPoolExecutor(max_workers=min(LOCK_WORKERS, len(calls))) as executor:
        futures = {executor.submit(method, *call): call for call in calls}
        for future in as_completed(futures):
            call = futures[future]
            try:
                outcomes.append(_Outcome(call, future.result(), None))
            except Exception as error:
                outcomes.append(_Outcome(call, None, error))
    return outcomes


def _log_errors(logger: logging.Logger, outcomes: list[_Outcome], message: str) -> None:
    """Logs each failed outcome's error with `message`, which may name the root."""
    for outcome in outcomes:
        if outcome.error is not None:
            logger.error(message.format(root=outcome.call[0], error=outcome.error))


def _lock_each(
    logger: logging.Logger, lock, unlock, root_ids, operation_id
) -> list[_Outcome]:
    """Locks every root with `lock`; if any refuses, releases them all with `unlock`."""
    calls = [(root_id, operation_id) for root_id in root_ids]
    locked = _fan_out(lock, calls)
    _log_errors(logger, locked, _LOCK_FAILED)
    if not all(outcome.ok for outcome in locked):
        _log_errors(logger, _fan_out(unlock, calls), _UNLOCK_FAILED)
    return locked


def log_lock_holders(logger: logging.Logger, prefix: str, cells) -> None:
    """Logs at debug the op ids a lock's `cells` hold, after `prefix`."""
    logger.debug(f"{prefix} operation ids: {[cell.value for cell in cells]}")


def _flatten_log(record: dict[attributes._Attribute, list]) -> datetime:
    """Flattens an op-log `record` to its newest values in place; returns its op time."""
    try:
        timestamp = record[attributes.OperationLogs.OperationTimeStamp][0].value
    except KeyError:
        timestamp = record[attributes.OperationLogs.RootID][0].timestamp
    record.update((column, cells[0].value) for column, cells in record.items())
    return timestamp


def _newest(row: dict[attributes._Attribute, list], column):
    """The newest value of `column` in a read `row`, or None when it holds none."""
    cells = row.get(column)
    return cells[0].value if cells else None


class SimpleClient(ABC):
    """
    Abstract class for interacting with backend data store.
    Eg., BigTableClient for using big table as storage, HBaseClient for Apache HBase.
    """

    backend_name = None
    config_class = None
    default_client_info = None
    # Whether this client reads a fork's table through its sources; plain ones refuse it.
    opens_forks = False
    # The open table's fork marker; None for a plain table.
    marker = None
    # Attribute holding backend's data handle, and that handle's read-only calls;
    # upstream of a fork, only those pass.
    DATA_HANDLE = ""
    READ_CALLS: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs):
        """Registers a subclass declaring ``backend_name``; an incomplete one fails."""
        super().__init_subclass__(**kwargs)
        if "backend_name" not in cls.__dict__:
            return
        missing = [
            name
            for name in ("config_class", "default_client_info")
            if getattr(cls, name, None) is None
        ]
        if missing:
            raise TypeError(
                f"backend client {cls.__name__} must define {', '.join(missing)} to register"
            )
        register_backend(cls)

    def _init_common(self, logger_name, table_meta, lock_expiry, max_row_key_count):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.WARNING)
        if not self.logger.handlers:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.WARNING)
            self.logger.addHandler(sh)
        self._table_meta = table_meta
        self._lock_expiry = lock_expiry
        self._version = None
        self._max_row_key_count = max_row_key_count
        self._root_ext = None

    @property
    def root_ext(self):
        if self._root_ext is None:
            self._root_ext = RootExtension(self)
        return self._root_ext

    def close(self):
        """Override in subclasses to release backend resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def create_table(self, meta, version: str, column_families=None) -> None:
        """Initialize the table and store associated meta."""
        if self._table_exists():
            raise ValueError(f"{self._table_id} already exists.")
        if column_families is None:
            column_families = DEFAULT_COLUMN_FAMILIES
        self._create_storage(column_families)
        self.add_table_version(version)
        self.update_table_meta(meta)

    @abstractmethod
    def _table_exists(self) -> bool:
        """Whether the table's storage exists."""

    @abstractmethod
    def _create_storage(self, column_families) -> None:
        """Create the table's storage with `column_families`."""

    @abstractmethod
    def mutate_row(self, row_key, val_dict, time_stamp=None):
        """Create a row mutation (does not write to storage).

        Returns a backend-specific mutation object consumed by write().
        """

    def write(
        self, rows, root_ids=None, operation_id=None, slow_retry=True, block_size=2000
    ):
        """Write a list of mutated rows in bulk."""
        if root_ids is not None and operation_id is not None:
            if isinstance(root_ids, int):
                root_ids = [root_ids]
            if not self.renew_locks(root_ids, operation_id):
                raise exceptions.LockingError(
                    f"Root lock renewal failed: operation {operation_id}"
                )
        self._write_rows(rows, slow_retry=slow_retry, block_size=block_size)

    @abstractmethod
    def _write_rows(self, rows, slow_retry=True, block_size=2000):
        """Backend-specific bulk write implementation."""

    def _lock_refused(self, row_key: bytes) -> bool:
        """Whether this table refuses a lock on `row_key` before any compare-and-set.

        A plain table never does; a fork refuses rows an inherited op left half-written.
        """
        return False

    def lock_root(self, node_id, operation_id) -> bool:
        """Locks root node with operation_id to prevent race conditions."""
        if self._lock_refused(serialize_uint64(node_id)):
            return False
        return self._lock_root(node_id, operation_id)

    @abstractmethod
    def _lock_root(self, node_id, operation_id) -> bool:
        """The backend's temporal root lock, asked only once the table allows it."""

    def lock_root_indefinitely(self, node_id, operation_id) -> bool:
        """Locks root node with operation_id to prevent race conditions."""
        if self._lock_refused(serialize_uint64(node_id)):
            return False
        return self._lock_root_indefinitely(node_id, operation_id)

    @abstractmethod
    def _lock_root_indefinitely(self, node_id, operation_id) -> bool:
        """The backend's indefinite root lock, asked only once the table allows it."""

    def unlock_root(self, root_id: np.uint64, operation_id: np.uint64):
        """Unlocks root node that is locked with operation_id."""
        return self.unlock_by_row_key(serialize_uint64(root_id), operation_id)

    def unlock_indefinitely_locked_root(
        self, root_id: np.uint64, operation_id: np.uint64
    ):
        """Unlocks root node that is indefinitely locked with operation_id."""
        return self.unlock_indefinitely_locked_by_row_key(
            serialize_uint64(root_id), operation_id
        )

    @abstractmethod
    def renew_lock(self, node_id, operation_id):
        """Renews existing node lock with operation_id for extended time."""

    @abstractmethod
    def lock_by_row_key(self, row_key, operation_id):
        """Lock an arbitrary row key.

        Same temporal-expiry semantics as `lock_root` but with no
        hierarchy or node-id coupling — the caller owns the row-key
        encoding. Returns True if acquired.
        """

    def lock_by_row_key_with_indefinite(self, row_key, operation_id) -> bool:
        """Temporal lock on a row key that also refuses if an indefinite
        lock cell exists on the same row.

        Row-key counterpart to `lock_root` (which unions both columns in
        its filter). Use for locks that will be upgraded to indefinite:
        without this guard, a fresh temporal acquire could silently
        succeed against a row whose indefinite cell was left set by a
        crashed worker.
        """
        if self._lock_refused(row_key):
            return False
        return self._lock_by_row_key_with_indefinite(row_key, operation_id)

    @abstractmethod
    def _lock_by_row_key_with_indefinite(self, row_key, operation_id) -> bool:
        """The backend's guarded temporal lock, asked only once the table allows it."""

    @abstractmethod
    def _put_if_absent(self, row_key: bytes, column, value) -> bool:
        """Writes `value` to `column`, stamped now, if absent; True if written."""

    def lock_by_row_key_indefinitely(self, row_key, operation_id) -> bool:
        """Acquire the indefinite-column cell on a row key. No expiry;
        persists until released or cleared by operator recovery. Row-key
        counterpart to `lock_root_indefinitely`.
        """
        if self._lock_refused(row_key):
            return False
        return self._lock_by_row_key_indefinitely(row_key, operation_id)

    def _lock_by_row_key_indefinitely(self, row_key, operation_id) -> bool:
        """Shared indefinite lock: never expires, so only an absent cell is taken."""
        return self._put_if_absent(
            row_key, attributes.Concurrency.IndefiniteLock, operation_id
        )

    @abstractmethod
    def unlock_by_row_key(self, row_key, operation_id):
        """Release a lock previously acquired via `lock_by_row_key`.

        Deletes the lock cell only if its value still matches
        `operation_id`, so an expired-then-stolen lock can't be released
        out from under the current holder.
        """

    @abstractmethod
    def unlock_indefinitely_locked_by_row_key(self, row_key, operation_id):
        """Release an indefinite row-key lock previously acquired by
        `lock_by_row_key_indefinitely`. Same value-match safety as
        `unlock_by_row_key`.
        """

    @abstractmethod
    def renew_lock_by_row_key(self, row_key, operation_id):
        """Extend the expiry of a lock acquired via `lock_by_row_key`."""

    @staticmethod
    def get_compatible_timestamp(time_stamp, round_up=False):
        """Datetime time stamp compatible with client's services."""
        return get_compatible_time_stamp(time_stamp, round_up=round_up)

    @abstractmethod
    def read_all_rows(self):
        """Read all rows from the table."""

    @abstractmethod
    def create_column_family(self, family_id, gc_rule=None):
        """Create a column family on the table."""

    @abstractmethod
    def delete_cells(self, mutations, row_keys_to_delete=None):
        """Delete specific cell versions and/or entire rows."""

    @abstractmethod
    def delete_row(self, row_key):
        """Delete an entire row."""

    @abstractmethod
    def _read_byte_row(
        self,
        row_key,
        columns=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single row (raw bytes, deserialized by column attribute)."""

    @abstractmethod
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
        """Read multiple rows (raw bytes, deserialized by column attribute)."""

    @abstractmethod
    def _delete_meta(self):
        """Delete the table meta row."""

    @property
    def table_meta(self):
        return self._table_meta

    def add_table_version(self, version: str, overwrite: bool = False):
        if not overwrite:
            assert self.read_table_version() is None, self.read_table_version()
        self._version = version
        self._write_cell(
            attributes.TableVersion.key, attributes.TableVersion.Version, version
        )

    def read_table_version(self) -> str:
        version = self._read_cell(
            attributes.TableVersion.key, attributes.TableVersion.Version
        )
        if version is not None:
            self._version = version
        return version

    def update_table_meta(self, meta, overwrite: typing.Optional[bool] = False):
        if overwrite:
            self._delete_meta()
        self._table_meta = meta
        self.write([self.mutate_row(attributes.TableMeta.key, self._meta_row(meta))])

    def _meta_row(self, meta) -> dict[attributes._Attribute, typing.Any]:
        """The meta row's values holding `meta`."""
        return {attributes.TableMeta.Meta: meta}

    def read_table_meta(self):
        row = self._read_byte_row(attributes.TableMeta.key)
        marker = _newest(row, attributes.TableMeta.Fork)
        if marker is not None:
            self._open_fork(marker)
        self._table_meta = _newest(row, attributes.TableMeta.Meta)
        return self._table_meta

    def _open_fork(self, marker: str) -> None:
        """Refuses a fork's table: this client cannot read below its fork time."""
        raise exceptions.ForkTableError(
            f"{self._table_id} is a fork; {type(self).__name__} reads only its own table"
        )

    def _write_cell(self, row_key: bytes, column, value) -> None:
        """Writes `value` to `column` of row `row_key`."""
        self.write([self.mutate_row(row_key, {column: value})])

    def _read_cell(self, row_key: bytes, column):
        """The newest value in `column` of row `row_key`, or None when it holds none."""
        return _newest(self._read_byte_row(row_key), column)

    def read_nodes(
        self,
        start_id=None,
        end_id=None,
        end_id_inclusive=False,
        user_id=None,
        node_ids=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
        attr_keys: bool = True,
    ):
        if node_ids is not None and len(node_ids) > self._max_row_key_count:
            node_ids = np.sort(node_ids)
        rows = self._read_byte_rows(
            start_key=(
                serialize_uint64(start_id, fake_edges=fake_edges)
                if start_id is not None
                else None
            ),
            end_key=(
                serialize_uint64(end_id, fake_edges=fake_edges)
                if end_id is not None
                else None
            ),
            end_key_inclusive=end_id_inclusive,
            row_keys=(
                serialize_uint64_batch(node_ids, fake_edges=fake_edges)
                if node_ids is not None
                else None
            ),
            columns=properties,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
            user_id=user_id,
        )
        if attr_keys:
            return {
                deserialize_uint64(row_key, fake_edges=fake_edges): data
                for (row_key, data) in rows.items()
            }
        return {
            deserialize_uint64(row_key, fake_edges=fake_edges): {
                k.key: v for k, v in data.items()
            }
            for (row_key, data) in rows.items()
        }

    def read_node(
        self,
        node_id: np.uint64,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
        fake_edges: bool = False,
    ):
        return self._read_byte_row(
            row_key=serialize_uint64(node_id, fake_edges=fake_edges),
            columns=properties,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )

    def write_nodes(self, nodes, root_ids=None, operation_id=None):
        pass

    def read_log_entry(
        self, operation_id: np.uint64
    ) -> typing.Tuple[typing.Dict, datetime]:
        log_record = self.read_node(
            operation_id, properties=attributes.OperationLogs.all()
        )
        if len(log_record) == 0:
            return {}, None
        return log_record, _flatten_log(log_record)

    def read_log_entries(
        self,
        operation_ids=None,
        user_id=None,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
    ):
        if properties is None:
            properties = attributes.OperationLogs.all()
        filters = dict(
            user_id=user_id,
            properties=properties,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        if operation_ids is None:
            logs_d = self.read_nodes(
                start_id=np.uint64(0),
                end_id=self.get_max_operation_id(),
                end_id_inclusive=True,
                **filters,
            )
        else:
            logs_d = self.read_nodes(node_ids=operation_ids, **filters)
        if not logs_d:
            return {}
        for record in logs_d.values():
            record["timestamp"] = _flatten_log(record)
        return logs_d

    def lock_roots(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: dict[np.uint64, np.ndarray],
        max_tries: int = 1,
        waittime_s: float = 0.5,
    ) -> typing.Tuple[bool, typing.Iterable]:
        i_try = 0
        while i_try < max_tries:
            root_ids = _with_future_roots(root_ids, future_root_ids_d.__getitem__)
            locked = _lock_each(
                self.logger, self.lock_root, self.unlock_root, root_ids, operation_id
            )
            if all(outcome.ok for outcome in locked):
                return True, root_ids
            time.sleep(waittime_s)
            i_try += 1
            self.logger.debug(f"Try {i_try}")
        return False, root_ids

    def lock_roots_indefinitely(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: dict[np.uint64, np.ndarray],
    ) -> typing.Tuple[bool, typing.Iterable, typing.Iterable]:
        root_ids = _with_future_roots(root_ids, future_root_ids_d.get)
        locked = _lock_each(
            self.logger,
            self.lock_root_indefinitely,
            self.unlock_indefinitely_locked_root,
            root_ids,
            operation_id,
        )
        failed_to_lock = [
            outcome.call[0]
            for outcome in locked
            if outcome.error is not None or outcome.result is False
        ]
        return all(outcome.ok for outcome in locked), root_ids, failed_to_lock

    def renew_locks(self, root_ids: np.uint64, operation_id: np.uint64) -> bool:
        calls = [(root_id, operation_id) for root_id in root_ids]
        for outcome in _fan_out(self.renew_lock, calls):
            root_id = outcome.call[0]
            if outcome.error is not None:
                self.logger.error(
                    f"Exception during renew_lock({root_id}): {outcome.error}"
                )
                return False
            if not outcome.result:
                self.logger.warning(f"renew_lock failed - {root_id}")
                return False
        return True

    def get_lock_timestamp(
        self, root_id: np.uint64, operation_id: np.uint64
    ) -> typing.Union[datetime, None]:
        row = self.read_node(root_id, properties=attributes.Concurrency.Lock)
        if len(row) == 0:
            self.logger.warning(f"No lock found for {root_id}")
            return None
        if row[0].value != operation_id:
            self.logger.warning(f"{root_id} not locked with {operation_id}")
            return None
        return row[0].timestamp

    def get_consolidated_lock_timestamp(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_ids: typing.Sequence[np.uint64],
    ) -> typing.Union[datetime, None]:
        if len(root_ids) == 0:
            return None
        calls = list(zip(root_ids, operation_ids))
        timestamps = []
        for outcome in _fan_out(self.get_lock_timestamp, calls):
            if outcome.error is not None:
                root_id, op_id = outcome.call
                self.logger.warning(f"({root_id}, {op_id}): {outcome.error}")
                return None
            if outcome.result is None:
                return None
            timestamps.append(outcome.result)
        if not timestamps:
            return None
        return np.min(timestamps)


class ClientWithIDGen(SimpleClient):
    """
    Abstract class for client to backend data store that has support for generating IDs.
    If not, something else can be used but these methods need to be implemented.
    Eg., Big Table row cells can be used to generate unique IDs.
    """

    @abstractmethod
    def _get_ids_range(self, key: bytes, size: int) -> typing.Tuple:
        """Atomically increment counter and return (min, max) range."""

    def create_node_ids(
        self, chunk_id: np.uint64, size: int, root_chunk=False
    ) -> np.ndarray:
        if root_chunk:
            new_ids = self._get_root_segment_ids_range(chunk_id, size)
        else:
            low, high = self._get_ids_range(counter_key(chunk_id), size)
            low, high = basetypes.SEGMENT_ID.type(low), basetypes.SEGMENT_ID.type(high)
            new_ids = np.arange(low, high + np.uint64(1), dtype=basetypes.SEGMENT_ID)
        return new_ids | chunk_id

    def create_node_id(
        self, chunk_id: np.uint64, root_chunk=False
    ) -> basetypes.NODE_ID:
        return self.create_node_ids(chunk_id, 1, root_chunk=root_chunk)[0]

    def set_max_node_id(self, chunk_id: np.uint64, node_id: np.uint64) -> None:
        """Set max segment ID for a given chunk."""
        size = int(np.uint64(chunk_id) ^ np.uint64(node_id))
        self._get_ids_range(counter_key(chunk_id), size)

    def get_max_node_id(
        self, chunk_id: basetypes.CHUNK_ID, root_chunk=False
    ) -> basetypes.NODE_ID:
        if root_chunk:
            max_value = 0
            for shard in range(COUNTER_SHARDS):
                value = self._read_counter(counter_shard_key(chunk_id, shard))
                val = shard_segment_id(basetypes.SEGMENT_ID.type(value), shard)
                max_value = val if val > max_value else max_value
            return chunk_id | basetypes.SEGMENT_ID.type(max_value)
        value = self._read_counter(counter_key(chunk_id))
        return chunk_id | basetypes.SEGMENT_ID.type(value)

    def create_operation_id(self):
        return self._get_ids_range(attributes.OperationLogs.key, 1)[1]

    def get_max_operation_id(self):
        return self._read_counter(attributes.OperationLogs.key)

    def _read_counter(self, key: bytes):
        """The newest value of the id counter in row `key`; zero when it holds none."""
        column = attributes.Concurrency.Counter
        row = self._read_byte_row(key, columns=column)
        return row[0].value if row else column.basetype(0)

    def _get_root_segment_ids_range(
        self, chunk_id: basetypes.CHUNK_ID, size: int = 1, counter: int = None
    ) -> np.ndarray:
        n_counters = np.uint64(COUNTER_SHARDS)
        counter = (
            np.uint64(counter % n_counters)
            if counter
            else np.uint64(np.random.randint(0, n_counters))
        )
        min_, max_ = self._get_ids_range(
            key=counter_shard_key(chunk_id, counter), size=size
        )
        return np.arange(
            shard_segment_id(min_, counter),
            shard_segment_id(max_, counter) + np.uint64(1),
            n_counters,
            dtype=basetypes.SEGMENT_ID,
        )


class OperationLogger(ABC):
    """
    Abstract class for the backend data store that holds the operation logs.
    Eg., BigTableClient can be used to store logs in Google BigTable.
    """
