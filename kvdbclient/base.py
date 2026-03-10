import sys
import time
import typing
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
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


class SimpleClient(ABC):
    """
    Abstract class for interacting with backend data store.
    Eg., BigTableClient for using big table as storage, HBaseClient for Apache HBase.
    """

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

    # ── Abstract: backend-specific primitives ────────────────────────────

    @abstractmethod
    def create_table(self, meta, version: str, column_families=None) -> None:
        """Initialize the table and store associated meta."""

    @abstractmethod
    def mutate_row(self, row_key, val_dict, time_stamp=None):
        """Create a row mutation (does not write to storage).

        Returns a backend-specific mutation object consumed by write().
        """

    def write(self, rows, root_ids=None, operation_id=None, slow_retry=True, block_size=2000):
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

    @abstractmethod
    def lock_root(self, node_id, operation_id):
        """Locks root node with operation_id to prevent race conditions."""

    @abstractmethod
    def lock_root_indefinitely(self, node_id, operation_id):
        """Locks root node with operation_id to prevent race conditions."""

    @abstractmethod
    def unlock_root(self, node_id, operation_id):
        """Unlocks root node that is locked with operation_id."""

    @abstractmethod
    def unlock_indefinitely_locked_root(self, node_id, operation_id):
        """Unlocks root node that is indefinitely locked with operation_id."""

    @abstractmethod
    def renew_lock(self, node_id, operation_id):
        """Renews existing node lock with operation_id for extended time."""

    @abstractmethod
    def get_compatible_timestamp(self, time_stamp, round_up=False):
        """Datetime time stamp compatible with client's services."""

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
    def _read_byte_row(self, row_key, columns=None, start_time=None, end_time=None, end_time_inclusive=False):
        """Read a single row (raw bytes, deserialized by column attribute)."""

    @abstractmethod
    def _read_byte_rows(self, start_key=None, end_key=None, end_key_inclusive=False, row_keys=None, columns=None, start_time=None, end_time=None, end_time_inclusive=False, user_id=None):
        """Read multiple rows (raw bytes, deserialized by column attribute)."""

    @abstractmethod
    def _delete_meta(self):
        """Delete the table meta row."""

    # ── Concrete: shared logic ───────────────────────────────────────────

    @property
    def table_meta(self):
        return self._table_meta

    def add_table_version(self, version: str, overwrite: bool = False):
        if not overwrite:
            assert self.read_table_version() is None, self.read_table_version()
        self._version = version
        row = self.mutate_row(
            attributes.TableVersion.key,
            {attributes.TableVersion.Version: version},
        )
        self.write([row])

    def read_table_version(self) -> str:
        try:
            row = self._read_byte_row(attributes.TableVersion.key)
            self._version = row[attributes.TableVersion.Version][0].value
            return self._version
        except KeyError:
            return None

    def update_table_meta(
        self, meta, overwrite: typing.Optional[bool] = False
    ):
        if overwrite:
            self._delete_meta()
        self._table_meta = meta
        row = self.mutate_row(
            attributes.TableMeta.key,
            {attributes.TableMeta.Meta: meta},
        )
        self.write([row])

    def read_table_meta(self):
        row = self._read_byte_row(attributes.TableMeta.key)
        try:
            self._table_meta = row[attributes.TableMeta.Meta][0].value
        except KeyError:
            self._table_meta = None
        return self._table_meta

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
        try:
            timestamp = log_record[attributes.OperationLogs.OperationTimeStamp][0].value
        except KeyError:
            timestamp = log_record[attributes.OperationLogs.RootID][0].timestamp
        log_record.update((column, v[0].value) for column, v in log_record.items())
        return log_record, timestamp

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

        if operation_ids is None:
            logs_d = self.read_nodes(
                start_id=np.uint64(0),
                end_id=self.get_max_operation_id(),
                end_id_inclusive=True,
                user_id=user_id,
                properties=properties,
                start_time=start_time,
                end_time=end_time,
                end_time_inclusive=end_time_inclusive,
            )
        else:
            logs_d = self.read_nodes(
                node_ids=operation_ids,
                properties=properties,
                start_time=start_time,
                end_time=end_time,
                end_time_inclusive=end_time_inclusive,
                user_id=user_id,
            )
        if not logs_d:
            return {}
        for operation_id in logs_d:
            log_record = logs_d[operation_id]
            try:
                timestamp = log_record[attributes.OperationLogs.OperationTimeStamp][
                    0
                ].value
            except KeyError:
                timestamp = log_record[attributes.OperationLogs.RootID][0].timestamp
            log_record.update((column, v[0].value) for column, v in log_record.items())
            log_record["timestamp"] = timestamp
        return logs_d

    # ── Locking orchestration ────────────────────────────────────────────

    def lock_roots(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: typing.Dict,
        max_tries: int = 1,
        waittime_s: float = 0.5,
    ) -> typing.Tuple[bool, typing.Iterable]:
        i_try = 0
        while i_try < max_tries:
            new_root_ids: typing.List[np.uint64] = []
            for root_id in root_ids:
                future_root_ids = future_root_ids_d[root_id]
                if not future_root_ids.size:
                    new_root_ids.append(root_id)
                else:
                    new_root_ids.extend(future_root_ids)

            lock_results = {}
            root_ids = np.unique(new_root_ids)
            max_workers = min(8, max(1, len(root_ids)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_root = {
                    executor.submit(self.lock_root, root_id, operation_id): root_id
                    for root_id in root_ids
                }
                for future in as_completed(future_to_root):
                    root_id = future_to_root[future]
                    try:
                        lock_results[root_id] = future.result()
                    except Exception as e:
                        self.logger.error(f"Failed to lock root {root_id}: {e}")
                        lock_results[root_id] = False

            all_locked = all(lock_results.values())
            if all_locked:
                return True, root_ids

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                unlock_futures = [
                    executor.submit(self.unlock_root, root_id, operation_id)
                    for root_id in root_ids
                ]
                for future in as_completed(unlock_futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Failed to unlock root: {e}")
            time.sleep(waittime_s)
            i_try += 1
            self.logger.debug(f"Try {i_try}")
        return False, root_ids

    def lock_roots_indefinitely(
        self,
        root_ids: typing.Sequence[np.uint64],
        operation_id: np.uint64,
        future_root_ids_d: typing.Dict,
    ) -> typing.Tuple[bool, typing.Iterable, typing.Iterable]:
        new_root_ids: typing.List[np.uint64] = []
        for _id in root_ids:
            future_root_ids = future_root_ids_d.get(_id)
            if not future_root_ids.size:
                new_root_ids.append(_id)
            else:
                new_root_ids.extend(future_root_ids)

        root_ids = np.unique(new_root_ids)
        lock_results = {}
        failed_to_lock = []
        max_workers = min(8, max(1, len(root_ids)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_root = {
                executor.submit(
                    self.lock_root_indefinitely, root_id, operation_id
                ): root_id
                for root_id in root_ids
            }
            for future in as_completed(future_to_root):
                root_id = future_to_root[future]
                try:
                    lock_results[root_id] = future.result()
                    if lock_results[root_id] is False:
                        failed_to_lock.append(root_id)
                except Exception as e:
                    self.logger.error(f"Failed to lock root {root_id}: {e}")
                    lock_results[root_id] = False
                    failed_to_lock.append(root_id)

        all_locked = all(lock_results.values())
        if all_locked:
            return True, root_ids, failed_to_lock

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            unlock_futures = [
                executor.submit(
                    self.unlock_indefinitely_locked_root, root_id, operation_id
                )
                for root_id in root_ids
            ]
            for future in as_completed(unlock_futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Failed to unlock root: {e}")
        return False, root_ids, failed_to_lock

    def renew_locks(self, root_ids: np.uint64, operation_id: np.uint64) -> bool:
        max_workers = min(8, max(1, len(root_ids)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.renew_lock, root_id, operation_id): root_id
                for root_id in root_ids
            }
            for future in as_completed(futures):
                root_id = futures[future]
                try:
                    result = future.result()
                    if not result:
                        self.logger.warning(f"renew_lock failed - {root_id}")
                        return False
                except Exception as e:
                    self.logger.error(f"Exception during renew_lock({root_id}): {e}")
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
        max_workers = min(8, max(1, len(root_ids)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_lock_timestamp, root_id, op_id): (
                    root_id,
                    op_id,
                )
                for root_id, op_id in zip(root_ids, operation_ids)
            }
            timestamps = []
            for future in as_completed(futures):
                root_id, op_id = futures[future]
                try:
                    ts = future.result()
                    if ts is None:
                        return None
                    timestamps.append(ts)
                except Exception as exc:
                    self.logger.warning(f"({root_id}, {op_id}): {exc}")
                    return None
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
            low, high = self._get_ids_range(
                serialize_uint64(chunk_id, counter=True), size
            )
            low, high = basetypes.SEGMENT_ID.type(low), basetypes.SEGMENT_ID.type(high)
            new_ids = np.arange(low, high + np.uint64(1), dtype=basetypes.SEGMENT_ID)
        return new_ids | chunk_id

    def create_node_id(
        self, chunk_id: np.uint64, root_chunk=False
    ) -> basetypes.NODE_ID:
        return self.create_node_ids(chunk_id, 1, root_chunk=root_chunk)[0]

    def set_max_node_id(
        self, chunk_id: np.uint64, node_id: np.uint64
    ) -> None:
        """Set max segment ID for a given chunk."""
        size = int(np.uint64(chunk_id) ^ np.uint64(node_id))
        self._get_ids_range(serialize_uint64(chunk_id, counter=True), size)

    def get_max_node_id(
        self, chunk_id: basetypes.CHUNK_ID, root_chunk=False
    ) -> basetypes.NODE_ID:
        if root_chunk:
            n_counters = np.uint64(2**8)
            max_value = 0
            for counter in range(n_counters):
                row = self._read_byte_row(
                    serialize_key(f"i{pad_node_id(chunk_id)}_{counter}"),
                    columns=attributes.Concurrency.Counter,
                )
                val = (
                    basetypes.SEGMENT_ID.type(row[0].value if row else 0) * n_counters
                    + counter
                )
                max_value = val if val > max_value else max_value
            return chunk_id | basetypes.SEGMENT_ID.type(max_value)
        column = attributes.Concurrency.Counter
        row = self._read_byte_row(
            serialize_uint64(chunk_id, counter=True), columns=column
        )
        return chunk_id | basetypes.SEGMENT_ID.type(row[0].value if row else 0)

    def create_operation_id(self):
        return self._get_ids_range(attributes.OperationLogs.key, 1)[1]

    def get_max_operation_id(self):
        column = attributes.Concurrency.Counter
        row = self._read_byte_row(attributes.OperationLogs.key, columns=column)
        return row[0].value if row else column.basetype(0)

    def _get_root_segment_ids_range(
        self, chunk_id: basetypes.CHUNK_ID, size: int = 1, counter: int = None
    ) -> np.ndarray:
        n_counters = np.uint64(2**8)
        counter = (
            np.uint64(counter % n_counters)
            if counter
            else np.uint64(np.random.randint(0, n_counters))
        )
        key = serialize_key(f"i{pad_node_id(chunk_id)}_{counter}")
        min_, max_ = self._get_ids_range(key=key, size=size)
        return np.arange(
            min_ * n_counters + counter,
            max_ * n_counters + np.uint64(1) + counter,
            n_counters,
            dtype=basetypes.SEGMENT_ID,
        )


class OperationLogger(ABC):
    """
    Abstract class for interacting with backend data store where the operation logs are stored.
    Eg., BigTableClient can be used to store logs in Google BigTable.
    """

    # read_log_entry and read_log_entries are now concrete in SimpleClient
