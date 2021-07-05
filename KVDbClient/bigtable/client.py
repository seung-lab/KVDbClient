import typing
from datetime import datetime

from google.cloud.bigtable import Client as BTClient
from google.cloud.bigtable.column_family import MaxVersionsGCRule
from google.cloud.bigtable.row import Row
from google.cloud.bigtable.row import DirectRow
from google.cloud.bigtable.row_set import RowSet
from google.cloud.bigtable.row_data import Cell
from google.cloud.bigtable.row_data import PartialRowData
from google.cloud.bigtable.row_filters import RowFilter
from google.cloud.bigtable.table import Table

from . import utils
from . import BigTableConfig
from . import attributes
from .. import serializers
from ..base import SimpleClient
from ..base import Entry
from ..base import EntryKey


class Client(BTClient, SimpleClient):
    def __init__(
        self,
        table_id: str = None,
        config: BigTableConfig = BigTableConfig(),
    ):
        if config.CREDENTIALS:
            super(Client, self).__init__(
                project=config.PROJECT,
                read_only=config.READ_ONLY,
                admin=config.ADMIN,
                credentials=config.CREDENTIALS,
            )
        else:
            super(Client, self).__init__(
                project=config.PROJECT,
                read_only=config.READ_ONLY,
                admin=config.ADMIN,
            )
        self._instance = self.instance(config.INSTANCE)
        self._table_meta = None
        if table_id is not None:
            self._table = self._instance.table(table_id)

    @property
    def meta(self) -> typing.Any:
        if self._table_meta is None:
            self._table_meta = self.read_metadata()
        return self._table_meta

    @property
    def table_id(self) -> str:
        return self._table.table_id

    def create_table(self, table_id: str, meta: typing.Any = {}) -> None:
        """Initialize a table and store associated meta."""
        self._table = self._instance.table(table_id)
        if self._table.exists():
            ValueError(f"{self._table.table_id} already exists.")
        self._table.create()
        self._table.column_family("0").create()
        self.write_metadata(meta)

    def create_column_family(self, name: str, max_version: int = None) -> None:
        """
        Creates column families for a table.
        Family '0' is created by default to store metadata.
        """
        if max_version is None:
            self._table.column_family(name).create()
            return
        self._table.column_family(name, gc_rule=MaxVersionsGCRule(max_version)).create()

    def write_metadata(self, meta: typing.Any) -> None:
        self._table_meta = meta
        row = self._mutate_row(
            attributes.TableMeta.key,
            {attributes.TableMeta.data: meta},
        )
        self._write([row])

    def read_metadata(self) -> typing.Any:
        row = self._read_byte_row(attributes.TableMeta.key)
        return row[attributes.TableMeta.data][0].value

    def read_entries(
        self,
        start_key=None,
        end_key=None,
        end_key_inclusive=False,
        keys=None,
        attributes=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
        key_serializer: serializers.Serializer = serializers.UInt64String(),
    ):
        """
        Read entries and their attributes.
        Accepts a range of keys or specific keys.
        Custom serializers can be used, `serializers.UInt64String()` is the default.
        """
        rows = self._read_byte_rows(
            start_key=key_serializer.serialize(start_key)
            if start_key is not None
            else None,
            end_key=key_serializer.serialize(end_key) if end_key is not None else None,
            end_key_inclusive=end_key_inclusive,
            row_keys=(key_serializer.serialize(key) for key in keys)
            if keys is not None
            else None,
            columns=attributes,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return {key_serializer.deserialize(key): data for (key, data) in rows.items()}

    def read_entry(
        self,
        key,
        attributes=None,
        start_time=None,
        end_time=None,
        end_time_inclusive: bool = False,
        key_serializer: serializers.Serializer = serializers.UInt64String(),
    ) -> typing.Union[
        typing.Dict[attributes.Attribute, typing.List[Cell]],
        typing.List[Cell],
    ]:
        """Convenience function for reading a single node from Bigtable."""
        return self._read_byte_row(
            row_key=key_serializer.serialize(key),
            columns=attributes,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )

    def write_entries(
        self,
        entries: typing.Iterable[Entry],
    ):
        """Writes/updates entries with attributes."""
        rows = []
        for entry in entries:
            rows.append(self._mutate_row(entry.key, entry.values(), entry.timestamp))
        self._write(rows)

    # PRIVATE METHODS
    def _write(
        self,
        rows: typing.Iterable[DirectRow],
        slow_retry: bool = True,
        block_size: int = 2000,
    ):
        """Writes a list of mutated rows in bulk."""
        from os import environ
        from google.api_core.retry import Retry
        from google.api_core.retry import if_exception_type
        from google.api_core.exceptions import Aborted
        from google.api_core.exceptions import DeadlineExceeded
        from google.api_core.exceptions import ServiceUnavailable

        initial = 1
        if slow_retry:
            initial = 5

        exception_types = (Aborted, DeadlineExceeded, ServiceUnavailable)
        retry = Retry(
            predicate=if_exception_type(exception_types),
            initial=initial,
            maximum=15.0,
            multiplier=2.0,
            deadline=float(environ.get("BIGTABLE_WRITE_RETRY_DEADLINE", 180.0)),
        )

        for i in range(0, len(rows), block_size):
            status = self._table.mutate_rows(rows[i : i + block_size], retry=retry)
            if not all(status):
                raise IOError("Bulk write failed.")

    def _mutate_row(
        self,
        row_key: bytes,
        val_dict: typing.Dict[attributes.Attribute, typing.Any],
        time_stamp: typing.Optional[datetime] = None,
    ) -> Row:
        """Mutates a single row (doesn't write to big table)."""
        row = self._table.direct_row(row_key)
        for attr, value in val_dict.items():
            row.set_cell(
                column_family_id=attr.family_id,
                column=attr.key,
                value=attr.serialize(value),
                timestamp=time_stamp,
            )
        return row

    def _read_byte_rows(
        self,
        start_key: typing.Optional[bytes] = None,
        end_key: typing.Optional[bytes] = None,
        end_key_inclusive: bool = False,
        row_keys: typing.Optional[typing.Iterable[bytes]] = None,
        columns: typing.Optional[
            typing.Union[typing.Iterable[attributes.Attribute], attributes.Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ) -> typing.Dict[
        bytes,
        typing.Union[
            typing.Dict[attributes.Attribute, typing.List[Cell]],
            typing.List[Cell],
        ],
    ]:
        """Main function for reading a row range or non-contiguous row sets."""
        # Create filters: Rows
        row_set = RowSet()
        if row_keys is not None:
            for row_key in row_keys:
                row_set.add_row_key(row_key)
        elif start_key is not None and end_key is not None:
            row_set.add_row_range_from_keys(
                start_key=start_key,
                start_inclusive=True,
                end_key=end_key,
                end_inclusive=end_key_inclusive,
            )
        else:
            raise ValueError("Invalid row keys.")
        filter_ = utils.get_time_range_and_column_filter(
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_inclusive=end_time_inclusive,
        )
        # Bigtable read with retries
        rows = self._read(row_set=row_set, row_filter=filter_)

        # Deserialize cells
        for row_key, column_dict in rows.items():
            for column, cell_entries in column_dict.items():
                for cell_entry in cell_entries:
                    cell_entry.value = column.deserialize(cell_entry.value)
            _column_dict = {}
            for column, cell_entries in column_dict.items():
                _column_dict[column.key] = column_dict[column]
            rows[row_key] = _column_dict
            # If no column array was requested, reattach single column's values directly to the row
            if isinstance(columns, attributes.Attribute):
                rows[row_key] = cell_entries
        return rows

    def _read_byte_row(
        self,
        row_key: bytes,
        columns: typing.Optional[
            typing.Union[typing.Iterable[attributes.Attribute], attributes.Attribute]
        ] = None,
        start_time: typing.Optional[datetime] = None,
        end_time: typing.Optional[datetime] = None,
        end_time_inclusive: bool = False,
    ) -> typing.Union[
        typing.Dict[attributes.Attribute, typing.List[Cell]],
        typing.List[Cell],
    ]:
        """Convenience function for reading a single row."""
        row = self._read_byte_rows(
            row_keys=[row_key],
            columns=columns,
            start_time=start_time,
            end_time=end_time,
            end_time_inclusive=end_time_inclusive,
        )
        return (
            row.get(row_key, [])
            if isinstance(columns, attributes.Attribute)
            else row.get(row_key, {})
        )

    def _execute_read_thread(self, args: typing.Tuple[Table, RowSet, RowFilter]):
        table, row_set, row_filter = args
        if not row_set.row_keys and not row_set.row_ranges:
            # Check for everything falsy, because Bigtable considers even empty
            # lists of row_keys as no upper/lower bound!
            return {}
        range_read = table.read_rows(row_set=row_set, filter_=row_filter)
        res = {v.row_key: utils.partial_row_data_to_column_dict(v) for v in range_read}
        return res

    def _read(
        self, row_set: RowSet, row_filter: RowFilter = None
    ) -> typing.Dict[bytes, typing.Dict[attributes.Attribute, PartialRowData]]:
        """Core function to read rows from Bigtable. Uses standard Bigtable retry logic."""
        # FIXME: Bigtable limits the length of the serialized request to 512 KiB. We should
        # calculate this properly (range_read.request.SerializeToString()), but this estimate is
        # good enough for now
        from numpy import ceil
        from multiwrapper.multiprocessing_utils import n_cpus
        from multiwrapper.multiprocessing_utils import multithread_func

        max_row_key_count = 1000
        n_subrequests = max(1, int(ceil(len(row_set.row_keys) / max_row_key_count)))
        n_threads = min(n_subrequests, 2 * n_cpus)

        row_sets = []
        for i in range(n_subrequests):
            r = RowSet()
            r.row_keys = row_set.row_keys[
                i * max_row_key_count : (i + 1) * max_row_key_count
            ]
            row_sets.append(r)

        # Don't forget the original RowSet's row_ranges
        row_sets[0].row_ranges = row_set.row_ranges
        responses = multithread_func(
            self._execute_read_thread,
            params=((self._table, r, row_filter) for r in row_sets),
            debug=n_threads == 1,
            n_threads=n_threads,
        )

        combined_response = {}
        for resp in responses:
            combined_response.update(resp)
        return combined_response
