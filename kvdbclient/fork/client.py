"""A bigtable client that reads a fork's table merged with its source's, capped at `TF`.

A fork holds only what was written after its fork point `TF`; older data is read upstream.
"""

import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from operator import attrgetter

from .. import attributes
from .. import exceptions
from ..bigtable import BigTableConfig
from ..bigtable.client import Client
from ..utils import get_compatible_time_stamp
from .marker import ForkMarker, MarkerState

# Source reads in flight at once per fork client; each overlaps the fork's own read.
SOURCE_WORKERS = 8

# A table's identity rows are its own: a fork never reads its source's meta or version.
OWN_ROWS = frozenset({attributes.TableMeta.key, attributes.TableVersion.key})

# The data-table calls an attached fork may make; every other call writes.
READS = frozenset({"read_rows", "read_row", "read_rows_stream"})


class _ReadOnlyTable:
    """A fork's data table, refusing every write until fork edits are supported."""

    def __init__(self, table, table_id: str):
        self._table, self._table_id = table, table_id

    def __getattr__(self, name):
        if name not in READS:
            raise exceptions.ForkTableError(
                f"fork {self._table_id} is read-only: fork edits are not supported yet"
            )
        return getattr(self._table, name)


def _merged(own, upstream):
    """Two reads' rows, columns or cells, newest first; a fork cell wins a tie."""
    if not upstream:
        return own
    if not own:
        return upstream
    if isinstance(own, dict):
        return {
            k: _merged(own.get(k), upstream.get(k))
            for k in own.keys() | upstream.keys()
        }
    taken = {cell.timestamp for cell in own}
    cells = own + [cell for cell in upstream if cell.timestamp not in taken]
    return sorted(cells, key=attrgetter("timestamp"), reverse=True)


class ForkClient(Client):
    """The plain client, reading through the fork chain when its table has a marker."""

    opens_forks = True

    def __init__(
        self,
        table_id: str,
        config: BigTableConfig = BigTableConfig(),
        table_meta=None,
        lock_expiry: timedelta = timedelta(minutes=3),
        data_client=None,
    ):
        super().__init__(
            table_id, config, table_meta, lock_expiry, data_client=data_client
        )
        self.marker = None
        self._source = None
        self._pool = None

    def attach(self, marker: ForkMarker) -> None:
        """Read below `TF` from the marker's source, a fork itself while chain lasts."""
        if marker.state is not MarkerState.READY:
            raise exceptions.ForkTableError(
                f"fork {self._table_id} is {marker.state.value}"
            )
        self.marker, self._tf = marker, get_compatible_time_stamp(marker.tf)
        self._source = ForkClient(
            marker.chain[0].table,
            self._config,
            lock_expiry=self._lock_expiry,
            data_client=self._data_client,
        )
        if len(marker.chain) > 1:
            self._source.attach(replace(marker, chain=marker.chain[1:]))
        self._pool = ThreadPoolExecutor(max_workers=SOURCE_WORKERS)
        self._table = _ReadOnlyTable(self._table, self._table_id)

    def close(self):
        if self._source is not None:
            self._pool.shutdown(wait=False)
            self._source.close()
        super().close()

    def read_table_meta(self):
        """Read the meta row once, attaching the fork chain its marker names."""
        row = self._read_byte_row(attributes.TableMeta.key)
        meta, fork = row.get(attributes.TableMeta.Meta), row.get(
            attributes.TableMeta.Fork
        )
        self._table_meta = meta[0].value if meta else None
        if fork and self._source is None:
            self.attach(ForkMarker.from_json(fork[0].value))
        return self._table_meta

    def update_table_meta(self, meta, overwrite: typing.Optional[bool] = False):
        """On a fork, write meta and marker in one mutation, so neither lands alone."""
        if self.marker is None:
            return super().update_table_meta(meta, overwrite)
        if overwrite:
            self._delete_meta()
        self._table_meta = meta
        values = {
            attributes.TableMeta.Meta: meta,
            attributes.TableMeta.Fork: self.marker.to_json(),
        }
        self.write([self.mutate_row(attributes.TableMeta.key, values)])

    def _source_end(
        self,
        start_time: typing.Optional[datetime],
        end_time: typing.Optional[datetime],
        end_inclusive: bool,
    ) -> typing.Optional[datetime]:
        """The source read's exclusive end; None with no source or a window past `TF`."""
        if self._source is None:
            return None
        if start_time is not None and get_compatible_time_stamp(start_time) > self._tf:
            return None
        cap = self._tf + timedelta(milliseconds=1)
        if end_time is None:
            return cap
        return min(get_compatible_time_stamp(end_time, round_up=end_inclusive), cap)

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
        query = dict(
            start_key=start_key,
            end_key=end_key,
            end_key_inclusive=end_key_inclusive,
            row_keys=row_keys,
            columns=columns,
            start_time=start_time,
            user_id=user_id,
        )
        end = self._source_end(start_time, end_time, end_time_inclusive)
        upstream = (
            None
            if end is None
            else self._pool.submit(
                self._source._read_byte_rows,
                end_time=end,
                end_time_inclusive=False,
                **query,
            )
        )
        own = super()._read_byte_rows(
            end_time=end_time, end_time_inclusive=end_time_inclusive, **query
        )
        return own if upstream is None else _merged(own, upstream.result())

    def _read_byte_row(
        self,
        row_key,
        columns=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        query = dict(columns=columns, start_time=start_time)
        end = None
        if row_key not in OWN_ROWS:
            end = self._source_end(start_time, end_time, end_time_inclusive)
        upstream = (
            None
            if end is None
            else self._pool.submit(
                self._source._read_byte_row,
                row_key,
                end_time=end,
                end_time_inclusive=False,
                **query,
            )
        )
        own = super()._read_byte_row(
            row_key, end_time=end_time, end_time_inclusive=end_time_inclusive, **query
        )
        return own if upstream is None else _merged(own, upstream.result())


def open_client(
    table_id: str,
    config: BigTableConfig = BigTableConfig(),
    lock_expiry: timedelta = timedelta(minutes=3),
) -> ForkClient:
    """A client for `table_id`, reading through its fork chain when it is a fork."""
    client = ForkClient(table_id, config, lock_expiry=lock_expiry)
    client.read_table_meta()
    return client


def create_fork_table(
    table_id: str,
    marker: ForkMarker,
    meta,
    version: str,
    config: BigTableConfig = BigTableConfig(),
    column_families=None,
    lock_expiry: timedelta = timedelta(minutes=3),
) -> ForkClient:
    """Create a fork's table, meta and marker in one mutation, then attach read-only."""
    client = ForkClient(table_id, config, lock_expiry=lock_expiry)
    client.marker = marker
    client.create_table(meta, version, column_families)
    client.attach(marker)
    return client
