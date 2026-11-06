"""Fork behaviour for any backend's client, mixed in before the backend's own class.

A fork holds only what was written after its fork point `TF`; older data is read upstream.
Every write lands in the fork's own table, and each table upstream refuses writes.
"""

import typing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from operator import attrgetter

from .. import attributes
from .. import exceptions
from ..serializers import serialize_uint64
from ..utils import get_compatible_time_stamp
from .counters import IdSpace
from .marker import ForkMarker, MarkerState

# Source reads in flight at once per fork client; each overlaps the fork's own read.
SOURCE_WORKERS = 8

# A table's identity rows are its own: a fork never reads its source's meta or version.
OWN_ROWS = frozenset({attributes.TableMeta.key, attributes.TableVersion.key})


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


class ForkMixin(ABC):
    """Fork reads and writes over a backend's client, which supplies the hooks below.

    On a plain table every method behaves as the backend's own.
    """

    opens_forks = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source = None
        self._pool = None
        self._seeded = set()

    @abstractmethod
    def _upstream_client(self, table_id: str) -> "ForkMixin":
        """A client of this backend for `table_id`, sharing this client's connection."""

    @abstractmethod
    def _refuse_writes(self) -> None:
        """Make this client refuse every write, as a table upstream of a fork must."""

    @abstractmethod
    def _put_counter_if_absent(self, key: bytes, value: int) -> None:
        """Set the id counter at `key` to `value` unless it holds one, atomically."""

    @classmethod
    def create_fork_table(
        cls,
        table_id: str,
        marker: ForkMarker,
        meta,
        version: str,
        config,
        column_families=None,
        lock_expiry: timedelta = timedelta(minutes=3),
    ) -> "ForkMixin":
        """Create a fork's table, meta and marker, attach it, and seed its op ids."""
        client = cls(table_id, config, lock_expiry=lock_expiry)
        client.marker = marker
        client.create_table(meta, version, column_families)
        client.attach(marker)
        # Op ids start above upstream's, so an op-log scan covers every inherited op.
        client.seed_counter(attributes.OperationLogs.key)
        return client

    def attach(self, marker: ForkMarker) -> None:
        """Read below `TF` from the marker's source, a fork itself while chain lasts."""
        if marker.state is not MarkerState.READY:
            raise exceptions.ForkTableError(
                f"fork {self._table_id} is {marker.state.value}"
            )
        self.marker, self._tf = marker, get_compatible_time_stamp(marker.tf)
        self._source = self._upstream_client(marker.chain[0].table)
        if len(marker.chain) > 1:
            self._source.attach(replace(marker, chain=marker.chain[1:]))
        self._source._refuse_writes()
        self._pool = ThreadPoolExecutor(max_workers=SOURCE_WORKERS)

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

    def _get_ids_range(self, key: bytes, size: int):
        """Allocate from this table's counter, seeded above upstream ids on first use."""
        if self._source is not None and key not in self._seeded:
            self.seed_counter(key)
        return super()._get_ids_range(key, size)

    def seed_counter(self, key: bytes) -> None:
        """Start the counter at `key` above every id upstream tables minted in its space.

        The write is conditional, so concurrent first allocations keep one seed.
        """
        space = IdSpace.of(key)
        highest = max(
            (space.highest(client._own_counters(space)) for client in self._upstream()),
            default=0,
        )
        self._put_counter_if_absent(key, space.start(highest))
        self._seeded.add(key)

    def _own_counters(self, space: IdSpace) -> dict[bytes, int]:
        """This table's own counters in `space`, unmerged and uncapped.

        A counter keeps one version, so a read capped at `TF` can miss it entirely.
        """
        rows = super()._read_byte_rows(
            row_keys=[key for key, _ in space.rows],
            columns=attributes.Concurrency.Counter,
        )
        return {key: int(cells[0].value) for key, cells in rows.items() if cells}

    def _upstream(self) -> typing.Iterator["ForkMixin"]:
        """Each table upstream of this one, direct source first."""
        client = self._source
        while client is not None:
            yield client
            client = client._source

    def lock_root(self, root_id, operation_id) -> bool:
        """Refused while an upstream op left the root indefinitely locked at `TF`."""
        if self._locked_upstream(serialize_uint64(root_id)):
            return False
        return super().lock_root(root_id, operation_id)

    def lock_root_indefinitely(self, root_id, operation_id) -> bool:
        """Refused while an upstream op left the root indefinitely locked at `TF`."""
        if self._locked_upstream(serialize_uint64(root_id)):
            return False
        return super().lock_root_indefinitely(root_id, operation_id)

    def lock_by_row_key_with_indefinite(self, row_key, operation_id) -> bool:
        """Refused while an upstream op left the row indefinitely locked at `TF`."""
        if self._locked_upstream(row_key):
            return False
        return super().lock_by_row_key_with_indefinite(row_key, operation_id)

    def _locked_upstream(self, row_key: bytes) -> bool:
        """Whether an upstream op that never finished holds `row_key` at `TF`.

        Its writes below that row may be half done, so a fork must not edit there.
        """
        if self._source is None:
            return False
        return bool(
            self._source._read_byte_row(
                row_key,
                columns=attributes.Concurrency.IndefiniteLock,
                end_time=self._source_end(None, None, False),
            )
        )
