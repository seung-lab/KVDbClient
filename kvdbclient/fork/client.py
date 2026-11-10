"""Fork reads for any backend's client, mixed in before the backend's own class.

A fork's table holds only what was written after its fork time `TF`; older data is read
upstream, through the chain of source tables its marker names.
"""

import typing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from operator import attrgetter

from .. import attributes
from .. import exceptions
from ..base import Cell
from ..serializers import serialize_uint64
from ..utils import TICK
from ..utils import get_compatible_time_range
from ..utils import get_compatible_time_stamp
from .counters import IdSpace
from .marker import ForkMarker, MarkerState

# Source reads in flight at once per fork client; each overlaps the fork's own read.
SOURCE_WORKERS = 8
T = typing.TypeVar("T")

# A table's identity rows are its own: a fork never reads its source's meta or version.
OWN_ROWS = frozenset({attributes.TableMeta.key, attributes.TableVersion.key})


def _merged(own, upstream):
    """Two reads of one row, as columns or one column's cells, newest first.

    A cell of the fork's own wins a timestamp tie.
    """
    if not upstream:
        return own
    if not own:
        return upstream
    if isinstance(own, dict):
        return {
            column: _merged(own.get(column), upstream.get(column))
            for column in own.keys() | upstream.keys()
        }
    taken = {cell.timestamp for cell in own}
    cells = own + [cell for cell in upstream if cell.timestamp not in taken]
    return sorted(cells, key=attrgetter("timestamp"), reverse=True)


def _merged_rows(
    own: dict[bytes, typing.Any], upstream: dict[bytes, typing.Any]
) -> dict[bytes, typing.Any]:
    """Two reads of many rows, merged row by row, in key order as a plain read returns."""
    return {
        key: _merged(own.get(key), upstream.get(key))
        for key in sorted(own.keys() | upstream.keys())
    }


class Upstream(typing.NamedTuple):
    """One table above a fork, and the time up to which the fork reads it."""

    client: "ForkMixin"
    tf: datetime


class _ReadOnly:
    """A backend's data handle on a table above a fork: reads pass, all else refuses."""

    def __init__(self, handle, reads: frozenset[str], table_id: str):
        self._handle, self._reads, self._table_id = handle, reads, table_id

    def __getattr__(self, name):
        if name not in self._reads:
            raise exceptions.ForkTableError(
                f"{self._table_id} is upstream of a fork, which never writes it"
            )
        return getattr(self._handle, name)


class ForkMixin(ABC):
    """Fork reads and writes over a backend's client, which supplies the hooks below.

    On a plain table every method behaves as the backend's own.
    """

    opens_forks = True
    # The client reading this table's source, and its read pool; None on a plain table.
    _source = None
    _pool = None

    @abstractmethod
    def _upstream_client(self, table_id: str) -> "ForkMixin":
        """A client of this backend for `table_id`, sharing this client's connection."""

    def _refuse_writes(self) -> None:
        """Makes this client refuse every write, as a table above a fork must."""
        handle = getattr(self, self.DATA_HANDLE)
        setattr(
            self, self.DATA_HANDLE, _ReadOnly(handle, self.READ_CALLS, self._table_id)
        )

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
        """Creates a fork's table, meta and marker in one row, attached, op ids seeded.

        `marker.tf` must be a safe time, below every edit that may still write upstream.
        """
        client = cls(table_id, config, lock_expiry=lock_expiry)
        client.marker = marker
        client.create_table(meta, version, column_families)
        client.attach(marker)
        # Op ids start above every upstream table's, so no inherited op shares one.
        client.seed_counter(attributes.OperationLogs.key)
        return client

    def attach(self, marker: ForkMarker) -> None:
        """Routes reads below `TF` to the marker's chain of source tables."""
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
        # Counter keys this fork has seeded above upstream ids, each once.
        self._seeded: set[bytes] = set()

    def close(self):
        if self._source is not None:
            self._pool.shutdown(wait=False)
            self._source.close()
        super().close()

    def _open_fork(self, marker: str) -> None:
        """Attaches the chain a fork's marker names, once."""
        if self._source is None:
            self.attach(ForkMarker.from_json(marker))

    def _meta_row(self, meta) -> dict[attributes._Attribute, typing.Any]:
        """A fork's marker rides in its meta row, so neither lands without the other."""
        row = super()._meta_row(meta)
        if self.marker is not None:
            row[attributes.TableMeta.Fork] = self.marker.to_json()
        return row

    def read_all_rows(self):
        """A plain table's every row; a fork's own table holds too few to answer."""
        if self._source is not None:
            raise exceptions.ForkTableError(
                f"{self._table_id} is a fork; read_all_rows reads one table alone"
            )
        return super().read_all_rows()

    def _delete_meta(self):
        """A fork keeps its meta row: reads take the newest version, so an overwrite
        only writes, and no crash can leave the table without its marker."""
        if self.marker is None:
            super()._delete_meta()

    def _lock_refused(self, row_key: bytes) -> bool:
        """Refused while an inherited op holds the row's indefinite lock upstream.

        Its writes below that row may be half done. Each upstream table answers for its
        own ops, read where this fork sees it.
        """
        if self._source is None:
            return False
        held = self._ask_upstream(lambda up: up.client._holds_inherited(row_key, up.tf))
        return any(held)

    def _ask_upstream(self, ask: typing.Callable[[Upstream], T]) -> list[T]:
        """`ask` of each table above this fork, all at once on the read pool."""
        return list(self._pool.map(ask, self._upstream()))

    def _upstream(self) -> typing.Iterator[Upstream]:
        """Each table above this fork, nearest first, with the time it is read up to."""
        client = self
        for hop in self.marker.chain:
            client = client._source
            yield Upstream(client, hop.tf)

    def _get_ids_range(self, key: bytes, size: int):
        """Allocates from this table's counter, seeded above upstream ids on first use."""
        if self._source is not None and key not in self._seeded:
            self.seed_counter(key)
        return super()._get_ids_range(key, size)

    def seed_counter(self, key: bytes) -> None:
        """Starts the counter at `key` above every id upstream tables minted in its space.

        The write is conditional, so concurrent first allocations keep one seed.
        """
        space = IdSpace.of(key)
        minted = self._ask_upstream(
            lambda up: space.highest(up.client._own_counters(space))
        )
        start = space.start(max(minted, default=0))
        self._put_if_absent(key, attributes.Concurrency.Counter, start)
        self._seeded.add(key)

    def _own_counters(self, space: IdSpace) -> dict[bytes, int]:
        """This table's own counters in `space`, uncapped: a counter keeps one version."""
        keys = [key for key, _ in space.rows]
        rows = self._own_rows(keys, attributes.Concurrency.Counter)
        return {key: int(cells[0].value) for key, cells in rows.items() if cells}

    def _holds_inherited(self, row_key: bytes, tf: datetime) -> bool:
        """Whether an op begun here by `tf` holds `row_key`'s indefinite lock here.

        An edit locks indefinitely after compute, so its lock cell is read uncapped.
        Its log row sits in this same table at its op time, so an id reused in another
        table never matches.
        """
        cap = get_compatible_time_stamp(tf) + TICK
        holders = self._own_cells(row_key, attributes.Concurrency.IndefiniteLock)
        status = attributes.OperationLogs.Status
        return any(
            self._own_cells(serialize_uint64(c.value), status, cap) for c in holders
        )

    def _own_cells(self, row_key: bytes, column, end_time=None) -> list[Cell]:
        """This table's own cells of `column` in `row_key`, never merged with a source."""
        return self._own_rows([row_key], column, end_time).get(row_key, [])

    def _own_rows(
        self, row_keys: list[bytes], column, end_time=None
    ) -> dict[bytes, list[Cell]]:
        """This table's own cells of `column` in each of `row_keys`, never merged."""
        return super()._read_byte_rows(
            row_keys=row_keys, columns=column, end_time=end_time
        )

    def _source_end(
        self,
        start_time: typing.Optional[datetime],
        end_time: typing.Optional[datetime],
        end_inclusive: bool,
    ) -> typing.Optional[datetime]:
        """The source read's exclusive end; None with no source or a window past `TF`."""
        if self._source is None:
            return None
        window = get_compatible_time_range(start_time, end_time, end_inclusive)
        if window.start is not None and window.start > self._tf:
            return None
        cap = self._tf + TICK
        return cap if window.end is None else min(window.end, cap)

    def _merged_read(
        self, read: str, merge, end, end_time, end_inclusive, *args, **query
    ):
        """This table's `read`, merged by `merge` with the source's read up to `end`.

        The source read runs beside this table's; an `end` of None skips it.
        """
        upstream = None
        if end is not None:
            upstream = self._pool.submit(
                getattr(self._source, read),
                *args,
                end_time=end,
                end_time_inclusive=False,
                **query,
            )
        own = getattr(super(), read)(
            *args, end_time=end_time, end_time_inclusive=end_inclusive, **query
        )
        return own if upstream is None else merge(own, upstream.result())

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
        return self._merged_read(
            "_read_byte_rows",
            _merged_rows,
            self._source_end(start_time, end_time, end_time_inclusive),
            end_time,
            end_time_inclusive,
            start_key=start_key,
            end_key=end_key,
            end_key_inclusive=end_key_inclusive,
            row_keys=row_keys,
            columns=columns,
            start_time=start_time,
            user_id=user_id,
        )

    def _read_byte_row(
        self,
        row_key,
        columns=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        end = None
        if row_key not in OWN_ROWS:
            end = self._source_end(start_time, end_time, end_time_inclusive)
        return self._merged_read(
            "_read_byte_row",
            _merged,
            end,
            end_time,
            end_time_inclusive,
            row_key,
            columns=columns,
            start_time=start_time,
        )
