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
from ..utils import TICK
from ..utils import get_compatible_time_range
from ..utils import get_compatible_time_stamp
from .marker import ForkMarker, MarkerState

# Source reads in flight at once per fork client; each overlaps the fork's own read.
SOURCE_WORKERS = 8

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


class ForkMixin(ABC):
    """Fork reads over a backend's client, which supplies the hook below.

    On a plain table every method behaves as the backend's own.
    """

    opens_forks = True
    # The client reading this table's source, and its read pool; None on a plain table.
    _source = None
    _pool = None

    @abstractmethod
    def _upstream_client(self, table_id: str) -> "ForkMixin":
        """A client of this backend for `table_id`, sharing this client's connection."""

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
        """Creates a fork's table, its meta and marker in one row, then attaches it."""
        client = cls(table_id, config, lock_expiry=lock_expiry)
        client.marker = marker
        client.create_table(meta, version, column_families)
        client.attach(marker)
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
        self._pool = ThreadPoolExecutor(max_workers=SOURCE_WORKERS)

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
