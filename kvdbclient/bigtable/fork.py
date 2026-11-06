"""The registered Bigtable client: a plain table as before, a fork through its sources."""

import numpy as np
from google.cloud.bigtable.data.mutations import SetCell

from .. import attributes
from .. import exceptions
from ..fork.client import ForkMixin
from ..serializers import serialize_uint64
from . import utils
from .client import Client

# The data-table calls a table upstream of a fork serves; every other call writes.
READS = frozenset({"read_rows", "read_row", "read_rows_stream"})


class _ReadOnlyTable:
    """A table upstream of a fork: the fork reads it and never writes it."""

    def __init__(self, table, table_id: str):
        self._table, self._table_id = table, table_id

    def __getattr__(self, name):
        if name not in READS:
            raise exceptions.ForkTableError(
                f"{self._table_id} is upstream of a fork, which never writes it"
            )
        return getattr(self._table, name)


class ForkClient(ForkMixin, Client):
    """The Bigtable backend's client, opening a fork through its chain of sources."""

    backend_name = "bigtable"

    def _upstream_client(self, table_id: str) -> "ForkClient":
        return ForkClient(
            table_id,
            self._config,
            lock_expiry=self._lock_expiry,
            data_client=self._data_client,
        )

    def _refuse_writes(self) -> None:
        self._table = _ReadOnlyTable(self._table, self._table_id)

    def _put_counter_if_absent(self, key: bytes, value: int) -> None:
        counter = attributes.Concurrency.Counter
        self._table.check_and_mutate_row(
            key,
            predicate=utils.get_has_column_filter(counter),
            # SetCell writes an int as the 8-byte big-endian value increments read.
            false_case_mutations=SetCell(counter.family_id, counter.key, int(value)),
            operation_timeout=30.0,
        )

    def renew_lock(self, root_id: np.uint64, operation_id: np.uint64) -> bool:
        """Renew only a lock this op holds; a fork's inherited root rows start empty."""
        if self.marker is None:
            return super().renew_lock(root_id, operation_id)
        lock = attributes.Concurrency.Lock
        return bool(
            self._table.check_and_mutate_row(
                serialize_uint64(root_id),
                predicate=utils.get_held_lock_renew_filter(lock, operation_id),
                true_case_mutations=self._set_lock_mutation(lock, operation_id),
                operation_timeout=30.0,
            )
        )
