"""Bigtable's fork client: a plain table as before, a fork read through its sources."""

from ..fork.client import ForkMixin
from .client import Client


class ForkClient(ForkMixin, Client):
    """The registered Bigtable client, reading a fork through its source tables."""

    backend_name = "bigtable"

    def _upstream_client(self, table_id: str) -> "ForkClient":
        return ForkClient(
            table_id,
            self._config,
            lock_expiry=self._lock_expiry,
            data_client=self._data_client,
        )
