"""HBase backend test harness: an in-process mock HBase REST server."""

from contextlib import contextmanager

from ..base import BackendHarness
from .mock_server import start_hbase_mock_server


class _HBaseMockHarness(BackendHarness):
    name = "hbase"

    @contextmanager
    def server(self):
        _data, server, port = start_hbase_mock_server()
        try:
            yield port
        finally:
            server.shutdown()

    def backend_client(self, port) -> dict:
        return {
            "TYPE": "hbase",
            "CONFIG": {
                "BASE_URL": f"http://127.0.0.1:{port}",
                "MAX_ROW_KEY_COUNT": 1000,
            },
        }

    def delete_table(self, graph) -> None:
        resp = graph.client._session.delete(graph.client._table_url("/schema"))
        if resp.status_code not in (200, 404):
            resp.raise_for_status()


harness = _HBaseMockHarness()
