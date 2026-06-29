"""Backend test-harness contract: start a local instance, give its client config, tear down."""

from contextlib import contextmanager


class BackendHarness:
    """Per-backend test harness.

    Subclasses set ``name`` and implement ``server`` (a session-scoped context manager
    that starts a local instance and yields a handle), ``backend_client`` (the
    ``{"TYPE", "CONFIG"}`` dict for a ChunkedGraph pointed at that handle), and
    ``delete_table`` (drop a test table). ``available`` reports whether the harness can
    run in the current environment.
    """

    name = ""

    def available(self) -> bool:
        return True

    @contextmanager
    def server(self):
        raise NotImplementedError
        yield  # pragma: no cover

    def backend_client(self, handle) -> dict:
        raise NotImplementedError

    def delete_table(self, graph) -> None:
        raise NotImplementedError
