"""Bigtable backend test harness: reuse a provided emulator, else start a local `gcloud` one."""

import os
import signal
import socket
import subprocess
from contextlib import contextmanager
from shutil import which
from time import monotonic, sleep

from ..base import BackendHarness

_PROJECT = "IGNORE_ENVIRONMENT_PROJECT"
_INSTANCE = "emulated_instance"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            sleep(0.2)
    raise RuntimeError(f"bigtable emulator on {host}:{port} not ready within {timeout}s")


@contextmanager
def bigtable_emulator():
    """Yield a ``BIGTABLE_EMULATOR_HOST``: reuse one if already set, else spawn a local gcloud one."""
    existing = os.environ.get("BIGTABLE_EMULATOR_HOST")
    if existing:
        host, _, port = existing.rpartition(":")
        _wait_for_port(host or "localhost", int(port))
        yield existing
        return

    port = _free_port()
    host_port = f"localhost:{port}"
    proc = subprocess.Popen(
        ["gcloud", "beta", "emulators", "bigtable", "start", f"--host-port={host_port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.environ["BIGTABLE_EMULATOR_HOST"] = host_port
    try:
        _wait_for_port("localhost", port)
        yield host_port
    finally:
        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        os.environ.pop("BIGTABLE_EMULATOR_HOST", None)


class _BigtableEmulatorHarness(BackendHarness):
    name = "bigtable"

    def available(self) -> bool:
        return bool(os.environ.get("BIGTABLE_EMULATOR_HOST")) or which("gcloud") is not None

    @contextmanager
    def server(self):
        with bigtable_emulator() as host_port:
            yield host_port

    def backend_client(self, handle) -> dict:
        from google.auth import credentials

        return {
            "TYPE": "bigtable",
            "CONFIG": {
                "ADMIN": True,
                "READ_ONLY": False,
                "PROJECT": _PROJECT,
                "INSTANCE": _INSTANCE,
                "CREDENTIALS": credentials.AnonymousCredentials(),
                "MAX_ROW_KEY_COUNT": 1000,
            },
        }

    def delete_table(self, graph) -> None:
        graph.client._admin_table.delete()


harness = _BigtableEmulatorHarness()
