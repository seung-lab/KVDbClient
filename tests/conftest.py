import os
import signal
import socket
import subprocess
import time
import uuid
from datetime import timedelta

import pytest

from kvdbclient.bigtable import BigTableConfig
from kvdbclient.bigtable.client import Client
from kvdbclient.hbase import HBaseConfig
from kvdbclient.hbase.client import Client as HBaseClient
from hbase_mock_server import start_hbase_mock_server


EMULATOR_PROJECT = "test-project"
EMULATOR_INSTANCE = "test-instance"


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host, port, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Emulator on {host}:{port} not ready within {timeout}s")


# ── BigTable fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def bigtable_emulator():
    """Start the BigTable emulator or use one already running (CI)."""
    existing = os.environ.get("BIGTABLE_EMULATOR_HOST")
    if existing:
        host, port = existing.rsplit(":", 1)
        _wait_for_port(host or "localhost", int(port))
        yield existing
        return

    port = _find_free_port()
    host_port = f"localhost:{port}"
    proc = subprocess.Popen(
        [
            "gcloud", "beta", "emulators", "bigtable", "start",
            f"--host-port={host_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.environ["BIGTABLE_EMULATOR_HOST"] = host_port
    _wait_for_port("localhost", port)
    yield host_port

    os.kill(proc.pid, signal.SIGTERM)
    proc.wait(timeout=10)
    os.environ.pop("BIGTABLE_EMULATOR_HOST", None)


@pytest.fixture(scope="session")
def bt_config(bigtable_emulator):
    return BigTableConfig(
        PROJECT=EMULATOR_PROJECT,
        INSTANCE=EMULATOR_INSTANCE,
        ADMIN=True,
        READ_ONLY=False,
        CREDENTIALS=None,
    )


@pytest.fixture()
def bt_client(bt_config):
    """Client with a fresh table already created."""
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = Client(table_id=table_id, config=bt_config)
    client.create_table(meta={"test": True}, version="0.0.1")
    yield client


@pytest.fixture()
def bt_client_no_table(bt_config):
    """Client bound to a table that does not yet exist."""
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = Client(table_id=table_id, config=bt_config)
    yield client


@pytest.fixture()
def bt_client_small_batch(bigtable_emulator):
    """Client with small MAX_ROW_KEY_COUNT to trigger sharded reads."""
    config = BigTableConfig(
        PROJECT=EMULATOR_PROJECT,
        INSTANCE=EMULATOR_INSTANCE,
        ADMIN=True,
        READ_ONLY=False,
        CREDENTIALS=None,
        MAX_ROW_KEY_COUNT=50,
    )
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = Client(table_id=table_id, config=config)
    client.create_table(meta={"test": True}, version="0.0.1")
    yield client


# ── HBase fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def hbase_server():
    _data, server, port = start_hbase_mock_server()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="session")
def hbase_config(hbase_server):
    return HBaseConfig(BASE_URL=hbase_server)


@pytest.fixture()
def hbase_client(hbase_config):
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = HBaseClient(table_id=table_id, config=hbase_config)
    client.create_table(meta={"test": True}, version="0.0.1")
    yield client


@pytest.fixture()
def hbase_client_no_table(hbase_config):
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = HBaseClient(table_id=table_id, config=hbase_config)
    yield client


@pytest.fixture()
def hbase_client_short_expiry(hbase_config):
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = HBaseClient(table_id=table_id, config=hbase_config, lock_expiry=timedelta(seconds=1))
    client.create_table(meta={"test": True}, version="0.0.1")
    yield client


@pytest.fixture()
def bt_client_short_expiry(bt_config):
    table_id = f"test_{uuid.uuid4().hex[:12]}"
    client = Client(table_id=table_id, config=bt_config, lock_expiry=timedelta(seconds=1))
    client.create_table(meta={"test": True}, version="0.0.1")
    yield client
