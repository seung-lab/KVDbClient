import uuid
from datetime import timedelta

import pytest

from kvdbclient.bigtable import BigTableConfig
from kvdbclient.bigtable.client import Client
from kvdbclient.hbase import HBaseConfig
from kvdbclient.hbase.client import Client as HBaseClient
from kvdbclient_testing.bigtable.harness import bigtable_emulator as _bigtable_emulator
from kvdbclient_testing.hbase.mock_server import start_hbase_mock_server


EMULATOR_PROJECT = "test-project"
EMULATOR_INSTANCE = "test-instance"


# ── BigTable fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def bigtable_emulator():
    """Start the BigTable emulator or reuse one already running (CI)."""
    with _bigtable_emulator() as host_port:
        yield host_port


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
