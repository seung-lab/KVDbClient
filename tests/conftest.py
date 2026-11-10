# pylint: disable=missing-docstring, redefined-outer-name
from datetime import timedelta
from typing import NamedTuple

import pytest

from kvdbclient import get_client_class
from kvdbclient import get_config_class
from kvdbclient_testing import backends

from helpers import SMALL_BATCH
from helpers import TEST_META
from helpers import TEST_VERSION
from helpers import new_table_id

_HARNESSES = {harness.name: harness for harness in backends()}


class Backend(NamedTuple):
    name: str
    config: object

    def client(self, create_table=True, max_row_key_count=None, **options):
        config = self.config
        if max_row_key_count is not None:
            config = config._replace(MAX_ROW_KEY_COUNT=max_row_key_count)
        client = get_client_class(self.name)(
            table_id=new_table_id(), config=config, **options
        )
        if create_table:
            client.create_table(meta=TEST_META, version=TEST_VERSION)
        return client


@pytest.fixture(scope="session", params=sorted(_HARNESSES))
def backend(request):
    harness = _HARNESSES[request.param]
    if not harness.available():
        pytest.skip(f"no local {request.param} instance can start here")
    with harness.server() as handle:
        info = harness.backend_client(handle)
        yield Backend(request.param, get_config_class(request.param)(**info["CONFIG"]))


@pytest.fixture
def client(backend):
    return backend.client()


@pytest.fixture
def client_no_table(backend):
    return backend.client(create_table=False)


@pytest.fixture
def client_short_expiry(backend):
    return backend.client(lock_expiry=timedelta(seconds=1))


@pytest.fixture
def client_small_batch(backend):
    return backend.client(max_row_key_count=SMALL_BATCH)
