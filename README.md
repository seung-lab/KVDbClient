[![codecov](https://codecov.io/gh/seung-lab/KVDbClient/graph/badge.svg)](https://app.codecov.io/gh/seung-lab/KVDbClient)

A Python client library providing a unified interface for key-value database backends. Currently supports Google Cloud BigTable and Apache HBase.

Built for:

- Node read/write operations with automatic serialization
- Concurrency control via row-level locking
- Atomic unique ID generation
- Operation logging and auditing
- Configurable column families with per-attribute serializers (NumPy arrays, JSON, Pickle, strings)

## Installation

```bash
pip install kvdbclient
pip install kvdbclient[extensions]
```

For development:

```bash
git clone https://github.com/seung-lab/KVDbClient.git
cd KVDbClient
pip install -e .
```

## Usage

```python
from kvdbclient import get_client_class, BigTableConfig

config = BigTableConfig(PROJECT="my-project", INSTANCE="my-instance", ADMIN=True, READ_ONLY=False)
client = get_client_class("bigtable")("my_table", config)
```

The backend is selected by passing `"bigtable"` or `"hbase"` to `get_client_class()`. Alternatively, `get_default_client_info()` reads configuration from environment variables automatically.

## Backends

**Google BigTable** — Uses the `google-cloud-bigtable` SDK. Configure with `BigTableConfig` or set `BIGTABLE_PROJECT` and `BIGTABLE_INSTANCE` environment variables.

**Apache HBase** — Communicates via the HBase REST API using HTTP. Configure with `HBaseConfig` or set the `HBASE_REST_URL` environment variable.

Set `PCG_BACKEND_TYPE` to `bigtable` or `hbase` to control which backend `get_default_client_info()` uses.

## Testing

```bash
pytest
```

## License

MIT
