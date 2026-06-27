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

## Release

Published to PyPI by a one-click workflow — no manual tag or version edit.

- **Release:** Actions → **CI** → **Run workflow** → choose `part` (`major`/`minor`/`patch`), or
  `gh workflow run test.yml -f part=patch`. It computes the next version from the latest `vX.Y.Z`
  tag, tags it, builds, and publishes to PyPI (OIDC trusted publishing).
- **Preview:** `dry-run=true` prints the next version without tagging or publishing.

The version is derived from the git tag by `setuptools_scm`; there is no version literal to bump.

## License

MIT
