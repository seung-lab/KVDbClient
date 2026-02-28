from collections import namedtuple
from os import environ

DEFAULT_REST_URL = "http://localhost:8080"

_hbaseconfig_fields = (
    "BASE_URL",
    "MAX_ROW_KEY_COUNT",
)
_hbaseconfig_defaults = (
    environ.get("HBASE_REST_URL", DEFAULT_REST_URL),
    1000,
)
HBaseConfig = namedtuple(
    "HBaseConfig", _hbaseconfig_fields, defaults=_hbaseconfig_defaults
)


def get_client_info(
    base_url: str = None,
):
    """Helper function to load config from env."""
    _base_url = environ.get("HBASE_REST_URL", DEFAULT_REST_URL)
    if base_url:
        _base_url = base_url

    return HBaseConfig(BASE_URL=_base_url)
