"""
Sub packages/modules for backend storage clients.
Supports Google BigTable and Apache HBase.

A simple client needs to be able to create the table,
store table meta and to write and read node information.
Also needs locking support to prevent race conditions
when modifying root/parent nodes.

In addition, clients with more features like generating unique IDs
and logging facilities can be implemented by inherting respective base classes.

These methods are in separate classes because they are logically related.
This also makes it possible to have different backend storage solutions,
making it possible to use any unique features these solutions may provide.

Please see `base.py` for more details.
"""

import warnings
from collections import namedtuple
from os import environ
from typing import Union

warnings.filterwarnings("ignore", message="urllib3", module="requests")

from .base import ColumnFamilyConfig, DEFAULT_COLUMN_FAMILIES
from .bigtable import BigTableConfig
from .bigtable.client import Client as BigTableClient
from .hbase import HBaseConfig
from .hbase.client import Client as HBaseClient
from .registry import available_backends, resolve_client

ClientType = Union[BigTableClient, HBaseClient]


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = ("bigtable", None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)


def get_client_class(backend_type: str = "bigtable"):
    """Return the client class for the given backend type."""
    return resolve_client(backend_type)


def get_config_class(backend_type: str = "bigtable"):
    """Return the config class for the given backend type."""
    return resolve_client(backend_type).config_class


def get_default_client_info():
    """Load client info for the env-selected backend."""
    name = environ.get("PCG_BACKEND_TYPE", "bigtable").lower()
    if name not in available_backends():
        name = "bigtable"
    return BackendClientInfo(TYPE=name, CONFIG=resolve_client(name).default_client_info())
