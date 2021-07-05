from collections import namedtuple

from .bigtable.client import Client as BigTableClient
from .bigtable.attributes import Attribute
from .serializers import Serializer
from .serializers import UInt64String


_backend_clientinfo_fields = ("TYPE", "CONFIG")
_backend_clientinfo_defaults = (None, None)
BackendClientInfo = namedtuple(
    "BackendClientInfo",
    _backend_clientinfo_fields,
    defaults=_backend_clientinfo_defaults,
)


def get_default_client_info():
    """
    Load client from env variables.
    """

    # TODO make dynamic after multiple platform support is added
    from .bigtable import get_client_config as get_bigtable_client_config

    return BackendClientInfo(
        CONFIG=get_bigtable_client_config(admin=True, read_only=False)
    )
