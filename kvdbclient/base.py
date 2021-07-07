from abc import ABC
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable
from typing import Iterable
from typing import Optional
from datetime import datetime

from .serializers import Serializer
from .serializers import UInt64String


class SimpleClient(ABC):
    from abc import abstractmethod

    """
    Abstract class for interacting with backend data storage system.
    Eg., BigTableClient for using big table as storage.
    """

    @abstractmethod
    def create_table(self) -> None:
        """Initialize the table and store associated meta."""

    @abstractmethod
    def write_metadata(self, metadata):
        """Update stored metadata."""

    @abstractmethod
    def read_metadata(self):
        """Read stored metadata."""

    @abstractmethod
    def read_entries(
        self,
        entry_ids,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read entries and their properties."""

    @abstractmethod
    def read_entry(
        self,
        entry_id,
        properties=None,
        start_time=None,
        end_time=None,
        end_time_inclusive=False,
    ):
        """Read a single entry and it's properties."""

    @abstractmethod
    def write_entries(self, entries):
        """Writes/updates entries (IDs along with properties)."""


class EntryKey:
    def __init__(self, key: Any, serializer: Optional[Serializer] = UInt64String()):
        self._key = key
        self._serializer = serializer.serialize

    def serialize(self) -> Any:
        return self._serializer(self._key)


class Entry:
    """
    Represents a single entry/record/entity in the database.
    """

    def __init__(
        self,
        key: EntryKey,
        val_dict: Dict[Any, Any],
        timestamp: Optional[datetime] = None,
    ):
        self._key = key
        self._val_dict = val_dict
        self._timestamp = timestamp

    @property
    def key(self) -> Any:
        return self._key.serialize()

    @property
    def values(self) -> Dict:
        return self._val_dict

    @property
    def timestamp(self):
        return self._timestamp
