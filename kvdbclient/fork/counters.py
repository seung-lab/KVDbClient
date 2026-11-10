"""Which counter rows mint into one id space, so a fork can allocate above all of them."""

from typing import NamedTuple, Optional

from .. import attributes
from ..base import COUNTER_SHARDS
from ..base import counter_key
from ..base import counter_row
from ..base import counter_shard_key
from ..base import shard_segment_id


class IdSpace(NamedTuple):
    """Every counter row minting operation ids, or one chunk's node ids.

    A chunk counts in one row, in `COUNTER_SHARDS` shard rows, or in both once sharded.
    """

    #: (row key, shard) per counter row; shard is None for a row counting ids itself.
    rows: tuple[tuple[bytes, Optional[int]], ...]
    #: Allocating row's shard; None when it counts ids directly.
    shard: Optional[int]

    @classmethod
    def of(cls, key: bytes) -> "IdSpace":
        """The id space the counter row `key` allocates in."""
        if key == attributes.OperationLogs.key:
            return cls(((key, None),), None)
        row = counter_row(key)
        shards = tuple(
            (counter_shard_key(row.chunk_id, s), s) for s in range(COUNTER_SHARDS)
        )
        return cls(((counter_key(row.chunk_id), None),) + shards, row.shard)

    def highest(self, counters: dict[bytes, int]) -> int:
        """The highest id these rows have minted, from each row's counter; 0 for none."""
        minted = (
            value if shard is None else int(shard_segment_id(value, shard))
            for key, shard in self.rows
            if (value := counters.get(key)) is not None
        )
        return max(minted, default=0)

    def start(self, highest: int) -> int:
        """The allocating row's counter value from which it mints only above `highest`."""
        return highest if self.shard is None else highest // COUNTER_SHARDS
