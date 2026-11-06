"""Which counter rows mint into one id space, so a fork can allocate above all of them."""

from typing import NamedTuple, Optional

import numpy as np

from .. import attributes
from ..base import COUNTER_SHARDS, counter_shard_key
from ..serializers import serialize_uint64


class IdSpace(NamedTuple):
    """Every counter row minting operation ids, or one chunk's node ids.

    A chunk counts in one row, in `COUNTER_SHARDS` shard rows, or in both once sharded.
    """

    # (row key, shard) per counter row; shard is None for a row counting ids directly.
    rows: tuple
    # The allocating row's shard, None when it counts ids directly.
    shard: Optional[int]

    @classmethod
    def of(cls, key: bytes) -> "IdSpace":
        """The id space the counter row `key` allocates in."""
        if key == attributes.OperationLogs.key:
            return cls(((key, None),), None)
        chunk, _, shard = key[1:].partition(b"_")
        chunk_id = np.uint64(int(chunk))
        shards = tuple(
            (counter_shard_key(chunk_id, s), s) for s in range(COUNTER_SHARDS)
        )
        own = (serialize_uint64(chunk_id, counter=True), None)
        return cls((own,) + shards, int(shard) if shard else None)

    def highest(self, counters: dict[bytes, int]) -> int:
        """The highest id these rows have minted, from each row's counter; 0 for none."""
        minted = (
            value if shard is None else value * COUNTER_SHARDS + shard
            for key, shard in self.rows
            if (value := counters.get(key)) is not None
        )
        return max(minted, default=0)

    def start(self, highest: int) -> int:
        """The allocating row's counter value from which it mints only above `highest`."""
        return highest if self.shard is None else highest // COUNTER_SHARDS
