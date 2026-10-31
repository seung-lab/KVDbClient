"""A fork's marker: its upstream chain, creation state and provenance, as JSON.

It lives in the fork's meta row, so the read every open already makes returns it.
"""

import json
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import NamedTuple


class MarkerState(Enum):
    """A fork opens only once `READY`; `CREATING` marks a half-built table."""

    CREATING = "creating"
    READY = "ready"


class Hop(NamedTuple):
    """One upstream table, and the instant the table below caps its reads of it at."""

    table: str
    tf: datetime


@dataclass(frozen=True)
class ForkMarker:
    """The resolved chain, direct source first, so opening a fork reads no upstream."""

    chain: tuple
    state: MarkerState
    generation: str
    source_max_op_id: int
    # The fork time its creator asked for; the chain's first `tf` is that time clamped.
    requested_at: datetime

    @property
    def source(self) -> str:
        return self.chain[0].table

    @property
    def tf(self) -> datetime:
        return self.chain[0].tf

    def with_state(self, state: MarkerState) -> "ForkMarker":
        return replace(self, state=state)

    def to_json(self) -> str:
        return json.dumps(
            {
                "chain": [[hop.table, hop.tf.isoformat()] for hop in self.chain],
                "state": self.state.value,
                "generation": self.generation,
                "source_max_op_id": self.source_max_op_id,
                "requested_at": self.requested_at.isoformat(),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "ForkMarker":
        fields = json.loads(text)
        return cls(
            chain=tuple(
                Hop(table, datetime.fromisoformat(tf)) for table, tf in fields["chain"]
            ),
            state=MarkerState(fields["state"]),
            generation=fields["generation"],
            source_max_op_id=int(fields["source_max_op_id"]),
            requested_at=datetime.fromisoformat(fields["requested_at"]),
        )
