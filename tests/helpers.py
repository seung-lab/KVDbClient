import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from kvdbclient import attributes
from kvdbclient import basetypes
from kvdbclient.attributes import OperationLogs
from kvdbclient.bigtable.fork import ForkClient
from kvdbclient.fork.marker import ForkMarker, MarkerState
from kvdbclient.serializers import serialize_uint64


def new_table_id() -> str:
    return f"test_{uuid.uuid4().hex[:12]}"


CHILD = attributes.Hierarchy.Child
TEST_META = {"test": True}
TEST_VERSION = "0.0.1"
SMALL_BATCH = 50
LOG_WRITTEN = datetime(2024, 1, 1, tzinfo=timezone.utc)
OP_TIME = LOG_WRITTEN + timedelta(days=1)


def node_ids(*ids) -> np.ndarray:
    return np.array(ids, dtype=basetypes.NODE_ID)


LOG_ROOTS = node_ids(100, 200)


def raise_backend_error(*_):
    raise RuntimeError("backend down")


def no_future(roots) -> dict[np.uint64, np.ndarray]:
    return {root: np.array([], dtype=np.uint64) for root in roots}


def write_node(client, node_id, val_dict, time_stamp=None) -> bytes:
    row_key = serialize_uint64(np.uint64(node_id))
    client.write([client.mutate_row(row_key, val_dict, time_stamp=time_stamp)])
    return row_key


def write_nodes(client, ids) -> np.ndarray:
    """Writes each id as its own one child, in one write."""
    ids = node_ids(*ids)
    client.write(
        [client.mutate_row(serialize_uint64(i), {CHILD: node_ids(i)}) for i in ids]
    )
    return ids


def write_log(client, op_id, user="alice", op_time=OP_TIME, status=None) -> None:
    record = {OperationLogs.UserID: user, OperationLogs.RootID: LOG_ROOTS}
    if op_time is not None:
        record[OperationLogs.OperationTimeStamp] = op_time
    if status is not None:
        record[OperationLogs.Status] = status.value
    write_node(client, op_id, record, LOG_WRITTEN)


PARENT = attributes.Hierarchy.Parent
#: Whole seconds an hour back: a fork at `TF1` falls between writes at `T1` and `T2`,
#: and a fork of that fork at `TF2` between `T2` and `T3`.
FORK_BASE = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
T1, TF1, T2, TF2, T3 = (FORK_BASE + timedelta(seconds=s) for s in (0, 10, 20, 30, 40))
NODE, OTHER = 7, 8


def fork_marker(*hops, state=MarkerState.READY) -> ForkMarker:
    return ForkMarker(
        chain=hops,
        state=state,
        generation="g",
        source_max_op_id=0,
        requested_at=hops[0].tf,
    )


def create_fork(backend, marker, meta=TEST_META, version=TEST_VERSION, table_id=None):
    return ForkClient.create_fork_table(
        table_id or new_table_id(), marker, meta, version, config=backend.config
    )


def open_fork(backend, table_id) -> ForkClient:
    client = ForkClient(table_id, backend.config)
    client.read_table_meta()
    return client


def write_parent(client, node, value, at) -> None:
    write_node(client, node, {PARENT: np.uint64(value)}, at)
