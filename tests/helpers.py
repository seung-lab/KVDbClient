from datetime import datetime, timedelta, timezone

import numpy as np

from kvdbclient import attributes
from kvdbclient import basetypes
from kvdbclient.attributes import OperationLogs
from kvdbclient.serializers import serialize_uint64

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


def write_log(client, op_id, user="alice", op_time=OP_TIME) -> None:
    record = {OperationLogs.UserID: user, OperationLogs.RootID: LOG_ROOTS}
    if op_time is not None:
        record[OperationLogs.OperationTimeStamp] = op_time
    write_node(client, op_id, record, LOG_WRITTEN)
