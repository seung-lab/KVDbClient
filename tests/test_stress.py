# pylint: disable=missing-docstring
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from helpers import CHILD
from helpers import no_future
from helpers import write_nodes

ROWS = int(os.environ.get("STRESS_ROW_COUNT", "10000"))
LOCKS = int(os.environ.get("STRESS_LOCK_COUNT", "1000"))


def test_many_rows_read_back_by_id(client):
    ids = write_nodes(client, range(2_000_000, 2_000_000 + ROWS))
    data = client.read_nodes(node_ids=ids, properties=CHILD)
    np.testing.assert_array_equal([data[i][0].value[0] for i in ids], ids)


def test_many_rows_read_back_by_range(client):
    ids = write_nodes(client, range(4_000_000, 4_000_000 + ROWS))
    data = client.read_nodes(
        start_id=ids[0], end_id=ids[-1], end_id_inclusive=True, properties=CHILD
    )
    assert len(data) == ROWS


def test_a_large_id_batch_is_distinct(client):
    ids = client.create_node_ids(np.uint64(1 << 32), ROWS)
    assert len(np.unique(ids)) == ROWS


def test_many_roots_lock_unlock_and_relock(client):
    roots = [np.uint64(5_000_000 + i) for i in range(LOCKS)]
    assert client.lock_roots(roots, np.uint64(1), no_future(roots))[0] is True
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda root: client.unlock_root(root, np.uint64(1)), roots))
    assert client.lock_roots(roots, np.uint64(2), no_future(roots))[0] is True
