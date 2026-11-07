# pylint: disable=missing-docstring
import numpy as np

CHUNK = np.uint64(1 << 32)


def _segment(node_id) -> int:
    return int(np.uint64(node_id) ^ CHUNK)


class TestNodeIds:
    def test_ids_are_unique(self, client):
        assert client.create_node_id(CHUNK) != client.create_node_id(CHUNK)

    def test_a_batch_holds_distinct_ids(self, client):
        assert len(np.unique(client.create_node_ids(CHUNK, 5))) == 5

    def test_the_max_is_the_last_created(self, client):
        ids = client.create_node_ids(CHUNK, 3)
        assert client.get_max_node_id(CHUNK) == ids.max()

    def test_set_max_on_a_fresh_counter_sets_it(self, client):
        client.set_max_node_id(CHUNK, CHUNK | np.uint64(10))
        assert client.get_max_node_id(CHUNK) == CHUNK | np.uint64(10)

    def test_ids_created_after_set_max_exceed_it(self, client):
        client.set_max_node_id(CHUNK, CHUNK | np.uint64(10))
        assert _segment(client.create_node_id(CHUNK)) > 10

    def test_set_max_adds_to_the_counter(self, client):
        client.create_node_ids(CHUNK, 3)
        client.set_max_node_id(CHUNK, CHUNK | np.uint64(7))
        assert _segment(client.get_max_node_id(CHUNK)) == 10

    def test_root_chunk_ids_are_distinct_and_counted(self, client):
        ids = client.create_node_ids(CHUNK, 5, root_chunk=True)
        assert len(np.unique(ids)) == 5
        assert client.get_max_node_id(CHUNK, root_chunk=True) == ids.max()


class TestOperationIds:
    def test_ids_are_unique(self, client):
        assert client.create_operation_id() != client.create_operation_id()

    def test_a_fresh_table_s_max_is_zero(self, client):
        assert client.get_max_operation_id() == 0

    def test_the_max_counts_created_ids(self, client):
        for _ in range(3):
            client.create_operation_id()
        assert client.get_max_operation_id() == 3
