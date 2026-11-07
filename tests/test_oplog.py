# pylint: disable=missing-docstring
import numpy as np

from kvdbclient.attributes import OperationLogs

from helpers import LOG_ROOTS
from helpers import LOG_WRITTEN
from helpers import OP_TIME
from helpers import node_ids
from helpers import write_log


class TestReadLogEntry:
    def test_an_absent_op_reads_empty(self, client):
        assert client.read_log_entry(np.uint64(99999)) == ({}, None)

    def test_reads_newest_values_and_the_op_time(self, client):
        write_log(client, 1)
        record, op_time = client.read_log_entry(np.uint64(1))
        assert record[OperationLogs.UserID] == "alice"
        np.testing.assert_array_equal(record[OperationLogs.RootID], LOG_ROOTS)
        assert op_time == OP_TIME

    def test_an_undated_record_dates_by_its_root_ids_cell(self, client):
        write_log(client, 1, op_time=None)
        assert client.read_log_entry(np.uint64(1))[1] == LOG_WRITTEN


class TestReadLogEntries:
    def test_by_ids_reads_each(self, client):
        write_log(client, 1)
        write_log(client, 2, user="bob")
        assert set(client.read_log_entries(operation_ids=node_ids(1, 2))) == {1, 2}

    def test_absent_ids_read_empty(self, client):
        assert client.read_log_entries(operation_ids=node_ids(99999)) == {}

    def test_without_ids_reads_every_created_op(self, client):
        for _ in range(2):
            write_log(client, client.create_operation_id())
        assert set(client.read_log_entries()) == {1, 2}
