# pylint: disable=missing-docstring
from datetime import timedelta

import numpy as np
import pytest

from kvdbclient.attributes import _Attribute
from kvdbclient.base import ColumnFamilyConfig
from kvdbclient.serializers import String

from helpers import TEST_META
from helpers import TEST_VERSION
from helpers import write_node

EXTRA_FAMILY = "5"
EXTRA_COLUMN = _Attribute(key=b"note", family_id=EXTRA_FAMILY, serializer=String())


def _holds_an_extra_cell(client) -> bool:
    write_node(client, 100, {EXTRA_COLUMN: "kept"})
    return client.read_node(np.uint64(100), properties=EXTRA_COLUMN)[0].value == "kept"


class TestCreateTable:
    def test_writes_version_and_meta(self, client_no_table):
        client_no_table.create_table({"chunk_size": 512}, "1.0")
        assert client_no_table.read_table_version() == "1.0"
        assert client_no_table.read_table_meta() == {"chunk_size": 512}

    def test_refuses_an_existing_table(self, client):
        with pytest.raises(ValueError, match="already exists"):
            client.create_table({}, "2.0")

    def test_custom_families_hold_cells(self, client_no_table):
        families = [
            ColumnFamilyConfig("0"),
            ColumnFamilyConfig("1", max_versions=3),
            ColumnFamilyConfig("2", max_age=timedelta(days=30)),
            ColumnFamilyConfig(EXTRA_FAMILY, max_versions=1, max_age=timedelta(days=1)),
        ]
        client_no_table.create_table({"custom": True}, "1.0", column_families=families)
        assert client_no_table.read_table_meta() == {"custom": True}
        assert _holds_an_extra_cell(client_no_table)


@pytest.mark.parametrize(
    "family",
    [EXTRA_FAMILY, ColumnFamilyConfig(EXTRA_FAMILY, max_versions=2)],
    ids=["id", "config"],
)
def test_a_created_family_holds_cells(client, family):
    client.create_column_family(family)
    assert _holds_an_extra_cell(client)


class TestTableMeta:
    def test_reads_the_created_meta(self, client):
        assert client.read_table_meta() == TEST_META

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_update_replaces_the_meta(self, client, overwrite):
        client.update_table_meta({"new_field": 42}, overwrite=overwrite)
        assert client.read_table_meta() == {"new_field": 42}

    def test_deleted_meta_reads_none(self, client):
        client._delete_meta()
        assert client.read_table_meta() is None

    def test_a_plain_table_has_no_fork_marker(self, client):
        client.read_table_meta()
        assert client.marker is None


class TestTableVersion:
    def test_reads_the_created_version(self, client):
        assert client.read_table_version() == TEST_VERSION

    def test_overwrite_replaces_the_version(self, client):
        client.add_table_version("2.0", overwrite=True)
        assert client.read_table_version() == "2.0"

    def test_refuses_a_second_version_without_overwrite(self, client):
        with pytest.raises(AssertionError):
            client.add_table_version("1.0", overwrite=False)
