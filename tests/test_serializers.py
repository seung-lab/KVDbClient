"""Tests for kvdbclient.serializers"""

import numpy as np
import pytest

from kvdbclient import basetypes
from kvdbclient.serializers import (
    JSON,
    NumPyArray,
    NumPyValue,
    Pickle,
    String,
    UInt64String,
    deserialize_key,
    deserialize_uint64,
    pad_node_id,
    serialize_key,
    serialize_uint64,
    serialize_uint64s_to_regex,
)

from helpers import node_ids

NODE_ID = basetypes.NODE_ID


@pytest.mark.parametrize(
    "serializer, value",
    [
        (NumPyArray(dtype=NODE_ID), node_ids(1, 2, 3)),
        (NumPyArray(dtype=NODE_ID, shape=(-1, 2)), node_ids([1, 2], [3, 4])),
        (NumPyArray(dtype=NODE_ID, compression_level=3), np.arange(5, dtype=NODE_ID)),
        (NumPyValue(dtype=NODE_ID), np.uint64(42)),
        (String(), "hello"),
        (JSON(), {"key": "value", "nested": [1, 2, 3]}),
        (Pickle(), {"complex": [1, 2], "nested": {"a": True}}),
        (UInt64String(), np.uint64(12345)),
    ],
    ids=[
        "array",
        "shaped",
        "compressed",
        "value",
        "string",
        "json",
        "pickle",
        "uint64",
    ],
)
def test_a_value_roundtrips(serializer, value):
    np.testing.assert_equal(serializer.deserialize(serializer.serialize(value)), value)


def test_an_array_serializer_names_its_dtype_as_basetype():
    assert NumPyArray(dtype=NODE_ID).basetype == NODE_ID.type


@pytest.mark.parametrize("node_id", [np.uint64(42), np.uint64(12345678901234567890)])
def test_node_ids_pad_to_twenty_digits(node_id):
    assert pad_node_id(node_id) == f"{int(node_id):020d}"


class TestSerializeUint64:
    def test_a_node_key_holds_its_padded_id(self):
        assert b"00000000000000000042" in serialize_uint64(np.uint64(42))

    @pytest.mark.parametrize("flag, prefix", [("counter", b"i"), ("fake_edges", b"f")])
    def test_a_flag_prefixes_the_key(self, flag, prefix):
        assert serialize_uint64(np.uint64(42), **{flag: True}).startswith(prefix)

    @pytest.mark.parametrize("fake_edges", [False, True])
    def test_a_key_roundtrips(self, fake_edges):
        key = serialize_uint64(np.uint64(42), fake_edges=fake_edges)
        assert deserialize_uint64(key, fake_edges=fake_edges) == np.uint64(42)


def test_several_ids_join_into_one_regex():
    assert b"|" in serialize_uint64s_to_regex([np.uint64(1), np.uint64(2)])


def test_a_string_key_roundtrips():
    assert deserialize_key(serialize_key("test_key_123")) == "test_key_123"
