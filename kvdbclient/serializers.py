from typing import Any, Iterable
import json
import pickle

import io

import numpy as np
import zstandard as zstd

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class _Serializer:
    def __init__(self, serializer, deserializer, basetype=Any, compression_level=None):
        self._serializer = serializer
        self._deserializer = deserializer
        self._basetype = basetype
        self._compression_level = compression_level

    def serialize(self, obj):
        content = self._serializer(obj)
        if self._compression_level:
            return zstd.ZstdCompressor(level=self._compression_level).compress(content)
        return content

    def deserialize(self, obj):
        if self._compression_level and obj[:4] == _ZSTD_MAGIC:
            obj = zstd.ZstdDecompressor().decompressobj().decompress(obj)
        return self._deserializer(obj)

    @property
    def basetype(self):
        return self._basetype


class NumPyArray(_Serializer):
    @staticmethod
    def _deserialize(val, dtype, shape=None, order=None):
        data = np.frombuffer(val, dtype=dtype)
        if shape is not None:
            return data.reshape(shape, order=order)
        if order is not None:
            return data.reshape(data.shape, order=order)
        return data

    def __init__(self, dtype, shape=None, order=None, compression_level=None):
        super().__init__(
            serializer=lambda x: np.asarray(x)
            .view(x.dtype.newbyteorder(dtype.byteorder))
            .tobytes(),
            deserializer=lambda x: NumPyArray._deserialize(
                x, dtype, shape=shape, order=order
            ),
            basetype=dtype.type,
            compression_level=compression_level,
        )


class NumPyValue(_Serializer):
    def __init__(self, dtype):
        super().__init__(
            serializer=lambda x: np.asarray(x)
            .view(np.dtype(type(x)).newbyteorder(dtype.byteorder))
            .tobytes(),
            deserializer=lambda x: np.frombuffer(x, dtype=dtype)[0],
            basetype=dtype.type,
        )


class String(_Serializer):
    def __init__(self, encoding="utf-8"):
        super().__init__(
            serializer=lambda x: x.encode(encoding),
            deserializer=lambda x: x.decode(),
            basetype=str,
        )


class JSON(_Serializer):
    def __init__(self):
        super().__init__(
            serializer=lambda x: json.dumps(x).encode("utf-8"),
            deserializer=lambda x: json.loads(x.decode()),
            basetype=str,
        )


class _Surrogate:
    """Stand-in for unknown classes during lenient unpickling."""
    def __new__(cls, *args):
        obj = object.__new__(cls)
        if args:
            obj._surrogate_args = args  # preserve namedtuple positional fields
        return obj
    def __init__(self, *_):
        pass
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _LenientUnpickler(pickle.Unpickler):
    """Unpickler that replaces unknown classes with _Surrogate."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return _Surrogate


class Pickle(_Serializer):
    def __init__(self):
        super().__init__(
            serializer=lambda x: pickle.dumps(x),
            deserializer=pickle.loads,
            basetype=str,
        )

    def deserialize(self, obj):
        try:
            return pickle.loads(obj)
        except (ModuleNotFoundError, ImportError, AttributeError):
            return _LenientUnpickler(io.BytesIO(obj)).load()


class UInt64String(_Serializer):
    def __init__(self):
        super().__init__(
            serializer=serialize_uint64,
            deserializer=deserialize_uint64,
            basetype=np.uint64,
        )


def pad_node_id(node_id: np.uint64) -> str:
    """Pad node id to 20 digits

    :param node_id: int
    :return: str
    """
    return "%.20d" % node_id


def serialize_uint64(node_id: np.uint64, counter=False, fake_edges=False) -> bytes:
    """Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    if counter:
        return serialize_key("i%s" % pad_node_id(node_id))  # type: ignore
    if fake_edges:
        return serialize_key("f%s" % pad_node_id(node_id))  # type: ignore
    return serialize_key(pad_node_id(node_id))  # type: ignore


def serialize_uint64_batch(node_ids: Iterable[np.uint64], fake_edges: bool = False) -> list:
    """Batch-serialize node IDs to row keys.

    Inlines pad_node_id + serialize_key into a single f-string per element
    to avoid per-ID function call overhead at scale (100K+ IDs).
    """
    prefix = "f" if fake_edges else ""
    return [f"{prefix}{int(nid):020d}".encode("utf-8") for nid in node_ids]


def serialize_uint64s_to_regex(node_ids: Iterable[np.uint64]) -> bytes:
    """Serializes an id to be ingested by a bigtable table row

    :param node_id: int
    :return: str
    """
    node_id_str = "".join(["%s|" % pad_node_id(node_id) for node_id in node_ids])[:-1]
    return serialize_key(node_id_str)  # type: ignore


def deserialize_uint64(node_id: bytes, fake_edges=False) -> np.uint64:
    """De-serializes a node id from a BigTable row

    :param node_id: bytes
    :return: np.uint64
    """
    if fake_edges:
        return np.uint64(node_id[1:].decode())  # type: ignore
    return np.uint64(node_id.decode())  # type: ignore


def serialize_key(key: str) -> bytes:
    """Serializes a key to be ingested by a bigtable table row

    :param key: str
    :return: bytes
    """
    return key.encode("utf-8")


def deserialize_key(key: bytes) -> str:
    """Deserializes a row key

    :param key: bytes
    :return: str
    """
    return key.decode()
