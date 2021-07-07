import json
import pickle
from typing import Any

import numpy as np
import zstandard as zstd


class Serializer:
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
        if self._compression_level:
            obj = zstd.ZstdDecompressor().decompressobj().decompress(obj)
        return self._deserializer(obj)

    @property
    def basetype(self):
        return self._basetype


class NumPyArray(Serializer):
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
            serializer=lambda x: x.newbyteorder(dtype.byteorder).tobytes(),
            deserializer=lambda x: NumPyArray._deserialize(
                x, dtype, shape=shape, order=order
            ),
            basetype=dtype.type,
            compression_level=compression_level,
        )


class NumPyValue(Serializer):
    def __init__(self, dtype):
        super().__init__(
            serializer=lambda x: x.newbyteorder(dtype.byteorder).tobytes(),
            deserializer=lambda x: np.frombuffer(x, dtype=dtype)[0],
            basetype=dtype.type,
        )


class String(Serializer):
    def __init__(self, encoding="utf-8"):
        super().__init__(
            serializer=lambda x: x.encode(encoding),
            deserializer=lambda x: x.decode(),
            basetype=str,
        )


class JSON(Serializer):
    def __init__(self):
        super().__init__(
            serializer=lambda x: json.dumps(x).encode("utf-8"),
            deserializer=lambda x: json.loads(x.decode()),
            basetype=str,
        )


class Pickle(Serializer):
    def __init__(self):
        super().__init__(
            serializer=lambda x: pickle.dumps(x),
            deserializer=lambda x: pickle.loads(x),
            basetype=str,
        )


class UInt64String(Serializer):
    def __init__(self):
        super().__init__(
            serializer=serialize_uint64,
            deserializer=deserialize_uint64,
            basetype=np.uint64,
        )


def pad_uint64(key: np.uint64) -> str:
    """Pad key id to 20 digits."""
    return "%.20d" % key


def serialize_uint64(key: np.uint64) -> bytes:
    """Serializes a numpy int for use as BigTable row key."""
    return serialize_key(pad_uint64(key))  # type: ignore


def deserialize_uint64(key: bytes) -> np.uint64:
    """De-serializes a key from a BigTable row."""
    return np.uint64(key.decode())  # type: ignore


def serialize_key(key: str) -> bytes:
    """Serializes a key for use as BigTable row key."""
    return key.encode("utf-8")


def deserialize_key(key: bytes) -> str:
    """Deserializes a row key."""
    return key.decode()
