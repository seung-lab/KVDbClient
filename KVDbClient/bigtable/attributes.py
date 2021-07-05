from typing import NamedTuple

from .. import serializers


class _AttributeType(NamedTuple):
    key: bytes
    family_id: str
    serializer: serializers.Serializer


class Attribute(_AttributeType):
    __slots__ = ()
    _attributes = {}

    def __init__(self, **kwargs):
        super().__init__()
        Attribute._attributes[(kwargs["family_id"], kwargs["key"])] = self

    def serialize(self, obj):
        return self.serializer.serialize(obj)

    def deserialize(self, stream):
        return self.serializer.deserialize(stream)

    @property
    def basetype(self):
        return self.serializer.basetype

    @property
    def index(self):
        return int(self.key.decode("utf-8").split("_")[-1])


class AttributeArray:
    _attributearrays = {}

    def __init__(self, pattern, family_id, serializer):
        self._pattern = pattern
        self._family_id = family_id
        self._serializer = serializer
        AttributeArray._attributearrays[(family_id, pattern)] = self

        # TODO: Add missing check in `fromkey(family_id, key)` and remove this
        #       loop (pre-creates `Attributes`, so that the inverse lookup works)
        for i in range(20):
            self[i]  # pylint: disable=W0104

    def __getitem__(self, item):
        return Attribute(
            key=self.pattern % item,
            family_id=self._family_id,
            serializer=self._serializer,
        )

    @property
    def pattern(self):
        return self._pattern

    @property
    def serialize(self):
        return self._serializer.serialize

    @property
    def deserialize(self):
        return self._serializer.deserialize

    @property
    def basetype(self):
        return self._serializer.basetype


class TableMeta:
    key = b"meta"
    family_id = "0"
    data = Attribute(key=key, family_id=family_id, serializer=serializers.Pickle())
