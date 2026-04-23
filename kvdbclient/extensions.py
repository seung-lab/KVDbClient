import typing
from datetime import datetime

import numpy as np

from . import attributes
from . import basetypes
from .utils import get_valid_timestamp

try:
    import cloudvolume as _cloudvolume
except ImportError:
    _cloudvolume = None


class RootExtension:
    """Root-related graph operations on top of a SimpleClient.

    Mirrors the root-traversal methods from PyChunkedGraph.
    Access via client.root_ext (lazily initialised on first access).

    Layer is encoded in the top `layer_id_bits` bits of each uint64 node ID:
        layer = int(node_id) >> (64 - layer_id_bits)

    `layer_count` is resolved from the table meta on first access.
    """

    def __init__(self, client, layer_id_bits: int = 8):
        self._client = client
        self._layer_id_bits = layer_id_bits
        self.__layer_count: typing.Optional[int] = None

    @property
    def _layer_count(self) -> int:
        if self.__layer_count is None:
            self.__layer_count = self._read_layer_count()
        return self.__layer_count

    def _read_layer_count(self) -> int:
        meta = self._client.table_meta
        if meta is None:
            meta = self._client.read_table_meta()
        if meta is not None:
            # Case 1: real ChunkedGraphMeta with pychunkedgraph installed —
            # call the property which connects to CloudVolume (same as PCG does)
            if hasattr(meta, "layer_count"):
                return int(meta.layer_count)
            # Case 2: _Surrogate from lenient unpickling (pychunkedgraph not installed /
            # version mismatch) — extract namedtuple fields and compute via cloudvolume
            lc = self._compute_layer_count_from_surrogate(meta)
            if lc is not None:
                return lc
            # Case 3: modern plain-dict meta
            if isinstance(meta, dict) and "layer_count" in meta:
                return int(meta["layer_count"])
        # Case 4: pcgv1 — stored in b'params' row, column b'n_layers'
        row = self._client._read_byte_row(
            attributes.GraphSettings.key, columns=attributes.GraphSettings.LayerCount
        )
        if row:
            return int(row[0].value)
        raise RuntimeError("layer_count not found in table")

    def _compute_layer_count_from_surrogate(self, meta) -> typing.Optional[int]:
        """Compute layer_count from a _Surrogate ChunkedGraphMeta via CloudVolume.

        GraphConfig namedtuple positions: 2=CHUNK_SIZE, 3=FANOUT
        DataSource namedtuple positions:  2=WATERSHED (path), 4=CV_MIP
        """
        if _cloudvolume is None:
            return None
        graph_config = getattr(meta, "graph_config", None)
        data_source = getattr(meta, "data_source", None)
        gc_args = getattr(graph_config, "_surrogate_args", None)
        ds_args = getattr(data_source, "_surrogate_args", None)
        if not gc_args or not ds_args:
            return None
        chunk_size = np.array(gc_args[2], dtype=int)
        fanout = int(gc_args[3])
        watershed = ds_args[2]
        cv_mip = ds_args[4] if len(ds_args) > 4 else 0
        cv = _cloudvolume.CloudVolume(watershed, mip=cv_mip)
        bounds = np.array(cv.bounds.to_list()).reshape(2, 3)
        voxel_counts = bounds[1] - bounds[0]
        n_chunks = np.ceil(voxel_counts / chunk_size).astype(int)
        return int(np.ceil(np.log(np.max(n_chunks)) / np.log(fanout))) + 2

    def get_chunk_layer(self, node_id: np.uint64) -> int:
        return int(int(node_id) >> (64 - self._layer_id_bits))

    def get_chunk_layers(self, node_ids: typing.Sequence[np.uint64]) -> np.ndarray:
        if len(node_ids) == 0:
            return np.array([], dtype=int)
        return np.array(node_ids, dtype=int) >> (64 - self._layer_id_bits)

    def get_parent(
        self,
        node_id: np.uint64,
        time_stamp: typing.Optional[datetime] = None,
    ) -> typing.Optional[np.uint64]:
        time_stamp = get_valid_timestamp(time_stamp)
        parents = self._client.read_node(
            node_id,
            properties=attributes.Hierarchy.Parent,
            end_time=time_stamp,
            end_time_inclusive=True,
        )
        return parents[0].value if parents else None

    def get_parents(
        self,
        node_ids: typing.Sequence[np.uint64],
        time_stamp: typing.Optional[datetime] = None,
        fail_to_zero: bool = False,
    ) -> np.ndarray:
        time_stamp = get_valid_timestamp(time_stamp)
        parent_rows = self._client.read_nodes(
            node_ids=node_ids,
            properties=attributes.Hierarchy.Parent,
            end_time=time_stamp,
            end_time_inclusive=True,
        )
        parents = []
        for node_id in node_ids:
            try:
                parents.append(parent_rows[node_id][0].value)
            except KeyError:
                if fail_to_zero:
                    parents.append(np.uint64(0))
                else:
                    raise
        return np.array(parents, dtype=basetypes.NODE_ID)

    def is_root(self, node_id: np.uint64) -> bool:
        return self.get_chunk_layer(node_id) == self._layer_count

    def get_root(
        self,
        node_id: np.uint64,
        time_stamp: typing.Optional[datetime] = None,
        get_all_parents: bool = False,
        stop_layer: typing.Optional[int] = None,
    ) -> typing.Union[np.uint64, typing.List[np.uint64]]:
        time_stamp = get_valid_timestamp(time_stamp)
        if stop_layer is None:
            stop_layer = self._layer_count
        current = node_id
        all_parents = [node_id]
        while self.get_chunk_layer(current) < stop_layer:
            parent = self.get_parent(current, time_stamp=time_stamp)
            if parent is None:
                break
            current = parent
            all_parents.append(current)
        return all_parents if get_all_parents else current

    def get_roots(
        self,
        node_ids: typing.Sequence[np.uint64],
        time_stamp: typing.Optional[datetime] = None,
        stop_layer: typing.Optional[int] = None,
        fail_to_zero: bool = False,
    ) -> np.ndarray:
        time_stamp = get_valid_timestamp(time_stamp)
        if stop_layer is None:
            stop_layer = self._layer_count
        result = np.array(node_ids, dtype=basetypes.NODE_ID)
        layer_mask = np.ones(len(node_ids), dtype=bool)
        layer_mask[self.get_chunk_layers(result) >= stop_layer] = False
        layer_mask[result == 0] = False
        for _ in range(stop_layer + 1):
            filtered = result[layer_mask]
            if not filtered.size:
                break
            unique_nodes, inverse = np.unique(filtered, return_inverse=True)
            temp_ids = self.get_parents(
                unique_nodes, time_stamp=time_stamp, fail_to_zero=fail_to_zero
            )
            temp_ids_i = temp_ids[inverse]
            new_layer_mask = layer_mask.copy()
            new_layer_mask[new_layer_mask] = self.get_chunk_layers(temp_ids_i) < stop_layer
            result[layer_mask] = temp_ids_i
            layer_mask = new_layer_mask
            if np.all(~layer_mask):
                break
        non_zero = result[result != 0]
        assert not np.any(self.get_chunk_layers(non_zero) < stop_layer), \
            "roots not found for some IDs"
        return result

    def is_latest_roots(
        self,
        root_ids: typing.Sequence[np.uint64],
        time_stamp: typing.Optional[datetime] = None,
    ) -> np.ndarray:
        time_stamp = get_valid_timestamp(time_stamp)
        root_ids = np.array(root_ids, dtype=basetypes.NODE_ID)
        rows = self._client.read_nodes(
            node_ids=root_ids,
            properties=attributes.Hierarchy.NewParent,
            end_time=time_stamp,
            end_time_inclusive=True,
        )
        superseded = (
            np.array(list(rows.keys()), dtype=basetypes.NODE_ID)
            if rows
            else np.array([], dtype=basetypes.NODE_ID)
        )
        return ~np.isin(root_ids, superseded)
