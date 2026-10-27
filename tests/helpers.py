import numpy as np

from kvdbclient.serializers import serialize_uint64


def write_node(client, node_id, val_dict, time_stamp=None):
    row_key = serialize_uint64(np.uint64(node_id))
    entry = client.mutate_row(row_key, val_dict, time_stamp=time_stamp)
    client.write([entry])
    return row_key
