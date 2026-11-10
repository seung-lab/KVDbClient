class KVDBClientError(Exception):
    pass


class LockingError(KVDBClientError):
    pass


class PreconditionError(KVDBClientError):
    pass


class ForkTableError(KVDBClientError):
    """A fork's table refused an open or a read it cannot serve."""
