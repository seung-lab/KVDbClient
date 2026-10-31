class KVDBClientError(Exception):
    pass


class LockingError(KVDBClientError):
    pass


class PreconditionError(KVDBClientError):
    pass


class ForkTableError(KVDBClientError):
    """A fork's table was opened by a client that reads only its own table."""
