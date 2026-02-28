class KVDBClientError(Exception):
    pass


class LockingError(KVDBClientError):
    pass


class PreconditionError(KVDBClientError):
    pass
