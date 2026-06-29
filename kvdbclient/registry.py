"""Backend registry: concrete backend client classes self-register here on definition."""

_REGISTRY = {}


def register_backend(client_cls) -> None:
    """Record a concrete backend client under its ``backend_name``."""
    name = client_cls.backend_name
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not client_cls:
        raise TypeError(f"backend name {name!r} already registered to {existing.__name__}")
    _REGISTRY[name] = client_cls


def available_backends() -> list:
    """Names of every registered backend."""
    return list(_REGISTRY)


def resolve_client(backend_type: str):
    """The client class registered for ``backend_type``; raises if unknown."""
    name = (backend_type or "bigtable").lower()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend type {backend_type!r}; registered: {available_backends()}"
        )
