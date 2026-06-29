"""Backend registry + self-registration behavior (no live backend needed)."""

import pytest

from kvdbclient import (
    available_backends,
    get_client_class,
    get_config_class,
    get_default_client_info,
)
from kvdbclient import registry
from kvdbclient.base import SimpleClient
from kvdbclient.bigtable import BigTableConfig
from kvdbclient.bigtable.client import Client as BigTableClient


@pytest.fixture
def clean_registry():
    """Restore the registry so a test that defines throwaway backends can't leak into others."""
    snapshot = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


def test_builtin_backends_resolve():
    assert set(available_backends()) >= {"bigtable", "hbase"}
    assert get_client_class("bigtable") is BigTableClient
    assert get_config_class("bigtable") is BigTableConfig


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_client_class("does-not-exist")


def test_unknown_env_backend_falls_back_to_bigtable(monkeypatch):
    monkeypatch.setenv("PCG_BACKEND_TYPE", "does-not-exist")
    assert get_default_client_info().TYPE == "bigtable"


def test_customization_subclass_neither_registers_nor_raises(clean_registry):
    before = set(available_backends())

    class Custom(BigTableClient):  # subclass for customization, not a new backend
        pass

    assert set(available_backends()) == before
    assert get_client_class("bigtable") is BigTableClient


def test_declaring_backend_name_registers_and_resolves(clean_registry):
    class FooClient(BigTableClient):
        backend_name = "foo"

    assert "foo" in available_backends()
    assert get_client_class("foo") is FooClient
    assert get_config_class("foo") is BigTableConfig


def test_incomplete_backend_fails_fast(clean_registry):
    with pytest.raises(TypeError):

        class BadClient(SimpleClient):
            backend_name = "bad"  # missing config_class / default_client_info


def test_duplicate_backend_name_fails_fast(clean_registry):
    with pytest.raises(TypeError):

        class DupClient(BigTableClient):
            backend_name = "bigtable"  # already registered
