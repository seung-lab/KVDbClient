"""Backend registry, resolution and self-registration; no live backend needed."""

import pytest

from kvdbclient import available_backends
from kvdbclient import get_client_class
from kvdbclient import get_config_class
from kvdbclient import get_default_client_info
from kvdbclient import registry
from kvdbclient.base import SimpleClient
from kvdbclient.bigtable import BigTableConfig
from kvdbclient.bigtable import get_client_info as get_bigtable_client_info
from kvdbclient.bigtable.client import Client as BigTableClient
from kvdbclient.hbase import HBaseConfig
from kvdbclient.hbase import get_client_info as get_hbase_client_info
from kvdbclient.hbase.client import Client as HBaseClient


@pytest.fixture
def clean_registry():
    """Restores the registry, so throwaway backends cannot leak into other tests."""
    snapshot = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(snapshot)


def test_builtin_backends_are_available():
    assert set(available_backends()) >= {"bigtable", "hbase"}


@pytest.mark.parametrize(
    "name, client_class, config_class",
    [
        ("bigtable", BigTableClient, BigTableConfig),
        ("hbase", HBaseClient, HBaseConfig),
        ("BigTable", BigTableClient, BigTableConfig),
        (None, BigTableClient, BigTableConfig),
    ],
    ids=["bigtable", "hbase", "any_case", "default"],
)
def test_a_name_resolves_its_classes(name, client_class, config_class):
    assert get_client_class(name) is client_class
    assert get_config_class(name) is config_class


def test_an_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_client_class("redis")


@pytest.mark.parametrize(
    "env, expected",
    [(None, "bigtable"), ("hbase", "hbase"), ("does-not-exist", "bigtable")],
    ids=["unset", "hbase", "unknown"],
)
def test_the_env_selects_the_default_backend(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv("PCG_BACKEND_TYPE", raising=False)
    else:
        monkeypatch.setenv("PCG_BACKEND_TYPE", env)
    info = get_default_client_info()
    assert info.TYPE == expected
    assert info.CONFIG is not None


class TestBigTableClientInfo:
    def test_defaults_to_a_read_only_non_admin_client(self):
        config = get_bigtable_client_info()
        assert config.PROJECT is not None
        assert config.INSTANCE is not None
        assert config.ADMIN is False
        assert config.READ_ONLY is True

    def test_arguments_override_the_defaults(self):
        config = get_bigtable_client_info(
            project="my-proj", instance="my-inst", admin=True, read_only=False
        )
        assert config == BigTableConfig("my-proj", "my-inst", True, False)


class TestHBaseClientInfo:
    def test_defaults_to_a_rest_url(self):
        assert get_hbase_client_info().BASE_URL is not None

    def test_the_argument_overrides_the_url(self):
        config = get_hbase_client_info(base_url="http://custom:9090")
        assert config.BASE_URL == "http://custom:9090"


def test_a_customizing_subclass_neither_registers_nor_raises(clean_registry):
    before = set(available_backends())

    class Custom(BigTableClient):
        pass

    assert set(available_backends()) == before
    assert get_client_class("bigtable") is BigTableClient


def test_declaring_a_backend_name_registers_it(clean_registry):
    class FooClient(BigTableClient):
        backend_name = "foo"

    assert "foo" in available_backends()
    assert get_client_class("foo") is FooClient
    assert get_config_class("foo") is BigTableConfig


def test_an_incomplete_backend_fails_fast(clean_registry):
    with pytest.raises(TypeError):

        class BadClient(SimpleClient):
            backend_name = "bad"


def test_a_duplicate_backend_name_fails_fast(clean_registry):
    with pytest.raises(TypeError):

        class DupClient(BigTableClient):
            backend_name = "bigtable"
