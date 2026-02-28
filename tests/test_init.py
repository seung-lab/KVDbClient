# pylint: disable=missing-docstring

import os

import pytest


class TestGetClientClass:
    def test_bigtable(self):
        from kvdbclient import get_client_class
        from kvdbclient.bigtable.client import Client as BigTableClient
        assert get_client_class("bigtable") is BigTableClient

    def test_hbase(self):
        from kvdbclient import get_client_class
        from kvdbclient.hbase.client import Client as HBaseClient
        assert get_client_class("hbase") is HBaseClient

    def test_unknown_raises(self):
        from kvdbclient import get_client_class
        with pytest.raises(ValueError, match="Unknown backend"):
            get_client_class("redis")

    def test_case_insensitive(self):
        from kvdbclient import get_client_class
        from kvdbclient.bigtable.client import Client as BigTableClient
        assert get_client_class("BigTable") is BigTableClient

    def test_none_defaults_to_bigtable(self):
        from kvdbclient import get_client_class
        from kvdbclient.bigtable.client import Client as BigTableClient
        assert get_client_class(None) is BigTableClient


class TestGetDefaultClientInfo:
    def test_default_is_bigtable(self):
        from kvdbclient import get_default_client_info
        old = os.environ.pop("PCG_BACKEND_TYPE", None)
        try:
            info = get_default_client_info()
            assert info.TYPE == "bigtable"
            assert info.CONFIG is not None
        finally:
            if old is not None:
                os.environ["PCG_BACKEND_TYPE"] = old

    def test_hbase_from_env(self):
        from kvdbclient import get_default_client_info
        old = os.environ.get("PCG_BACKEND_TYPE")
        os.environ["PCG_BACKEND_TYPE"] = "hbase"
        try:
            info = get_default_client_info()
            assert info.TYPE == "hbase"
        finally:
            if old is not None:
                os.environ["PCG_BACKEND_TYPE"] = old
            else:
                os.environ.pop("PCG_BACKEND_TYPE", None)


class TestBigTableGetClientInfo:
    def test_defaults(self):
        from kvdbclient.bigtable import get_client_info
        config = get_client_info()
        assert config.PROJECT is not None
        assert config.INSTANCE is not None
        assert config.ADMIN is False
        assert config.READ_ONLY is True

    def test_overrides(self):
        from kvdbclient.bigtable import get_client_info
        config = get_client_info(project="my-proj", instance="my-inst", admin=True, read_only=False)
        assert config.PROJECT == "my-proj"
        assert config.INSTANCE == "my-inst"
        assert config.ADMIN is True
        assert config.READ_ONLY is False


class TestHBaseGetClientInfo:
    def test_defaults(self):
        from kvdbclient.hbase import get_client_info
        config = get_client_info()
        assert config.BASE_URL is not None

    def test_override(self):
        from kvdbclient.hbase import get_client_info
        config = get_client_info(base_url="http://custom:9090")
        assert config.BASE_URL == "http://custom:9090"
