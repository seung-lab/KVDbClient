# pylint: disable=missing-docstring
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from helpers import no_future
from helpers import node_ids
from helpers import raise_backend_error

ROOT, OTHER_ROOT = np.uint64(100), np.uint64(200)
ROOTS = [ROOT, OTHER_ROOT]
OP, OTHER_OP = np.uint64(1), np.uint64(2)
ROW, OTHER_ROW = b"chunk\x01\x02\x03", b"chunk_b"


class TestRootLock:
    def test_an_unheld_root_locks(self, client):
        assert client.lock_root(ROOT, OP) is True

    def test_a_held_root_refuses_another_op(self, client):
        client.lock_root(ROOT, OP)
        assert client.lock_root(ROOT, OTHER_OP) is False

    def test_roots_lock_independently(self, client):
        client.lock_root(ROOT, OP)
        assert client.lock_root(OTHER_ROOT, OTHER_OP) is True

    def test_an_indefinite_lock_refuses_it(self, client):
        client.lock_root_indefinitely(ROOT, OP)
        assert client.lock_root(ROOT, OTHER_OP) is False

    def test_unlock_frees_it_for_another_op(self, client):
        client.lock_root(ROOT, OP)
        client.unlock_root(ROOT, OP)
        assert client.lock_root(ROOT, OTHER_OP) is True

    def test_unlock_by_another_op_keeps_it(self, client):
        client.lock_root(ROOT, OP)
        client.unlock_root(ROOT, OTHER_OP)
        assert client.lock_root(ROOT, OTHER_OP) is False

    def test_renew_by_its_holder_succeeds(self, client):
        client.lock_root(ROOT, OP)
        assert client.renew_lock(ROOT, OP) is True

    def test_renew_by_another_op_fails(self, client):
        client.lock_root(ROOT, OP)
        assert client.renew_lock(ROOT, OTHER_OP) is False

    def test_a_refusal_logs_the_holders_at_debug(self, client, caplog):
        caplog.set_level(logging.DEBUG, logger=client.logger.name)
        client.lock_root(ROOT, OP)
        client.lock_root(ROOT, OTHER_OP)
        assert "Locked operation ids" in caplog.text


class TestIndefiniteRootLock:
    def test_an_unheld_root_locks(self, client):
        assert client.lock_root_indefinitely(ROOT, OP) is True

    def test_a_held_root_refuses_another_op(self, client):
        client.lock_root_indefinitely(ROOT, OP)
        assert client.lock_root_indefinitely(ROOT, OTHER_OP) is False

    def test_unlock_frees_it_for_another_op(self, client):
        client.lock_root_indefinitely(ROOT, OP)
        client.unlock_indefinitely_locked_root(ROOT, OP)
        assert client.lock_root_indefinitely(ROOT, OTHER_OP) is True

    def test_unlock_by_another_op_keeps_it(self, client):
        client.lock_root_indefinitely(ROOT, OP)
        client.unlock_indefinitely_locked_root(ROOT, OTHER_OP)
        assert client.lock_root_indefinitely(ROOT, OTHER_OP) is False

    def test_a_refusal_logs_the_holders_at_debug(self, client, caplog):
        caplog.set_level(logging.DEBUG, logger=client.logger.name)
        client.lock_root_indefinitely(ROOT, OP)
        client.lock_root_indefinitely(ROOT, OTHER_OP)
        assert "Indefinitely locked operation ids" in caplog.text


class TestRowKeyLock:
    def test_a_held_row_refuses_another_op(self, client):
        assert client.lock_by_row_key(ROW, OP) is True
        assert client.lock_by_row_key(ROW, OTHER_OP) is False

    def test_unlock_frees_it_for_another_op(self, client):
        client.lock_by_row_key(ROW, OP)
        client.unlock_by_row_key(ROW, OP)
        assert client.lock_by_row_key(ROW, OTHER_OP) is True

    def test_renew_by_its_holder_succeeds(self, client):
        client.lock_by_row_key(ROW, OP)
        assert client.renew_lock_by_row_key(ROW, OP) is True

    def test_renew_by_another_op_fails(self, client):
        client.lock_by_row_key(ROW, OP)
        assert client.renew_lock_by_row_key(ROW, OTHER_OP) is False


class TestRowKeyLockWithIndefinite:
    def test_a_clean_row_locks(self, client):
        assert client.lock_by_row_key_with_indefinite(ROW, OP) is True

    @pytest.mark.parametrize(
        "hold",
        ["lock_by_row_key_with_indefinite", "lock_by_row_key_indefinitely"],
        ids=["temporal", "indefinite"],
    )
    def test_either_held_lock_refuses_it(self, client, hold):
        getattr(client, hold)(ROW, OP)
        assert client.lock_by_row_key_with_indefinite(ROW, OTHER_OP) is False


class TestIndefiniteRowKeyLock:
    def test_a_held_row_refuses_another_op(self, client):
        assert client.lock_by_row_key_indefinitely(ROW, OP) is True
        assert client.lock_by_row_key_indefinitely(ROW, OTHER_OP) is False

    def test_rows_lock_independently(self, client):
        client.lock_by_row_key_indefinitely(ROW, OP)
        assert client.lock_by_row_key_indefinitely(OTHER_ROW, OTHER_OP) is True

    def test_unlock_frees_it_for_another_op(self, client):
        client.lock_by_row_key_indefinitely(ROW, OP)
        client.unlock_indefinitely_locked_by_row_key(ROW, OP)
        assert client.lock_by_row_key_indefinitely(ROW, OTHER_OP) is True

    def test_unlock_by_another_op_keeps_it(self, client):
        client.lock_by_row_key_indefinitely(ROW, OP)
        client.unlock_indefinitely_locked_by_row_key(ROW, OTHER_OP)
        assert client.lock_by_row_key_indefinitely(ROW, OTHER_OP) is False


class TestLockRoots:
    def test_locks_every_root(self, client):
        success, locked = client.lock_roots(ROOTS, OP, no_future(ROOTS))
        assert success is True
        assert set(locked) == set(ROOTS)

    def test_a_refusal_releases_the_rest(self, client):
        client.lock_root(OTHER_ROOT, np.uint64(999))
        success, _ = client.lock_roots(ROOTS, OP, no_future(ROOTS), waittime_s=0)
        assert success is False
        assert client.lock_root(ROOT, OTHER_OP) is True

    def test_locks_future_roots_in_place_of_old(self, client):
        success, locked = client.lock_roots([ROOT], OP, {ROOT: node_ids(300)})
        assert success is True
        np.testing.assert_array_equal(locked, [300])
        assert client.lock_root(np.uint64(300), OTHER_OP) is False

    def test_no_roots_lock_at_once(self, client):
        success, locked = client.lock_roots([], OP, {})
        assert success is True
        assert len(locked) == 0

    def test_a_raising_lock_call_fails_and_logs(self, client, monkeypatch, caplog):
        monkeypatch.setattr(client, "lock_root", raise_backend_error)
        success, _ = client.lock_roots(ROOTS, OP, no_future(ROOTS), waittime_s=0)
        assert success is False
        assert "Failed to lock root" in caplog.text


class TestLockRootsIndefinitely:
    def test_locks_every_root(self, client):
        success, locked, failed = client.lock_roots_indefinitely(
            ROOTS, OP, no_future(ROOTS)
        )
        assert success is True
        assert set(locked) == set(ROOTS)
        assert failed == []

    def test_a_refusal_releases_the_rest_and_names_it(self, client):
        client.lock_root_indefinitely(OTHER_ROOT, np.uint64(999))
        success, _, failed = client.lock_roots_indefinitely(ROOTS, OP, no_future(ROOTS))
        assert success is False
        assert failed == [OTHER_ROOT]
        assert client.lock_root_indefinitely(ROOT, OTHER_OP) is True


class TestRenewLocks:
    def test_renews_every_held_root(self, client):
        for root in ROOTS:
            client.lock_root(root, OP)
        assert client.renew_locks(ROOTS, OP) is True

    def test_a_root_held_by_another_op_fails_it(self, client):
        client.lock_root(ROOT, OP)
        client.lock_root(OTHER_ROOT, OTHER_OP)
        assert client.renew_locks(ROOTS, OP) is False

    def test_a_raising_renew_call_fails_and_logs(self, client, monkeypatch, caplog):
        monkeypatch.setattr(client, "renew_lock", raise_backend_error)
        assert client.renew_locks(ROOTS, OP) is False
        assert "Exception during renew_lock" in caplog.text


class TestLockTimestamp:
    def test_is_when_the_lock_was_taken(self, client):
        before = datetime.now(timezone.utc) - timedelta(milliseconds=1)
        client.lock_root(ROOT, OP)
        after = datetime.now(timezone.utc)
        assert before <= client.get_lock_timestamp(ROOT, OP) <= after

    def test_an_unheld_root_has_none(self, client):
        assert client.get_lock_timestamp(ROOT, OP) is None

    def test_another_op_s_lock_gives_none(self, client):
        client.lock_root(ROOT, OP)
        assert client.get_lock_timestamp(ROOT, OTHER_OP) is None


class TestConsolidatedLockTimestamp:
    def test_is_the_earliest_lock(self, client):
        client.lock_root(ROOT, OP)
        client.lock_root(OTHER_ROOT, OTHER_OP)
        earliest = min(
            client.get_lock_timestamp(ROOT, OP),
            client.get_lock_timestamp(OTHER_ROOT, OTHER_OP),
        )
        assert client.get_consolidated_lock_timestamp(ROOTS, [OP, OTHER_OP]) == earliest

    def test_no_roots_give_none(self, client):
        assert client.get_consolidated_lock_timestamp([], []) is None

    def test_a_root_held_by_another_op_gives_none(self, client):
        client.lock_root(ROOT, OP)
        assert client.get_consolidated_lock_timestamp([ROOT], [OTHER_OP]) is None

    def test_a_raising_lookup_gives_none_and_logs(self, client, monkeypatch, caplog):
        monkeypatch.setattr(client, "get_lock_timestamp", raise_backend_error)
        assert client.get_consolidated_lock_timestamp([ROOT], [OP]) is None
        assert "backend down" in caplog.text


class TestLockExpiry:
    def test_an_expired_lock_frees_the_root(self, client_short_expiry):
        client_short_expiry.lock_root(ROOT, OP)
        time.sleep(1.5)
        assert client_short_expiry.lock_root(ROOT, OTHER_OP) is True

    def test_an_unexpired_lock_holds(self, client_short_expiry):
        client_short_expiry.lock_root(ROOT, OP)
        assert client_short_expiry.lock_root(ROOT, OTHER_OP) is False
