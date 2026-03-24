"""
tests/test_supabase.py — Unit tests for utils/supabase_utils.py retry helpers.

The supabase-py client is fully mocked in every test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from utils.supabase_utils import (
    retry_insert,
    retry_update,
    retry_fetch,
    SupabaseWriteError,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_client(*, fail_n_times: int = 0, return_data=None):
    """
    Build a mock Supabase client where the first *fail_n_times* calls
    to ``.execute()`` raise an exception, then succeeding calls return data.

    Parameters
    ----------
    fail_n_times : int  Number of initial failures before success.
    return_data      : Data object returned inside ``MagicMock(data=...)``.
    """
    if return_data is None:
        return_data = [{"id": "ok"}]

    call_count = {"n": 0}

    def _execute_side_effect():
        call_count["n"] += 1
        if call_count["n"] <= fail_n_times:
            raise Exception(f"DB error on attempt {call_count['n']}")
        return MagicMock(data=return_data)

    chain = MagicMock()
    for m in ("select", "insert", "update", "eq", "maybe_single"):
        getattr(chain, m).return_value = chain
    chain.execute = MagicMock(side_effect=_execute_side_effect)

    client = MagicMock()
    client.table.return_value = chain
    return client, chain


# =============================================================================
# Tests — retry_insert
# =============================================================================


class TestRetryInsert:
    """Tests for retry_insert()."""

    @patch("utils.supabase_utils._backoff_sleep")  # skip actual sleeps
    def test_success_first_try(self, mock_sleep):
        """Insert succeeds on the first attempt — called once."""
        client, chain = _make_client(fail_n_times=0)
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_insert("detections", {"job_id": "j1"})
        chain.insert.assert_called_once()
        chain.execute.assert_called_once()
        assert result is not None

    @patch("utils.supabase_utils._backoff_sleep")
    def test_retries_on_failure(self, mock_sleep):
        """First call fails, second succeeds — execute called exactly twice."""
        client, chain = _make_client(fail_n_times=1)
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_insert("detections", {"job_id": "j2"})
        assert chain.execute.call_count == 2
        # Backoff was called once (between attempt 1 and 2)
        mock_sleep.assert_called_once_with(0)

    @patch("utils.supabase_utils._backoff_sleep")
    def test_exhausted_raises(self, mock_sleep):
        """All 3 retries fail → raises SupabaseWriteError."""
        client, chain = _make_client(fail_n_times=100)
        with patch("utils.supabase_utils._get_client", return_value=client):
            with pytest.raises(SupabaseWriteError) as exc_info:
                retry_insert("detections", {"job_id": "j3"}, max_retries=3)
        assert exc_info.value.table == "detections"
        assert exc_info.value.operation == "INSERT"
        assert exc_info.value.attempts == 3
        assert chain.execute.call_count == 3

    @patch("utils.supabase_utils._backoff_sleep")
    def test_custom_max_retries(self, mock_sleep):
        """max_retries=5 → allows up to 5 attempts."""
        client, chain = _make_client(fail_n_times=4)
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_insert("detections", {"job_id": "j4"}, max_retries=5)
        assert chain.execute.call_count == 5
        assert result is not None


# =============================================================================
# Tests — retry_update
# =============================================================================


class TestRetryUpdate:
    """Tests for retry_update()."""

    @patch("utils.supabase_utils._backoff_sleep")
    def test_update_success(self, mock_sleep):
        """Update succeeds on first try."""
        client, chain = _make_client(fail_n_times=0)
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_update(
                "detections",
                {"status": "completed"},
                eq_column="job_id",
                eq_value="j5",
            )
        chain.update.assert_called_once()
        chain.eq.assert_called_once_with("job_id", "j5")
        assert result is not None

    @patch("utils.supabase_utils._backoff_sleep")
    def test_update_exhausted(self, mock_sleep):
        """All retries fail → raises SupabaseWriteError with operation=UPDATE."""
        client, chain = _make_client(fail_n_times=100)
        with patch("utils.supabase_utils._get_client", return_value=client):
            with pytest.raises(SupabaseWriteError) as exc_info:
                retry_update(
                    "detections",
                    {"status": "error"},
                    eq_column="job_id",
                    eq_value="j6",
                    max_retries=2,
                )
        assert exc_info.value.operation == "UPDATE"
        assert exc_info.value.attempts == 2


# =============================================================================
# Tests — retry_fetch
# =============================================================================


class TestRetryFetch:
    """Tests for retry_fetch()."""

    @patch("utils.supabase_utils._backoff_sleep")
    def test_fetch_success(self, mock_sleep):
        """Fetch succeeds on first try."""
        client, chain = _make_client(fail_n_times=0, return_data=[{"job_id": "j7"}])
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_fetch("detections", {"job_id": "j7"})
        assert result is not None

    @patch("utils.supabase_utils._backoff_sleep")
    def test_fetch_single(self, mock_sleep):
        """Fetch with single=True calls maybe_single()."""
        client, chain = _make_client(fail_n_times=0, return_data={"job_id": "j8"})
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_fetch("detections", {"job_id": "j8"}, single=True)
        chain.maybe_single.assert_called_once()

    @patch("utils.supabase_utils._backoff_sleep")
    def test_fetch_exhausted(self, mock_sleep):
        """All retries fail → raises SupabaseWriteError with operation=SELECT."""
        client, chain = _make_client(fail_n_times=100)
        with patch("utils.supabase_utils._get_client", return_value=client):
            with pytest.raises(SupabaseWriteError) as exc_info:
                retry_fetch("detections", {"job_id": "j9"}, max_retries=3)
        assert exc_info.value.operation == "SELECT"

    @patch("utils.supabase_utils._backoff_sleep")
    def test_fetch_retries_then_succeeds(self, mock_sleep):
        """First two calls fail, third succeeds."""
        client, chain = _make_client(fail_n_times=2, return_data=[{"ok": True}])
        with patch("utils.supabase_utils._get_client", return_value=client):
            result = retry_fetch("agent_logs", {"job_id": "j10"}, max_retries=3)
        assert chain.execute.call_count == 3
        assert result is not None


# =============================================================================
# Tests — SupabaseWriteError
# =============================================================================


class TestSupabaseWriteError:
    """Tests for the custom exception class."""

    def test_error_attributes(self):
        """SupabaseWriteError stores table, operation, attempts, last_error."""
        err = SupabaseWriteError("detections", "INSERT", 3, ValueError("boom"))
        assert err.table == "detections"
        assert err.operation == "INSERT"
        assert err.attempts == 3
        assert isinstance(err.last_error, ValueError)

    def test_error_message(self):
        """String representation includes table and attempt count."""
        err = SupabaseWriteError("agent_logs", "UPDATE", 5, RuntimeError("net"))
        assert "agent_logs" in str(err)
        assert "5" in str(err)
