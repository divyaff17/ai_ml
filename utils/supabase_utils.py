"""
utils/supabase_utils.py — Supabase CRUD helpers with exponential-backoff retry logic.

Classes
-------
SupabaseWriteError — raised when all retries are exhausted.

Functions
---------
retry_insert(table, data, max_retries=3)
retry_fetch(table, filters, max_retries=3)
"""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger
from supabase import create_client, Client as SupabaseClient

import config

# ── Custom exception ──────────────────────────────────────────────────────────


class SupabaseWriteError(Exception):
    """Raised when a Supabase write or read operation fails after all retries."""

    def __init__(
        self, table: str, operation: str, attempts: int, last_error: Exception
    ) -> None:
        self.table = table
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Supabase {operation} on '{table}' failed after {attempts} attempts: {last_error}"
        )


# ── Client singleton ─────────────────────────────────────────────────────────

_client: Optional[SupabaseClient] = None


def _get_client() -> SupabaseClient:
    """Return (and lazily create) the Supabase client singleton."""
    global _client
    if _client is None:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _client


# ── Retry helpers ─────────────────────────────────────────────────────────────


def _backoff_sleep(attempt: int) -> None:
    """
    Sleep for ``2^attempt`` seconds (exponential backoff).

    Parameters
    ----------
    attempt : int
        Zero-indexed retry attempt number.
    """
    delay = 2**attempt
    logger.debug("Backing off for {}s before retry #{}", delay, attempt + 1)
    time.sleep(delay)


def retry_insert(
    table: str,
    data: dict[str, Any],
    *,
    max_retries: int = 3,
) -> dict:
    """
    Insert a row into a Supabase table with exponential-backoff retry.

    Parameters
    ----------
    table       : Name of the Supabase table (e.g. ``"detections"``).
    data        : Dict of column→value pairs to insert.
    max_retries : Maximum number of attempts before raising.

    Returns
    -------
    dict
        The row data returned by Supabase on success.

    Raises
    ------
    SupabaseWriteError
        If all *max_retries* attempts fail.
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = _get_client().table(table).insert(data).execute()
            logger.info(
                "Supabase INSERT succeeded — table='{}' attempt={}/{}",
                table,
                attempt + 1,
                max_retries,
            )
            return resp.data
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Supabase INSERT failed — table='{}' attempt={}/{}: {}",
                table,
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)

    logger.error(
        "Supabase INSERT exhausted all {} retries on table '{}': {}",
        max_retries,
        table,
        last_err,
    )
    raise SupabaseWriteError(table, "INSERT", max_retries, last_err)


def retry_update(
    table: str,
    data: dict[str, Any],
    *,
    eq_column: str,
    eq_value: str,
    max_retries: int = 3,
) -> dict:
    """
    Update rows in a Supabase table with exponential-backoff retry.

    Parameters
    ----------
    table      : Name of the Supabase table.
    data       : Dict of column→value updates.
    eq_column  : Column name for the equality filter (e.g. ``"job_id"``).
    eq_value   : Value the *eq_column* must equal.
    max_retries: Maximum number of attempts.

    Returns
    -------
    dict
        The updated row data on success.

    Raises
    ------
    SupabaseWriteError
        If all retries fail.
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            resp = (
                _get_client()
                .table(table)
                .update(data)
                .eq(eq_column, eq_value)
                .execute()
            )
            logger.info(
                "Supabase UPDATE succeeded — table='{}' {}='{}' attempt={}/{}",
                table,
                eq_column,
                eq_value,
                attempt + 1,
                max_retries,
            )
            return resp.data
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Supabase UPDATE failed — table='{}' attempt={}/{}: {}",
                table,
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)

    logger.error(
        "Supabase UPDATE exhausted all {} retries on table '{}': {}",
        max_retries,
        table,
        last_err,
    )
    raise SupabaseWriteError(table, "UPDATE", max_retries, last_err)


def retry_fetch(
    table: str,
    filters: dict[str, Any],
    *,
    select: str = "*",
    single: bool = False,
    max_retries: int = 3,
) -> Any:
    """
    Fetch rows from a Supabase table with exponential-backoff retry.

    Parameters
    ----------
    table       : Name of the Supabase table.
    filters     : Dict of ``{column: value}`` equality filters.
    select      : Columns to select (default ``"*"``).
    single      : If ``True``, use ``.maybe_single()`` and return one dict or ``None``.
    max_retries : Maximum number of attempts.

    Returns
    -------
    list of dict  or  dict or None
        Matching rows, or a single row / ``None`` if *single* is ``True``.

    Raises
    ------
    SupabaseWriteError
        If all retries fail (the class name is reused for read errors too).
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            query = _get_client().table(table).select(select)
            for col, val in filters.items():
                query = query.eq(col, val)
            if single:
                query = query.maybe_single()
            resp = query.execute()
            logger.debug(
                "Supabase SELECT succeeded — table='{}' attempt={}/{}",
                table,
                attempt + 1,
                max_retries,
            )
            return resp.data
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Supabase SELECT failed — table='{}' attempt={}/{}: {}",
                table,
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)

    logger.error(
        "Supabase SELECT exhausted all {} retries on table '{}': {}",
        max_retries,
        table,
        last_err,
    )
    raise SupabaseWriteError(table, "SELECT", max_retries, last_err)
