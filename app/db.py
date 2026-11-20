"""Database connection helper.

Connections are short-lived and used as context managers: the transaction is
committed when the block succeeds and rolled back when it raises. Every
connection has a connect timeout and a statement timeout, so a slow or
unreachable database fails fast instead of hanging a request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from app.config import settings
from app.ingestion.errors import DatabaseUnavailableError

CONNECT_TIMEOUT_SECONDS = 5
STATEMENT_TIMEOUT_MS = 30_000


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Open a database connection for one unit of work.

    Raises:
        DatabaseUnavailableError: If the database is unreachable or drops the connection.
    """
    try:
        conn = psycopg.connect(
            settings.database_url,
            connect_timeout=CONNECT_TIMEOUT_SECONDS,
            options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
        )
        with conn:  # commits on success, rolls back on error, then closes
            yield conn
    except psycopg.OperationalError as exc:
        raise DatabaseUnavailableError("The database is not reachable or timed out.") from exc
