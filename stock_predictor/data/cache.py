"""SQLite-backed cache for external data."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class SQLiteCache:
    """Small SQLite cache for JSON-serializable payloads."""

    _locks: dict[Path, threading.Lock] = {}
    _locks_mutex = threading.Lock()

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _get_lock(self) -> threading.Lock:
        with SQLiteCache._locks_mutex:
            if self.db_path not in SQLiteCache._locks:
                SQLiteCache._locks[self.db_path] = threading.Lock()
            return SQLiteCache._locks[self.db_path]

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_lock():
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL
                )
                """
            )

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT value, expires_at
                FROM cache_entries
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        if row["expires_at"] is not None and row["expires_at"] < now:
            self.delete(key)
            return None
        return json.loads(row["value"])

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = None if ttl_seconds is None else time.time() + ttl_seconds
        payload = json.dumps(value, default=str)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (cache_key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET
                    value = excluded.value,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (key, payload, time.time(), expires_at),
            )

    def delete(self, key: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (key,))

    def clear(self) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM cache_entries")
