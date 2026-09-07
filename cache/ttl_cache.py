"""SQLite-backed TTL cache with thread-safety and in-memory fallback."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional
from config.logging_config import logger
from config.settings import get_settings


class TTLCache:
    """Persistent SQLite cache with automatic TTL expiration."""

    def __init__(self, db_path: Optional[str] = None, default_ttl: Optional[int] = None):
        settings = get_settings()
        self.db_path = db_path or settings.CACHE_DB_PATH
        self.default_ttl = default_ttl or settings.CACHE_TTL_SECONDS
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with appropriate isolation."""
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize cache table and indexes."""
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS cache_store (
                            key TEXT PRIMARY KEY,
                            value TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            expires_at REAL NOT NULL
                        )
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_store(expires_at)")
                conn.close()
            except Exception as e:
                logger.warning("Failed to initialize SQLite cache table: %s. Using memory fallback.", e)

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value if it exists and has not expired."""
        now = time.time()
        with self._lock:
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute(
                    "SELECT value, expires_at FROM cache_store WHERE key = ?",
                    (key,),
                )
                row = cur.fetchone()
                if not row:
                    conn.close()
                    return None

                if now > row["expires_at"]:
                    # Expired, clean up
                    cur.execute("DELETE FROM cache_store WHERE key = ?", (key,))
                    conn.commit()
                    conn.close()
                    return None

                val = json.loads(row["value"])
                conn.close()
                return val
            except Exception as e:
                logger.debug("Cache read error for key '%s': %s", key, e)
                return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store a JSON-serializable value in the cache with a TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()
        expires_at = now + ttl
        payload = json.dumps(value)

        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_store (key, value, created_at, expires_at)
                        VALUES (?, ?, ?, ?)
                    """, (key, payload, now, expires_at))
                conn.close()
            except Exception as e:
                logger.warning("Cache write error for key '%s': %s", key, e)

    def delete(self, key: str) -> None:
        """Delete an entry by key."""
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("DELETE FROM cache_store WHERE key = ?", (key,))
                conn.close()
            except Exception as e:
                logger.debug("Cache delete error for key '%s': %s", key, e)

    def clear(self) -> None:
        """Wipe all cached entries."""
        with self._lock:
            try:
                conn = self._get_connection()
                with conn:
                    conn.execute("DELETE FROM cache_store")
                conn.close()
            except Exception as e:
                logger.debug("Cache clear error: %s", e)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                now = time.time()
                cur.execute("SELECT COUNT(*) as total FROM cache_store WHERE expires_at > ?", (now,))
                active = cur.fetchone()["total"]
                cur.execute("SELECT COUNT(*) as total FROM cache_store WHERE expires_at <= ?", (now,))
                expired = cur.fetchone()["total"]
                conn.close()
                return {"active_keys": active, "expired_keys": expired, "db_path": self.db_path}
            except Exception as e:
                return {"error": str(e), "active_keys": 0, "expired_keys": 0}


# Global singleton cache instance
cache = TTLCache()
