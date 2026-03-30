"""SQLite-backed persistence for company tracking and notifications."""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("BARISTA_DB_PATH", "barista_ci.sqlite")
_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_company(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "url": row["url"],
        "unread_count": row["unread_count"] or 0,
        "last_scanned_at": row["last_scanned_at"],
        "next_scanned_at": row["next_scanned_at"],
        "last_run_status": row["last_run_status"] if "last_run_status" in row.keys() else None,
        "last_error": row["last_error"] if "last_error" in row.keys() else None,
        "last_duration_ms": row["last_duration_ms"] if "last_duration_ms" in row.keys() else None,
        "last_trigger": row["last_trigger"] if "last_trigger" in row.keys() else None,
        "last_run_at": row["last_run_at"] if "last_run_at" in row.keys() else None,
    }


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str):
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {c[1] for c in cols}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def create_db_and_tables():
    """Initialize database tables used by company tracking."""
    with _LOCK, _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT,
                unread_count INTEGER NOT NULL DEFAULT 0,
                last_scanned_at TEXT,
                next_scanned_at TEXT,
                last_run_status TEXT,
                last_error TEXT,
                last_duration_ms INTEGER,
                last_trigger TEXT,
                last_run_at TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_column(conn, "companies", "last_run_status", "TEXT")
        _ensure_column(conn, "companies", "last_error", "TEXT")
        _ensure_column(conn, "companies", "last_duration_ms", "INTEGER")
        _ensure_column(conn, "companies", "last_trigger", "TEXT")
        _ensure_column(conn, "companies", "last_run_at", "TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS company_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                snippet TEXT,
                source_type TEXT NOT NULL,
                published_date TEXT,
                is_read INTEGER NOT NULL DEFAULT 0,
                discovered_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_company_updates_unique_url
            ON company_updates(company_id, url)
            WHERE url IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE SET NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS report_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                report_json TEXT,
                report_pdf TEXT,
                selected_update_ids_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(company_id) REFERENCES companies(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
    logger.info("Database initialized at %s", os.path.abspath(DB_PATH))


def get_companies() -> list[dict]:
    with _LOCK, _get_conn() as conn:
        rows = conn.execute("SELECT * FROM companies ORDER BY id DESC").fetchall()
        return [_row_to_company(r) for r in rows]


def add_company(name: str, url: str | None = None) -> dict:
    now = _utc_now_iso()
    with _LOCK, _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO companies(name, url, unread_count, created_at, next_scanned_at)
            VALUES (?, ?, 0, ?, ?)
            """,
            (name, url, now, (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM companies WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_company(row)


def get_company(company_id: int) -> dict | None:
    with _LOCK, _get_conn() as conn:
        row = conn.execute("SELECT * FROM companies WHERE id = ?", (company_id,)).fetchone()
        return _row_to_company(row) if row else None


def update_company_scan_timestamps(company_id: int, *, last_scanned_at: str, next_scanned_at: str):
    with _LOCK, _get_conn() as conn:
        conn.execute(
            """
            UPDATE companies
            SET last_scanned_at = ?, next_scanned_at = ?
            WHERE id = ?
            """,
            (last_scanned_at, next_scanned_at, company_id),
        )
        conn.commit()


def update_company_scan_telemetry(
    company_id: int,
    *,
    last_run_status: str,
    last_error: str | None,
    last_duration_ms: int | None,
    last_trigger: str | None,
):
    with _LOCK, _get_conn() as conn:
        conn.execute(
            """
            UPDATE companies
            SET last_run_status = ?,
                last_error = ?,
                last_duration_ms = ?,
                last_trigger = ?,
                last_run_at = ?
            WHERE id = ?
            """,
            (
                last_run_status,
                last_error,
                last_duration_ms,
                last_trigger,
                _utc_now_iso(),
                company_id,
            ),
        )
        conn.commit()


def get_due_companies(now_iso: str | None = None) -> list[dict]:
    now_iso = now_iso or _utc_now_iso()
    with _LOCK, _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM companies
            WHERE next_scanned_at IS NULL OR next_scanned_at <= ?
            ORDER BY id ASC
            """,
            (now_iso,),
        ).fetchall()
        return [_row_to_company(r) for r in rows]


def get_company_updates(company_id: int) -> list[dict]:
    with _LOCK, _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM company_updates
            WHERE company_id = ?
            ORDER BY discovered_at DESC, id DESC
            """,
            (company_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "company_id": r["company_id"],
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "source_type": r["source_type"],
                "published_date": r["published_date"],
                "is_read": bool(r["is_read"]),
                "discovered_at": r["discovered_at"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            }
            for r in rows
        ]


def get_company_updates_by_ids(company_id: int, update_ids: list[int]) -> list[dict]:
    if not update_ids:
        return []

    placeholders = ",".join(["?"] * len(update_ids))
    params = [company_id] + [int(u) for u in update_ids]

    with _LOCK, _get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM company_updates
            WHERE company_id = ? AND id IN ({placeholders})
            ORDER BY discovered_at DESC, id DESC
            """,
            params,
        ).fetchall()

    return [
        {
            "id": r["id"],
            "company_id": r["company_id"],
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "source_type": r["source_type"],
            "published_date": r["published_date"],
            "is_read": bool(r["is_read"]),
            "discovered_at": r["discovered_at"],
            "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
        }
        for r in rows
    ]


def add_company_update(company_id: int, update: dict) -> dict | None:
    """Add an update for a company; skips duplicate URLs per company."""
    discovered_at = update.get("discovered_at") or _utc_now_iso()
    metadata = update.get("metadata") or {}

    with _LOCK, _get_conn() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO company_updates(
                    company_id, title, url, snippet, source_type,
                    published_date, is_read, discovered_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id,
                    update.get("title") or "Untitled",
                    update.get("url"),
                    update.get("snippet"),
                    update.get("source_type") or "trusted",
                    update.get("published_date"),
                    int(bool(update.get("is_read", False))),
                    discovered_at,
                    json.dumps(metadata),
                ),
            )
        except sqlite3.IntegrityError:
            return None

        if not bool(update.get("is_read", False)):
            conn.execute(
                "UPDATE companies SET unread_count = unread_count + 1 WHERE id = ?",
                (company_id,),
            )

        row = conn.execute("SELECT * FROM company_updates WHERE id = ?", (cur.lastrowid,)).fetchone()
        conn.commit()

    return {
        "id": row["id"],
        "company_id": row["company_id"],
        "title": row["title"],
        "url": row["url"],
        "snippet": row["snippet"],
        "source_type": row["source_type"],
        "published_date": row["published_date"],
        "is_read": bool(row["is_read"]),
        "discovered_at": row["discovered_at"],
        "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
    }


def mark_updates_read(company_id: int):
    with _LOCK, _get_conn() as conn:
        conn.execute("UPDATE company_updates SET is_read = 1 WHERE company_id = ?", (company_id,))
        conn.execute("UPDATE companies SET unread_count = 0 WHERE id = ?", (company_id,))
        conn.commit()


def mark_update_read(company_id: int, update_id: int) -> bool:
    with _LOCK, _get_conn() as conn:
        existing = conn.execute(
            "SELECT is_read FROM company_updates WHERE id = ? AND company_id = ?",
            (update_id, company_id),
        ).fetchone()
        if not existing:
            return False

        if existing["is_read"]:
            return True

        conn.execute(
            "UPDATE company_updates SET is_read = 1 WHERE id = ? AND company_id = ?",
            (update_id, company_id),
        )
        conn.execute(
            """
            UPDATE companies
            SET unread_count = (
                SELECT COUNT(*) FROM company_updates
                WHERE company_id = ? AND is_read = 0
            )
            WHERE id = ?
            """,
            (company_id, company_id),
        )
        conn.commit()
        return True


def add_notification(title: str, message: str, company_id: int | None = None) -> dict:
    created_at = _utc_now_iso()
    with _LOCK, _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO notifications(company_id, title, message, is_read, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (company_id, title, message, created_at),
        )
        row = conn.execute("SELECT * FROM notifications WHERE id = ?", (cur.lastrowid,)).fetchone()
        conn.commit()
    return {
        "id": row["id"],
        "company_id": row["company_id"],
        "title": row["title"],
        "message": row["message"],
        "is_read": bool(row["is_read"]),
        "created_at": row["created_at"],
    }


def get_notifications(limit: int = 100, unread_only: bool = False) -> list[dict]:
    with _LOCK, _get_conn() as conn:
        if unread_only:
            rows = conn.execute(
                """
                SELECT * FROM notifications
                WHERE is_read = 0
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notifications ORDER BY created_at DESC, id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "company_id": r["company_id"],
                "title": r["title"],
                "message": r["message"],
                "is_read": bool(r["is_read"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]


def get_unread_notification_count() -> int:
    with _LOCK, _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM notifications WHERE is_read = 0").fetchone()
        return int(row["c"] if row else 0)


def mark_notification_read(notification_id: int) -> bool:
    with _LOCK, _get_conn() as conn:
        cur = conn.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = ?",
            (notification_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def add_report_event(
    company_id: int,
    *,
    session_id: str,
    report_json: str,
    report_pdf: str,
    selected_update_ids: list[int],
) -> dict:
    created_at = _utc_now_iso()
    with _LOCK, _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO report_events(
                company_id, session_id, report_json, report_pdf,
                selected_update_ids_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                session_id,
                report_json,
                report_pdf,
                json.dumps([int(x) for x in selected_update_ids]),
                created_at,
            ),
        )
        row = conn.execute("SELECT * FROM report_events WHERE id = ?", (cur.lastrowid,)).fetchone()
        conn.commit()

    return {
        "id": row["id"],
        "company_id": row["company_id"],
        "session_id": row["session_id"],
        "report_json": row["report_json"],
        "report_pdf": row["report_pdf"],
        "selected_update_ids": json.loads(row["selected_update_ids_json"] or "[]"),
        "created_at": row["created_at"],
    }


def get_company_report_events(company_id: int) -> list[dict]:
    with _LOCK, _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM report_events
            WHERE company_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (company_id,),
        ).fetchall()

    return [
        {
            "id": r["id"],
            "company_id": r["company_id"],
            "session_id": r["session_id"],
            "report_json": r["report_json"],
            "report_pdf": r["report_pdf"],
            "selected_update_ids": json.loads(r["selected_update_ids_json"] or "[]"),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
