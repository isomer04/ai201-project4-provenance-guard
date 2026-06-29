import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

# Path to the SQLite audit database (defined in config.py)
from config import AUDIT_DB_PATH

os.makedirs(os.path.dirname(AUDIT_DB_PATH), exist_ok=True)

# Schema creation SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    content_id TEXT NOT NULL,
    creator_id TEXT,
    text TEXT,
    signals TEXT,
    combined_score REAL,
    label TEXT,
    status TEXT NOT NULL,
    degraded TEXT,
    payload TEXT NOT NULL,
    links_to TEXT,
    timestamp TEXT NOT NULL
);
"""

CREATE_INDEX_CONTENT_ID = (
    "CREATE INDEX IF NOT EXISTS idx_content_id ON events (content_id);"
)
CREATE_INDEX_TIMESTAMP = (
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON events (timestamp);"
)


def get_connection() -> sqlite3.Connection:
    """Create and return a connection to the audit database."""
    conn = sqlite3.connect(AUDIT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def bootstrap_db():
    """Create the events table and indexes if they do not exist. Idempotent."""
    with get_connection() as conn:
        conn.executescript(
            CREATE_TABLE_SQL + CREATE_INDEX_CONTENT_ID + CREATE_INDEX_TIMESTAMP
        )
        conn.commit()


def _current_timestamp() -> str:
    """Get the current UTC timestamp in ISO format with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_event(
    event_type: str,
    content_id: str,
    creator_id: str | None,
    status: str,
    *,
    text: str | None = None,
    signals: dict | None = None,
    combined_score: float | None = None,
    label: dict | None = None,
    degraded: dict | None = None,
    payload: dict | None = None,
    links_to: str | None = None,
    timestamp: str | None = None,
) -> str:
    """Insert a new audit event row.

    Returns the generated event_id (UUID4).
    """
    event_id = str(uuid.uuid4())
    ts = timestamp or _current_timestamp()
    payload_json = json.dumps(payload or {})
    signals_json = json.dumps(signals) if signals is not None else None
    label_json = json.dumps(label) if label is not None else None
    degraded_json = json.dumps(degraded) if degraded is not None else None
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO events (
                event_id, event_type, content_id, creator_id, text, signals,
                combined_score, label, status, degraded, payload, links_to, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                content_id,
                creator_id,
                text,
                signals_json,
                combined_score,
                label_json,
                status,
                degraded_json,
                payload_json,
                links_to,
                ts,
            ),
        )
        conn.commit()
    return event_id


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a database row to a dictionary, deserializing JSON columns."""

    def _load(col):
        val = row[col]
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val
        return val

    return {
        "event_id": row["event_id"],
        "event_type": row["event_type"],
        "content_id": row["content_id"],
        "creator_id": row["creator_id"],
        "text": row["text"],
        "signals": _load("signals"),
        "combined_score": row["combined_score"],
        "label": _load("label"),
        "status": row["status"],
        "degraded": _load("degraded"),
        "payload": _load("payload"),
        "links_to": row["links_to"],
        "timestamp": row["timestamp"],
    }


def get_log() -> list[dict]:
    """Return all events ordered newest first."""
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM events ORDER BY timestamp DESC, rowid DESC")
        return [_row_to_dict(row) for row in cur]


def get_latest_event_for(content_id: str) -> dict | None:
    """Return the most recent event for a given content_id, or None if none exist."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM events WHERE content_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT 1",
            (content_id,),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def get_events_for(content_id: str) -> list[dict]:
    """Return all events for a content_id, oldest first."""
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM events WHERE content_id = ? ORDER BY timestamp ASC, rowid ASC",
            (content_id,),
        )
        return [_row_to_dict(row) for row in cur]


def get_original_classification_for(content_id: str) -> dict | None:
    """Return the earliest 'classification' event for a given content_id, or None.

    This is used to determine the original creator_id and original classification
    details for ownership checks and audit linking.
    """
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM events WHERE content_id = ? AND event_type = 'classification' ORDER BY timestamp ASC, rowid ASC LIMIT 1",
            (content_id,),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def get_appeal_queue() -> list[dict]:
    """Return the latest event per content_id whose status is 'under_review'."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT e.* FROM events e
            WHERE e.rowid = (
                SELECT rowid FROM events e2
                WHERE e2.content_id = e.content_id
                ORDER BY timestamp DESC, rowid DESC LIMIT 1
            )
            AND e.status = 'under_review'
            ORDER BY e.timestamp DESC, e.rowid DESC
            """
        )
        return [_row_to_dict(row) for row in cur]
