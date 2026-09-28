"""SQLite ticket log powering the dashboard's history and analytics."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .pii import redact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    text        TEXT NOT NULL,
    intent      TEXT NOT NULL,
    confidence  REAL NOT NULL,
    category    TEXT NOT NULL,
    team        TEXT NOT NULL,
    priority    TEXT NOT NULL,
    escalate    INTEGER NOT NULL,
    latency_ms  REAL NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at);
"""


class TicketStore:
    def __init__(self, path: Path | str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def add(self, text: str, result: dict) -> int:
        # Tickets are stored redacted: the log should never become a PII store.
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO tickets (created_at, text, intent, confidence, category, team, priority, escalate,"
                " latency_ms, result_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    redact(text).text,
                    result["intent"],
                    result["confidence"],
                    result["category"],
                    result["team"],
                    result["priority"],
                    int(result["escalate"]),
                    result["latency_ms"]["total"],
                    json.dumps(result),
                ),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, created_at, text, intent, confidence, category, team, priority, escalate"
            " FROM tickets ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) | {"escalate": bool(r["escalate"])} for r in rows]

    def get(self, ticket_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["result"] = json.loads(data.pop("result_json"))
        data["escalate"] = bool(data["escalate"])
        return data

    def stats(self) -> dict:
        c = self._conn
        total, escalated, avg_conf, avg_latency = c.execute(
            "SELECT COUNT(*), COALESCE(SUM(escalate),0), AVG(confidence), AVG(latency_ms) FROM tickets"
        ).fetchone()

        def breakdown(column: str) -> dict[str, int]:
            rows = c.execute(f"SELECT {column}, COUNT(*) FROM tickets GROUP BY {column} ORDER BY 2 DESC").fetchall()
            return {k: n for k, n in rows}

        return {
            "total": total,
            "escalation_rate": round(escalated / total, 3) if total else 0.0,
            "avg_confidence": round(avg_conf or 0.0, 3),
            "avg_latency_ms": round(avg_latency or 0.0, 1),
            "by_priority": breakdown("priority"),
            "by_team": breakdown("team"),
            "by_intent": breakdown("intent"),
        }

    def close(self) -> None:
        self._conn.close()
