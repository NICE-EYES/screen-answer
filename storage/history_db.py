"""Локальная история вопросов/ответов и сводок экрана в SQLite."""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    question     TEXT    NOT NULL,
    answer       TEXT    NOT NULL,
    context_kind TEXT    NOT NULL DEFAULT '',
    ocr_text     TEXT    NOT NULL DEFAULT '',
    model        TEXT    NOT NULL DEFAULT '',
    image_sent   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS screen_summaries (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    hint    TEXT NOT NULL,
    excerpt TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts);
"""

MAX_SUMMARIES = 500


@dataclass
class HistoryEntry:
    id: int
    timestamp: str
    question: str
    answer: str
    context_kind: str = ""
    model: str = ""
    image_sent: bool = False

    @property
    def time_label(self) -> str:
        try:
            dt = datetime.fromisoformat(self.timestamp)
            if dt.date() == datetime.now().date():
                return dt.strftime("%H:%M")
            return dt.strftime("%d.%m %H:%M")
        except ValueError:
            return self.timestamp


class HistoryDB:
    """Потокобезопасная обёртка над SQLite. Данные никуда не отправляются."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def add_entry(self, question: str, answer: str, context_kind: str = "", ocr_text: str = "",
                  model: str = "", image_sent: bool = False) -> HistoryEntry:
        ts = self._now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO history(ts, question, answer, context_kind, ocr_text, model, image_sent)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, question, answer, context_kind, ocr_text[:20000], model, int(image_sent)),
            )
            self._conn.commit()
            row_id = int(cur.lastrowid or 0)
        return HistoryEntry(row_id, ts, question, answer, context_kind, model, image_sent)

    def recent(self, limit: int = 30) -> list[HistoryEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, question, answer, context_kind, model, image_sent"
                " FROM history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [HistoryEntry(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6])) for r in rows]

    def add_summary(self, kind: str, hint: str, excerpt: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO screen_summaries(ts, kind, hint, excerpt) VALUES (?, ?, ?, ?)",
                (self._now(), kind, hint, excerpt[:1000]),
            )
            self._conn.execute(
                "DELETE FROM screen_summaries WHERE id NOT IN"
                " (SELECT id FROM screen_summaries ORDER BY id DESC LIMIT ?)",
                (MAX_SUMMARIES,),
            )
            self._conn.commit()

    def clear(self) -> None:
        """Полностью стирает историю и сжимает файл базы."""
        with self._lock:
            self._conn.execute("DELETE FROM history")
            self._conn.execute("DELETE FROM screen_summaries")
            self._conn.execute("DELETE FROM sqlite_sequence")
            self._conn.commit()
            self._conn.execute("VACUUM")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                log.warning("Ошибка закрытия базы: %s", exc)
