"""Which run belongs to which WhatsApp chat.

LangGraph already persists each run's state. What it cannot know is who asked
for the run, so a reply arriving from a phone number can be matched to the run
waiting on it, including after a restart. This is a small SQLite file beside
the checkpoints.

It also remembers which webhook messages were handled, because Meta re-delivers
a message when it does not get a fast ``200``, and answering a gate twice would
be wrong.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    video_id   TEXT PRIMARY KEY,
    phone      TEXT NOT NULL,
    created_at REAL NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS runs_phone ON runs (phone, active);
CREATE TABLE IF NOT EXISTS assets (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    phone      TEXT NOT NULL,
    video_id   TEXT,
    path       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    mime       TEXT NOT NULL,
    caption    TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS assets_phone ON assets (phone, video_id);
CREATE TABLE IF NOT EXISTS seen_messages (
    message_id TEXT PRIMARY KEY,
    seen_at    REAL NOT NULL
);
"""

DAY_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class StoredAsset:
    """A file a person sent, and where it is."""

    path: str
    kind: str
    mime: str
    caption: str


class ConversationStore:
    """Phone number to active run, plus webhook de-duplication."""

    def __init__(self, path: str) -> None:
        # One connection, used only from the event loop thread.
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def start(self, phone: str, video_id: str) -> None:
        """Record a new run as the phone's active one."""
        with self._db:
            self._db.execute(
                "UPDATE runs SET active = 0 WHERE phone = ?", (phone,)
            )
            self._db.execute(
                "INSERT INTO runs (video_id, phone, created_at) VALUES (?, ?, ?)",
                (video_id, phone, time.time()),
            )

    def active_run(self, phone: str) -> str | None:
        """The run this phone is currently talking about, if any."""
        row = self._db.execute(
            "SELECT video_id FROM runs WHERE phone = ? AND active = 1 "
            "ORDER BY created_at DESC LIMIT 1",
            (phone,),
        ).fetchone()
        return row[0] if row else None

    def phone_for(self, video_id: str) -> str | None:
        """Who asked for ``video_id``. None for runs that did not come from chat."""
        row = self._db.execute(
            "SELECT phone FROM runs WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row[0] if row else None

    def finish(self, video_id: str) -> None:
        """Mark a run over, so the phone's next message starts a new one."""
        with self._db:
            self._db.execute(
                "UPDATE runs SET active = 0 WHERE video_id = ?", (video_id,)
            )

    def runs_in_last_day(self, phone: str) -> int:
        """How many runs this phone started in the last 24 hours."""
        row = self._db.execute(
            "SELECT COUNT(*) FROM runs WHERE phone = ? AND created_at > ?",
            (phone, time.time() - DAY_SECONDS),
        ).fetchone()
        return int(row[0])

    # ── Uploads ───────────────────────────────────────────────────────────
    # A person sends photos, video or audio first and says what to make after, so
    # a file is "staged" for the phone until a request claims it for a run. Every
    # method that removes rows returns the paths, because deleting the files is
    # the caller's job: this class only keeps the records.

    def stage_asset(self, phone: str, path: str, kind: str, mime: str, caption: str = "") -> None:
        """Record a file the phone sent that no run has claimed yet."""
        with self._db:
            self._db.execute(
                "INSERT INTO assets (phone, path, kind, mime, caption, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (phone, path, kind, mime, caption, time.time()),
            )

    def staged_assets(self, phone: str) -> list[StoredAsset]:
        """Files the phone sent that are waiting for a request, oldest first."""
        rows = self._db.execute(
            "SELECT path, kind, mime, caption FROM assets "
            "WHERE phone = ? AND video_id IS NULL ORDER BY id",
            (phone,),
        ).fetchall()
        return [StoredAsset(*row) for row in rows]

    def claim_assets(self, phone: str, video_id: str) -> None:
        """Hand the phone's staged files to a run, so they go when the run does."""
        with self._db:
            self._db.execute(
                "UPDATE assets SET video_id = ? WHERE phone = ? AND video_id IS NULL",
                (video_id, phone),
            )

    def release_assets(self, video_id: str) -> list[str]:
        """Forget a run's files. Returns their paths, to be deleted."""
        return self._remove("video_id = ?", (video_id,))

    def release_staged(self, phone: str) -> list[str]:
        """Forget the phone's unclaimed files. Returns their paths, to be deleted."""
        return self._remove("phone = ? AND video_id IS NULL", (phone,))

    def stale_assets(self, staged_after: float, claimed_after: float) -> list[str]:
        """Forget files nobody came back for. Returns their paths.

        Args:
            staged_after: Seconds an unclaimed file may wait for a request.
            claimed_after: Seconds a run's files may outlive their run, for a chat
                that was left at a review and never answered.
        """
        now = time.time()
        return self._remove(
            "(video_id IS NULL AND created_at < ?) OR (video_id IS NOT NULL AND created_at < ?)",
            (now - staged_after, now - claimed_after),
        )

    def _remove(self, where: str, args: tuple) -> list[str]:
        with self._db:
            rows = self._db.execute(f"SELECT path FROM assets WHERE {where}", args).fetchall()
            self._db.execute(f"DELETE FROM assets WHERE {where}", args)
        return [row[0] for row in rows]

    def first_sighting(self, message_id: str) -> bool:
        """Record a webhook message id; True the first time it is seen."""
        with self._db:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO seen_messages (message_id, seen_at) VALUES (?, ?)",
                (message_id, time.time()),
            )
            # Keep the table small. Meta retries an unacknowledged delivery for
            # up to 7 days, so remember a message for longer than that.
            self._db.execute(
                "DELETE FROM seen_messages WHERE seen_at < ?",
                (time.time() - 8 * DAY_SECONDS,),
            )
        return cursor.rowcount == 1
