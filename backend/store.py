import json
import sqlite3
from pathlib import Path

from backend.models import Track


class LibraryStore:
    def __init__(self, db_path: str = "data/dropit.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY, title TEXT, artist TEXT, path TEXT UNIQUE,
            duration_sec INTEGER, bpm REAL, musical_key TEXT, energy REAL,
            mood TEXT, role TEXT)"""
        )
        self.connection.commit()

    def replace(self, tracks: list[Track]) -> None:
        self.connection.execute("DELETE FROM tracks")
        self.connection.executemany(
            "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(t.id, t.title, t.artist, t.path, t.duration_sec, t.bpm, t.key,
              t.energy, json.dumps(t.mood, ensure_ascii=False), t.role) for t in tracks],
        )
        self.connection.commit()

    def all(self) -> list[Track]:
        rows = self.connection.execute("SELECT * FROM tracks ORDER BY artist, title").fetchall()
        return [Track(id=r["id"], title=r["title"], artist=r["artist"], path=r["path"],
                      duration_sec=r["duration_sec"], bpm=r["bpm"], key=r["musical_key"],
                      energy=r["energy"], mood=json.loads(r["mood"]), role=r["role"])
                for r in rows]
