import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from backend.models import ChatMessage, Conversation, Job, LibrarySource, Playlist, Project, ToolEvent, Track, TrackUpdate


GLOBAL_PROJECT_ID = "global-chat"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteRepository:
    """Concrete repository implementation shared by API requests and workers."""

    def __init__(self, db_path: str = "data/dropit.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.lock = threading.RLock()
        self._migrate()

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def _migrate(self) -> None:
        with self.lock:
            has_migrations = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            if not has_migrations:
                has_legacy = self.connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracks'"
                ).fetchone()
                self.connection.execute(
                    "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
                )
                if has_legacy:
                    self.connection.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (1, ?)", (_now(),)
                    )
                self.connection.commit()

            applied = {row["version"] for row in self.connection.execute("SELECT version FROM schema_migrations")}
            migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
            for migration in sorted(migrations_dir.glob("*.sql")):
                version = int(migration.stem.split("_", 1)[0])
                if version in applied:
                    continue
                self.connection.executescript(migration.read_text(encoding="utf-8"))
                self.connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, _now())
                )
                self.connection.commit()

    def create_project(self, name: str, description: str = "") -> Project:
        project_id, now = uuid4().hex, _now()
        with self.lock:
            self.connection.execute(
                "INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (project_id, name.strip(), description.strip(), now, now),
            )
            self._insert_conversation(project_id, "新对话", now)
            self.connection.commit()
        return self.get_project(project_id)  # type: ignore[return-value]

    def update_project(self, project_id: str, name: str, description: str = "") -> Project | None:
        with self.lock:
            cursor = self.connection.execute(
                "UPDATE projects SET name=?, description=?, updated_at=? WHERE id=?",
                (name.strip(), description.strip(), _now(), project_id),
            )
            self.connection.commit()
        return self.get_project(project_id) if cursor.rowcount else None

    def list_projects(self) -> list[Project]:
        with self.lock:
            rows = self.connection.execute(
                """SELECT p.*, COUNT(pt.track_id) AS track_count
                   FROM projects p LEFT JOIN project_tracks pt ON pt.project_id=p.id
                   WHERE p.scope='project'
                   GROUP BY p.id ORDER BY p.updated_at DESC"""
            ).fetchall()
        return [Project(**dict(row)) for row in rows]

    def get_project(self, project_id: str) -> Project | None:
        with self.lock:
            row = self.connection.execute(
                """SELECT p.*, COUNT(pt.track_id) AS track_count
                   FROM projects p LEFT JOIN project_tracks pt ON pt.project_id=p.id
                   WHERE p.id=? GROUP BY p.id""", (project_id,),
            ).fetchone()
        return Project(**dict(row)) if row else None

    def get_or_create_source(self, folder_key: str, name: str) -> tuple[LibrarySource, bool]:
        with self.lock:
            row = self.connection.execute(
                "SELECT id FROM library_sources WHERE folder_key=?", (folder_key,)
            ).fetchone()
            if row:
                return self.get_source(row["id"]), True  # type: ignore[return-value]
            source_id, now = uuid4().hex, _now()
            self.connection.execute(
                """INSERT INTO library_sources (id, folder_key, name, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (source_id, folder_key, name.strip(), now, now),
            )
            self.connection.commit()
        return self.get_source(source_id), False  # type: ignore[return-value]

    def get_source(self, source_id: str) -> LibrarySource | None:
        with self.lock:
            row = self.connection.execute(
                """SELECT s.*, COUNT(st.track_id) AS track_count
                   FROM library_sources s LEFT JOIN source_tracks st ON st.source_id=s.id
                   WHERE s.id=? GROUP BY s.id""", (source_id,),
            ).fetchone()
        return LibrarySource(**dict(row)) if row else None

    def list_sources(self, project_id: str) -> list[LibrarySource]:
        with self.lock:
            rows = self.connection.execute(
                """SELECT s.*, COUNT(st.track_id) AS track_count
                   FROM library_sources s
                   JOIN project_sources ps ON ps.source_id=s.id
                   LEFT JOIN source_tracks st ON st.source_id=s.id
                   WHERE ps.project_id=? GROUP BY s.id ORDER BY ps.added_at DESC""", (project_id,),
            ).fetchall()
        return [LibrarySource(**dict(row)) for row in rows]

    def bind_source(self, project_id: str, source_id: str) -> None:
        now = _now()
        with self.lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO project_sources (project_id, source_id, added_at) VALUES (?, ?, ?)",
                (project_id, source_id, now),
            )
            self.connection.execute(
                """INSERT OR IGNORE INTO project_tracks (project_id, track_id, added_at)
                   SELECT ?, track_id, ? FROM source_tracks WHERE source_id=?""",
                (project_id, now, source_id),
            )
            self.connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self.connection.commit()

    def project_source_ids(self, project_id: str) -> list[str]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT source_id FROM project_sources WHERE project_id=? ORDER BY added_at", (project_id,)
            ).fetchall()
        return [str(row["source_id"]) for row in rows]

    def source_tracks(self, source_id: str) -> list[Track]:
        with self.lock:
            rows = self.connection.execute(
                """SELECT t.* FROM tracks t JOIN source_tracks st ON st.track_id=t.id
                   WHERE st.source_id=? ORDER BY t.artist, t.title""", (source_id,),
            ).fetchall()
        return [self._track(row) for row in rows]

    def source_ids_for_tracks(self, track_ids: list[str], project_id: str | None = None) -> dict[str, list[str]]:
        if not track_ids:
            return {}
        placeholders = ",".join("?" for _ in track_ids)
        params: list[Any] = list(track_ids)
        project_clause = ""
        if project_id:
            project_clause = " AND ps.project_id=?"
            params.append(project_id)
        with self.lock:
            rows = self.connection.execute(
                f"""SELECT st.source_id, st.track_id FROM source_tracks st
                    JOIN project_sources ps ON ps.source_id=st.source_id
                    WHERE st.track_id IN ({placeholders}){project_clause}""", params,
            ).fetchall()
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(str(row["source_id"]), []).append(str(row["track_id"]))
        return grouped

    def update_source_status(self, source_id: str, status: str) -> None:
        with self.lock:
            self.connection.execute(
                "UPDATE library_sources SET status=?, updated_at=? WHERE id=?", (status, _now(), source_id)
            )
            self.connection.commit()

    def delete_project(self, project_id: str) -> bool:
        with self.lock:
            cursor = self.connection.execute("DELETE FROM projects WHERE id=?", (project_id,))
            self.connection.commit()
        return cursor.rowcount > 0

    def create_conversation(self, project_id: str, title: str = "新对话") -> Conversation:
        with self.lock:
            conversation = self._insert_conversation(project_id, title, _now())
            self.connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (_now(), project_id))
            self.connection.commit()
        return conversation

    def ensure_default_conversation(self, project_id: str) -> Conversation:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM conversations WHERE project_id=? ORDER BY updated_at DESC LIMIT 1", (project_id,)
            ).fetchone()
            if row:
                return Conversation(**dict(row))
            conversation = self._insert_conversation(project_id, "新对话", _now())
            self.connection.commit()
            return conversation

    def _insert_conversation(self, project_id: str, title: str, now: str) -> Conversation:
        conversation_id = uuid4().hex
        title = title.strip() or "新对话"
        self.connection.execute(
            "INSERT INTO conversations (id, project_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, project_id, title, now, now),
        )
        return Conversation(id=conversation_id, project_id=project_id, title=title, created_at=now, updated_at=now)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return Conversation(**dict(row)) if row else None

    def list_conversations(self, project_id: str) -> list[Conversation]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM conversations WHERE project_id=? ORDER BY updated_at DESC", (project_id,)
            ).fetchall()
        return [Conversation(**dict(row)) for row in rows]

    def delete_conversation(self, project_id: str, conversation_id: str) -> bool:
        with self.lock:
            count = self.connection.execute(
                "SELECT COUNT(*) AS total FROM conversations WHERE project_id=?", (project_id,)
            ).fetchone()["total"]
            if count <= 1:
                return False
            cursor = self.connection.execute(
                "DELETE FROM conversations WHERE id=? AND project_id=?", (conversation_id, project_id)
            )
            self.connection.commit()
        return cursor.rowcount > 0

    def add_message(self, project_id: str, conversation_id: str, role: str, content: str,
                    tool_events: list[ToolEvent] | None = None) -> ChatMessage:
        message_id, now = uuid4().hex, _now()
        events = [event.model_dump() for event in (tool_events or [])]
        with self.lock:
            self.connection.execute(
                """INSERT INTO messages
                   (id, project_id, conversation_id, role, content, tool_events, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (message_id, project_id, conversation_id, role, content,
                 json.dumps(events, ensure_ascii=False), now),
            )
            if role == "user":
                total = self.connection.execute(
                    "SELECT COUNT(*) AS total FROM messages WHERE conversation_id=?", (conversation_id,)
                ).fetchone()["total"]
                if total == 1:
                    title = " ".join(content.strip().split())[:36] or "新对话"
                    self.connection.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))
            self.connection.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            self.connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self.connection.commit()
        return ChatMessage(id=message_id, project_id=project_id, conversation_id=conversation_id,
                           role=role, content=content, tool_events=tool_events or [], created_at=now)

    def list_messages(self, project_id: str, conversation_id: str, limit: int = 80) -> list[ChatMessage]:
        with self.lock:
            rows = self.connection.execute(
                """SELECT * FROM (
                       SELECT * FROM messages WHERE project_id=? AND conversation_id=?
                       ORDER BY created_at DESC LIMIT ?
                   ) ORDER BY created_at""", (project_id, conversation_id, limit),
            ).fetchall()
        return [ChatMessage(id=row["id"], project_id=row["project_id"],
                            conversation_id=row["conversation_id"], role=row["role"],
                            content=row["content"], tool_events=json.loads(row["tool_events"] or "[]"),
                            created_at=row["created_at"]) for row in rows]

    def upsert_track(self, track: Track, file_hash: str, project_id: str,
                     source_id: str | None = None) -> Track:
        now = _now()
        with self.lock:
            existing = self.connection.execute("SELECT id FROM tracks WHERE file_hash=?", (file_hash,)).fetchone()
            if existing:
                track_id = existing["id"]
            else:
                track_id = track.id
                self.connection.execute(
                    """INSERT INTO tracks (
                        id, file_hash, title, artist, filename, path, duration_sec, bpm,
                        musical_key, camelot_key, energy, bpm_confidence, key_confidence,
                        analysis_status, analysis_error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (track_id, file_hash, track.title, track.artist, track.filename, track.path,
                     track.duration_sec, track.bpm, track.key, track.camelot_key, track.energy,
                     track.bpm_confidence, track.key_confidence, track.analysis_status,
                     track.analysis_error, now, now),
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO project_tracks (project_id, track_id, added_at) VALUES (?, ?, ?)",
                (project_id, track_id, now),
            )
            if source_id:
                self.connection.execute(
                    "INSERT OR IGNORE INTO source_tracks (source_id, track_id, added_at) VALUES (?, ?, ?)",
                    (source_id, track_id, now),
                )
                self.connection.execute(
                    """INSERT OR IGNORE INTO project_tracks (project_id, track_id, added_at)
                       SELECT project_id, ?, ? FROM project_sources WHERE source_id=?""",
                    (track_id, now, source_id),
                )
            self.connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (now, project_id))
            self.connection.commit()
        return self.get_track(track_id)  # type: ignore[return-value]

    def update_track_analysis(self, track: Track) -> Track:
        with self.lock:
            self.connection.execute(
                """UPDATE tracks SET title=?, artist=?, duration_sec=?, bpm=?, musical_key=?,
                   camelot_key=?, energy=?, bpm_confidence=?, key_confidence=?, analysis_status=?,
                   analysis_error=?, analyzer=?, analysis_details=?, updated_at=? WHERE id=?""",
                (track.title, track.artist, track.duration_sec, track.bpm, track.key, track.camelot_key,
                 track.energy, track.bpm_confidence, track.key_confidence, track.analysis_status,
                 track.analysis_error, track.analyzer,
                 json.dumps(track.analysis_details), _now(), track.id),
            )
            self.connection.commit()
        return self.get_track(track.id)  # type: ignore[return-value]

    def update_track(self, track_id: str, update: TrackUpdate) -> Track | None:
        with self.lock:
            cursor = self.connection.execute(
                """UPDATE tracks SET title=?, artist=?, bpm=?, musical_key=?, camelot_key=?,
                   energy=?, updated_at=? WHERE id=?""",
                (update.title.strip(), update.artist.strip(), update.bpm, update.key.strip(),
                 update.camelot_key.strip(), update.energy, _now(), track_id),
            )
            self.connection.commit()
        return self.get_track(track_id) if cursor.rowcount else None

    def mark_track_analyzing(self, track_id: str) -> None:
        with self.lock:
            self.connection.execute(
                """UPDATE tracks SET analysis_status='analyzing', analysis_error=NULL,
                   embedding_status='pending', embedding_error=NULL, updated_at=? WHERE id=?""",
                (_now(), track_id),
            )
            self.connection.commit()

    def get_track(self, track_id: str) -> Track | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        return self._track(row) if row else None

    def save_embedding(self, track_id: str, model: str, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype="<f4")
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("无效的音乐向量")
        norm = float(np.linalg.norm(values))
        if norm < 1e-8:
            raise ValueError("音乐向量不能为零")
        values = values / norm
        with self.lock, self.connection:
            self.connection.execute(
                """INSERT INTO music_embeddings (track_id, model, dimensions, vector, updated_at)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(track_id, model) DO UPDATE SET
                   dimensions=excluded.dimensions, vector=excluded.vector, updated_at=excluded.updated_at""",
                (track_id, model, values.size, values.tobytes(), _now()),
            )
            self.connection.execute(
                """UPDATE tracks SET embedding_status='ready', embedding_model=?,
                   embedding_error=NULL, updated_at=? WHERE id=?""", (model, _now(), track_id),
            )

    def fail_embedding(self, track_id: str, error: str) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE tracks SET embedding_status='failed', embedding_error=?, updated_at=? WHERE id=?",
                (error[:500], _now(), track_id),
            )

    def music_vectors(self, project_id: str, model: str) -> dict[str, np.ndarray]:
        """Membership is checked in SQL before any vectors enter the retriever."""
        join = "" if project_id == GLOBAL_PROJECT_ID else "JOIN project_tracks pt ON pt.track_id=t.id"
        scope = "" if project_id == GLOBAL_PROJECT_ID else "AND pt.project_id=?"
        params = (model,) if project_id == GLOBAL_PROJECT_ID else (model, project_id)
        with self.lock:
            rows = self.connection.execute(
                f"""SELECT e.track_id, e.vector FROM music_embeddings e
                    JOIN tracks t ON t.id=e.track_id {join}
                    WHERE e.model=? AND t.analysis_status='analyzed'
                    AND t.embedding_status='ready' AND t.embedding_model=e.model {scope}""", params,
            ).fetchall()
        return {row["track_id"]: np.frombuffer(row["vector"], dtype="<f4").copy() for row in rows}

    def all_tracks(self, project_id: str | None = None) -> list[Track]:
        with self.lock:
            if project_id:
                rows = self.connection.execute(
                    """SELECT t.* FROM tracks t JOIN project_tracks pt ON pt.track_id=t.id
                       WHERE pt.project_id=? ORDER BY t.artist, t.title""", (project_id,),
                ).fetchall()
            else:
                rows = self.connection.execute("SELECT * FROM tracks ORDER BY artist, title").fetchall()
        return [self._track(row) for row in rows]

    def tracks_by_ids(self, track_ids: list[str], project_id: str | None = None) -> list[Track]:
        allowed = {track.id: track for track in self.all_tracks(project_id)}
        return [allowed[track_id] for track_id in track_ids if track_id in allowed]

    def remove_track_from_project(self, project_id: str, track_id: str) -> bool:
        with self.lock:
            cursor = self.connection.execute(
                "DELETE FROM project_tracks WHERE project_id=? AND track_id=?", (project_id, track_id)
            )
            if cursor.rowcount:
                self.connection.execute("UPDATE projects SET updated_at=? WHERE id=?", (_now(), project_id))
            self.connection.commit()
        return cursor.rowcount > 0

    def save_playlist(self, playlist: Playlist) -> Playlist:
        now = _now()
        if playlist.created_at is None:
            playlist.created_at = datetime.fromisoformat(now)
        with self.lock:
            self.connection.execute(
                """INSERT INTO playlists (id, project_id, payload, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at""",
                (playlist.id, playlist.project_id, playlist.model_dump_json(), now, now),
            )
            self.connection.commit()
        return playlist

    def get_playlist(self, playlist_id: str) -> Playlist | None:
        with self.lock:
            row = self.connection.execute("SELECT payload FROM playlists WHERE id=?", (playlist_id,)).fetchone()
        return Playlist.model_validate_json(row["payload"]) if row else None

    def list_playlists(self, project_id: str) -> list[Playlist]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT payload FROM playlists WHERE project_id=? ORDER BY updated_at DESC", (project_id,)
            ).fetchall()
        return [Playlist.model_validate_json(row["payload"]) for row in rows]

    def delete_playlist(self, playlist_id: str) -> bool:
        with self.lock:
            cursor = self.connection.execute("DELETE FROM playlists WHERE id=?", (playlist_id,))
            self.connection.commit()
        return cursor.rowcount > 0

    def create_job(self, project_id: str, kind: str, payload: dict[str, Any], total: int = 0) -> Job:
        job_id, now = uuid4().hex, _now()
        with self.lock:
            self.connection.execute(
                """INSERT INTO jobs
                   (id, project_id, kind, status, progress, total, message, payload, error, created_at, updated_at)
                   VALUES (?, ?, ?, 'queued', 0, ?, ?, ?, NULL, ?, ?)""",
                (job_id, project_id, kind, total, "等待处理", json.dumps(payload), now, now),
            )
            self.connection.commit()
        return self.get_job(job_id)  # type: ignore[return-value]

    def update_job(self, job_id: str, **changes: Any) -> Job | None:
        allowed = {"status", "progress", "total", "message", "error"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        if not updates:
            return self.get_job(job_id)
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self.lock:
            self.connection.execute(f"UPDATE jobs SET {assignments} WHERE id=?", [*updates.values(), job_id])
            self.connection.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job | None:
        with self.lock:
            row = self.connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job(row) if row else None

    def list_jobs(self, project_id: str, limit: int = 20) -> list[Job]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ?", (project_id, limit)
            ).fetchall()
        return [self._job(row) for row in rows]

    def recover_jobs(self) -> list[Job]:
        with self.lock:
            self.connection.execute(
                "UPDATE jobs SET status='queued', message='服务重启，重新排队', updated_at=? WHERE status='running'",
                (_now(),),
            )
            self.connection.commit()
            rows = self.connection.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at").fetchall()
        return [self._job(row) for row in rows]

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        data = dict(row)
        data["payload"] = json.loads(data["payload"] or "{}")
        return Job(**data)

    @staticmethod
    def _track(row: sqlite3.Row) -> Track:
        keys = set(row.keys())
        return Track(
            id=row["id"], title=row["title"], artist=row["artist"], filename=row["filename"],
            path=row["path"], duration_sec=row["duration_sec"], bpm=row["bpm"],
            key=row["musical_key"], camelot_key=row["camelot_key"], energy=row["energy"],
            bpm_confidence=row["bpm_confidence"], key_confidence=row["key_confidence"],
            analysis_status=row["analysis_status"], analysis_error=row["analysis_error"],
            analyzer=row["analyzer"], analysis_details=json.loads(row["analysis_details"]),
            embedding_status=row["embedding_status"], embedding_model=row["embedding_model"],
            embedding_error=row["embedding_error"],
            created_at=row["created_at"], updated_at=row["updated_at"] if "updated_at" in keys else None,
        )


DropItStore = SqliteRepository
LibraryStore = SqliteRepository
