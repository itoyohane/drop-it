import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from backend.models import (
    ChatMessage, Conversation, Job, LibrarySource, Playlist, Project, ToolEvent, Track,
    TrackUpdate, derive_agent_playlist_id,
)
from backend.repositories.chroma import ChromaEmbeddingsRepository


GLOBAL_PROJECT_ID = "global-chat"


class StaleAgentRunError(RuntimeError):
    """A write was attempted by a run owner whose lease/token is no longer current."""

    def __init__(self, run_id: str):
        super().__init__(f"Agent run {run_id} 的 claim 已失效，拒绝 stale owner 写入")
        self.run_id = run_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteRepository:
    """Concrete repository implementation shared by API requests and workers."""

    def __init__(self, db_path: str = "data/dropit.db", vector_store_path: str | Path | None = None) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.lock = threading.RLock()
        self._closed = False
        chroma_path = vector_store_path or (":memory:" if db_path == ":memory:" else path.parent / "chroma")
        self.vector_store: ChromaEmbeddingsRepository | None = None
        try:
            self.vector_store = ChromaEmbeddingsRepository(chroma_path)
            self._migrate()
        except BaseException:
            try:
                if self.vector_store is not None:
                    self.vector_store.close()
            except Exception:
                pass
            self.connection.close()
            self._closed = True
            raise

    def close(self) -> None:
        with self.lock:
            if self._closed:
                return
            try:
                if self.vector_store is not None:
                    self.vector_store.close()
            finally:
                self.connection.close()
                self._closed = True

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
                if version == 10:
                    self.connection.execute("BEGIN IMMEDIATE")
                    try:
                        self._apply_agent_run_claims_migration()
                        self.connection.execute(
                            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                            (version, _now()),
                        )
                        self.connection.commit()
                    except BaseException:
                        self.connection.rollback()
                        raise
                else:
                    self.connection.executescript(migration.read_text(encoding="utf-8"))
                    self.connection.execute(
                        "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                        (version, _now()),
                    )
                    self.connection.commit()

    def _migration_fault_point(self, stage: str) -> None:
        """No-op hook used to verify that schema upgrades are transactional."""

    def _apply_agent_run_claims_migration(self) -> None:
        """Upgrade both the original 009 schema and an early 009 implementation.

        The first implementation of 009 shipped some of the columns that now
        belong to 010.  Introspection keeps existing installations upgradeable
        while a fresh database still follows the documented 009 -> 010 path.
        """

        def table_exists(table: str) -> bool:
            return self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is not None

        def columns(table: str) -> set[str]:
            return {str(row["name"]) for row in self.connection.execute(f"PRAGMA table_info({table})")}

        def versioned_steps(table: str) -> bool:
            step_columns = columns(table)
            step_pk = [
                str(row["name"])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
                if row["pk"]
            ]
            return "version" in step_columns and step_pk == ["run_id", "step", "version"]

        def create_steps(table: str) -> None:
            self.connection.execute(
                f"""CREATE TABLE {table} (
                    run_id TEXT NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
                    step TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, step, version)
                )"""
            )

        def copy_steps(source: str, target: str = "agent_run_steps") -> None:
            source_columns = columns(source)
            version_expr = "version" if "version" in source_columns else "0"
            self.connection.execute(
                f"""INSERT OR IGNORE INTO {target} (run_id, step, version, state, created_at)
                    SELECT run_id, step, {version_expr}, state, created_at FROM {source}"""
            )

        run_columns = columns("agent_runs")
        for definition in (
            ("latest_step", "TEXT"),
            ("latest_version", "INTEGER NOT NULL DEFAULT 0"),
            ("claim_owner", "TEXT"),
            ("claim_token", "INTEGER NOT NULL DEFAULT 0"),
            ("claim_expires_at", "REAL"),
        ):
            if definition[0] not in run_columns:
                self.connection.execute(f"ALTER TABLE agent_runs ADD COLUMN {definition[0]} {definition[1]}")

        current_exists = table_exists("agent_run_steps")
        legacy_exists = table_exists("agent_run_steps_legacy")
        if current_exists and not versioned_steps("agent_run_steps") and not legacy_exists:
            self.connection.execute("ALTER TABLE agent_run_steps RENAME TO agent_run_steps_legacy")
            self._migration_fault_point("agent_run_steps_renamed")
            current_exists = False
            legacy_exists = True

        if legacy_exists and current_exists and not versioned_steps("agent_run_steps"):
            # Tolerate an unusual partially-applied state containing two old
            # tables by merging both into a fresh versioned table.
            create_steps("agent_run_steps_v10")
            copy_steps("agent_run_steps", "agent_run_steps_v10")
            copy_steps("agent_run_steps_legacy", "agent_run_steps_v10")
            self.connection.execute("DROP TABLE agent_run_steps")
            self.connection.execute("ALTER TABLE agent_run_steps_v10 RENAME TO agent_run_steps")
            self._migration_fault_point("agent_run_steps_created")
            current_exists = True
        elif not current_exists:
            create_steps("agent_run_steps")
            self._migration_fault_point("agent_run_steps_created")
            current_exists = True

        if legacy_exists:
            copy_steps("agent_run_steps_legacy")
            self.connection.execute("DROP TABLE agent_run_steps_legacy")
        self.connection.execute("DROP INDEX IF EXISTS idx_agent_run_steps_run_created")
        self.connection.execute(
            "CREATE INDEX idx_agent_run_steps_run_created "
            "ON agent_run_steps(run_id, created_at)"
        )

        playlist_columns = columns("playlists")
        if "agent_run_id" not in playlist_columns:
            self.connection.execute("ALTER TABLE playlists ADD COLUMN agent_run_id TEXT")
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_playlists_agent_run "
            "ON playlists(agent_run_id) WHERE agent_run_id IS NOT NULL"
        )

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

    # Agent runs/checkpoints are deliberately repository methods rather than a
    # second persistence implementation.  The agent graph can therefore use
    # the same transaction/lock as Playlist writes.
    def create_agent_run(self, run_id: str, project_id: str, conversation_id: str,
                         route: str, user_text: str,
                         history: list[dict[str, str]]) -> dict[str, Any]:
        now = _now()
        with self.lock:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO agent_runs
                   (run_id, project_id, conversation_id, route, user_text, history,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
                (run_id, project_id, conversation_id, route, user_text,
                 json.dumps(history, ensure_ascii=False), now, now),
            )
            inserted = cursor.rowcount == 1
            self.connection.commit()
            existing = self.connection.execute(
                "SELECT * FROM agent_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if not existing:
                raise RuntimeError("无法创建 Agent run")
            if (existing["project_id"], existing["conversation_id"], existing["user_text"]) != (
                project_id, conversation_id, user_text
            ):
                raise ValueError("run_id 已绑定到另一条 Agent 任务")
        result = self.get_agent_run(run_id)
        if result is None:
            raise RuntimeError("无法读取 Agent run")
        result["created"] = bool(inserted)
        return result

    def _begin_write(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def _claim_row_locked(self, run_id: str, owner: str, token: int | None) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT claim_owner, claim_token, claim_expires_at FROM agent_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        now = time.time()
        if (
            not row
            or token is None
            or row["claim_owner"] != owner
            or int(row["claim_token"] or 0) != int(token)
            or row["claim_expires_at"] is None
            or float(row["claim_expires_at"]) <= now
        ):
            raise StaleAgentRunError(run_id)
        return row

    def set_agent_run_history(self, run_id: str, history: list[dict[str, str]],
                              owner: str | None = None, token: int | None = None) -> None:
        with self.lock:
            self._begin_write()
            try:
                self._claim_row_locked(run_id, owner or "", token)
                self.connection.execute(
                    "UPDATE agent_runs SET history=?, updated_at=? "
                    "WHERE run_id=? AND claim_owner=? AND claim_token=? AND claim_expires_at>?",
                    (json.dumps(history, ensure_ascii=False), _now(), run_id, owner, token, time.time()),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise

    def claim_agent_run(self, run_id: str, owner: str, lease_seconds: float = 15.0) -> int | None:
        """Claim or renew a run and return its monotonic fencing token."""

        if lease_seconds <= 0:
            raise ValueError("Agent run lease 必须为正数")
        now = time.time()
        with self.lock:
            self._begin_write()
            try:
                row = self.connection.execute(
                    "SELECT claim_owner, claim_token, claim_expires_at FROM agent_runs WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if not row:
                    self.connection.rollback()
                    return None
                expires = row["claim_expires_at"]
                current_token = int(row["claim_token"] or 0)
                if row["claim_owner"] not in (None, owner) and expires is not None and float(expires) > now:
                    self.connection.rollback()
                    return None
                if row["claim_owner"] == owner and expires is not None and float(expires) > now:
                    token = current_token
                else:
                    token = current_token + 1
                self.connection.execute(
                    """UPDATE agent_runs
                       SET claim_owner=?, claim_token=?, claim_expires_at=?, updated_at=?
                       WHERE run_id=?""",
                    (owner, token, now + lease_seconds, _now(), run_id),
                )
                self.connection.commit()
                return token
            except Exception:
                self.connection.rollback()
                raise

    def renew_agent_run(self, run_id: str, owner: str, token: int,
                        lease_seconds: float = 15.0) -> bool:
        """Extend a live claim without changing its fencing token."""

        if lease_seconds <= 0:
            raise ValueError("Agent run lease 必须为正数")
        now = time.time()
        with self.lock:
            self._begin_write()
            try:
                cursor = self.connection.execute(
                    """UPDATE agent_runs
                       SET claim_expires_at=?, updated_at=?
                       WHERE run_id=? AND claim_owner=? AND claim_token=?
                         AND claim_expires_at>?""",
                    (now + lease_seconds, _now(), run_id, owner, token, now),
                )
                self.connection.commit()
                return cursor.rowcount == 1
            except Exception:
                self.connection.rollback()
                raise

    def release_agent_run(self, run_id: str, owner: str, token: int | None = None) -> None:
        with self.lock:
            self._begin_write()
            try:
                self._claim_row_locked(run_id, owner, token)
                self.connection.execute(
                    """UPDATE agent_runs SET claim_owner=NULL, claim_expires_at=NULL, updated_at=?
                       WHERE run_id=? AND claim_owner=? AND claim_token=?""",
                    (_now(), run_id, owner, token),
                )
                self.connection.commit()
            except StaleAgentRunError:
                self.connection.rollback()
                # A lease may expire naturally while unwinding a crashed or
                # cancelled stream; release is best-effort, not a stale write.
                return
            except Exception:
                self.connection.rollback()
                raise

    def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM agent_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["history"] = json.loads(result.get("history") or "[]")
        return result

    def update_agent_run(self, run_id: str, *, owner: str | None = None,
                         token: int | None = None, **changes: Any) -> dict[str, Any] | None:
        allowed = {"status", "error_code", "latest_step", "latest_version"}
        updates = {key: value for key, value in changes.items() if key in allowed}
        if not updates:
            return self.get_agent_run(run_id)
        updates["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self.lock:
            self._begin_write()
            try:
                self._claim_row_locked(run_id, owner or "", token)
                cursor = self.connection.execute(
                    f"UPDATE agent_runs SET {assignments} "
                    "WHERE run_id=? AND claim_owner=? AND claim_token=? AND claim_expires_at>?",
                    [*updates.values(), run_id, owner, token, time.time()],
                )
                if cursor.rowcount != 1:
                    raise StaleAgentRunError(run_id)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return self.get_agent_run(run_id)

    def save_agent_checkpoint(self, run_id: str, step: str, state: str,
                              version: int = 0, owner: str | None = None,
                              token: int | None = None) -> dict[str, Any]:
        now = _now()
        with self.lock:
            self._begin_write()
            try:
                self._claim_row_locked(run_id, owner or "", token)
                self.connection.execute(
                    """INSERT OR IGNORE INTO agent_run_steps (run_id, step, version, state, created_at)
                       VALUES (?, ?, ?, ?, ?)""", (run_id, step, version, state, now)
                )
                inserted = self.connection.execute("SELECT changes() AS count").fetchone()["count"]
                if inserted:
                    cursor = self.connection.execute(
                        """UPDATE agent_runs SET latest_step=?, latest_version=?, updated_at=?
                           WHERE run_id=? AND claim_owner=? AND claim_token=? AND claim_expires_at>?""",
                        (step, version, now, run_id, owner, token, time.time()),
                    )
                    if cursor.rowcount != 1:
                        raise StaleAgentRunError(run_id)
                row = self.connection.execute(
                    """SELECT * FROM agent_run_steps
                       WHERE run_id=? AND step=? AND version=?""", (run_id, step, version)
                ).fetchone()
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        if not row:
            raise ValueError(f"Agent run 不存在：{run_id}")
        return dict(row)

    def get_agent_checkpoint(self, run_id: str, step: str,
                             version: int | None = None) -> dict[str, Any] | None:
        with self.lock:
            if version is None:
                row = self.connection.execute(
                    """SELECT * FROM agent_run_steps WHERE run_id=? AND step=?
                       ORDER BY version DESC LIMIT 1""", (run_id, step)
                ).fetchone()
            else:
                row = self.connection.execute(
                    """SELECT * FROM agent_run_steps
                       WHERE run_id=? AND step=? AND version=?""", (run_id, step, version)
                ).fetchone()
        return dict(row) if row else None

    def get_latest_agent_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                """SELECT s.* FROM agent_run_steps s
                   JOIN agent_runs r ON r.run_id=s.run_id
                       AND r.latest_step=s.step AND r.latest_version=s.version
                   WHERE s.run_id=?""", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_agent_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM agent_run_steps WHERE run_id=? ORDER BY created_at, rowid",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_agent_run_message(self, run_id: str, role: str) -> ChatMessage | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM messages WHERE agent_run_id=? AND role=? LIMIT 1",
                (run_id, role),
            ).fetchone()
        return self._message(row) if row else None

    def add_message(self, project_id: str, conversation_id: str, role: str, content: str,
                    tool_events: list[ToolEvent] | None = None,
                    agent_run_id: str | None = None,
                    agent_owner: str | None = None,
                    agent_token: int | None = None) -> ChatMessage:
        message_id, now = uuid4().hex, _now()
        events = [event.model_dump() for event in (tool_events or [])]
        with self.lock:
            run = None
            if agent_run_id:
                run = self.connection.execute(
                    "SELECT project_id, conversation_id FROM agent_runs WHERE run_id=?",
                    (agent_run_id,),
                ).fetchone()
                if not run or (run["project_id"], run["conversation_id"]) != (
                    project_id, conversation_id
                ):
                    raise ValueError("agent_run_id 已绑定到另一项目或会话")
                existing = self.connection.execute(
                    "SELECT * FROM messages WHERE agent_run_id=? AND role=? LIMIT 1",
                    (agent_run_id, role),
                ).fetchone()
                if existing:
                    if (existing["project_id"], existing["conversation_id"], existing["content"]) != (
                        project_id, conversation_id, content
                    ):
                        raise ValueError("run_id 对应消息与当前项目、会话或内容不匹配")
                    if role == "assistant" and (
                        existing["content"] != content or existing["tool_events"] != json.dumps(events, ensure_ascii=False)
                    ):
                        self._begin_write()
                        try:
                            self._claim_row_locked(agent_run_id, agent_owner or "", agent_token)
                            self.connection.execute(
                                "UPDATE messages SET content=?, tool_events=? WHERE id=?",
                                (content, json.dumps(events, ensure_ascii=False), existing["id"]),
                            )
                            self.connection.commit()
                            existing = self.connection.execute(
                                "SELECT * FROM messages WHERE id=?", (existing["id"],)
                            ).fetchone()
                        except Exception:
                            self.connection.rollback()
                            raise
                    return self._message(existing)
                # User-message setup is intentionally allowed before a worker
                # claims the run.  Any final assistant insert/update must be
                # fenced by the live owner/token.
                if role != "user":
                    self._begin_write()
                    try:
                        self._claim_row_locked(agent_run_id, agent_owner or "", agent_token)
                    except Exception:
                        self.connection.rollback()
                        raise
            elif agent_owner is not None or agent_token is not None:
                raise ValueError("agent_owner/agent_token 只能与 agent_run_id 一起使用")
            transaction_started = False
            try:
                if agent_run_id and role != "user":
                    transaction_started = True
                elif not agent_run_id:
                    self._begin_write()
                    transaction_started = True
                self.connection.execute(
                    """INSERT INTO messages
                       (id, project_id, conversation_id, role, content, tool_events, created_at, agent_run_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (message_id, project_id, conversation_id, role, content,
                     json.dumps(events, ensure_ascii=False), now, agent_run_id),
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
            except Exception:
                if transaction_started:
                    self.connection.rollback()
                raise
        return ChatMessage(id=message_id, project_id=project_id, conversation_id=conversation_id,
                           role=role, content=content, tool_events=tool_events or [], created_at=now)

    @staticmethod
    def _message(row: sqlite3.Row) -> ChatMessage:
        return ChatMessage(
            id=row["id"], project_id=row["project_id"], conversation_id=row["conversation_id"],
            role=row["role"], content=row["content"],
            tool_events=json.loads(row["tool_events"] or "[]"), created_at=row["created_at"],
        )

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
                   analysis_error=?, analyzer=?, analysis_details=?, description=?, description_model=?,
                   updated_at=? WHERE id=?""",
                (track.title, track.artist, track.duration_sec, track.bpm, track.key, track.camelot_key,
                 track.energy, track.bpm_confidence, track.key_confidence, track.analysis_status,
                 track.analysis_error, track.analyzer,
                 json.dumps(track.analysis_details), track.description, track.description_model,
                 _now(), track.id),
            )
            self.connection.commit()
        return self.get_track(track.id)  # type: ignore[return-value]

    def update_track(self, track_id: str, update: TrackUpdate) -> Track | None:
        with self.lock:
            cursor = self.connection.execute(
                """UPDATE tracks SET title=?, artist=?, bpm=?, musical_key=?, camelot_key=?,
                   energy=?, description='', description_model='', embedding_status='pending',
                   embedding_model='', embedding_error=NULL, updated_at=? WHERE id=?""",
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
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("无效的音乐向量")
        norm = float(np.linalg.norm(values))
        if norm < 1e-8:
            raise ValueError("音乐向量不能为零")
        # Vectors are intentionally not serialized into SQLite anymore. The
        # relational DB stores only readiness/model metadata; Chroma owns the
        # vector payload and persists it under the configured data directory.
        self.vector_store.save_embedding(track_id, model, values)
        with self.lock, self.connection:
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
        """Read authorized vectors from Chroma after checking membership in SQL."""
        join = "" if project_id == GLOBAL_PROJECT_ID else "JOIN project_tracks pt ON pt.track_id=t.id"
        scope = "" if project_id == GLOBAL_PROJECT_ID else "AND pt.project_id=?"
        params = (model,) if project_id == GLOBAL_PROJECT_ID else (model, project_id)
        with self.lock:
            rows = self.connection.execute(
                f"""SELECT t.id FROM tracks t {join}
                    WHERE t.analysis_status='analyzed'
                    AND t.embedding_status='ready' AND t.embedding_model=? {scope}""", params,
            ).fetchall()
        track_ids = [str(row["id"]) for row in rows]
        return self.vector_store.vectors(track_ids, model)

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
            self._begin_write()
            try:
                existing = self.connection.execute(
                    "SELECT project_id, payload, agent_run_id FROM playlists WHERE id=?",
                    (playlist.id,),
                ).fetchone()
                if existing and existing["project_id"] != playlist.project_id:
                    raise ValueError("Playlist ID 已绑定到另一项目")
                self.connection.execute(
                    """INSERT INTO playlists (id, project_id, payload, created_at, updated_at, agent_run_id)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,
                                                     updated_at=excluded.updated_at,
                                                     agent_run_id=excluded.agent_run_id""",
                    (playlist.id, playlist.project_id, playlist.model_dump_json(), now, now,
                     playlist.agent_run_id),
                )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
        return playlist

    def save_agent_playlist(self, run_id: str, project_id: str, playlist: Playlist,
                            owner: str, token: int) -> Playlist:
        """Atomically validate the claim and persist one canonical Agent playlist."""

        expected_id = derive_agent_playlist_id(project_id, run_id)
        if playlist.project_id != project_id:
            raise ValueError("Agent Playlist 项目范围与当前 run 不一致")
        if playlist.id != expected_id:
            raise ValueError("Agent Playlist ID 不是由 project_id+run_id 规范派生")
        if playlist.agent_run_id != run_id:
            raise ValueError("Agent Playlist payload 缺少匹配的 agent_run_id")
        now = _now()
        if playlist.created_at is None:
            playlist.created_at = datetime.fromisoformat(now)
        with self.lock:
            self._begin_write()
            try:
                self._claim_row_locked(run_id, owner, token)
                existing = self.connection.execute(
                    "SELECT id, project_id, payload, agent_run_id FROM playlists "
                    "WHERE agent_run_id=? OR id=? LIMIT 1",
                    (run_id, expected_id),
                ).fetchone()
                if existing:
                    if existing["id"] != expected_id or existing["project_id"] != project_id \
                            or existing["agent_run_id"] != run_id:
                        raise ValueError("Agent Playlist ID 或 agent_run_id 发生冲突")
                    try:
                        persisted = Playlist.model_validate_json(existing["payload"])
                    except Exception as exc:
                        raise ValueError("已有 Agent Playlist payload 无法验证") from exc
                    if (
                        persisted.id != expected_id
                        or persisted.project_id != project_id
                        or persisted.agent_run_id != run_id
                    ):
                        raise ValueError("已有 Agent Playlist payload 与规范不匹配")
                    self.connection.commit()
                    return persisted
                self.connection.execute(
                    """INSERT INTO playlists
                       (id, project_id, payload, created_at, updated_at, agent_run_id)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (expected_id, project_id, playlist.model_dump_json(), now, now, run_id),
                )
                self.connection.commit()
                return playlist
            except Exception:
                self.connection.rollback()
                raise

    def get_playlist(self, playlist_id: str) -> Playlist | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT project_id, payload, agent_run_id FROM playlists WHERE id=?", (playlist_id,)
            ).fetchone()
        if not row:
            return None
        playlist = Playlist.model_validate_json(row["payload"])
        if playlist.project_id != row["project_id"] or playlist.agent_run_id != row["agent_run_id"]:
            raise ValueError("Playlist payload 与数据库范围不匹配")
        return playlist

    def list_playlists(self, project_id: str) -> list[Playlist]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT project_id, payload, agent_run_id FROM playlists "
                "WHERE project_id=? ORDER BY updated_at DESC", (project_id,)
            ).fetchall()
        playlists = []
        for row in rows:
            playlist = Playlist.model_validate_json(row["payload"])
            if playlist.project_id != row["project_id"] or playlist.agent_run_id != row["agent_run_id"]:
                raise ValueError("Playlist payload 与数据库范围不匹配")
            playlists.append(playlist)
        return playlists

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
            description=row["description"] if "description" in keys else "",
            description_model=row["description_model"] if "description_model" in keys else "",
            embedding_status=row["embedding_status"], embedding_model=row["embedding_model"],
            embedding_error=row["embedding_error"],
            created_at=row["created_at"], updated_at=row["updated_at"] if "updated_at" in keys else None,
        )


DropItStore = SqliteRepository
LibraryStore = SqliteRepository
