"""Explicit configuration and disposable catalog snapshots for offline/live evals."""

from pathlib import Path
from contextlib import closing
import shutil
import sqlite3
import tempfile

from backend.config import Settings
from backend.repositories import DropItStore


def evaluation_settings(args) -> Settings:
    overrides = {}
    if getattr(args, "env_file", None) is not None:
        env_file = args.env_file.resolve()
        if not env_file.is_file():
            raise FileNotFoundError(f"env file does not exist: {env_file}")
        overrides["_env_file"] = env_file
    for option, alias in (("data_dir", "DROPIT_DATA_DIR"), ("chroma_dir", "DROPIT_CHROMA_DIR")):
        value = getattr(args, option, None)
        if value is not None:
            overrides[alias] = value.resolve()
    return Settings(**overrides)


def _backup(source: Path, target: Path) -> None:
    # Online backup includes committed WAL pages; never migrate the source DB.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as reader:
        with closing(sqlite3.connect(target)) as writer, writer:
            reader.backup(writer)


class EvaluationSnapshot:
    """Keep all Agent messages, leases, checkpoints and playlists off production."""

    def __init__(self, settings: Settings):
        self.source_database = (settings.data_dir / "dropit.db").resolve()
        source_chroma = settings.resolved_chroma_dir.resolve()
        if not self.source_database.is_file():
            raise FileNotFoundError(
                f"source database does not exist: {self.source_database}; cwd={Path.cwd()}. "
                "Relative DROPIT_DATA_DIR is resolved from cwd, not from the .env file. "
                "Use --data-dir <directory containing dropit.db> and, if needed, "
                "--env-file <.env> / --chroma-dir <chroma directory>."
            )
        if not (source_chroma / "chroma.sqlite3").is_file():
            raise FileNotFoundError(f"source Chroma database does not exist: {source_chroma}; use --chroma-dir")
        self._temporary = tempfile.TemporaryDirectory(prefix="dropit-eval-")
        self.store = None
        try:
            root = Path(self._temporary.name)
            _backup(self.source_database, root / "dropit.db")
            # Chroma clients can run migrations even for read calls. Copy both
            # its SQLite metadata (including WAL) and persisted vector segments.
            shutil.copytree(source_chroma, root / "chroma", ignore=shutil.ignore_patterns("chroma.sqlite3*"))
            _backup(source_chroma / "chroma.sqlite3", root / "chroma" / "chroma.sqlite3")
            self.store = DropItStore(str(root / "dropit.db"), vector_store_path=root / "chroma")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        self._temporary.cleanup()
