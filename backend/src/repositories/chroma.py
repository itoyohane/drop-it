"""Chroma-backed storage for versioned music vectors.

The relational catalog still owns track membership and embedding status. Chroma
owns only the high-dimensional vectors; callers pass the already-authorized track
IDs when reading them, so project scoping remains enforced by SQLite.
"""

from hashlib import sha256
from pathlib import Path
from threading import RLock

import numpy as np


class ChromaEmbeddingsRepository:
    def __init__(self, path: str | Path = "data/chroma") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - exercised in minimal installs
            raise RuntimeError(
                "Chroma 向量库未安装，请运行 python -m pip install chromadb。"
            ) from exc

        self.path = str(path)
        if self.path == ":memory:":
            self.client = chromadb.EphemeralClient()
        else:
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.path)
        self.lock = RLock()
        self._closed = False

    def close(self) -> None:
        """Release Chroma's background system/telemetry resources."""

        with self.lock:
            if self._closed:
                return
            close = getattr(self.client, "close", None)
            if close is not None:
                close()
            self._closed = True

    @staticmethod
    def _collection_name(model: str) -> str:
        # Chroma collection names have stricter characters/length rules than model
        # identifiers (which commonly contain @, /, and :).
        digest = sha256(model.encode("utf-8")).hexdigest()[:32]
        return f"music_{digest}"

    def _collection(self, model: str):
        return self.client.get_or_create_collection(
            name=self._collection_name(model),
            metadata={"hnsw:space": "cosine", "model": model},
        )

    @staticmethod
    def _id(track_id: str) -> str:
        return f"track:{track_id}"

    def save_embedding(self, track_id: str, model: str, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("无效的音乐向量")
        norm = float(np.linalg.norm(values))
        if norm < 1e-8:
            raise ValueError("音乐向量不能为零")
        values = (values / norm).astype(np.float32)
        with self.lock:
            self._collection(model).upsert(
                ids=[self._id(track_id)],
                embeddings=[values.tolist()],
                metadatas=[{"track_id": track_id, "model": model, "dimensions": values.size}],
            )

    def vectors(self, track_ids: list[str], model: str) -> dict[str, np.ndarray]:
        if not track_ids:
            return {}
        with self.lock:
            result = self._collection(model).get(
                ids=[self._id(track_id) for track_id in track_ids],
                include=["embeddings", "metadatas"],
            )
        ids = result.get("ids") or []
        embeddings = result.get("embeddings")
        embeddings = [] if embeddings is None else embeddings
        metadatas = result.get("metadatas")
        metadatas = [] if metadatas is None else metadatas
        output: dict[str, np.ndarray] = {}
        for index, (stored_id, embedding) in enumerate(zip(ids, embeddings)):
            metadata = metadatas[index] if index < len(metadatas) else None
            track_id = str((metadata or {}).get("track_id") or stored_id.removeprefix("track:"))
            values = np.asarray(embedding, dtype=np.float32)
            if values.ndim == 1 and values.size and np.isfinite(values).all():
                output[track_id] = values.copy()
        return output
