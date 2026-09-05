"""Repository contract for versioned music embeddings."""

from typing import Protocol

import numpy as np


class EmbeddingsRepository(Protocol):
    def save_embedding(self, track_id: str, model: str, vector: np.ndarray) -> None: ...
    def fail_embedding(self, track_id: str, error: str) -> None: ...
    def music_vectors(self, project_id: str, model: str) -> dict[str, np.ndarray]: ...
