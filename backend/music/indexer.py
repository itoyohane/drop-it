"""Persist a track-description vector using the active text model version."""

from backend.music.text_models import TextEmbedder
from backend.repositories.embeddings import EmbeddingsRepository


class MusicIndexer:
    def __init__(self, embeddings: EmbeddingsRepository, embedder: TextEmbedder):
        self.embeddings = embeddings
        self.embedder = embedder

    def index(self, track_id: str, description: str) -> None:
        vector = self.embedder.text(description)
        if vector.shape != (self.embedder.dimensions,):
            raise ValueError("文本模型返回的音乐向量维度不正确")
        self.embeddings.save_embedding(track_id, self.embedder.model_key, vector)
