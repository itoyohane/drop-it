"""Persist one track's CLAP audio vector using the active model version."""

from backend.music.clap_embedder import MusicEmbedder
from backend.repositories.embeddings import EmbeddingsRepository


class MusicIndexer:
    def __init__(self, embeddings: EmbeddingsRepository, embedder: MusicEmbedder):
        self.embeddings = embeddings
        self.embedder = embedder

    def index(self, track_id: str, path: str) -> None:
        vector = self.embedder.audio(path)
        if vector.shape != (self.embedder.dimensions,):
            raise ValueError("CLAP 返回的音乐向量维度不正确")
        self.embeddings.save_embedding(track_id, self.embedder.model_key, vector)
