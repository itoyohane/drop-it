"""CLAP cosine recall plus DJ-aware reranking."""

import numpy as np

from backend.models import MusicMatch, Track
from backend.music.clap_embedder import MusicEmbedder, unit_vector
from backend.repositories.embeddings import EmbeddingsRepository
from backend.services.set_planner import is_camelot_compatible


class SimilarityService:
    def __init__(self, embeddings: EmbeddingsRepository, embedder: MusicEmbedder):
        self.embeddings = embeddings
        self.embedder = embedder

    def search(self, project_id: str, tracks: list[Track], query: str) -> list[MusicMatch]:
        vectors = self.embeddings.music_vectors(project_id, self.embedder.model_key)
        if not any(track.id in vectors for track in tracks):
            raise ValueError("候选曲目尚未建立当前 CLAP 音乐索引，请先完成音频分析。")
        return self._rank(tracks, vectors, self.embedder.text(query))

    def similar(self, project_id: str, reference: Track, candidates: list[Track]) -> list[MusicMatch]:
        vectors = self.embeddings.music_vectors(project_id, self.embedder.model_key)
        if reference.id not in vectors:
            raise ValueError("参考歌曲的 CLAP 索引尚未就绪，请先完成分析。")
        matches = self._rank(candidates, vectors, vectors[reference.id])
        for match in matches:
            track = match.track
            tempo = max(0, 1 - abs(track.bpm - reference.bpm) / 20)
            key = float(is_camelot_compatible(track.camelot_key, reference.camelot_key))
            energy = 1 - abs(track.energy - reference.energy)
            match.score = round(.8 * match.audio_similarity + .1 * tempo + .05 * key + .05 * energy, 6)
        return sorted(matches, key=lambda match: (-match.score, match.track.id))

    def _rank(self, tracks: list[Track], vectors: dict[str, np.ndarray], query) -> list[MusicMatch]:
        query = unit_vector(query)
        if len(query) != self.embedder.dimensions:
            raise ValueError("CLAP 向量维度与配置不一致，请重建音乐索引。")
        tracks = [track for track in tracks if track.id in vectors]
        if not tracks:
            return []
        if any(len(vectors[track.id]) != len(query) for track in tracks):
            raise ValueError("音乐索引损坏或维度不一致，请重新分析。")
        matrix = np.stack([unit_vector(vectors[track.id]) for track in tracks])
        scores = np.clip(matrix @ query, -1, 1)
        matches = [MusicMatch(track=track, score=round(float(score), 6),
                              audio_similarity=round(float(score), 6))
                   for track, score in zip(tracks, scores)]
        return sorted(matches, key=lambda match: (-match.score, match.track.id))
