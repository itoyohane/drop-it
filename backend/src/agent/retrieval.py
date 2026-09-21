"""Project-scoped retrieval and Set candidate selection."""

from __future__ import annotations

import numpy as np

from backend.agent.set_planning import is_camelot_compatible
from backend.models import MusicFilters, MusicMatch, Track
from backend.music.text_models import TextEmbedder, unit_vector
from backend.repositories import GLOBAL_PROJECT_ID, DropItStore


class MissingTrackReferenceError(ValueError):
    """The requested reference is not present in the selected library."""


class AmbiguousTrackReferenceError(ValueError):
    """The requested reference resolves to more than one scoped track."""


class DropItToolRegistry:
    """Search one server-selected music library; never choose project scope."""

    names = ["search_library", "find_similar_tracks", "generate_dj_set"]
    provider = "librosa-deepseek-dashscope-chroma-rag"

    def __init__(self, store: DropItStore, embedder: TextEmbedder):
        self.store, self.embedder = store, embedder

    def catalog(self, project_id: str, filters: MusicFilters | None = None) -> list[Track]:
        constraints = filters or MusicFilters()
        scope = None if project_id == GLOBAL_PROJECT_ID else project_id
        return [track for track in self.store.all_tracks(scope) if constraints.matches(track)]

    def search(self, project_id: str, query: str = "", filters: MusicFilters | None = None,
               k: int = 20) -> list[MusicMatch]:
        candidates = self.catalog(project_id, filters)
        if not query.strip():
            return [MusicMatch(track=track) for track in candidates[:k]]
        if not candidates:
            return []
        vectors = self.store.music_vectors(project_id, self.embedder.model_key)
        if not any(track.id in vectors for track in candidates):
            raise ValueError("候选曲目尚未建立当前歌曲描述索引，请先完成音频分析。")
        return self._rank(candidates, vectors, self.embedder.text(query))[:k]

    def similar(self, project_id: str, track_id: str, filters: MusicFilters | None = None,
                k: int = 3) -> list[MusicMatch]:
        catalog = self.catalog(project_id)
        reference = next((track for track in catalog if track.id == track_id), None)
        if reference is None:
            raise ValueError("当前曲库没有这个 track_id，请先调用 search_library 确认曲目。")
        constraints = filters or MusicFilters()
        candidates = [
            track for track in catalog
            if track.id != track_id and constraints.matches(track)
        ]
        vectors = self.store.music_vectors(project_id, self.embedder.model_key)
        if reference.id not in vectors:
            raise ValueError("参考歌曲的描述索引尚未就绪，请先完成分析。")
        matches = self._rank(candidates, vectors, vectors[reference.id])
        for match in matches:
            track = match.track
            tempo = max(0, 1 - abs(track.bpm - reference.bpm) / 20)
            key = float(is_camelot_compatible(track.camelot_key, reference.camelot_key))
            energy = 1 - abs(track.energy - reference.energy)
            match.score = round(
                .8 * (match.description_similarity or 0)
                + .1 * tempo + .05 * key + .05 * energy,
                6,
            )
        return sorted(matches, key=lambda item: (-float(item.score or 0), item.track.id))[:k]

    def resolve_reference(self, project_id: str, reference: str) -> Track:
        value = reference.strip()
        if value.lower().startswith("track_id:"):
            value = value.split(":", 1)[1].strip()
        catalog = self.catalog(project_id)
        by_id = [track for track in catalog if track.id.casefold() == value.casefold()]
        if len(by_id) == 1:
            return by_id[0]
        exact_titles = [
            track for track in catalog if track.title.casefold() == value.casefold()
        ]
        if len(exact_titles) == 1:
            return exact_titles[0]
        if len(exact_titles) > 1:
            labels = ", ".join(
                f"{track.title} ({track.artist}, {track.id})" for track in exact_titles[:8]
            )
            raise AmbiguousTrackReferenceError(
                f"参考歌曲名称不唯一，请指定 track_id：{labels}"
            )
        raise MissingTrackReferenceError(
            "当前项目曲库没有找到这首参考歌曲，请先搜索确认歌曲名称或 track_id。"
        )

    def retrieve_set_candidates(
        self,
        project_id: str,
        *,
        bpm_min: float,
        bpm_max: float,
        style_query: str = "",
        track_ids: list[str] | None = None,
    ) -> list[Track]:
        if project_id == GLOBAL_PROJECT_ID:
            raise ValueError("请进入具体项目后生成 Set。")
        filters = MusicFilters(bpm_min=bpm_min, bpm_max=bpm_max)
        candidates = (
            [
                match.track
                for match in self.search(project_id, style_query, filters, k=200)
            ]
            if style_query.strip()
            else self.catalog(project_id, filters)
        )
        if track_ids is not None:
            allowed = {track.id for track in self.catalog(project_id)}
            if not track_ids or not set(track_ids) <= allowed:
                raise ValueError("选曲列表为空或包含当前项目之外的曲目。")
            selected_ids = set(track_ids)
            candidates = [track for track in candidates if track.id in selected_ids]
        return candidates

    def _rank(
        self, tracks: list[Track], vectors: dict[str, np.ndarray], query: np.ndarray
    ) -> list[MusicMatch]:
        query = unit_vector(query)
        if len(query) != self.embedder.dimensions:
            raise ValueError("文本向量维度与配置不一致，请重建歌曲描述索引。")
        tracks = [track for track in tracks if track.id in vectors]
        if not tracks:
            return []
        if any(len(vectors[track.id]) != len(query) for track in tracks):
            raise ValueError("歌曲描述索引损坏或维度不一致，请重新分析。")
        matrix = np.stack([unit_vector(vectors[track.id]) for track in tracks])
        scores = np.clip(matrix @ query, -1, 1)
        matches = [
            MusicMatch(
                track=track,
                score=round(float(score), 6),
                description_similarity=round(float(score), 6),
            )
            for track, score in zip(tracks, scores)
        ]
        return sorted(matches, key=lambda item: (-float(item.score or 0), item.track.id))


__all__ = [
    "AmbiguousTrackReferenceError",
    "DropItToolRegistry",
    "MissingTrackReferenceError",
]
