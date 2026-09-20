"""All three Agent tool implementations and their local music-planning helpers."""

import csv
import io
import json
import math
import re
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from langchain.tools import tool
from pydantic import Field

from backend.agent.commands import GenerateSetCommand
from backend.models import (
    AgentEvent, Brief, MusicFilters, MusicMatch, Playlist, PlaylistTrack, ToolResult, Track,
)
from backend.music.text_models import TextEmbedder, unit_vector
from backend.repositories import GLOBAL_PROJECT_ID, DropItStore


class MissingTrackReferenceError(ValueError):
    """The requested reference is not present in the server-selected library."""


class AmbiguousTrackReferenceError(ValueError):
    """The requested reference resolves to more than one scoped track."""


class DropItToolRegistry:
    """Bind the Agent's tool functions to a server-selected project and local store."""

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
        candidates = [track for track in catalog if track.id != track_id and constraints.matches(track)]
        vectors = self.store.music_vectors(project_id, self.embedder.model_key)
        if reference.id not in vectors:
            raise ValueError("参考歌曲的描述索引尚未就绪，请先完成分析。")
        matches = self._rank(candidates, vectors, vectors[reference.id])
        for match in matches:
            track = match.track
            tempo = max(0, 1 - abs(track.bpm - reference.bpm) / 20)
            key = float(is_camelot_compatible(track.camelot_key, reference.camelot_key))
            energy = 1 - abs(track.energy - reference.energy)
            match.score = round(.8 * (match.description_similarity or 0) + .1 * tempo
                                + .05 * key + .05 * energy, 6)
        return sorted(matches, key=lambda item: (-float(item.score or 0), item.track.id))[:k]

    def resolve_reference(self, project_id: str, reference: str) -> Track:
        """Resolve a user/model reference only against the current project scope."""

        value = reference.strip()
        if value.lower().startswith("track_id:"):
            value = value.split(":", 1)[1].strip()
        catalog = self.catalog(project_id)
        by_id = [track for track in catalog if track.id.casefold() == value.casefold()]
        if len(by_id) == 1:
            return by_id[0]
        exact_titles = [track for track in catalog if track.title.casefold() == value.casefold()]
        if len(exact_titles) == 1:
            return exact_titles[0]
        if len(exact_titles) > 1:
            labels = ", ".join(f"{track.title} ({track.artist}, {track.id})" for track in exact_titles[:8])
            raise AmbiguousTrackReferenceError(f"参考歌曲名称不唯一，请指定 track_id：{labels}")
        raise MissingTrackReferenceError("当前项目曲库没有找到这首参考歌曲，请先搜索确认歌曲名称或 track_id。")

    def retrieve_set_candidates(self, project_id: str, command: GenerateSetCommand) -> list[Track]:
        """Retrieve Set candidates without planning or writing a playlist."""

        if project_id == GLOBAL_PROJECT_ID:
            raise ValueError("请进入具体项目后生成 Set。")
        filters = MusicFilters(bpm_min=command.bpm_min, bpm_max=command.bpm_max)
        candidates = (
            [match.track for match in self.search(project_id, command.style_query, filters, k=200)]
            if command.style_query.strip() else self.catalog(project_id, filters)
        )
        if command.track_ids is not None:
            allowed = {track.id for track in self.catalog(project_id)}
            if not command.track_ids or not set(command.track_ids) <= allowed:
                raise ValueError("选曲列表为空或包含当前项目之外的曲目。")
            selected_ids = set(command.track_ids)
            candidates = [track for track in candidates if track.id in selected_ids]
        return candidates

    def plan_set(self, project_id: str, command: GenerateSetCommand,
                 candidates: list[Track]) -> Playlist:
        brief = Brief(
            title=command.request[:60] or "Untitled set",
            duration_min=command.duration_min,
            bpm_min=command.bpm_min,
            bpm_max=command.bpm_max,
            energy=command.energy_curve,
            style=command.style_query,
            notes=command.request,
        )
        return generate_playlist(project_id, brief, candidates)

    def persist_set(self, project_id: str, playlist: Playlist) -> Playlist:
        if playlist.project_id != project_id:
            raise ValueError("Set 项目范围与当前对话不一致。")
        return self.store.save_playlist(playlist)

    def _rank(self, tracks: list[Track], vectors: dict[str, np.ndarray], query) -> list[MusicMatch]:
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
        matches = [MusicMatch(track=track, score=round(float(score), 6),
                              description_similarity=round(float(score), 6))
                   for track, score in zip(tracks, scores)]
        return sorted(matches, key=lambda item: (-float(item.score or 0), item.track.id))

    def tools_for(self, project_id: str):
        store = self.store

        @tool
        def search_library(
            query: str = "",
            filters: MusicFilters | None = None,
            limit: Annotated[int, Field(ge=1, le=100)] = 20,
        ) -> str:
            """Retrieve facts from the current music library.

            Use query for semantic search over DeepSeek descriptions generated from librosa
            features and indexed by DashScope Qwen embeddings in Chroma. For title/artist/BPM/key/energy constraints or an overview,
            leave query empty and use filters. Only returned track_ids are valid references.
            """
            try:
                return _matches_result(self.search(project_id, query, filters, k=limit))
            except (ValueError, RuntimeError) as exc:
                return _failure(exc)

        @tool
        def find_similar_tracks(
            track_id: str,
            limit: Annotated[int, Field(ge=1, le=100)] = 3,
            filters: MusicFilters | None = None,
        ) -> str:
            """Find songs with similar generated descriptions, then rerank for DJ transitions.

            Resolve track_id with search_library first. Similarity and combined scores are
            ranking signals, not match percentages.
            """
            try:
                return _matches_result(self.similar(project_id, track_id, filters, k=limit))
            except (ValueError, RuntimeError) as exc:
                return _failure(exc)

        @tool
        def generate_dj_set(
            request: str,
            duration_min: Annotated[int, Field(ge=10, le=240)] = 45,
            bpm_min: Annotated[int, Field(ge=60, le=220)] = 110,
            bpm_max: Annotated[int, Field(ge=60, le=220)] = 140,
            energy_curve: Literal["steady", "build", "peak", "wave"] = "build",
            style_query: str = "",
            track_ids: Annotated[list[str] | None, Field(max_length=500)] = None,
        ) -> str:
            """Generate and persist a DJ set from this project's analyzed songs.

            style_query searches generated song descriptions. Numerical constraints belong
            in bpm_min/bpm_max/duration_min. Never silently relax requested constraints.
            Pass track_ids to restrict the set to songs selected by an earlier tool result.
            """
            if project_id == GLOBAL_PROJECT_ID:
                return _failure(ValueError("请进入具体项目后生成 Set。"))
            try:
                command = GenerateSetCommand(
                    request=request, duration_min=duration_min, bpm_min=bpm_min,
                    bpm_max=bpm_max, energy_curve=energy_curve, style_query=style_query,
                    track_ids=track_ids,
                )
                candidates = self.retrieve_set_candidates(project_id, command)
                playlist = self.plan_set(project_id, command, candidates)
                self.persist_set(project_id, playlist)
                return ToolResult(
                    ok=True,
                    summary=f"已生成 {len(playlist.tracks)} 首、约 {playlist.duration_sec // 60} 分钟的 Set。",
                    data={"playlist_id": playlist.id, "report": playlist.report,
                          "tracks": [dict(MusicMatch(track=row.track).context(), reason=row.reason)
                                     for row in playlist.tracks]},
                ).model_dump_json()
            except (ValueError, RuntimeError) as exc:
                return _failure(exc)

        return [search_library, find_similar_tracks, generate_dj_set]


def _failure(error: Exception) -> str:
    return ToolResult(ok=False, summary=str(error)).model_dump_json()


def _matches_result(matches: list[MusicMatch]) -> str:
    return ToolResult(
        ok=True, summary=f"找到 {len(matches)} 首曲目。",
        data={"tracks": [match.context() for match in matches],
              "score_note": "song-description cosine / DJ reranking score; not a probability"},
    ).model_dump_json()


def generate_playlist(project_id: str, brief: Brief, library: list[Track]) -> Playlist:
    if brief.bpm_min > brief.bpm_max:
        raise ValueError("BPM 下限不能高于上限")
    candidates = [track for track in library if track.analysis_status == "analyzed"
                  and brief.bpm_min <= track.bpm <= brief.bpm_max]
    if not candidates:
        raise ValueError("项目曲库里没有满足 BPM 范围且已完成分析的歌曲")
    target = brief.duration_min * 60
    ordered = _order_for_curve(candidates, brief.energy, target)
    selected: list[Track] = []
    total = 0
    for track in ordered:
        if total >= target * .94:
            break
        selected.append(track)
        total += track.duration_sec
    rows = []
    for index, track in enumerate(selected):
        previous = selected[index - 1] if index else None
        alternatives = [item.id for item in candidates
                        if item.id != track.id and abs(item.bpm - track.bpm) <= 3][:2]
        rows.append(PlaylistTrack(track=track, reason=_selection_reason(track, previous, brief.energy),
                                  alternatives=alternatives))
    duration_error = abs(total - target) / target if target else 0
    bpm_jumps = sum(abs(a.bpm - b.bpm) > 8 for a, b in zip(selected, selected[1:]))
    harmonic_breaks = sum(not is_camelot_compatible(a.camelot_key, b.camelot_key)
                          for a, b in zip(selected, selected[1:]))
    trace = [
        AgentEvent(agent="Library", status="done", message=f"从项目曲库取得 {len(candidates)} 首候选曲目。"),
        AgentEvent(agent="Set Planner", status="done", message=f"按 {brief.energy} 能量曲线编排 {len(rows)} 首曲目。"),
        AgentEvent(agent="Transition Rules", status="failed" if bpm_jumps or harmonic_breaks else "approved",
                   message=f"检查重复、时长和 BPM 跳跃；发现 {bpm_jumps} 处大于 8 BPM 的跳跃。"),
    ]
    report = [
        f"Camelot 相邻兼容 {len(selected) - 1 - harmonic_breaks}/{max(0, len(selected) - 1)} 处",
        "无重复曲目", f"目标 {brief.duration_min} 分钟，当前误差 {duration_error * 100:.1f}%",
        f"相邻 BPM 大跳跃 {bpm_jumps} 处", "所有 track_id 均来自当前项目曲库",
    ]
    return Playlist(id=uuid4().hex, project_id=project_id, brief=brief, tracks=rows,
                    duration_sec=total, report=report, trace=trace)


def _order_for_curve(candidates: list[Track], curve: str, target_seconds: int) -> list[Track]:
    approximate_count = max(1, min(len(candidates), math.ceil(
        target_seconds / max(1, sum(track.duration_sec for track in candidates) / len(candidates))
    )))
    energies = sorted(track.energy for track in candidates)
    low = energies[max(0, len(energies) // 5)]
    high = energies[min(len(energies) - 1, len(energies) * 4 // 5)]

    def target_energy(position: int) -> float:
        ratio = position / max(1, approximate_count - 1)
        if curve == "steady":
            return energies[len(energies) // 2]
        if curve == "wave":
            return low + (high - low) * (.5 + .5 * math.sin(ratio * math.pi * 2 - math.pi / 2))
        if curve == "peak":
            return high
        return low + (high - low) * ratio

    remaining = candidates[:]
    ordered: list[Track] = []
    while remaining:
        previous = ordered[-1] if ordered else None
        compatible = [track for track in remaining if previous and
                      is_camelot_compatible(previous.camelot_key, track.camelot_key)]
        if previous and not compatible:
            break
        pool = compatible if previous else remaining
        desired = target_energy(len(ordered))

        def score(track: Track) -> float:
            energy_cost = abs(track.energy - desired) * 8
            if previous is None:
                return energy_cost + track.bpm / 1000
            bpm_cost = min(abs(track.bpm - previous.bpm), abs(track.bpm * 2 - previous.bpm),
                           abs(track.bpm - previous.bpm * 2)) / 5
            return energy_cost + bpm_cost + camelot_distance(previous.camelot_key, track.camelot_key)

        selected = min(pool, key=score)
        ordered.append(selected)
        remaining.remove(selected)
    return ordered


def camelot_distance(first: str, second: str) -> float:
    match_a = re.fullmatch(r"(\d{1,2})([AB])", (first or "").strip().upper())
    match_b = re.fullmatch(r"(\d{1,2})([AB])", (second or "").strip().upper())
    if not match_a or not match_b:
        return 1.5
    number_a, mode_a = int(match_a.group(1)), match_a.group(2)
    number_b, mode_b = int(match_b.group(1)), match_b.group(2)
    ring = min((number_a - number_b) % 12, (number_b - number_a) % 12)
    if ring == 0 and mode_a == mode_b:
        return 0
    if ring == 0 or (ring == 1 and mode_a == mode_b):
        return .2
    return 1 + ring * .25 + (mode_a != mode_b) * .3


def is_camelot_compatible(first: str, second: str) -> bool:
    match_a = re.fullmatch(r"(\d{1,2})([AB])", (first or "").strip().upper())
    match_b = re.fullmatch(r"(\d{1,2})([AB])", (second or "").strip().upper())
    if not match_a or not match_b:
        return False
    number_a, mode_a = int(match_a.group(1)), match_a.group(2)
    number_b, mode_b = int(match_b.group(1)), match_b.group(2)
    same_number = number_a == number_b
    adjacent_number = min((number_a - number_b) % 12, (number_b - number_a) % 12) == 1
    return same_number or (adjacent_number and mode_a == mode_b)


def _selection_reason(track: Track, previous: Track | None, curve: str) -> str:
    energy = "高" if track.energy >= .72 else "中" if track.energy >= .45 else "低"
    if previous:
        transition = f"与上一首相差 {track.bpm - previous.bpm:+.1f} BPM"
        if previous.camelot_key == track.camelot_key:
            transition += "，同 Camelot 调性"
    else:
        transition = "作为开场锚点"
    return f"{transition}；{energy}能量，符合 {curve} 曲线。"


def render_export(playlist: Playlist, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(playlist.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if fmt == "m3u":
        return "#EXTM3U\n" + "\n".join(
            f"#EXTINF:{row.track.duration_sec},{row.track.artist} - {row.track.title}\n{row.track.path}"
            for row in playlist.tracks
        )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["position", "title", "artist", "bpm", "key", "camelot", "energy", "path"])
    for index, row in enumerate(playlist.tracks, 1):
        track = row.track
        writer.writerow([index, track.title, track.artist, track.bpm, track.key,
                         track.camelot_key, track.energy, track.path])
    return output.getvalue()
