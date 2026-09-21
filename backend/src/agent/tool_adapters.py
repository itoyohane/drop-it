"""LangChain tool adapters for compatibility tests and offline evaluation.

The controlled LangGraph calls :class:`DropItToolRegistry` methods directly.
These decorated adapters preserve the standalone tool schema without mixing
framework concerns into the deterministic music operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from langchain.tools import tool
from pydantic import Field

from backend.agent.commands import GenerateSetCommand
from backend.models import MusicFilters, MusicMatch, ToolResult
from backend.repositories import GLOBAL_PROJECT_ID

if TYPE_CHECKING:
    from backend.agent.tools import DropItToolRegistry


def _failure(error: Exception) -> str:
    return ToolResult(ok=False, summary=str(error)).model_dump_json()


def _matches_result(matches: list[MusicMatch]) -> str:
    return ToolResult(
        ok=True,
        summary=f"找到 {len(matches)} 首曲目。",
        data={
            "tracks": [match.context() for match in matches],
            "score_note": "song-description cosine / DJ reranking score; not a probability",
        },
    ).model_dump_json()


def build_tools(registry: DropItToolRegistry, project_id: str):
    """Bind the stable LangChain schemas to one server-selected project."""

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
            return _matches_result(registry.search(project_id, query, filters, k=limit))
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
            return _matches_result(registry.similar(project_id, track_id, filters, k=limit))
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
        required_tracks: Annotated[list[str] | None, Field(max_length=500)] = None,
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
                request=request,
                duration_min=duration_min,
                bpm_min=bpm_min,
                bpm_max=bpm_max,
                energy_curve=energy_curve,
                style_query=style_query,
                track_ids=track_ids,
                required_tracks=required_tracks,
            )
            candidates = registry.retrieve_set_candidates(project_id, command)
            playlist, _, _ = registry.validate_and_repair_set(project_id, command, candidates)
            registry.persist_set(project_id, playlist)
            return ToolResult(
                ok=True,
                summary=(
                    f"已生成 {len(playlist.tracks)} 首、"
                    f"约 {playlist.duration_sec // 60} 分钟的 Set。"
                ),
                data={
                    "playlist_id": playlist.id,
                    "report": playlist.report,
                    "tracks": [
                        dict(MusicMatch(track=row.track).context(), reason=row.reason)
                        for row in playlist.tracks
                    ],
                },
            ).model_dump_json()
        except (ValueError, RuntimeError) as exc:
            return _failure(exc)

    return [search_library, find_similar_tracks, generate_dj_set]
