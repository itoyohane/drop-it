"""The Agent's three capabilities, all bound to a server-selected library."""

from typing import Annotated, Literal

from langchain.tools import tool
from pydantic import Field

from backend.models import Brief, MusicFilters, MusicMatch, ToolResult
from backend.agent.retriever import RagLibrary
from backend.services.set_planner import generate_playlist
from backend.repositories import GLOBAL_PROJECT_ID, DropItStore


class DropItToolRegistry:
    names = ["search_library", "find_similar_tracks", "generate_dj_set"]

    def __init__(self, store: DropItStore, rag: RagLibrary):
        self.store, self.rag = store, rag

    def tools_for(self, project_id: str):
        store, rag = self.store, self.rag

        @tool
        def search_library(
            query: str = "",
            filters: MusicFilters | None = None,
            limit: Annotated[int, Field(ge=1, le=100)] = 20,
        ) -> str:
            """Retrieve music facts from the current library for answering the user.

            Use an English sound/style description as query for CLAP semantic search.
            For titles/artists, exact BPM/key/energy constraints or a library overview,
            leave query empty and use filters. Only returned track_ids are valid references.
            """
            try:
                return _matches_result(rag.search(project_id, query, filters, k=limit))
            except (ValueError, RuntimeError) as exc:
                return _failure(exc)

        @tool
        def find_similar_tracks(
            track_id: str,
            limit: Annotated[int, Field(ge=1, le=100)] = 3,
            filters: MusicFilters | None = None,
        ) -> str:
            """Find local songs with similar audio, reranked by BPM, Camelot and energy.

            Resolve the reference track_id using search_library first. Ask the user if
            multiple versions match. Scores are ranking signals, not match percentages.
            """
            try:
                return _matches_result(rag.similar(project_id, track_id, filters, k=limit))
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

            style_query is an optional English musical description for CLAP recall;
            leave it empty if no style was requested. Numerical constraints belong
            in bpm_min/bpm_max/duration_min. Never silently relax requested constraints.
            Pass track_ids to restrict the set to songs selected from earlier tool results.
            """
            if project_id == GLOBAL_PROJECT_ID:
                return _failure(ValueError("请进入具体项目后生成 Set。"))
            try:
                filters = MusicFilters(bpm_min=bpm_min, bpm_max=bpm_max)
                candidates = (
                    [match.track for match in rag.search(project_id, style_query, filters, k=200)]
                    if style_query.strip() else rag.catalog(project_id, filters)
                )
                if track_ids is not None:
                    allowed = {track.id for track in rag.catalog(project_id)}
                    if not track_ids or not set(track_ids) <= allowed:
                        raise ValueError("选曲列表为空或包含当前项目之外的曲目。")
                    chosen = set(track_ids)
                    candidates = [track for track in candidates if track.id in chosen]
                brief = Brief(title=request[:60] or "Untitled set", duration_min=duration_min,
                              bpm_min=bpm_min, bpm_max=bpm_max, energy=energy_curve,
                              style=style_query, notes=request)
                playlist = generate_playlist(project_id, brief, candidates)
                store.save_playlist(playlist)
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
    return ToolResult(ok=True, summary=f"找到 {len(matches)} 首曲目。",
                      data={"tracks": [match.context() for match in matches],
                            "score_note": "CLAP cosine / reranking score; not a probability"}).model_dump_json()
