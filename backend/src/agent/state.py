"""Typed state and server-only runtime context for the controlled Agent graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

if TYPE_CHECKING:
    from backend.agent.tools import DropItToolRegistry
    from backend.repositories import DropItStore

from backend.agent.commands import CommandPayload
from backend.models import MusicMatch, Playlist, ToolEvent, Track


AgentRoute = Literal[
    "search_library",
    "find_similar_tracks",
    "generate_dj_set",
    "music_chat",
    "reject",
]


class AgentState(TypedDict, total=False):
    """Raw structured values produced and consumed by graph nodes.

    Project and conversation scope intentionally do not live in this state. They
    are immutable, server-selected fields in :class:`AgentRuntimeContext`.
    """

    run_id: str
    route: AgentRoute
    resume_node: str
    user_text: str
    history: list[dict[str, str]]
    command: CommandPayload | None
    reference_track: Track | None
    search_matches: list[MusicMatch]
    similar_matches: list[MusicMatch]
    candidate_tracks: list[Track]
    candidate_track_ids: list[str]
    playlist: Playlist | None
    validation: dict[str, Any] | None
    repair_attempts: int
    repaired_issue_codes: list[str]
    playlist_persisted: bool
    tool_events: list[ToolEvent]
    final_response: str
    error_code: str | None
    error_detail: str | None


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Immutable per-turn dependencies that the model cannot override."""

    project_id: str
    conversation_id: str
    store: DropItStore
    registry: DropItToolRegistry
    model_factory: Callable[[], Any]
    run_id: str | None = None
    claim_owner: str | None = None
    claim_token: int | None = None
