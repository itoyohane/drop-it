"""Persistent Agent node checkpoints and state (de)serialization.

LangGraph owns the in-process topology, while this small repository-backed
layer owns restart/resume semantics.  The unique ``run_id + step`` key makes a
completed node reusable even when the caller retries the request.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from backend.agent.commands import GenerateSetCommand, SearchCommand, SimilarCommand
from backend.models import MusicMatch, Playlist, ToolEvent, Track


CHECKPOINT_STEPS = (
    "command_parsed",
    "candidates_retrieved",
    "set_planned",
    "set_validated",
    "set_repaired",
    "playlist_persisted",
    "completed",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def serialize_state(state: dict[str, Any]) -> str:
    return json.dumps(_jsonable(state), ensure_ascii=False, separators=(",", ":"))


def _model_list(value: Any, model: type[BaseModel]) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [model.model_validate(item) if not isinstance(item, model) else item for item in value]


def hydrate_state(value: dict[str, Any] | str | None) -> dict[str, Any]:
    """Restore the typed values graph nodes use for ``isinstance`` checks."""

    if value is None:
        return {}
    data = json.loads(value) if isinstance(value, str) else dict(value)
    route = data.get("route")
    command = data.get("command")
    command_type = {
        "search_library": SearchCommand,
        "find_similar_tracks": SimilarCommand,
        "generate_dj_set": GenerateSetCommand,
    }.get(route)
    if command_type and isinstance(command, dict):
        data["command"] = command_type.model_validate(command)
    for field in ("reference_track",):
        if isinstance(data.get(field), dict):
            data[field] = Track.model_validate(data[field])
    for field in ("candidate_tracks",):
        data[field] = _model_list(data.get(field), Track)
    for field in ("search_matches", "similar_matches"):
        data[field] = _model_list(data.get(field), MusicMatch)
    if isinstance(data.get("playlist"), dict):
        data["playlist"] = Playlist.model_validate(data["playlist"])
    data["tool_events"] = _model_list(data.get("tool_events"), ToolEvent)
    return data


class AgentCheckpointStore:
    """Repository adapter kept separate from graph code for unit testing."""

    def __init__(self, store: Any):
        self.store = store

    def start_run(self, run_id: str, project_id: str, conversation_id: str,
                  route: str, user_text: str, history: list[dict[str, str]]) -> dict[str, Any]:
        return self.store.create_agent_run(
            run_id, project_id, conversation_id, route, user_text, history
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.store.get_agent_run(run_id)

    def save(self, run_id: str, step: str, state: dict[str, Any],
             version: int = 0, *, owner: str | None = None,
             token: int | None = None) -> dict[str, Any]:
        if step not in CHECKPOINT_STEPS:
            raise ValueError(f"不支持的 Agent checkpoint 节点：{step}")
        checkpoint = self.store.save_agent_checkpoint(
            run_id, step, serialize_state(state), version, owner, token
        )
        # INSERT OR IGNORE makes the first completed writer authoritative.  A
        # racing caller always receives and uses that persisted winner state.
        return {**checkpoint, "state": hydrate_state(checkpoint.get("state"))}

    def get(self, run_id: str, step: str, version: int | None = None) -> dict[str, Any] | None:
        checkpoint = self.store.get_agent_checkpoint(run_id, step, version)
        if not checkpoint:
            return None
        return {**checkpoint, "state": hydrate_state(checkpoint.get("state"))}

    def latest(self, run_id: str) -> dict[str, Any] | None:
        checkpoint = self.store.get_latest_agent_checkpoint(run_id)
        if not checkpoint:
            return None
        return {**checkpoint, "state": hydrate_state(checkpoint.get("state"))}

    def list(self, run_id: str) -> list[dict[str, Any]]:
        return [
            {**checkpoint, "state": hydrate_state(checkpoint.get("state"))}
            for checkpoint in self.store.list_agent_checkpoints(run_id)
        ]

    def mark_completed(self, run_id: str, state: dict[str, Any], *, owner: str,
                       token: int) -> dict[str, Any]:
        checkpoint = self.save(run_id, "completed", state, 0, owner=owner, token=token)
        self.store.update_agent_run(run_id, owner=owner, token=token, status="completed")
        return checkpoint


def next_node_for_checkpoint(route: str, checkpoint: dict[str, Any] | None) -> str:
    """Return the first node that has not completed for a durable run."""

    initial = {
        "search_library": "search_library",
        "find_similar_tracks": "resolve_reference",
        "generate_dj_set": "retrieve_candidates",
        "music_chat": "respond_chat",
        "reject": "reject_response",
    }
    if not checkpoint:
        return initial.get(route, "reject_response")
    step = checkpoint.get("step")
    state = checkpoint.get("state") or {}
    if step == "command_parsed":
        return initial.get(route, "reject_response")
    if step == "candidates_retrieved":
        return {
            "search_library": "respond",
            "find_similar_tracks": "find_similar_tracks",
            "generate_dj_set": "plan_set",
        }.get(route, initial.get(route, "reject_response"))
    if step == "set_planned":
        return "validate_set"
    if step == "set_validated":
        validation = state.get("validation") or {}
        if validation.get("valid"):
            return "persist_set"
        return "repair_set" if int(state.get("repair_attempts", 0)) < 2 else "persist_set"
    if step == "set_repaired":
        return "validate_set"
    if step == "playlist_persisted":
        return "respond"
    if step == "completed":
        return {
            "music_chat": "respond_chat",
            "reject": "reject_response",
        }.get(route, "respond")
    return initial.get(route, "reject_response")


# Both names are useful at call sites and keep the module friendly to callers
# that refer to this as a checkpoint manager rather than a repository.
AgentCheckpointManager = AgentCheckpointStore
CheckpointStore = AgentCheckpointStore


__all__ = [
    "CHECKPOINT_STEPS", "AgentCheckpointManager", "AgentCheckpointStore",
    "CheckpointStore", "hydrate_state", "next_node_for_checkpoint", "serialize_state",
]
