"""The directly compiled, deterministic LangGraph execution graph."""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from backend.agent.commands import GenerateSetCommand, SimilarCommand, extract_command, message_text
from backend.agent.prompts import RESPONSE_SYSTEM_PROMPT
from backend.agent.state import AgentRuntimeContext, AgentState
from backend.agent.tools import (
    AmbiguousTrackReferenceError,
    MissingTrackReferenceError,
)
from backend.models import MusicMatch, Playlist, ToolEvent, Track


def _context(runtime: Runtime[AgentRuntimeContext]) -> AgentRuntimeContext:
    context = runtime.context
    if not isinstance(context, AgentRuntimeContext):
        raise TypeError("Agent graph requires its server-only runtime context")
    return context


def _events(state: AgentState, event: ToolEvent) -> list[ToolEvent]:
    return [*state.get("tool_events", []), event]


def _failure(state: AgentState, name: str, code: str, detail: str) -> dict[str, Any]:
    return {
        "error_code": code,
        "error_detail": detail,
        "tool_events": _events(state, ToolEvent(name=name, status="failed", summary=detail)),
    }


def _exception_detail(exc: Exception) -> str:
    return str(exc) or "执行失败，请重试。"


def _track_context(track: Track) -> dict[str, Any]:
    return {
        "track_id": track.id,
        "title": track.title,
        "artist": track.artist,
        "bpm": track.bpm,
        "key": track.key,
        "camelot_key": track.camelot_key,
        "energy": track.energy,
        "duration_sec": track.duration_sec,
        "analysis_status": track.analysis_status,
    }


def _match_context(match: MusicMatch) -> dict[str, Any]:
    value = match.context()
    if match.score is not None:
        value["score"] = match.score
    if match.description_similarity is not None:
        value["description_similarity"] = match.description_similarity
    return value


def _command_dump(command: Any) -> dict[str, Any] | None:
    return command.model_dump(mode="json") if hasattr(command, "model_dump") else None


async def search_library(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Extract a SearchCommand and perform exactly one deterministic search."""

    context = _context(runtime)
    try:
        command = await extract_command(
            context.model_factory(), "search_library", state["history"], state["user_text"]
        )
        matches = context.registry.search(
            context.project_id, command.query, command.filters, k=command.limit
        )
        return {
            "command": command,
            "search_matches": matches,
            "tool_events": _events(
                state,
                ToolEvent(name="search_library", status="done", summary=f"找到 {len(matches)} 首曲目。"),
            ),
        }
    except Exception as exc:
        return _failure(state, "search_library", "search_failed", _exception_detail(exc))


async def resolve_reference(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Extract a SimilarCommand and resolve its reference within server scope."""

    context = _context(runtime)
    try:
        command = await extract_command(
            context.model_factory(), "find_similar_tracks", state["history"], state["user_text"]
        )
        if not isinstance(command, SimilarCommand) or not command.reference.strip():
            detail = "请提供要参考的歌曲名称或 track_id。"
            return {
                "command": command,
                **_failure(state, "find_similar_tracks", "missing_reference", detail),
            }
        reference = context.registry.resolve_reference(context.project_id, command.reference)
        return {"command": command, "reference_track": reference}
    except AmbiguousTrackReferenceError as exc:
        detail = str(exc)
        return {
            "command": locals().get("command"),
            **_failure(state, "find_similar_tracks", "ambiguous_reference", detail),
        }
    except MissingTrackReferenceError as exc:
        detail = str(exc)
        return {
            "command": locals().get("command"),
            **_failure(state, "find_similar_tracks", "missing_reference", detail),
        }
    except Exception as exc:
        return {
            **_failure(state, "find_similar_tracks", "reference_resolution_failed", _exception_detail(exc)),
        }


async def find_similar_tracks(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Run deterministic similarity only after reference resolution succeeds."""

    if state.get("error_code") or not state.get("reference_track"):
        return {}
    context = _context(runtime)
    command = state.get("command")
    if not isinstance(command, SimilarCommand):
        return _failure(state, "find_similar_tracks", "missing_command", "未能解析相似歌曲参数。")
    try:
        matches = context.registry.similar(
            context.project_id,
            state["reference_track"].id,
            command.filters,
            k=command.limit,
        )
        return {
            "similar_matches": matches,
            "tool_events": _events(
                state,
                ToolEvent(name="find_similar_tracks", status="done", summary=f"找到 {len(matches)} 首相似曲目。"),
            ),
        }
    except Exception as exc:
        return _failure(state, "find_similar_tracks", "similarity_failed", _exception_detail(exc))


async def retrieve_candidates(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Extract a GenerateSetCommand and retrieve candidates without persisting."""

    context = _context(runtime)
    try:
        command = await extract_command(
            context.model_factory(), "generate_dj_set", state["history"], state["user_text"]
        )
        if not isinstance(command, GenerateSetCommand):
            raise ValueError("未能解析 DJ Set 参数")
        candidates = context.registry.retrieve_set_candidates(context.project_id, command)
        return {
            "command": command,
            "candidate_tracks": candidates,
            "candidate_track_ids": [track.id for track in candidates],
        }
    except Exception as exc:
        return _failure(state, "generate_dj_set", "candidate_retrieval_failed", _exception_detail(exc))


async def plan_set(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Plan a Set deterministically; P0 deliberately has no validation/repair node."""

    if state.get("error_code"):
        return {}
    context = _context(runtime)
    command = state.get("command")
    if not isinstance(command, GenerateSetCommand):
        return _failure(state, "generate_dj_set", "missing_command", "未能解析 DJ Set 参数。")
    try:
        playlist = context.registry.plan_set(
            context.project_id, command, state.get("candidate_tracks", [])
        )
        return {"playlist": playlist}
    except Exception as exc:
        return _failure(state, "generate_dj_set", "set_planning_failed", _exception_detail(exc))


async def persist_set(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Persist exactly one planned playlist after deterministic planning."""

    if state.get("error_code") or not state.get("playlist"):
        return {}
    context = _context(runtime)
    playlist = state["playlist"]
    try:
        if playlist.project_id != context.project_id:
            raise ValueError("Set 项目范围与当前对话不一致。")
        context.registry.persist_set(context.project_id, playlist)
        return {
            "tool_events": _events(
                state,
                ToolEvent(
                    name="generate_dj_set",
                    status="done",
                    summary=f"已生成 {len(playlist.tracks)} 首、约 {playlist.duration_sec // 60} 分钟的 Set。",
                ),
            )
        }
    except Exception as exc:
        return _failure(state, "generate_dj_set", "set_persistence_failed", _exception_detail(exc))


def _response_evidence(state: AgentState) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "route": state.get("route"),
        "command": _command_dump(state.get("command")),
        "error_code": state.get("error_code"),
        "error_detail": state.get("error_detail"),
    }
    if state.get("search_matches") is not None:
        evidence["tracks"] = [_match_context(item) for item in state.get("search_matches", [])]
    if state.get("reference_track") is not None:
        evidence["reference_track"] = _track_context(state["reference_track"])
    if state.get("similar_matches") is not None:
        evidence["similar_tracks"] = [_match_context(item) for item in state.get("similar_matches", [])]
    if state.get("candidate_track_ids") is not None:
        evidence["candidate_track_ids"] = state.get("candidate_track_ids", [])
    if state.get("playlist") is not None:
        playlist: Playlist = state["playlist"]
        evidence["playlist"] = {
            "playlist_id": playlist.id,
            "duration_sec": playlist.duration_sec,
            "report": playlist.report,
            "tracks": [_track_context(row.track) for row in playlist.tracks],
        }
    return evidence


async def _stream_response(model: Any, messages: list[tuple[str, str]], writer: Any) -> str:
    chunks: list[str] = []
    astream = getattr(model, "astream", None)
    if astream is not None:
        async for chunk in astream(messages):
            text = message_text(chunk)
            if text:
                writer({"type": "response_token", "content": text})
                chunks.append(text)
    if not chunks:
        response = await model.ainvoke(messages)
        text = message_text(response).strip()
        if text:
            writer({"type": "response_token", "content": text})
            chunks.append(text)
    return "".join(chunks).strip()


async def respond(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Use the model only for the final natural-language response."""

    context = _context(runtime)
    evidence = json.dumps(_response_evidence(state), ensure_ascii=False)
    messages = [("system", RESPONSE_SYSTEM_PROMPT), *[
        (item["role"], item["content"])
        for item in state.get("history", [])
        if item.get("role") in {"user", "assistant"}
    ], ("human", f"基于以下已验证的结构化证据回答当前用户。不要声称执行了未记录的操作：\n{evidence}")]
    try:
        text = await _stream_response(
            context.model_factory(), messages, runtime.stream_writer
        )
        if not text:
            raise ValueError("模型返回了空回答")
        return {"final_response": text}
    except Exception:
        detail = state.get("error_detail") or "模型调用失败，请检查服务端日志或稍后重试。"
        runtime.stream_writer({"type": "response_token", "content": detail})
        return {
            "final_response": detail,
            "error_code": state.get("error_code") or "response_failed",
            "error_detail": state.get("error_detail") or detail,
        }


async def respond_chat(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Answer music_chat with an unbound model; no business tools are exposed."""

    context = _context(runtime)
    messages = [("system", RESPONSE_SYSTEM_PROMPT + "\n当前是一般音乐问答，不要声称检索了本地曲库。"), *[
        (item["role"], item["content"])
        for item in state.get("history", [])
        if item.get("role") in {"user", "assistant"}
    ]]
    try:
        text = await _stream_response(
            context.model_factory(), messages, runtime.stream_writer
        )
        if not text:
            raise ValueError("模型返回了空回答")
        return {"final_response": text}
    except Exception:
        detail = "模型调用失败，请检查服务端日志或稍后重试。"
        runtime.stream_writer({"type": "response_token", "content": detail})
        return {"final_response": detail,
                "error_code": "response_failed", "error_detail": detail}


async def reject_response(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    from backend.agent.intent import OVERSTEP_RESPONSE

    return {"final_response": OVERSTEP_RESPONSE}


def _route(state: AgentState) -> str:
    route = state.get("route")
    return route if route in {
        "search_library", "find_similar_tracks", "generate_dj_set", "music_chat", "reject"
    } else "reject"


def build_graph():
    """Compile one graph topology; per-turn scope is supplied through context."""

    builder = StateGraph(AgentState, context_schema=AgentRuntimeContext)
    builder.add_node("search_library", search_library)
    builder.add_node("respond", respond)
    builder.add_node("resolve_reference", resolve_reference)
    builder.add_node("find_similar_tracks", find_similar_tracks)
    builder.add_node("retrieve_candidates", retrieve_candidates)
    builder.add_node("plan_set", plan_set)
    builder.add_node("persist_set", persist_set)
    builder.add_node("respond_chat", respond_chat)
    builder.add_node("reject_response", reject_response)

    builder.add_conditional_edges(START, _route, {
        "search_library": "search_library",
        "find_similar_tracks": "resolve_reference",
        "generate_dj_set": "retrieve_candidates",
        "music_chat": "respond_chat",
        "reject": "reject_response",
    })
    builder.add_edge("search_library", "respond")
    builder.add_edge("respond", END)
    builder.add_edge("resolve_reference", "find_similar_tracks")
    builder.add_edge("find_similar_tracks", "respond")
    builder.add_edge("retrieve_candidates", "plan_set")
    builder.add_edge("plan_set", "persist_set")
    builder.add_edge("persist_set", "respond")
    builder.add_edge("respond_chat", END)
    builder.add_edge("reject_response", END)
    return builder.compile()
