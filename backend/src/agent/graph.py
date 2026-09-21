"""The directly compiled, deterministic LangGraph execution graph."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from backend.agent.commands import GenerateSetCommand, SimilarCommand, extract_command, message_text
from backend.agent.checkpoints import AgentCheckpointStore
from backend.agent.prompts import RESPONSE_SYSTEM_PROMPT
from backend.agent.set_repair import SetRepairer
from backend.agent.set_validation import SetValidationResult, SetValidator
from backend.agent.state import AgentRuntimeContext, AgentState
from backend.agent.tools import (
    AmbiguousTrackReferenceError,
    MissingTrackReferenceError,
    SetConstraintConflictError,
)
from backend.models import MusicMatch, Playlist, ToolEvent, Track, derive_agent_playlist_id
from backend.repositories import StaleAgentRunError


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


def _checkpoints(runtime: Runtime[AgentRuntimeContext]) -> AgentCheckpointStore:
    return AgentCheckpointStore(_context(runtime).store)


def _checkpoint_state(state: AgentState, delta: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    merged.update(delta)
    return merged


def _load_checkpoint(state: AgentState, runtime: Runtime[AgentRuntimeContext], step: str,
                     version: int | None = None) -> dict[str, Any] | None:
    run_id = state.get("run_id")
    if not run_id:
        return None
    checkpoint = _checkpoints(runtime).get(run_id, step, version)
    return checkpoint.get("state") if checkpoint else None


def _save_checkpoint(state: AgentState, runtime: Runtime[AgentRuntimeContext],
                     step: str, delta: dict[str, Any], version: int = 0) -> dict[str, Any]:
    run_id = state.get("run_id")
    if run_id:
        checkpoint = _checkpoints(runtime).save(
            run_id, step, _checkpoint_state(state, delta), version,
            owner=_context(runtime).claim_owner, token=_context(runtime).claim_token,
        )
        return checkpoint["state"]
    return _checkpoint_state(state, delta)


def _conflict_detail(result: SetValidationResult, attempts: int) -> str:
    codes = ", ".join(dict.fromkeys(issue.code for issue in result.issues)) or "unknown"
    conditions = {
        "track_membership": "曲目必须属于当前项目曲库",
        "duplicate_tracks": "不重复曲目",
        "bpm_range": "BPM 范围",
        "duration_tolerance": "目标时长或可接受误差",
        "bpm_transition": "相邻 BPM 转场阈值",
        "camelot_compatibility": "Camelot 调性兼容",
        "energy_curve": "能量曲线",
        "required_tracks": "必选曲目",
    }
    relax = "、".join(conditions.get(code, code) for code in dict.fromkeys(issue.code for issue in result.issues))
    return f"Set 约束冲突：最多修复 {attempts} 轮后仍未满足 {codes}。如需继续，请放宽：{relax}。"


async def search_library(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Extract a SearchCommand and perform exactly one deterministic search."""

    cached = _load_checkpoint(state, runtime, "candidates_retrieved")
    if cached:
        return cached
    context = _context(runtime)
    try:
        parsed = _load_checkpoint(state, runtime, "command_parsed")
        command = parsed.get("command") if parsed else None
        if command is None:
            command = await extract_command(
                context.model_factory(), "search_library", state["history"], state["user_text"]
            )
            parsed_delta = {"command": command}
            command = _save_checkpoint(
                state, runtime, "command_parsed", parsed_delta
            ).get("command", command)
        matches = await asyncio.to_thread(
            context.registry.search,
            context.project_id, command.query, command.filters, k=command.limit
        )
        delta = {
            "command": command,
            "search_matches": matches,
            "tool_events": _events(
                state,
                ToolEvent(name="search_library", status="done", summary=f"找到 {len(matches)} 首曲目。"),
            ),
        }
        return _save_checkpoint(state, runtime, "candidates_retrieved", delta)
    except StaleAgentRunError:
        raise
    except Exception as exc:
        return _failure(state, "search_library", "search_failed", _exception_detail(exc))


async def extract_and_resolve_reference(
    state: AgentState, runtime: Runtime[AgentRuntimeContext]
) -> dict[str, Any]:
    """Extract a SimilarCommand and resolve its reference within server scope."""

    cached = _load_checkpoint(state, runtime, "candidates_retrieved")
    if cached:
        return cached
    context = _context(runtime)
    try:
        parsed = _load_checkpoint(state, runtime, "command_parsed")
        command = parsed.get("command") if parsed else None
        if command is None:
            command = await extract_command(
                context.model_factory(), "find_similar_tracks", state["history"], state["user_text"]
            )
            command = _save_checkpoint(
                state, runtime, "command_parsed", {"command": command}
            ).get("command", command)
        if not isinstance(command, SimilarCommand) or not command.reference.strip():
            detail = "请提供要参考的歌曲名称或 track_id。"
            return {
                "command": command,
                **_failure(state, "find_similar_tracks", "missing_reference", detail),
            }
        reference = await asyncio.to_thread(
            context.registry.resolve_reference, context.project_id, command.reference
        )
        delta = {"command": command, "reference_track": reference}
        return _save_checkpoint(state, runtime, "candidates_retrieved", delta)
    except StaleAgentRunError:
        raise
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
        matches = await asyncio.to_thread(
            context.registry.similar,
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
    except StaleAgentRunError:
        raise
    except Exception as exc:
        return _failure(state, "find_similar_tracks", "similarity_failed", _exception_detail(exc))


async def retrieve_candidates(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Extract a GenerateSetCommand and retrieve candidates without persisting."""

    cached = _load_checkpoint(state, runtime, "candidates_retrieved")
    if cached:
        return cached
    context = _context(runtime)
    try:
        parsed = _load_checkpoint(state, runtime, "command_parsed")
        command = parsed.get("command") if parsed else None
        if command is None:
            command = await extract_command(
                context.model_factory(), "generate_dj_set", state["history"], state["user_text"]
            )
            command = _save_checkpoint(
                state, runtime, "command_parsed", {"command": command}
            ).get("command", command)
        if not isinstance(command, GenerateSetCommand):
            raise ValueError("未能解析 DJ Set 参数")
        candidates = await asyncio.to_thread(
            context.registry.retrieve_set_candidates, context.project_id, command
        )
        delta = {
            "command": command,
            "candidate_tracks": candidates,
            "candidate_track_ids": [track.id for track in candidates],
        }
        return _save_checkpoint(state, runtime, "candidates_retrieved", delta)
    except StaleAgentRunError:
        raise
    except Exception as exc:
        return _failure(state, "generate_dj_set", "candidate_retrieval_failed", _exception_detail(exc))


async def plan_set(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Plan a Set deterministically without persistence."""

    if state.get("error_code"):
        return {}
    cached = _load_checkpoint(state, runtime, "set_planned")
    if cached:
        return cached
    context = _context(runtime)
    command = state.get("command")
    if not isinstance(command, GenerateSetCommand):
        return _failure(state, "generate_dj_set", "missing_command", "未能解析 DJ Set 参数。")
    try:
        playlist = await asyncio.to_thread(
            context.registry.plan_set,
            context.project_id, command, state.get("candidate_tracks", []),
            playlist_id=derive_agent_playlist_id(context.project_id, state["run_id"]),
        )
        playlist = playlist.model_copy(update={"agent_run_id": state["run_id"]})
        delta = {"playlist": playlist}
        return _save_checkpoint(state, runtime, "set_planned", delta)
    except StaleAgentRunError:
        raise
    except Exception as exc:
        return _failure(state, "generate_dj_set", "set_planning_failed", _exception_detail(exc))


async def persist_set(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Persist exactly one planned playlist after deterministic planning."""

    if state.get("error_code") or not state.get("playlist"):
        return {}
    cached = _load_checkpoint(state, runtime, "playlist_persisted")
    if cached:
        return cached
    context = _context(runtime)
    playlist = state["playlist"]
    try:
        if playlist.project_id != context.project_id:
            raise ValueError("Set 项目范围与当前对话不一致。")
        persisted = await asyncio.to_thread(
            context.registry.persist_set,
            context.project_id, playlist, run_id=context.run_id or state.get("run_id"),
            owner=context.claim_owner, token=context.claim_token,
        )
        delta = {
            "playlist": persisted,
            "playlist_persisted": True,
            "tool_events": _events(
                state,
                ToolEvent(
                    name="generate_dj_set",
                    status="done",
                    summary=f"已生成 {len(playlist.tracks)} 首、约 {playlist.duration_sec // 60} 分钟的 Set。",
                ),
            )
        }
        return _save_checkpoint(state, runtime, "playlist_persisted", delta)
    except StaleAgentRunError:
        raise
    except SetConstraintConflictError as exc:
        return {
            "error_code": "constraint_conflict",
            "error_detail": _conflict_detail(exc.result, exc.attempts),
        }
    except Exception as exc:
        return _failure(state, "generate_dj_set", "set_persistence_failed", _exception_detail(exc))


async def validate_set(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Validate a planned/repaired Set and checkpoint the structured result."""

    if state.get("error_code"):
        return {}
    attempt = int(state.get("repair_attempts", 0))
    cached = _load_checkpoint(state, runtime, "set_validated", attempt)
    if cached:
        return cached
    command = state.get("command")
    playlist = state.get("playlist")
    if not isinstance(command, GenerateSetCommand) or not playlist:
        return _failure(state, "generate_dj_set", "missing_set", "未能取得待校验的 DJ Set。")
    validator = SetValidator()
    result = validator.validate(
        playlist, state.get("candidate_track_ids", []),
        required_tracks=command.required_tracks or [],
    )
    delta: dict[str, Any] = {"validation": result.model_dump(mode="json")}
    if not result.valid and state.get("repair_attempts", 0) >= 2:
        delta.update({
            "error_code": "constraint_conflict",
            "error_detail": _conflict_detail(result, state.get("repair_attempts", 0)),
        })
    return _save_checkpoint(state, runtime, "set_validated", delta, attempt)


async def repair_set(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Apply the remaining deterministic repair rounds in one idempotent node."""

    if state.get("error_code"):
        return {}
    attempts = int(state.get("repair_attempts", 0))
    if attempts >= 2:
        return {}
    version = attempts + 1
    cached = _load_checkpoint(state, runtime, "set_repaired", version)
    if cached:
        return cached
    command = state.get("command")
    playlist = state.get("playlist")
    validation = SetValidationResult.model_validate(state.get("validation") or {})
    if not isinstance(command, GenerateSetCommand) or not playlist:
        return _failure(state, "generate_dj_set", "missing_set", "未能取得待修复的 DJ Set。")
    repairer = SetRepairer(validator=SetValidator())
    repaired = repairer.repair(
        playlist, validation.issues, state.get("candidate_tracks", []),
        required_tracks=command.required_tracks or [], attempt=version,
    )
    delta = {
        "playlist": repaired.playlist,
        "repair_attempts": version,
        "repaired_issue_codes": list(repaired.repaired_codes),
    }
    return _save_checkpoint(state, runtime, "set_repaired", delta, version)


def _after_validation(state: AgentState) -> str:
    result = state.get("validation") or {}
    if result.get("valid"):
        return "persist"
    if state.get("error_code") == "constraint_conflict" or state.get("repair_attempts", 0) >= 2:
        return "persist"
    return "repair"


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


async def _generate_response(
    context: AgentRuntimeContext,
    messages: list[tuple[str, str]],
    writer: Any,
    *,
    error_code: str,
    error_detail: str,
) -> dict[str, Any]:
    """Stream one model response and preserve the graph's error contract."""

    try:
        text = await _stream_response(
            context.model_factory(), messages, writer
        )
        if not text:
            raise ValueError("模型返回了空回答")
        return {"final_response": text}
    except StaleAgentRunError:
        raise
    except Exception:
        writer({"type": "response_token", "content": error_detail})
        return {
            "final_response": error_detail,
            "error_code": error_code,
            "error_detail": error_detail,
        }


async def respond(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Use the model only for the final natural-language response."""

    cached = _load_checkpoint(state, runtime, "completed")
    if cached:
        return cached
    context = _context(runtime)
    evidence = json.dumps(_response_evidence(state), ensure_ascii=False)
    messages = [("system", RESPONSE_SYSTEM_PROMPT), *[
        (item["role"], item["content"])
        for item in state.get("history", [])
        if item.get("role") in {"user", "assistant"}
    ], ("human", f"基于以下已验证的结构化证据回答当前用户。不要声称执行了未记录的操作：\n{evidence}")]
    detail = state.get("error_detail") or "模型调用失败，请检查服务端日志或稍后重试。"
    delta = await _generate_response(
        context,
        messages,
        runtime.stream_writer,
        error_code=state.get("error_code") or "response_failed",
        error_detail=detail,
    )
    return _save_checkpoint(state, runtime, "completed", delta)


async def respond_chat(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    """Answer music_chat with an unbound model; no business tools are exposed."""

    cached = _load_checkpoint(state, runtime, "completed")
    if cached:
        return cached
    context = _context(runtime)
    messages = [("system", RESPONSE_SYSTEM_PROMPT + "\n当前是一般音乐问答，不要声称检索了本地曲库。"), *[
        (item["role"], item["content"])
        for item in state.get("history", [])
        if item.get("role") in {"user", "assistant"}
    ]]
    detail = "模型调用失败，请检查服务端日志或稍后重试。"
    delta = await _generate_response(
        context,
        messages,
        runtime.stream_writer,
        error_code="response_failed",
        error_detail=detail,
    )
    return _save_checkpoint(state, runtime, "completed", delta)


async def reject_response(state: AgentState, runtime: Runtime[AgentRuntimeContext]) -> dict[str, Any]:
    from backend.agent.intent import OVERSTEP_RESPONSE

    cached = _load_checkpoint(state, runtime, "completed")
    if cached:
        return cached
    delta = {"final_response": OVERSTEP_RESPONSE}
    return _save_checkpoint(state, runtime, "completed", delta)


def _route(state: AgentState) -> str:
    allowed = {
        "search_library", "resolve_reference", "find_similar_tracks", "retrieve_candidates",
        "plan_set", "validate_set", "repair_set", "persist_set", "respond",
        "respond_chat", "reject_response",
    }
    resume_node = state.get("resume_node")
    if resume_node in allowed:
        return resume_node
    return {
        "search_library": "search_library",
        "find_similar_tracks": "resolve_reference",
        "generate_dj_set": "retrieve_candidates",
        "music_chat": "respond_chat",
        "reject": "reject_response",
    }.get(state.get("route"), "reject_response")


def build_graph():
    """Compile one graph topology; per-turn scope is supplied through context."""

    builder = StateGraph(AgentState, context_schema=AgentRuntimeContext)
    builder.add_node("search_library", search_library)
    builder.add_node("respond", respond)
    builder.add_node("resolve_reference", extract_and_resolve_reference)
    builder.add_node("find_similar_tracks", find_similar_tracks)
    builder.add_node("retrieve_candidates", retrieve_candidates)
    builder.add_node("plan_set", plan_set)
    builder.add_node("validate_set", validate_set)
    builder.add_node("repair_set", repair_set)
    builder.add_node("persist_set", persist_set)
    builder.add_node("respond_chat", respond_chat)
    builder.add_node("reject_response", reject_response)

    builder.add_conditional_edges(START, _route, {
        "search_library": "search_library",
        "resolve_reference": "resolve_reference",
        "find_similar_tracks": "find_similar_tracks",
        "retrieve_candidates": "retrieve_candidates",
        "plan_set": "plan_set",
        "validate_set": "validate_set",
        "repair_set": "repair_set",
        "persist_set": "persist_set",
        "respond": "respond",
        "respond_chat": "respond_chat",
        "reject_response": "reject_response",
    })
    builder.add_edge("search_library", "respond")
    builder.add_edge("respond", END)
    builder.add_edge("resolve_reference", "find_similar_tracks")
    builder.add_edge("find_similar_tracks", "respond")
    builder.add_edge("retrieve_candidates", "plan_set")
    builder.add_edge("plan_set", "validate_set")
    builder.add_conditional_edges("validate_set", _after_validation, {
        "persist": "persist_set",
        "repair": "repair_set",
    })
    builder.add_edge("repair_set", "validate_set")
    builder.add_edge("persist_set", "respond")
    builder.add_edge("respond_chat", END)
    builder.add_edge("reject_response", END)
    return builder.compile()
