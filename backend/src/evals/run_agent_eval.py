"""Evaluate production LangGraph tasks, typed commands, validation and repair."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Protocol
from uuid import uuid4


CASES_PATH = Path(__file__).with_name("cases") / "agent_tasks.jsonl"
CASE_CATEGORIES = ("search", "similar", "set", "multi_turn", "overstep")
REQUIRED_CASE_FIELDS = {
    "id", "category", "turns", "expected_command", "expected_tool_sequence",
    "tool_argument_assertions", "expected_completion", "expected_interception",
    "core", "repeats",
    "expected_graph_nodes",
}


class CaseValidationError(ValueError):
    """Raised when the versioned Agent case data is malformed."""


def _lookup(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(path)
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def _resolve_string(value: str, fixtures: Mapping[str, Any]) -> Any:
    """Resolve ``{{fixture.foo}}`` while preserving non-string fixture types."""
    import re

    exact = re.fullmatch(r"\{\{\s*fixture\.([\w.]+)\s*\}\}", value)
    if exact:
        return _lookup(fixtures, exact.group(1))

    def replacement(match: re.Match[str]) -> str:
        return str(_lookup(fixtures, match.group(1)))

    return re.sub(r"\{\{\s*fixture\.([\w.]+)\s*\}\}", replacement, value)


def resolve_templates(value: Any, fixtures: Mapping[str, Any]) -> Any:
    """Recursively resolve case templates without mutating the source case."""
    if isinstance(value, str):
        return _resolve_string(value, fixtures)
    if isinstance(value, list):
        return [resolve_templates(item, fixtures) for item in value]
    if isinstance(value, dict):
        return {key: resolve_templates(item, fixtures) for key, item in value.items()}
    return value


def validate_case(case: Mapping[str, Any]) -> None:
    missing = REQUIRED_CASE_FIELDS - set(case)
    if missing:
        raise CaseValidationError(f"{case.get('id', '<unknown>')}: missing {sorted(missing)}")
    if not isinstance(case["id"], str) or not case["id"]:
        raise CaseValidationError("case id must be a non-empty string")
    if case["category"] not in CASE_CATEGORIES:
        raise CaseValidationError(f"{case['id']}: unknown category {case['category']!r}")
    turns = case["turns"]
    if not isinstance(turns, list) or not turns or any(
        not isinstance(turn, dict) or not isinstance(turn.get("user"), str) or not turn["user"]
        for turn in turns
    ):
        raise CaseValidationError(f"{case['id']}: turns must contain user messages")
    if not isinstance(case["expected_tool_sequence"], list):
        raise CaseValidationError(f"{case['id']}: expected_tool_sequence must be a list")
    if not isinstance(case["expected_graph_nodes"], list) or any(
        not isinstance(node, str) for node in case["expected_graph_nodes"]
    ):
        raise CaseValidationError(f"{case['id']}: expected_graph_nodes must contain node names")
    if not isinstance(case["tool_argument_assertions"], list):
        raise CaseValidationError(f"{case['id']}: tool_argument_assertions must be a list")
    for assertion in case["tool_argument_assertions"]:
        if not isinstance(assertion, dict) or not {"tool", "path", "op", "value"} <= set(assertion):
            raise CaseValidationError(f"{case['id']}: malformed argument assertion")
    if not isinstance(case["repeats"], int) or case["repeats"] < 1:
        raise CaseValidationError(f"{case['id']}: repeats must be >= 1")
    if case["core"] and case["repeats"] != 3:
        raise CaseValidationError(f"{case['id']}: core cases must repeat exactly 3 times")


def load_cases(path: str | Path = CASES_PATH) -> list[dict[str, Any]]:
    """Load and validate the fixed 60-case dataset."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseValidationError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(case, dict):
            raise CaseValidationError(f"{source}:{line_number}: case must be an object")
        validate_case(case)
        rows.append(case)
    ids = [case["id"] for case in rows]
    if len(rows) != 60:
        raise CaseValidationError(f"expected exactly 60 cases, got {len(rows)}")
    if len(set(ids)) != len(ids):
        raise CaseValidationError("case IDs must be unique")
    counts = Counter(case["category"] for case in rows)
    expected = {"search": 15, "similar": 10, "set": 20, "multi_turn": 10, "overstep": 5}
    if dict(counts) != expected:
        raise CaseValidationError(f"case distribution mismatch: {dict(counts)} != {expected}")
    return rows


def make_fixtures(tracks: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build runtime-safe placeholders from tracks; no fixed track ID is assumed."""
    rows = list(tracks or [])
    if rows:
        first = rows[0]
        get = lambda key, default="": first.get(key, default)  # noqa: E731
        track_ids = [str(row.get("id", row.get("track_id", ""))) for row in rows]
        title = str(get("title", "reference"))
        artist = str(get("artist", "artist"))
        bpm = float(get("bpm", 124))
        camelot = str(get("camelot_key", "8A"))
        energy = float(get("energy", .6))
        track_id = str(get("id", get("track_id", "reference")))
    else:
        # Used only by offline tests and template validation; these are not real IDs.
        track_ids, title, artist, bpm, camelot, energy, track_id = [], "Reference", "Artist", 120, "8A", .6, "fixture-reference"
    return {
        "reference_track_id": track_id,
        "reference_title": title,
        "reference_title_prefix": title[:8],
        "artist": artist,
        "track_ids": track_ids,
        "bpm_min": int(max(60, math.floor(bpm - 4))),
        "bpm_max": int(min(220, math.ceil(bpm + 8))),
        "camelot_key": camelot,
        "energy_min": round(max(0, energy - .1), 2),
        "energy_max": round(min(1, energy + .1), 2),
        "duration_min": 45,
    }


def expand_cases(cases: Iterable[Mapping[str, Any]], fixtures: Mapping[str, Any]) -> list[dict[str, Any]]:
    expanded = []
    for source in cases:
        case = resolve_templates(dict(source), fixtures)
        for repeat_index in range(int(source.get("repeats", 1))):
            expanded.append({**case, "case_id": source["id"], "repeat_index": repeat_index + 1})
    return expanded


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceBuilder:
    case_id: str
    repeat_index: int = 1
    run_id: str = field(default_factory=lambda: uuid4().hex)
    execution_mode: str = "langgraph"
    started_at: str = field(default_factory=_now_iso)
    started: float = field(default_factory=time.perf_counter)
    command: str | None = None
    graph_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] | None = None
    error_code: str | None = None
    completed: bool = False
    intercepted: bool = False
    constraints_passed: bool | None = None
    constraint_issues: list[Any] = field(default_factory=list)
    repair_attempts: int = 0
    repair_success: bool | None = None
    turn_traces: list[dict[str, Any]] = field(default_factory=list)
    output: Any = None

    def finish(self) -> dict[str, Any]:
        latency = round((time.perf_counter() - self.started) * 1000, 3)
        usage = self.token_usage
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "repeat_index": self.repeat_index,
            "started_at": self.started_at,
            "execution_mode": self.execution_mode,
            "command": self.command,
            "graph_steps": self.graph_steps,
            "tool_calls": self.tool_calls,
            "latency_ms": latency,
            "token_usage": usage,
            "token_usage_status": "available" if usage else "unavailable",
            "estimated_cost": None,
            "repair_attempts": self.repair_attempts,
            "repair_success": self.repair_success,
            "turn_traces": self.turn_traces,
            "error_code": self.error_code,
            "completed": self.completed,
            "intercepted": self.intercepted,
            "constraints_passed": self.constraints_passed,
            "constraint_issues": self.constraint_issues,
            "task_completed": False,
            "output": self.output,
        }


class EvalRunner(Protocol):
    def run(self, case: Mapping[str, Any], *, repeat_index: int = 1) -> Mapping[str, Any]: ...


class FixtureTraceRunner:
    """Small deterministic runner used by offline tests and metric examples.

    It models the observable contract, not a production result, and is never used
    by the live CLI unless explicitly selected with ``--mode fixture``.
    """

    execution_mode = "fixture"

    def run(self, case: Mapping[str, Any], *, repeat_index: int = 1) -> Mapping[str, Any]:
        trace = TraceBuilder(str(case.get("case_id", case.get("id", "fixture"))), repeat_index,
                             execution_mode="fixture")
        trace.command = str(case["expected_command"])
        trace.graph_steps.append({"name": "route", "status": "done"})
        if trace.command != "overstep":
            trace.graph_steps.append({"name": "model", "status": "done"})
        arguments_by_tool: dict[str, dict[str, Any]] = defaultdict(dict)
        for assertion in case.get("tool_argument_assertions", []):
            op = assertion.get("op")
            if op in {"eq", "contains", "in", "subset_of", "lte", "gte"}:
                expected = assertion["value"]
                if op == "in" and isinstance(expected, list):
                    expected = expected[0]
                _set_path(arguments_by_tool[assertion["tool"]], assertion["path"], expected)
        for name in case.get("expected_tool_sequence", []):
            trace.tool_calls.append({"name": name, "arguments": arguments_by_tool.get(name, {})})
            trace.graph_steps.append({"name": "tool", "tool": name, "status": "done"})
        trace.graph_steps.append({"name": "complete", "status": "done"})
        trace.completed = bool(case.get("expected_completion", True))
        trace.intercepted = bool(case.get("expected_interception", False))
        trace.constraints_passed = True if "generate_dj_set" in case.get("expected_tool_sequence", []) else None
        trace.graph_steps = [{"name": node, "status": "done"} for node in case.get("expected_graph_nodes", [])]
        return trace.finish()


def _get_tool_args(trace: Mapping[str, Any], name: str) -> dict[str, Any]:
    calls = [item for item in trace.get("tool_calls", []) if item.get("name") == name]
    return (calls[-1].get("arguments") or {}) if calls else {}


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted mapping path for deterministic fixture traces."""
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _model_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, Mapping) else None
    return None


def validate_set_constraints(
    playlist: Any, arguments: Mapping[str, Any], project_track_ids: Iterable[str],
    tracks_by_id: Mapping[str, Any], *, project_id: str | None = None,
) -> dict[str, Any]:
    """Use production SetValidator with catalog facts and requested constraints."""
    from backend.agent.set_planning import SetValidator
    from backend.models import Playlist, Track

    payload = _model_mapping(playlist)
    if not payload:
        return {"passed": False, "issues": [{"code": "playlist_missing"}]}
    try:
        actual = Playlist.model_validate(payload)
    except ValueError:
        return {"passed": False, "issues": [{"code": "invalid_playlist"}]}
    allowed = set(project_track_ids)
    if arguments.get("track_ids") is not None:
        allowed &= set(arguments["track_ids"])
    issues = []
    if not actual.tracks:
        issues.append({"code": "empty_playlist"})
    if project_id is not None and actual.project_id != project_id:
        issues.append({"code": "project_mismatch"})
    brief = actual.brief.model_copy(update={
        "duration_min": arguments.get("duration_min", 45),
        "bpm_min": arguments.get("bpm_min", 110),
        "bpm_max": arguments.get("bpm_max", 140),
        "energy": arguments.get("energy_curve", "build"),
    })
    rows = []
    for row in actual.tracks:
        canonical = tracks_by_id.get(row.track.id)
        if canonical is not None:
            track = canonical if isinstance(canonical, Track) else Track.model_validate(canonical)
            row = row.model_copy(update={"track": track})
        rows.append(row)
    actual = actual.model_copy(update={"brief": brief, "tracks": rows})
    result = SetValidator().validate(actual, allowed, arguments.get("required_tracks") or [])
    issues.extend(result.model_dump(mode="json")["issues"])
    return {"passed": not issues, "issues": issues, "checks": result.checks}


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    remaining = iter(actual)
    return all(any(node == wanted for node in remaining) for wanted in expected)


def _assertion_passes(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected or (isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and math.isclose(actual, expected))
    if op == "contains":
        return isinstance(actual, str) and str(expected).casefold() in actual.casefold()
    if op == "in":
        return actual in expected
    if op == "lte":
        return actual is not None and actual <= expected
    if op == "gte":
        return actual is not None and actual >= expected
    if op == "subset_of":
        return isinstance(actual, list) and set(actual) <= set(expected)
    if op == "present":
        return actual is not None
    raise CaseValidationError(f"unknown assertion op {op!r}")


def evaluate_trace(case: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    expected_command = case["expected_command"]
    actual_sequence = [str(item.get("name")) for item in trace.get("tool_calls", [])]
    expected_sequence = list(case.get("expected_tool_sequence", []))
    assertions = []
    for spec in case.get("tool_argument_assertions", []):
        actual_args = _get_tool_args(trace, spec["tool"])
        try:
            actual = _lookup(actual_args, spec["path"])
            passed = _assertion_passes(actual, spec["op"], spec["value"])
        except (KeyError, TypeError, CaseValidationError):
            actual, passed = None, False
        assertions.append({**spec, "actual": actual, "passed": passed})
    argument_passed = all(item["passed"] for item in assertions)
    intercepted = bool(trace.get("intercepted"))
    no_tools = not actual_sequence
    command_passed = trace.get("command") == expected_command
    tool_sequence_passed = actual_sequence == expected_sequence
    completion_event_passed = bool(trace.get("completed")) == bool(case.get("expected_completion"))
    interception_passed = intercepted == bool(case.get("expected_interception")) and (not intercepted or no_tools)
    has_set = "generate_dj_set" in expected_sequence
    constraints_passed = bool(trace.get("constraints_passed")) if has_set else True
    graph_passed = _is_subsequence(case.get("expected_graph_nodes", []), [
        str(step.get("name")) for step in trace.get("graph_steps", [])
    ])
    error_code = trace.get("error_code")
    if not error_code and any(item.get("status") == "failed" for item in trace.get("tool_calls", [])):
        error_code = "tool_failed"
    failures: list[dict[str, str]] = []
    if error_code:
        failures.append({
            "code": str(error_code),
            "node": trace.get("failure_node") or ("tool" if error_code == "tool_failed" else "model"),
        })
    if not command_passed:
        failures.append({"code": "command_mismatch", "node": "route"})
    if not argument_passed:
        failures.append({"code": "argument_mismatch", "node": "tool_arguments"})
    if not tool_sequence_passed:
        failures.append({"code": "tool_sequence_mismatch", "node": "tool_sequence"})
    if not graph_passed:
        failures.append({"code": "graph_sequence_mismatch", "node": "graph"})
    if not completion_event_passed:
        failures.append({"code": "completion_mismatch", "node": "complete"})
    if not interception_passed:
        failures.append({"code": "interception_mismatch", "node": "route"})
    if not constraints_passed:
        failures.append({"code": "set_constraint_failure", "node": "set_validation"})
    if not error_code and failures:
        error_code = failures[0]["code"]
    task_completed = bool(
        command_passed and argument_passed and tool_sequence_passed and
        completion_event_passed and interception_passed and constraints_passed and graph_passed and
        not error_code
    )
    return {
        "command_passed": command_passed,
        "argument_passed": argument_passed,
        "argument_assertions": assertions,
        "tool_sequence_passed": tool_sequence_passed,
        "completion_event_passed": completion_event_passed,
        "completion_passed": completion_event_passed,
        "interception_passed": interception_passed,
        "set_constraints_passed": bool(trace.get("constraints_passed")) if has_set else None,
        "graph_sequence_passed": graph_passed,
        "task_completed": task_completed,
        "error_code": error_code,
        "failure_node": failures[0]["node"] if failures else None,
        "failures": failures,
        "actual_tool_sequence": actual_sequence,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1]), 3) if len(values) > 1 else round(values[0], 3)


def _cost_for_trace(trace: Mapping[str, Any], input_price: float | None, output_price: float | None) -> float | None:
    usage = trace.get("token_usage")
    if not usage or input_price is None or output_price is None:
        return None
    return round((usage.get("input_tokens", 0) * input_price + usage.get("output_tokens", 0) * output_price) / 1_000_000, 8)


def _validate_prices(input_price: float | None, output_price: float | None) -> None:
    if (input_price is None) != (output_price is None):
        raise ValueError("input and output token prices must be provided together")
    if input_price is not None and (not math.isfinite(input_price) or not math.isfinite(output_price) or input_price < 0 or output_price < 0):
        raise ValueError("token prices must be non-negative")


def _normalized_arguments(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_arguments(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_normalized_arguments(item) for item in value]
    return value


def _stability_signature(row: Mapping[str, Any]) -> str:
    trace = row.get("trace", {})
    calls = [
        {
            "name": item.get("name"),
            "arguments": _normalized_arguments(item.get("arguments") or {}),
            "status": item.get("status"),
        }
        for item in trace.get("tool_calls", [])
    ]
    return json.dumps({
        "command": trace.get("command"),
        "tool_calls": calls,
        "task_completed": row.get("evaluation", {}).get("task_completed", trace.get("task_completed")),
        "error_code": trace.get("error_code") or row.get("evaluation", {}).get("error_code"),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_metrics(results: Iterable[Mapping[str, Any]], *, input_price_per_million: float | None = None,
                    output_price_per_million: float | None = None) -> dict[str, Any]:
    _validate_prices(input_price_per_million, output_price_per_million)
    rows = list(results)
    evaluated = [row.get("evaluation", {}) for row in rows]
    def count(key: str) -> tuple[int, int]:
        return sum(bool(item.get(key)) for item in evaluated), len(evaluated)
    command_n, total = count("command_passed")
    argument_assertions = [assertion for item in evaluated for assertion in item.get("argument_assertions", [])]
    argument_n = sum(bool(item.get("passed")) for item in argument_assertions)
    argument_total = len(argument_assertions)
    sequence_n, _ = count("tool_sequence_passed")
    completion_n, _ = count("task_completed")
    sets = [item for item in evaluated if item.get("set_constraints_passed") is not None]
    oversteps = [item for item, row in zip(evaluated, rows) if row.get("category") == "overstep"]
    latencies = [float(row["trace"]["latency_ms"]) for row in rows if row.get("trace", {}).get("latency_ms") is not None]
    by_case: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id"))].append(row)
    stable_groups = [group for group in by_case.values() if len(group) > 1]
    stable = sum(len({_stability_signature(x) for x in group}) == 1 for group in stable_groups)
    token_available = [row["trace"].get("token_usage") for row in rows if row.get("trace", {}).get("token_usage")]
    total_tokens = sum(int(usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0))) for usage in token_available)
    costs = [_cost_for_trace(row.get("trace", {}), input_price_per_million, output_price_per_million) for row in rows]
    costs = [cost for cost in costs if cost is not None]
    usage_status = ("unavailable" if not token_available else "available" if
                    len(token_available) == len(rows) and all(
                        row.get("trace", {}).get("token_usage_status") != "partial" for row in rows
                    ) else "partial")
    repairs_attempted = sum(bool(row.get("trace", {}).get("repair_attempts", 0)) for row in rows)
    repairs_succeeded = sum(bool(row.get("trace", {}).get("repair_success")) for row in rows if row.get("trace", {}).get("repair_attempts", 0))
    return {
        "sample_count": total,
        "command_accuracy": {"value_pct": _rate(command_n, total), "passed": command_n, "total": total},
        "argument_accuracy": {"value_pct": _rate(argument_n, argument_total), "passed": argument_n, "total": argument_total},
        "tool_sequence_accuracy": {"value_pct": _rate(sequence_n, total), "passed": sequence_n, "total": total},
        "graph_sequence_accuracy": {"value_pct": _rate(sum(bool(x.get("graph_sequence_passed")) for x in evaluated), total), "passed": sum(bool(x.get("graph_sequence_passed")) for x in evaluated), "total": total},
        "task_completion_rate": {"value_pct": _rate(completion_n, total), "passed": completion_n, "total": total},
        "set_constraint_pass_rate": {"value_pct": _rate(sum(bool(x.get("set_constraints_passed")) for x in sets), len(sets)), "passed": sum(bool(x.get("set_constraints_passed")) for x in sets), "total": len(sets)},
        "unauthorized_action_block_rate": {"value_pct": _rate(sum(bool(x.get("interception_passed")) for x in oversteps), len(oversteps)), "passed": sum(bool(x.get("interception_passed")) for x in oversteps), "total": len(oversteps)},
        "stability": {"value_pct": _rate(stable, len(stable_groups)), "stable_groups": stable, "total_groups": len(stable_groups)},
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95), "sample_count": len(latencies)},
        "token_usage": {"status": usage_status, "scope": "chat_model_only", "input_tokens": sum(int(u.get("input_tokens", 0)) for u in token_available), "output_tokens": sum(int(u.get("output_tokens", 0)) for u in token_available), "total_tokens": total_tokens, "runs_with_usage": len(token_available), "runs": len(rows)},
        "estimated_cost": {"status": usage_status if costs else "unavailable", "scope": "chat_model_only", "total": round(sum(costs), 8) if costs else None, "currency": "USD", "input_price_per_million": input_price_per_million, "output_price_per_million": output_price_per_million},
        "repair_success_rate": {"status": "available" if repairs_attempted else "not_applicable", "value_pct": _rate(repairs_succeeded, repairs_attempted) if repairs_attempted else None, "succeeded": repairs_succeeded, "attempted": repairs_attempted},
    }


def evaluate_cases(cases: Iterable[Mapping[str, Any]], runner: EvalRunner | Callable[..., Mapping[str, Any]], *, fixtures: Mapping[str, Any] | None = None,
                   input_price_per_million: float | None = None, output_price_per_million: float | None = None,
                   rounds: int | None = None, on_result: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    if rounds is not None and rounds < 1:
        raise ValueError("rounds must be >= 1")
    _validate_prices(input_price_per_million, output_price_per_million)
    source = list(cases)
    fixture_values = dict(fixtures or make_fixtures())
    results = []
    for case in source:
        resolved = resolve_templates(case, fixture_values)
        for repeat_index in range(rounds if rounds is not None else int(case.get("repeats", 1))):
            if hasattr(runner, "run"):
                raw = runner.run({**resolved, "case_id": case["id"]}, repeat_index=repeat_index + 1)
            else:
                raw = runner({**resolved, "case_id": case["id"]}, repeat_index=repeat_index + 1)
            trace = dict(raw)
            trace.setdefault("case_id", case["id"])
            trace.setdefault("repeat_index", repeat_index + 1)
            trace["estimated_cost"] = _cost_for_trace(trace, input_price_per_million, output_price_per_million)
            evaluation = evaluate_trace(resolved, trace)
            trace["task_completed"] = evaluation["task_completed"]
            if evaluation["error_code"] and not trace.get("error_code"):
                trace["error_code"] = evaluation["error_code"]
            trace["failure_node"] = trace.get("failure_node") or evaluation["failure_node"]
            trace["evaluation_failures"] = evaluation["failures"]
            results.append({"case_id": case["id"], "category": case["category"], "repeat_index": repeat_index + 1, "trace": trace, "evaluation": evaluation})
            if on_result is not None:
                on_result(results[-1])
    metrics = compute_metrics(results, input_price_per_million=input_price_per_million, output_price_per_million=output_price_per_million)
    return {"schema_version": 2, "evaluation": "agent_eval_2.0", "generated_at": _now_iso(), "execution_mode": getattr(runner, "execution_mode", "offline"), "case_count": len(source), "run_count": len(results), "fixtures": {key: value for key, value in fixture_values.items() if key not in {"track_ids"}}, "pricing": {"input_price_per_million": input_price_per_million, "output_price_per_million": output_price_per_million, "note": "Prices are supplied by the caller; no provider price is hard-coded."}, "metrics": metrics, "results": results}


def compare_baseline(payload: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not baseline:
        return None
    current = payload.get("metrics", {})
    prior = baseline.get("metrics", baseline)
    delta = {}
    for name, value in current.items():
        old = prior.get(name, {}) if isinstance(prior, Mapping) else {}
        if isinstance(value, Mapping) and isinstance(old, Mapping) and isinstance(value.get("value_pct"), (int, float)) and isinstance(old.get("value_pct"), (int, float)):
            delta[name] = {"current_pct": value["value_pct"], "baseline_pct": old["value_pct"], "delta_pct_points": round(value["value_pct"] - old["value_pct"], 2)}
    return {"baseline_generated_at": baseline.get("generated_at"), "metrics": delta}


def render_markdown(payload: Mapping[str, Any], *, baseline: Mapping[str, Any] | None = None, json_name: str = "agent-eval.json") -> str:
    metrics = payload["metrics"]
    lines = ["# DropIt Agent Eval 2.0", "", f"Generated: `{payload.get('generated_at', '')}`. Raw traces: `{json_name}`.", "", "## Metrics", "", "| Metric | Result | Denominator |", "| --- | ---: | ---: |"]
    for name, item in metrics.items():
        if not isinstance(item, Mapping):
            lines.append(f"| {name} | {item} | — |")
            continue
        if name == "latency_ms":
            result, denominator = f"P50 {item['p50']} ms / P95 {item['p95']} ms", item["sample_count"]
        elif name == "repair_success_rate":
            result, denominator = ("N/A" if item["status"] == "not_applicable" else f"{item['value_pct']:.2f}%"), item["attempted"]
        elif name in {"token_usage", "estimated_cost"}:
            result = ((f"{item['total_tokens']} tokens" if item["status"] != "unavailable" else "unavailable") if name == "token_usage" else ("unavailable" if item["total"] is None else f"${item['total']:.8f}"))
            denominator = item.get("runs", "—")
        else:
            result = f"{item.get('value_pct', 0):.2f}%"
            denominator = f"{item.get('passed', item.get('stable_groups', '—'))}/{item.get('total', item.get('total_groups', '—'))}"
        lines.append(f"| {name} | {result} | {denominator} |")
    pricing = payload.get("pricing", {})
    failures = Counter(
        (row.get("trace", {}).get("error_code") or "unclassified",
         row.get("trace", {}).get("failure_node") or "unknown")
        for row in payload.get("results", [])
        if not row.get("evaluation", {}).get("task_completed")
    )
    if failures:
        lines.extend([
            "", "## Failure diagnostics", "",
            "| Error code | Failure node | Runs |", "| --- | --- | ---: |",
        ])
        lines.extend(
            f"| {code} | {node} | {count} |"
            for (code, node), count in sorted(failures.items())
        )
    execution_mode = payload.get("execution_mode", "unknown")
    lines.extend(["", "Chat-model usage/cost excludes embedding and intent-classification requests; partial usage is a known subtotal, not a complete bill."])
    lines.extend(["", "## Boundaries", "", f"- Current execution mode is `{execution_mode}`; live traces contain actual compiled LangGraph node updates.", "- Token usage is `unavailable` when provider metadata is missing. Estimated cost remains null unless both input and output prices are explicitly supplied.", f"- Pricing note: {pricing.get('note', 'caller supplied pricing')}"])
    comparison = compare_baseline(payload, baseline)
    if comparison:
        lines.extend(["", "## Baseline comparison", "", "| Metric | Current | Baseline | Delta |", "| --- | ---: | ---: | ---: |"])
        for name, item in comparison["metrics"].items():
            lines.append(f"| {name} | {item['current_pct']:.2f}% | {item['baseline_pct']:.2f}% | {item['delta_pct_points']:+.2f} pp |")
    return "\n".join(lines) + "\n"


def check_thresholds(metrics: Mapping[str, Any], thresholds: Mapping[str, float] | None) -> dict[str, Any]:
    checks = []
    for name, threshold in (thresholds or {}).items():
        if not 0 <= threshold <= 100:
            raise ValueError(f"threshold for {name} must be between 0 and 100")
        item = metrics.get(name, {})
        actual = item.get("value_pct") if isinstance(item, Mapping) else None
        checks.append({"metric": name, "threshold_pct": threshold, "actual_pct": actual, "passed": actual is not None and actual >= threshold})
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def parse_thresholds(values: Iterable[str]) -> dict[str, float]:
    parsed = {}
    for value in values:
        try:
            name, threshold = value.split("=", 1)
            parsed[name] = float(threshold)
            if not 0 <= parsed[name] <= 100:
                raise ValueError(f"threshold for {name} must be between 0 and 100")
        except ValueError as exc:
            raise ValueError(f"threshold must be metric=percentage, got {value!r}") from exc
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "live"), default="live")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--json-out", type=Path, default=Path("eval-results/agent-eval.json"))
    parser.add_argument("--report-out", type=Path, default=Path("eval-results/agent-eval.md"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--rounds", type=int, help="Override repeats for every selected case.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases.")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--project-id")
    parser.add_argument("--traces-out", type=Path)
    parser.add_argument("--threshold", action="append", default=[])
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    return parser.parse_args()



def _build_live_runner(settings: Any, project_id: str | None = None) -> "LiveAgentRunner":
    from backend.agent.agent import DropItAgent
    from backend.agent.graph import IntentRecognizer, OllamaIntentFallback
    from backend.agent.retrieval import DropItToolRegistry
    from backend.evals.runtime import EvaluationSnapshot
    from backend.music.text_models import DashScopeTextEmbedder

    snapshot = EvaluationSnapshot(settings)
    try:
        if not settings.model_configured or not settings.dashscope_api_key:
            raise RuntimeError("Live eval requires DEEPSEEK_API_KEY and DASHSCOPE_API_KEY; use --env-file.")
        embedder = DashScopeTextEmbedder(settings)
        store = snapshot.store
        projects = store.list_projects()
        if project_id:
            projects = [project for project in projects if project.id == project_id]
        eligible = []
        for project in projects:
            tracks = store.all_tracks(project.id)
            vectors = store.music_vectors(project.id, embedder.model_key)
            ready = [track for track in tracks if track.analysis_status == "analyzed"
                     and track.embedding_status == "ready" and track.embedding_model == embedder.model_key
                     and track.id in vectors and vectors[track.id].size == embedder.dimensions]
            if len(ready) >= 2:
                eligible.append((len(ready), project.id, tracks, sorted(ready, key=lambda track: track.id)))
        if not eligible:
            raise RuntimeError("No selected project has at least two analyzed tracks with current-model Chroma vectors.")
        _, selected_id, tracks, ready = max(eligible, key=lambda row: row[0])
        fallback = (OllamaIntentFallback(settings.ollama_base_url, settings.ollama_model,
                                        settings.ollama_timeout_seconds)
                    if settings.intent_fallback_enabled else None)
        registry = DropItToolRegistry(store, embedder)
        agent = DropItAgent(store, registry, settings, intent=IntentRecognizer(fallback=fallback))
        runner = LiveAgentRunner(agent, store, selected_id, ready, tracks, snapshot=snapshot)
        runner.metadata = {"project_id": selected_id, "chat_model": settings.model_name,
                           "embedding_model": embedder.model_key,
                           "intent_model": settings.ollama_model if fallback else None,
                           "source_database": str(snapshot.source_database)}
        return runner
    except BaseException:
        snapshot.close()
        raise


class LiveAgentRunner:
    """Run the compiled production graph, observing node updates and checkpoints."""

    execution_mode = "langgraph"

    def __init__(self, agent: Any, store: Any, project_id: str,
                 reference_tracks: Iterable[Any], tracks: Iterable[Any], *, snapshot: Any = None):
        self.agent, self.store, self.project_id = agent, store, project_id
        self.snapshot = snapshot
        self.tracks_by_id = {track.id: track for track in tracks}
        self.project_track_ids = set(self.tracks_by_id)
        self.fixtures = make_fixtures([track.model_dump(mode="json") for track in reference_tracks])
        self.metadata = {"project_id": project_id}

    def close(self) -> None:
        if self.snapshot is not None:
            self.snapshot.close()

    def run(self, case: Mapping[str, Any], *, repeat_index: int = 1) -> Mapping[str, Any]:
        return asyncio.run(self._run(case, repeat_index))

    async def _run(self, case: Mapping[str, Any], repeat_index: int) -> Mapping[str, Any]:
        from backend.evals.tracing import ModelUsageRecorder

        trace = TraceBuilder(str(case.get("case_id", case["id"])), repeat_index)
        conversation = self.store.create_conversation(self.project_id, f"eval {trace.run_id}")
        original_observer = self.agent.trace_callback
        original_model_factory = self.agent._chat_model
        recorder = ModelUsageRecorder()
        failure_node = None
        wrapped_models = {}

        def model_factory():
            model = original_model_factory()
            if id(model) in wrapped_models:
                return wrapped_models[id(model)][1]
            # Set callbacks on the underlying model, so with_structured_output()
            # retains them too (a RunnableBinding alone can lose this config).
            if hasattr(model, "model_copy"):
                callbacks = list(model.callbacks or []) if isinstance(model.callbacks, list) else []
                wrapped = model.model_copy(update={"callbacks": [*callbacks, recorder]})
            else:
                wrapped = model.with_config(callbacks=[recorder])
            wrapped_models[id(model)] = (model, wrapped)
            return wrapped

        self.agent._chat_model = model_factory
        try:
            for index, turn in enumerate(case["turns"], 1):
                state: dict[str, Any] = {}
                turn_run_id = uuid4().hex
                completed = False
                turn_error = None

                def observe(update):
                    nonlocal failure_node
                    state.update(update["state"])
                    code = update["delta"].get("error_code")
                    trace.graph_steps.append({
                        "name": update["node"], "turn": index, "run_id": turn_run_id,
                        "status": "failed" if code else "done", "error_code": code,
                    })
                    if code and failure_node is None:
                        failure_node = update["node"]

                self.agent.trace_callback = observe
                async for event in self.agent.stream_chat(
                    self.project_id, conversation.id, turn["user"], run_id=turn_run_id,
                ):
                    if event["type"] == "status" and event.get("intent"):
                        trace.command = event["intent"]
                    elif event["type"] == "complete":
                        completed = True
                        trace.output = event
                        turn_error = turn_error or event.get("error_code")
                    elif event["type"] == "error":
                        turn_error = turn_error or event.get("error_code") or "agent_error"

                command = dict(_model_mapping(state.get("command")) or {})
                arguments = dict(command)
                reference = state.get("reference_track")
                if reference is not None:
                    arguments["track_id"] = reference.id
                for tool_event in state.get("tool_events", []):
                    record = dict(_model_mapping(tool_event) or {})
                    trace.tool_calls.append({
                        "name": record.get("name"), "arguments": arguments,
                        "status": record.get("status"), "error_code": record.get("error_code"),
                        "turn": index, "run_id": turn_run_id,
                    })
                turn_trace = {
                    "turn": index, "run_id": turn_run_id, "route": state.get("route"),
                    "command": command, "completed": completed,
                    "repair_attempts": state.get("repair_attempts", 0),
                    "error_code": state.get("error_code") or turn_error,
                    "checkpoints": [row["step"] for row in self.store.list_agent_checkpoints(turn_run_id)],
                }
                if state.get("route") == "generate_dj_set":
                    constraints = validate_set_constraints(
                        state.get("playlist"), command, self.project_track_ids,
                        self.tracks_by_id, project_id=self.project_id,
                    )
                    turn_trace["constraints_passed"] = bool(
                        constraints["passed"] and state.get("playlist_persisted") and not state.get("error_code")
                    )
                    turn_trace["constraint_issues"] = constraints["issues"]
                    if state.get("validation"):
                        turn_trace["validation"] = state["validation"]
                    trace.constraint_issues.extend(constraints["issues"])
                trace.turn_traces.append(turn_trace)
                trace.error_code = trace.error_code or state.get("error_code") or turn_error
            trace.completed = all(turn["completed"] for turn in trace.turn_traces)
            trace.intercepted = bool(trace.turn_traces) and all(
                turn["route"] == "reject" for turn in trace.turn_traces
            ) and not trace.tool_calls
            sets = [turn for turn in trace.turn_traces if "constraints_passed" in turn]
            trace.constraints_passed = all(turn["constraints_passed"] for turn in sets) if sets else None
            trace.repair_attempts = sum(turn["repair_attempts"] for turn in trace.turn_traces)
            repaired = [turn for turn in sets if turn["repair_attempts"]]
            trace.repair_success = all(turn["constraints_passed"] for turn in repaired) if repaired else None
        except Exception as exc:
            trace.error_code = trace.error_code or "runner_error"
            trace.output = {"exception_type": type(exc).__name__}
        finally:
            self.agent.trace_callback = original_observer
            self.agent._chat_model = original_model_factory
            self.agent.memory.forget(f"{self.project_id}:{conversation.id}")
        trace.token_usage = recorder.usage
        result = trace.finish()
        result["token_usage_status"] = recorder.summary["status"]
        result["model_usage"] = recorder.summary
        result["failure_node"] = failure_node
        return result


def main() -> int:
    from backend.evals.runtime import evaluation_settings

    args = parse_args()
    _validate_prices(args.input_price_per_million, args.output_price_per_million)
    thresholds = parse_thresholds(args.threshold)
    unknown = set(thresholds) - {
        "command_accuracy", "argument_accuracy", "tool_sequence_accuracy", "graph_sequence_accuracy",
        "task_completion_rate", "set_constraint_pass_rate", "unauthorized_action_block_rate",
        "stability", "repair_success_rate",
    }
    if unknown:
        raise ValueError(f"Unknown threshold metrics: {sorted(unknown)}")
    if args.rounds is not None and args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    traces_out = args.traces_out or args.json_out.with_suffix(".runs.jsonl")
    paths = [args.json_out.resolve(), args.report_out.resolve(), traces_out.resolve()]
    if len(set(paths)) != len(paths) or (args.baseline and args.baseline.resolve() in paths):
        raise ValueError("Output paths and baseline must be distinct.")
    if traces_out.exists():
        raise FileExistsError(f"Trace output already exists; choose a new --json-out or --traces-out: {traces_out}")
    runner = FixtureTraceRunner() if args.mode == "fixture" else _build_live_runner(
        evaluation_settings(args), args.project_id,
    )
    try:
        traces_out.parent.mkdir(parents=True, exist_ok=True)
        total = sum(args.rounds if args.rounds is not None else int(case["repeats"]) for case in cases)
        with traces_out.open("x", encoding="utf-8") as trace_file:
            count = 0

            def record(row):
                nonlocal count
                trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                trace_file.flush()
                count += 1
                print(f"[{count}/{total}] {row['case_id']} repeat={row['repeat_index']} "
                      f"passed={row['evaluation']['task_completed']} "
                      f"latency_ms={row['trace']['latency_ms']}", flush=True)

            payload = evaluate_cases(
                cases, runner, fixtures=getattr(runner, "fixtures", None),
                input_price_per_million=args.input_price_per_million,
                output_price_per_million=args.output_price_per_million,
                rounds=args.rounds, on_result=record,
            )
        payload["rounds_override"] = args.rounds
        payload["evaluation_scope"] = getattr(runner, "metadata", {})
        payload["baseline_comparison"] = compare_baseline(payload, baseline)
        payload["thresholds"] = {"configured": thresholds, **check_thresholds(payload["metrics"], thresholds)}
        for path in (args.json_out, args.report_out):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        args.report_out.write_text(render_markdown(payload, baseline=baseline, json_name=args.json_out.name), encoding="utf-8")
        print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
        return 0 if payload["thresholds"]["passed"] else 1
    finally:
        close = getattr(runner, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
