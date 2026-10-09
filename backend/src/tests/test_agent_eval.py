import json
from pathlib import Path

import pytest

from backend.evals.run_agent_eval import (
    CaseValidationError,
    FixtureTraceRunner,
    check_thresholds,
    compare_baseline,
    compute_metrics,
    evaluate_cases,
    evaluate_trace,
    load_cases,
    make_fixtures,
    render_markdown,
    resolve_templates,
    TraceBuilder,
    parse_thresholds,
    validate_set_constraints,
)


def test_agent_case_dataset_is_exactly_60_and_balanced():
    cases = load_cases()
    assert len(cases) == 60
    assert {category: sum(row["category"] == category for row in cases)
            for category in ("search", "similar", "set", "multi_turn", "overstep")} == {
                "search": 15, "similar": 10, "set": 20, "multi_turn": 10, "overstep": 5,
            }
    assert all(row["core"] is False or row["repeats"] == 3 for row in cases)


def test_template_resolution_preserves_fixture_types_and_never_requires_real_track_id():
    fixtures = make_fixtures([{"id": "runtime-id", "title": "Signal", "artist": "Mira", "bpm": 124}])
    value = resolve_templates({"id": "{{fixture.reference_track_id}}", "text": "{{fixture.reference_title}}"}, fixtures)
    assert value == {"id": "runtime-id", "text": "Signal"}


def test_fixture_runner_exposes_trace_and_all_metrics():
    cases = load_cases()[:2] + load_cases()[-1:]
    payload = evaluate_cases(cases, FixtureTraceRunner())
    assert payload["run_count"] == 7  # two core cases (3 each) plus one overstep
    trace = payload["results"][0]["trace"]
    assert {"run_id", "command", "graph_steps", "tool_calls", "latency_ms", "token_usage",
            "estimated_cost", "repair_attempts", "error_code"} <= trace.keys()
    assert payload["metrics"]["command_accuracy"]["value_pct"] == 100.0
    assert payload["metrics"]["repair_success_rate"]["status"] == "not_applicable"


def test_metrics_percentiles_stability_and_cost_missing_or_supplied():
    cases = [load_cases()[0]]
    def runner(case, *, repeat_index):
        return {
            "command": case["expected_command"], "tool_calls": [{"name": "search_library", "arguments": {"filters": {"bpm_min": 116, "bpm_max": 128}}}],
            "completed": True, "intercepted": False, "constraints_passed": None,
            "graph_steps": [], "latency_ms": 10 * repeat_index, "token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "repair_attempts": 0,
        }
    payload = evaluate_cases(cases, runner, fixtures=make_fixtures(), input_price_per_million=1, output_price_per_million=2)
    assert payload["metrics"]["latency_ms"]["p50"] == 20
    assert payload["metrics"]["latency_ms"]["p95"] == 29
    assert payload["metrics"]["estimated_cost"]["total"] == .00042
    assert payload["metrics"]["token_usage"]["total_tokens"] == 360


def test_thresholds_write_failure_semantics_and_baseline_markdown():
    payload = evaluate_cases(load_cases()[-1:], FixtureTraceRunner())
    payload["metrics"]["command_accuracy"]["value_pct"] = 0
    checks = check_thresholds(payload["metrics"], {"command_accuracy": 99})
    assert checks["passed"] is False
    baseline = {"generated_at": "before", "metrics": {"command_accuracy": {"value_pct": 100}}}
    comparison = compare_baseline(payload, baseline)
    assert comparison["metrics"]["command_accuracy"]["delta_pct_points"] == -100
    report = render_markdown(payload, baseline=baseline)
    assert "Baseline comparison" in report
    assert "unavailable" in report


def test_argument_subset_and_unauthorized_interception():
    case = load_cases()[-1]
    evaluation = evaluate_trace(case, {"command": "overstep", "tool_calls": [], "completed": True, "intercepted": True})
    assert evaluation["interception_passed"] is True
    set_case = next(row for row in load_cases() if row["id"] == "set-09")
    resolved = resolve_templates(set_case, make_fixtures())
    evaluation = evaluate_trace(resolved, {"command": "generate_dj_set", "tool_calls": [{"name": "generate_dj_set", "arguments": {"track_ids": []}}], "completed": True, "intercepted": False, "constraints_passed": True})
    assert evaluation["argument_passed"] is True


def test_invalid_case_schema_is_rejected(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"id": "bad"}) + "\n", encoding="utf-8")
    with pytest.raises(CaseValidationError):
        load_cases(path)


def test_set_constraints_validate_playlist_and_report_structured_failures():
    from backend.models import Track, Playlist, PlaylistTrack, Brief

    tracks = {
        key: Track(id=key, title=key, artist="artist", filename=f"{key}.wav", path=key,
                   bpm=bpm, camelot_key="8A", energy=energy, duration_sec=300)
        for key, bpm, energy in (("a", 120, .4), ("b", 122, .6))
    }
    playlist = Playlist(id="playlist", project_id="project", duration_sec=600,
                        brief=Brief(duration_min=10),
                        tracks=[PlaylistTrack(track=track, reason="eval") for track in tracks.values()])
    arguments = {"duration_min": 10, "bpm_min": 118, "bpm_max": 130, "energy_curve": "build"}
    assert validate_set_constraints(playlist, arguments, tracks, tracks)["passed"] is True

    invalid = playlist.model_copy(update={"duration_sec": 100, "tracks": [playlist.tracks[0]] * 3})
    result = validate_set_constraints(invalid, {**arguments, "track_ids": ["a"]}, ["a", "b"], tracks)
    assert result["passed"] is False
    assert {"duplicate_tracks", "duration_summary_mismatch", "duration_tolerance"} <= {x["code"] for x in result["issues"]}
    # Canonical durations and BPM override fabricated embedded facts.
    fake = playlist.model_copy(update={"tracks": [
        row.model_copy(update={"track": row.track.model_copy(update={"duration_sec": 1, "bpm": 999})})
        for row in playlist.tracks
    ]})
    assert validate_set_constraints(fake, arguments, tracks, tracks)["passed"] is True
    assert validate_set_constraints(playlist, {**arguments, "track_ids": ["a"]}, tracks, tracks)["passed"] is False
    assert validate_set_constraints(playlist, arguments, tracks, tracks, project_id="other")["passed"] is False


def test_task_completion_is_stricter_than_completion_event_and_tool_failure_is_stable():
    case = next(row for row in load_cases() if row["id"] == "overstep-01")
    trace = {"command": "overstep", "tool_calls": [], "completed": True, "intercepted": True, "error_code": "agent_error"}
    evaluation = evaluate_trace(case, trace)
    assert evaluation["completion_event_passed"] is True
    assert evaluation["task_completed"] is False
    assert evaluation["error_code"] == "agent_error"
    assert evaluation["failure_node"] == "model"

    search = load_cases()[0]
    failed = {"command": search["expected_command"], "tool_calls": [{"name": "search_library", "arguments": {}, "status": "failed"}], "completed": True, "intercepted": False}
    evaluation = evaluate_trace(search, failed)
    assert evaluation["error_code"] == "tool_failed"
    assert evaluation["failure_node"] == "tool"
    assert evaluation["task_completed"] is False

    mismatch = evaluate_trace(search, {
        "command": "music_chat", "tool_calls": [], "completed": True,
        "intercepted": False,
    })
    assert mismatch["error_code"] == "command_mismatch"
    assert mismatch["failure_node"] == "route"
    assert {item["code"] for item in mismatch["failures"]} >= {
        "command_mismatch", "argument_mismatch", "tool_sequence_mismatch",
    }


def test_stability_includes_normalized_arguments_completion_and_error():
    case = load_cases()[-1]
    def runner(case, *, repeat_index):
        return {"command": "overstep", "tool_calls": [], "completed": True, "intercepted": True,
                "error_code": None if repeat_index == 1 else "agent_error", "latency_ms": 1}
    payload = evaluate_cases([case | {"repeats": 2}], runner)
    assert payload["metrics"]["stability"]["value_pct"] == 0.0


def test_trace_start_and_price_threshold_validation():
    builder = TraceBuilder("case")
    assert builder.finish()["started_at"] == builder.started_at
    with pytest.raises(ValueError):
        compute_metrics([], input_price_per_million=1)
    with pytest.raises(ValueError):
        compute_metrics([], input_price_per_million=-1, output_price_per_million=1)
    with pytest.raises(ValueError):
        check_thresholds({}, {"command_accuracy": 101})
    with pytest.raises(ValueError):
        parse_thresholds(["command_accuracy=-1"])


def test_markdown_summarizes_failure_code_and_node():
    case = load_cases()[0]

    def runner(case, *, repeat_index):
        return {
            "command": "music_chat", "tool_calls": [], "completed": True,
            "intercepted": False, "latency_ms": 1,
        }

    payload = evaluate_cases([case | {"repeats": 1}], runner)
    report = render_markdown(payload)
    assert "Failure diagnostics" in report
    assert "command_mismatch" in report
    assert "route" in report
