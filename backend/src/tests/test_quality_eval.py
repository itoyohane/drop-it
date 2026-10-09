from backend.evals.run_quality_eval import _aggregate, _fact_equal, _pct


def test_percentage_and_fact_comparison_are_stable():
    assert _pct(3, 4) == 75.0
    assert _pct(0, 0) == 0.0
    assert _fact_equal("bpm", 120.0000001, 120.0)
    assert not _fact_equal("title", "Signal", "signal")


def test_aggregate_uses_micro_averages_and_minimum_vector_count():
    rounds = [
        {
            "vector_audit": {"valid_vectorized_tracks": 10, "integrity_rate_pct": 100.0},
            "intent": {"passed": 4, "total": 5, "accuracy_pct": 80.0},
            "tools": {"passed": 3, "total": 4},
            "rag": {"hits_at_3": 2, "queries": 3, "facts_verified": 20, "facts_checked": 24},
        },
        {
            "vector_audit": {"valid_vectorized_tracks": 9, "integrity_rate_pct": 90.0},
            "intent": {"passed": 5, "total": 5, "accuracy_pct": 100.0},
            "tools": {"passed": 4, "total": 4},
            "rag": {"hits_at_3": 3, "queries": 3, "facts_verified": 24, "facts_checked": 24},
        },
    ]
    result = _aggregate(rounds)
    assert result["vectorized_tracks"] == 9
    assert result["intent_accuracy_pct"] == 90.0
    assert result["tool_success_rate_pct"] == 87.5
    assert result["rag_accuracy_pct"] == 83.33
    assert result["rag_truthfulness_pct"] == 91.67


def test_report_and_component_execution_follow_current_registry_and_round_count(library, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from backend.config import Settings
    from backend.evals import run_quality_eval as quality
    from backend.agent.graph import IntentRecognizer
    from backend.agent.retrieval import DropItToolRegistry
    from backend.tests.conftest import add_track

    store, project, _, embedder, _ = library
    for index, energy in enumerate((.2, .5, .8)):
        add_track(store, project, f"t{index}", [1, .1 * index, 0], duration_sec=200, energy=energy)
    proxy = quality.ReadOnlyEvaluationStore(store)
    tracks = store.all_tracks(project.id)
    closed = []
    settings = Settings(_env_file=None, DROPIT_INTENT_FALLBACK_ENABLED=False)
    context = quality.EvaluationContext(
        settings, store, proxy, DropItToolRegistry(proxy, embedder), IntentRecognizer(),
        project.id, project.name, tracks, {track.id: track for track in tracks},
        snapshot=SimpleNamespace(close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(quality, "build_context", lambda settings: context)
    payload = quality.run(1, 2, settings)
    assert closed == [True]
    assert payload["aggregate"]["rounds"] == 1
    assert payload["rounds"][0]["tools"]["total"] == 8
    assert payload["rounds"][0]["rag"]["queries"] == 2
    payload["agent_eval"] = {"path": "agent.json", "generated_at": "now", "execution_mode": "fixture",
                             "metrics": {"task_completion_rate": {"value_pct": 50}}}
    report = quality.render_report(payload, Path("quality.json"))
    assert "# DropIt 1 轮质量评测报告" in report
    assert report.count("### 第 1 轮") == 1
    assert "LangChain Tool schema" not in report
    assert "意图路由每轮固定错" not in report
    assert "task_completion_rate: 50%" in report
    assert "--rounds 1 --rag-cases-per-round 2" in report
