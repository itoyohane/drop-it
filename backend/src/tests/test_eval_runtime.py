import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from backend.config import Settings
from backend.evals.runtime import EvaluationSnapshot, evaluation_settings
from backend.evals import run_agent_eval
from backend.tests.conftest import add_track


def test_missing_database_has_actionable_absolute_path(tmp_path):
    settings = Settings(_env_file=None, DROPIT_DATA_DIR=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="--data-dir") as exc:
        EvaluationSnapshot(settings)
    assert str(tmp_path / "absent" / "dropit.db") in str(exc.value)
    assert not (tmp_path / "absent").exists()


def test_explicit_configuration_overrides_env_file_and_environment(tmp_path, monkeypatch):
    env_file = tmp_path / "eval.env"
    env_file.write_text("DROPIT_MODEL=test-model\nDROPIT_DATA_DIR=relative-catalog\n", encoding="utf-8")
    monkeypatch.setenv("DROPIT_DATA_DIR", "wrong-environment-directory")
    settings = evaluation_settings(SimpleNamespace(env_file=env_file, data_dir=tmp_path / "catalog", chroma_dir=None))
    assert settings.data_dir == (tmp_path / "catalog").resolve()
    assert settings.model_name == "test-model"
    with pytest.raises(FileNotFoundError, match="env file"):
        evaluation_settings(SimpleNamespace(env_file=tmp_path / "missing.env"))


def test_catalog_snapshot_keeps_writes_off_source_and_copies_vectors(library):
    store, project, _, embedder, _ = library
    add_track(store, project, "Signal", [1, 0, 0])
    source_db = Path(store.connection.execute("PRAGMA database_list").fetchone()[2])
    settings = Settings(_env_file=None, DROPIT_DATA_DIR=source_db.parent)
    # Fixture DB is named test.db; use its existing path by making a read-only
    # online backup with the production filename inside this test directory.
    import sqlite3
    with sqlite3.connect(source_db.parent / "dropit.db") as destination:
        store.connection.backup(destination)
    source_messages = store.connection.execute("select count(*) from messages").fetchone()[0]
    snapshot = EvaluationSnapshot(settings)
    snapshot_root = Path(snapshot.store.connection.execute("PRAGMA database_list").fetchone()[2]).parent
    try:
        assert snapshot.store.music_vectors(project.id, embedder.model_key)["Signal"].shape == (3,)
        conversation = snapshot.store.create_conversation(project.id, "isolated")
        snapshot.store.add_message(project.id, conversation.id, "user", "eval")
        assert store.connection.execute("select count(*) from messages").fetchone()[0] == source_messages
        with sqlite3.connect(source_db.parent / "dropit.db") as source:
            assert source.execute("select count(*) from messages").fetchone()[0] == source_messages
    finally:
        snapshot.close()
    assert not snapshot_root.exists()


def test_cli_round_override_progress_incremental_traces_and_threshold_exit(tmp_path, monkeypatch):
    output = tmp_path / "agent.json"
    report = tmp_path / "agent.md"
    monkeypatch.setattr(sys, "argv", ["eval", "--mode", "fixture", "--rounds", "2", "--limit", "1",
                                     "--json-out", str(output), "--report-out", str(report),
                                     "--threshold", "task_completion_rate=80"])
    assert run_agent_eval.main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_count"] == 2
    assert payload["rounds_override"] == 2
    assert payload["execution_mode"] == "fixture"
    assert len(output.with_suffix(".runs.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert "fixture" in report.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_agent_eval.main()

    original_run = run_agent_eval.FixtureTraceRunner.run

    def broken(self, case, **kwargs):
        return {**original_run(self, case, **kwargs), "error_code": "intentional-test-failure"}

    monkeypatch.setattr(run_agent_eval.FixtureTraceRunner, "run", broken)
    monkeypatch.setattr(sys, "argv", ["eval", "--mode", "fixture", "--rounds", "1", "--limit", "1",
                                     "--json-out", str(tmp_path / "failure.json"), "--report-out", str(report),
                                     "--threshold", "task_completion_rate=80"])
    assert run_agent_eval.main() == 1


@pytest.mark.parametrize("argument,value", [("--rounds", "0"), ("--limit", "0"),
                                           ("--threshold", "typo=90"),
                                           ("--input-price-per-million", "nan")])
def test_invalid_cli_options_fail_before_live_initialization(argument, value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval", argument, value])
    monkeypatch.setattr(run_agent_eval, "_build_live_runner", lambda *args: pytest.fail("must not start live eval"))
    with pytest.raises(ValueError):
        run_agent_eval.main()


def test_usage_callback_handles_provider_metadata_and_marks_partial_subtotal():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    from backend.evals.tracing import ModelUsageRecorder

    recorder = ModelUsageRecorder()
    recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="no usage"))]]))
    assert recorder.usage is None
    assert recorder.summary["status"] == "unavailable"
    recorder.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(
        content="answer", response_metadata={"token_usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    ))]]))
    assert recorder.usage["total_tokens"] == 8
    assert recorder.summary["status"] == "partial"
    metrics = run_agent_eval.compute_metrics([{"trace": {"token_usage": recorder.usage,
                                                          "token_usage_status": "partial"}}],
                                             input_price_per_million=1, output_price_per_million=2)
    assert metrics["token_usage"]["status"] == "partial"
    assert metrics["estimated_cost"]["status"] == "partial"


@pytest.mark.parametrize("valid_project", [True, False])
def test_live_builder_selects_current_vector_scope_and_cleans_up_on_failure(library, monkeypatch, valid_project):
    from backend.evals import runtime
    from backend.music import text_models

    store, project, other, embedder, _ = library
    add_track(store, project, "Signal", [1, 0, 0])
    add_track(store, project, "Echo", [.9, .1, 0])
    # A ready SQLite row alone is insufficient to select the reference song.
    unindexed = add_track(store, project, "MissingVector")
    store.update_track_analysis(unindexed.model_copy(update={"embedding_status": "ready", "embedding_model": embedder.model_key}))
    closed = []
    monkeypatch.setattr(runtime, "EvaluationSnapshot", lambda settings: SimpleNamespace(
        store=store, source_database=Path("source/dropit.db"), close=lambda: closed.append(True)))
    monkeypatch.setattr(text_models, "DashScopeTextEmbedder", lambda settings: embedder)
    settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-only", DASHSCOPE_API_KEY="test-only",
                        DROPIT_INTENT_FALLBACK_ENABLED=False)
    if valid_project:
        runner = run_agent_eval._build_live_runner(settings, project.id)
        assert set(runner.fixtures["track_ids"]) == {"Echo", "Signal"}
        assert runner.project_id == project.id
        assert runner.agent.intent.fallback is None
        runner.close()
    else:
        with pytest.raises(RuntimeError, match="at least two"):
            run_agent_eval._build_live_runner(settings, other.id)
    assert closed == [True]
