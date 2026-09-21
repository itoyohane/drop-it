import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
import pytest

from backend.agent.agent import DropItAgent
from backend.agent.checkpoints import AgentCheckpointStore, next_node_for_checkpoint
from backend.agent.commands import GenerateSetCommand
from backend.agent.intent import IntentRecognizer
from backend.agent.set_repair import SetRepairer
from backend.agent.set_validation import SetValidator
from backend.agent.tools import (
    DropItToolRegistry, SetConstraintConflictError, generate_playlist,
)
from backend.config import Settings
from backend.main import create_app
from backend.models import Brief, Playlist, PlaylistTrack, Track, derive_agent_playlist_id
from backend.repositories import DropItStore, StaleAgentRunError
from backend.tests.conftest import FakeAnalyzer, FakeDescriptor, FakeEmbedder, add_track


class _JsonModel(FakeMessagesListChatModel):
    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError

    def bind_tools(self, tools, **kwargs):
        raise AssertionError("controlled graph must not bind tools")


def _track(track_id: str, *, bpm: float = 124, key: str = "8A",
           energy: float = .5, duration_sec: int = 200) -> Track:
    return Track(
        id=track_id, title=track_id, artist="Mira", filename=f"{track_id}.wav",
        path=f"/music/{track_id}.wav", duration_sec=duration_sec, bpm=bpm,
        key="A minor", camelot_key=key, energy=energy, analysis_status="analyzed",
    )


def _playlist(project_id: str, tracks: list[Track], *, duration_min: int = 10,
              energy: str = "build") -> Playlist:
    return Playlist(
        id="playlist-p0", project_id=project_id,
        brief=Brief(duration_min=duration_min, bpm_min=118, bpm_max=132, energy=energy),
        tracks=[PlaylistTrack(track=track, reason="test") for track in tracks],
        duration_sec=sum(track.duration_sec for track in tracks),
    )


def _create_009_database(db_path: Path) -> None:
    """Create a deployed-v9 database by executing the checked-in SQL verbatim."""

    migrations = Path(__file__).parents[1] / "migrations"
    connection = sqlite3.connect(db_path)
    try:
        for version in range(1, 10):
            migration = next(migrations.glob(f"{version:03d}_*.sql"))
            connection.executescript(migration.read_text(encoding="utf-8"))
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO schema_migrations VALUES (?, datetime('now'))",
            [(version,) for version in range(1, 10)],
        )
        connection.execute(
            """INSERT INTO agent_runs
               (run_id, project_id, conversation_id, route, user_text, history,
                status, created_at, updated_at)
               VALUES ('legacy-run', 'global-chat', 'global-chat-default', 'music_chat',
                       'old', '[]', 'running', datetime('now'), datetime('now'))"""
        )
        connection.execute(
            """INSERT INTO agent_run_steps (run_id, step, state, created_at)
               VALUES ('legacy-run', 'command_parsed',
                       '{"run_id":"legacy-run","marker":"preserved"}', datetime('now'))"""
        )
        connection.commit()
    finally:
        connection.close()


def test_set_validator_reports_all_structured_constraint_families():
    inside = _track("inside", bpm=120, key="1A", energy=.8, duration_sec=60)
    bad_bpm = _track("bad-bpm", bpm=150, key="4B", energy=.2, duration_sec=60)
    outsider = _track("outsider", bpm=120, key="4B", energy=.2, duration_sec=60)
    playlist = _playlist("project", [inside, bad_bpm, outsider, inside], duration_min=10)

    result = SetValidator().validate(
        playlist, ["inside", "bad-bpm"], required_tracks=["required"]
    )

    assert not result.valid
    assert {
        "track_membership", "duplicate_tracks", "bpm_range", "duration_tolerance",
        "bpm_transition", "camelot_compatibility", "energy_curve", "required_tracks",
    } <= set(result.failed_codes)
    assert all(issue.code and issue.message for issue in result.issues)
    assert result.model_dump()["valid"] is False


def test_set_repairer_is_deterministic_and_can_repair_without_writes():
    candidates = [
        _track("a", energy=.2), _track("b", energy=.5), _track("c", energy=.8),
    ]
    invalid = _playlist("project", [candidates[2], candidates[2], candidates[0]])
    validator = SetValidator()
    before = validator.validate(invalid, [track.id for track in candidates])
    repaired_a = SetRepairer(validator=validator).repair(
        invalid, before.issues, candidates, attempt=1
    ).playlist
    repaired_b = SetRepairer(validator=validator).repair(
        invalid, before.issues, candidates, attempt=1
    ).playlist

    assert repaired_a.model_dump(mode="json") == repaired_b.model_dump(mode="json")
    assert validator.validate(repaired_a, [track.id for track in candidates]).valid


def test_unrepairable_set_uses_exactly_two_rounds_and_is_not_persisted(library):
    store, project, _, _, registry = library
    add_track(store, project, "only", duration_sec=600)
    command = GenerateSetCommand(
        request="required conflict", duration_min=10, required_tracks=["missing"]
    )
    candidates = registry.retrieve_set_candidates(project.id, command)

    with pytest.raises(SetConstraintConflictError) as caught:
        registry.validate_and_repair_set(project.id, command, candidates)

    assert caught.value.attempts == 2
    assert "required_tracks" in caught.value.result.failed_codes
    assert store.list_playlists(project.id) == []


def test_checkpoint_step_is_unique_and_reuses_first_result(library):
    store, project, _, _, _ = library
    conversation = store.ensure_default_conversation(project.id)
    manager = AgentCheckpointStore(store)
    run_id = "run-checkpoint-idempotent"
    manager.start_run(run_id, project.id, conversation.id, "generate_dj_set", "set", [])
    owner = "checkpoint-test-owner"
    token = store.claim_agent_run(run_id, owner, lease_seconds=5)

    first = manager.save(run_id, "command_parsed", {"run_id": run_id, "command": {"request": "first"}},
                         owner=owner, token=token)
    second = manager.save(run_id, "command_parsed", {"run_id": run_id, "command": {"request": "second"}},
                          owner=owner, token=token)

    assert first["state"] == second["state"]
    assert [item["step"] for item in manager.list(run_id)] == ["command_parsed"]


def test_claim_fencing_uses_two_connections_and_allows_expired_takeover(tmp_path):
    db_path = tmp_path / "claims.db"
    first = DropItStore(str(db_path), vector_store_path=tmp_path / "chroma-a")
    second = DropItStore(str(db_path), vector_store_path=tmp_path / "chroma-b")
    project = first.create_project("Claims")
    conversation = first.ensure_default_conversation(project.id)
    run_id = "claim-barrier-run"
    first.create_agent_run(run_id, project.id, conversation.id, "music_chat", "claim", [])
    barrier = threading.Barrier(2)
    claims: dict[str, int | None] = {}

    def contender(store, owner):
        barrier.wait()
        claims[owner] = store.claim_agent_run(run_id, owner, lease_seconds=.08)

    left = threading.Thread(target=contender, args=(first, "owner-a"))
    right = threading.Thread(target=contender, args=(second, "owner-b"))
    left.start(); right.start(); left.join(); right.join()
    winner = next(owner for owner, token in claims.items() if token is not None)
    winner_token = claims[winner]
    loser = "owner-b" if winner == "owner-a" else "owner-a"
    assert claims[loser] is None
    time.sleep(.11)
    takeover = second if winner == "owner-a" else first
    takeover_token = takeover.claim_agent_run(run_id, loser, lease_seconds=.2)
    assert takeover_token is not None and takeover_token > winner_token

    with pytest.raises(StaleAgentRunError, match="stale owner"):
        first.save_agent_checkpoint(run_id, "command_parsed", "{}", owner=winner, token=winner_token)
    with pytest.raises(StaleAgentRunError, match="stale owner"):
        first.update_agent_run(run_id, owner=winner, token=winner_token, status="completed")
    with pytest.raises(StaleAgentRunError, match="stale owner"):
        first.add_message(project.id, conversation.id, "assistant", "stale", agent_run_id=run_id,
                           agent_owner=winner, agent_token=winner_token)
    stale_playlist = Playlist(
        id=derive_agent_playlist_id(project.id, run_id), project_id=project.id,
        agent_run_id=run_id, brief=Brief(title="stale"), tracks=[], duration_sec=0,
    )
    with pytest.raises(StaleAgentRunError, match="stale owner"):
        first.save_agent_playlist(
            run_id, project.id, stale_playlist, winner, winner_token
        )
    assert takeover.renew_agent_run(run_id, loser, takeover_token, lease_seconds=.2)
    first.close(); second.close()


def test_blocking_registry_call_does_not_starve_run_heartbeat(tmp_path):
    db_path = tmp_path / "heartbeat.db"
    store = DropItStore(str(db_path), vector_store_path=tmp_path / "chroma-heartbeat-a")
    observer = DropItStore(str(db_path), vector_store_path=tmp_path / "chroma-heartbeat-b")
    try:
        project = store.create_project("Heartbeat")
        conversation = store.ensure_default_conversation(project.id)
        track = add_track(store, project, "slow-retrieval", duration_sec=600)
        registry = DropItToolRegistry(store, FakeEmbedder())
        original = registry.retrieve_set_candidates
        started = threading.Event()

        def slow_retrieve(*args, **kwargs):
            started.set()
            time.sleep(.25)
            return original(*args, **kwargs)

        registry.retrieve_set_candidates = slow_retrieve
        run_id = "run-heartbeat-during-blocking-registry"
        model = _JsonModel(responses=[
            AIMessage(content=json.dumps({
                "request": "slow", "duration_min": 10, "bpm_min": 110,
                "bpm_max": 140, "energy_curve": "build", "track_ids": [track.id],
            })),
            AIMessage(content="已完成。"),
        ])
        agent = DropItAgent(
            store, registry,
            Settings(
                _env_file=None, DEEPSEEK_API_KEY="test-only",
                DROPIT_AGENT_RUN_LEASE_SECONDS=.09,
            ),
            intent=IntentRecognizer(),
        )
        agent._chat_model = lambda: model

        async def collect():
            return [event async for event in agent.stream_chat(
                project.id, conversation.id, "生成一个十分钟 set", run_id
            )]

        async def attempt_takeover():
            assert await asyncio.to_thread(started.wait, .5)
            await asyncio.sleep(.14)
            snapshot = await asyncio.to_thread(observer.get_agent_run, run_id)
            token = await asyncio.to_thread(
                observer.claim_agent_run, run_id, "observer", lease_seconds=.2
            )
            return snapshot, token

        async def run_both():
            return await asyncio.gather(collect(), attempt_takeover())

        events, (snapshot, takeover_token) = asyncio.run(run_both())
        assert snapshot["claim_token"] == 1
        assert snapshot["claim_owner"] is not None
        assert takeover_token is None
        assert events[-1]["playlist"]["agent_run_id"] == run_id
        assert store.get_agent_run(run_id)["claim_token"] == 1
    finally:
        store.close()
        observer.close()


def test_repository_close_releases_chroma_and_is_idempotent():
    store = DropItStore(":memory:")
    client = store.vector_store.client
    store.close()
    store.close()
    assert getattr(client, "_closed", False) is True


def test_agent_playlist_id_is_project_scoped_and_raw_run_id_collision_is_harmless(library):
    store, project, other, _, _ = library
    run_id = "client-retry-id"
    assert derive_agent_playlist_id(project.id, run_id) != derive_agent_playlist_id(other.id, run_id)
    conversation = store.ensure_default_conversation(project.id)
    store.create_agent_run(run_id, project.id, conversation.id, "music_chat", "set", [])
    owner = "playlist-owner"
    token = store.claim_agent_run(run_id, owner, lease_seconds=5)
    brief = Brief(title="collision", duration_min=10, bpm_min=110, bpm_max=140)
    raw_collision = Playlist(id=run_id, project_id=project.id, brief=brief, tracks=[], duration_sec=0)
    store.save_playlist(raw_collision)
    playlist = Playlist(
        id=derive_agent_playlist_id(project.id, run_id), project_id=project.id,
        agent_run_id=run_id, brief=brief, tracks=[], duration_sec=0,
    )
    saved = store.save_agent_playlist(run_id, project.id, playlist, owner, token)
    assert saved.id != run_id
    assert saved.agent_run_id == run_id
    assert store.get_playlist(run_id).id == run_id
    store.release_agent_run(run_id, owner, token)


def test_old_009_database_upgrades_to_010_without_losing_runs(tmp_path):
    db_path = tmp_path / "old-009.db"
    _create_009_database(db_path)
    store = DropItStore(str(db_path), vector_store_path=tmp_path / "chroma-upgrade")
    run = store.get_agent_run("legacy-run")
    assert run and run["latest_version"] == 0 and run["claim_token"] == 0
    checkpoint = AgentCheckpointStore(store).get("legacy-run", "command_parsed")
    assert checkpoint["version"] == 0
    assert checkpoint["state"]["marker"] == "preserved"
    columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(playlists)")}
    assert "agent_run_id" in columns
    assert {row["version"] for row in store.connection.execute(
        "SELECT version FROM schema_migrations") if row["version"] == 10} == {10}
    store.close()


def test_010_recovers_an_existing_legacy_steps_table(tmp_path):
    db_path = tmp_path / "partial-010.db"
    _create_009_database(db_path)
    connection = sqlite3.connect(db_path)
    connection.execute("ALTER TABLE agent_run_steps RENAME TO agent_run_steps_legacy")
    connection.commit()
    connection.close()

    store = DropItStore(str(db_path), vector_store_path=tmp_path / "chroma-partial")
    try:
        checkpoint = AgentCheckpointStore(store).get("legacy-run", "command_parsed")
        assert checkpoint["state"]["marker"] == "preserved"
        tables = {row["name"] for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "agent_run_steps" in tables
        assert "agent_run_steps_legacy" not in tables
    finally:
        store.close()


@pytest.mark.parametrize("fault_stage", [
    "agent_run_steps_renamed",
    "agent_run_steps_created",
])
def test_010_failure_rolls_back_and_can_retry_without_data_loss(
    tmp_path, monkeypatch, fault_stage
):
    db_path = tmp_path / f"fault-{fault_stage}.db"
    _create_009_database(db_path)

    def fail_at_stage(self, stage):
        if stage == fault_stage:
            raise RuntimeError(f"injected migration failure: {stage}")

    monkeypatch.setattr(DropItStore, "_migration_fault_point", fail_at_stage)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        DropItStore(str(db_path), vector_store_path=tmp_path / f"chroma-{fault_stage}")

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version=10"
        ).fetchone() is None
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "agent_run_steps" in tables
        assert "agent_run_steps_legacy" not in tables
        columns = {row[1] for row in connection.execute("PRAGMA table_info(agent_run_steps)")}
        assert "version" not in columns
        state = connection.execute(
            "SELECT state FROM agent_run_steps WHERE run_id='legacy-run'"
        ).fetchone()[0]
        assert json.loads(state)["marker"] == "preserved"
    finally:
        connection.close()

    monkeypatch.setattr(DropItStore, "_migration_fault_point", lambda self, stage: None)
    store = DropItStore(
        str(db_path), vector_store_path=tmp_path / f"chroma-retry-{fault_stage}"
    )
    try:
        assert AgentCheckpointStore(store).get(
            "legacy-run", "command_parsed"
        )["state"]["marker"] == "preserved"
        assert store.connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version=10"
        ).fetchone() is not None
    finally:
        store.close()


def test_agent_resumes_from_checkpoint_and_repeated_run_writes_one_playlist(library):
    store, project, _, _, registry = library
    track = add_track(store, project, "resume", duration_sec=600)
    conversation = store.ensure_default_conversation(project.id)
    run_id = "run-restart-resume"
    command = GenerateSetCommand(request="resume", duration_min=10, track_ids=[track.id])
    playlist = Playlist(
        id=derive_agent_playlist_id(project.id, run_id), project_id=project.id,
        agent_run_id=run_id,
        brief=Brief(title="resume", duration_min=10, bpm_min=110, bpm_max=140),
        tracks=[PlaylistTrack(track=track, reason="test")], duration_sec=600,
    )
    manager = AgentCheckpointStore(store)
    history = [{"role": "user", "content": "resume"}]
    manager.start_run(run_id, project.id, conversation.id, "generate_dj_set", "resume", history)
    owner = "resume-test-owner"
    token = store.claim_agent_run(run_id, owner, lease_seconds=5)
    store.add_message(project.id, conversation.id, "user", "resume", agent_run_id=run_id)
    base = {
        "run_id": run_id, "route": "generate_dj_set", "user_text": "resume",
        "history": history, "command": command, "candidate_tracks": [track],
        "candidate_track_ids": [track.id], "playlist": playlist, "tool_events": [],
    }
    manager.save(run_id, "command_parsed", base, owner=owner, token=token)
    manager.save(run_id, "candidates_retrieved", base, owner=owner, token=token)
    manager.save(run_id, "set_planned", base, owner=owner, token=token)
    store.release_agent_run(run_id, owner, token)

    model = FakeMessagesListChatModel(responses=[AIMessage(content="已从 checkpoint 恢复。")])
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model

    async def run_once(current_agent):
        return [event async for event in current_agent.stream_chat(
            project.id, conversation.id, "resume", run_id
        )]

    first_events = asyncio.run(run_once(agent))
    assert first_events[-1]["playlist"]["id"] == derive_agent_playlist_id(project.id, run_id)
    assert len(store.list_playlists(project.id)) == 1
    assert {item["step"] for item in manager.list(run_id)} >= {
        "command_parsed", "candidates_retrieved", "set_planned", "set_validated",
        "playlist_persisted", "completed",
    }

    class ExplodingModel:
        async def ainvoke(self, messages):
            raise AssertionError("completed checkpoint must avoid a model call")

    restarted = DropItAgent(
        store, DropItToolRegistry(store, FakeEmbedder()),
        Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    restarted._chat_model = lambda: ExplodingModel()
    second_events = asyncio.run(run_once(restarted))
    assert second_events[-1]["resumed"] is True
    assert second_events[-1]["message"]["id"] == first_events[-1]["message"]["id"]
    assert len(store.list_playlists(project.id)) == 1


def test_duration_validation_uses_track_sum_and_reports_summary_mismatch():
    tracks = [
        _track("a", energy=.2, duration_sec=200),
        _track("b", energy=.5, duration_sec=200),
        _track("c", energy=.8, duration_sec=200),
    ]
    playlist = _playlist("project", tracks)
    playlist.duration_sec = 999

    result = SetValidator().validate(playlist, [track.id for track in tracks])

    assert "duration_summary_mismatch" in result.failed_codes
    assert "duration_tolerance" not in result.failed_codes
    duration_issue = next(issue for issue in result.issues
                          if issue.code == "duration_summary_mismatch")
    assert duration_issue.details == {
        "summary_duration_sec": 999,
        "actual_duration_sec": 600,
    }

    falsely_matching_summary = _playlist("project", [tracks[0]])
    falsely_matching_summary.duration_sec = 600
    failed = SetValidator().validate(falsely_matching_summary, [tracks[0].id])
    assert "duration_tolerance" in failed.failed_codes
    assert next(issue for issue in failed.issues
                if issue.code == "duration_tolerance").details["duration_sec"] == 200


def test_peak_planner_and_validator_share_a_middle_late_peak_definition():
    tracks = [
        _track("low", energy=.2, duration_sec=150),
        _track("mid", energy=.45, duration_sec=150),
        _track("peak", energy=.8, duration_sec=150),
        _track("release", energy=.55, duration_sec=150),
    ]

    playlist = generate_playlist(
        "project", Brief(duration_min=10, energy="peak"), tracks,
        playlist_id="peak-feasible",
    )
    energies = [row.track.energy for row in playlist.tracks]

    assert energies == [.2, .45, .8, .55]
    assert SetValidator().validate(playlist, [track.id for track in tracks]).valid


@pytest.mark.parametrize(("step", "state", "expected"), [
    ("command_parsed", {}, "retrieve_candidates"),
    ("candidates_retrieved", {}, "plan_set"),
    ("set_planned", {}, "validate_set"),
    ("set_validated", {"validation": {"valid": False}, "repair_attempts": 0}, "repair_set"),
    ("set_repaired", {"repair_attempts": 1}, "validate_set"),
    ("set_validated", {"validation": {"valid": True}, "repair_attempts": 1}, "persist_set"),
    ("playlist_persisted", {}, "respond"),
    ("completed", {}, "respond"),
])
def test_latest_checkpoint_routes_to_the_next_node(step, state, expected):
    assert next_node_for_checkpoint(
        "generate_dj_set", {"step": step, "state": state}
    ) == expected


def test_restart_after_set_repaired_continues_at_validation(library):
    store, project, _, _, registry = library
    track = add_track(store, project, "repaired", duration_sec=600)
    conversation = store.ensure_default_conversation(project.id)
    run_id = "run-after-repair"
    command = GenerateSetCommand(request="repair resume", duration_min=10,
                                 track_ids=[track.id])
    playlist = Playlist(
        id=derive_agent_playlist_id(project.id, run_id), project_id=project.id,
        agent_run_id=run_id,
        brief=Brief(title="repair resume", duration_min=10, bpm_min=110, bpm_max=140),
        tracks=[PlaylistTrack(track=track, reason="repaired")], duration_sec=600,
    )
    manager = AgentCheckpointStore(store)
    history = [{"role": "user", "content": "repair resume"}]
    manager.start_run(run_id, project.id, conversation.id, "generate_dj_set",
                      "repair resume", history)
    owner = "repair-resume-owner"
    token = store.claim_agent_run(run_id, owner, lease_seconds=5)
    store.add_message(project.id, conversation.id, "user", "repair resume",
                      agent_run_id=run_id)
    base = {
        "run_id": run_id, "route": "generate_dj_set", "user_text": "repair resume",
        "history": history, "command": command, "candidate_tracks": [track],
        "candidate_track_ids": [track.id], "playlist": playlist,
        "tool_events": [], "repair_attempts": 1,
    }
    manager.save(run_id, "command_parsed", base, owner=owner, token=token)
    manager.save(run_id, "candidates_retrieved", base, owner=owner, token=token)
    manager.save(run_id, "set_planned", base, owner=owner, token=token)
    manager.save(run_id, "set_repaired", base, version=1, owner=owner, token=token)
    store.release_agent_run(run_id, owner, token)

    def forbidden(*args, **kwargs):
        raise AssertionError("resume must not rerun retrieval or planning")

    registry.retrieve_set_candidates = forbidden
    registry.plan_set = forbidden
    model = _JsonModel(responses=[AIMessage(content="已恢复并保存。")])
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model

    async def collect():
        return [event async for event in agent.stream_chat(
            project.id, conversation.id, "repair resume", run_id
        )]

    events = asyncio.run(collect())
    checkpoints = manager.list(run_id)
    assert events[-1]["playlist"]["id"] == derive_agent_playlist_id(project.id, run_id)
    assert [(item["step"], item["version"]) for item in checkpoints].count(
        ("set_repaired", 1)
    ) == 1
    assert ("set_validated", 1) in {
        (item["step"], item["version"]) for item in checkpoints
    }
    assert len(store.list_playlists(project.id)) == 1


def test_constraint_conflict_sse_has_no_playlist_and_two_versioned_repairs(library):
    store, project, _, _, registry = library
    track = add_track(store, project, "too-short", duration_sec=200)
    conversation = store.ensure_default_conversation(project.id)
    run_id = "run-invalid-set"
    model = _JsonModel(responses=[
        AIMessage(content=json.dumps({
            "request": "invalid", "duration_min": 10, "bpm_min": 110,
            "bpm_max": 140, "energy_curve": "build", "track_ids": [track.id],
        })),
        AIMessage(content="需要放宽目标时长。"),
    ])
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model

    async def collect():
        return [event async for event in agent.stream_chat(
            project.id, conversation.id, "生成一个十分钟 set", run_id
        )]

    events = asyncio.run(collect())
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["playlist"] is None
    assert complete["error_code"] == "constraint_conflict"
    error = next(event for event in events if event["type"] == "error")
    assert error["error_code"] == "constraint_conflict"
    assert "放宽" in error["detail"]
    assert store.list_playlists(project.id) == []
    versions = [(item["step"], item["version"])
                for item in AgentCheckpointStore(store).list(run_id)]
    assert [(step, version) for step, version in versions if step == "set_repaired"] == [
        ("set_repaired", 1), ("set_repaired", 2),
    ]
    assert [(step, version) for step, version in versions if step == "set_validated"] == [
        ("set_validated", 0), ("set_validated", 1), ("set_validated", 2),
    ]


def test_default_run_ids_are_random_and_explicit_run_id_resumes(library):
    store, project, _, _, registry = library
    conversation = store.ensure_default_conversation(project.id)
    model = _JsonModel(responses=[AIMessage(content="one"), AIMessage(content="two")])
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model

    async def collect(run_id=None):
        return [event async for event in agent.stream_chat(
            project.id, conversation.id, "解释一下 Camelot wheel", run_id
        )]

    first = asyncio.run(collect())[-1]
    second = asyncio.run(collect())[-1]
    resumed = asyncio.run(collect(first["run_id"]))[-1]
    assert first["run_id"] != second["run_id"]
    assert first["resumed"] is False and second["resumed"] is False
    assert resumed["run_id"] == first["run_id"] and resumed["resumed"] is True
    assert resumed["message"]["id"] == first["message"]["id"]


def test_same_run_concurrency_persists_one_deterministic_playlist(library):
    store, project, _, _, registry = library
    track = add_track(store, project, "concurrent", duration_sec=600)
    conversation = store.ensure_default_conversation(project.id)
    run_id = "run-concurrent-playlist"
    model = _JsonModel(responses=[
        AIMessage(content=json.dumps({
            "request": "concurrent", "duration_min": 10, "bpm_min": 110,
            "bpm_max": 140, "energy_curve": "build", "track_ids": [track.id],
        })),
        AIMessage(content="并发任务已完成。"),
    ])
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model

    async def collect_one():
        return [event async for event in agent.stream_chat(
            project.id, conversation.id, "生成并发 set", run_id
        )]

    async def collect_both():
        return await asyncio.gather(collect_one(), collect_one())

    results = asyncio.run(collect_both())
    completes = [events[-1] for events in results]
    expected_id = derive_agent_playlist_id(project.id, run_id)
    assert {item["playlist"]["id"] for item in completes} == {expected_id}
    assert len(store.list_playlists(project.id)) == 1
    assert store.list_playlists(project.id)[0].id == expected_id
    assert len(store.list_messages(project.id, conversation.id)) == 2
    assert sorted(item["resumed"] for item in completes) == [False, True]


def test_run_scope_is_validated_before_message_write(library):
    store, project, other, _, registry = library
    first_conversation = store.ensure_default_conversation(project.id)
    other_conversation = store.ensure_default_conversation(other.id)
    run_id = "run-scope-guard"
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )

    async def collect(project_id, conversation_id):
        return [event async for event in agent.stream_chat(
            project_id, conversation_id, "帮我写一段 Python 股票代码", run_id
        )]

    asyncio.run(collect(project.id, first_conversation.id))
    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(collect(other.id, other_conversation.id))
    assert store.list_messages(other.id, other_conversation.id) == []


def test_api_sync_sse_and_playlist_conflict_contracts(tmp_path):
    settings = Settings(
        _env_file=None, DROPIT_DATA_DIR=tmp_path / "api",
        DEEPSEEK_API_KEY="test-only", DROPIT_INTENT_FALLBACK_ENABLED=False,
    )
    app = create_app(
        settings, embedder=FakeEmbedder(), descriptor=FakeDescriptor(), analyzer=FakeAnalyzer()
    )
    store = app.state.store
    project = store.create_project("P0 API")
    sync_conversation = store.ensure_default_conversation(project.id)
    stream_conversation = store.create_conversation(project.id, "stream")
    track = add_track(store, project, "api-short", duration_sec=200)

    command_response = AIMessage(content=json.dumps({
        "request": "invalid", "duration_min": 10, "bpm_min": 110,
        "bpm_max": 140, "energy_curve": "build", "track_ids": [track.id],
    }))
    with TestClient(app) as client:
        app.state.copilot._chat_model = lambda: _JsonModel(responses=[
            command_response, AIMessage(content="同步冲突说明。")
        ])
        sync = client.post(
            f"/api/projects/{project.id}/conversations/{sync_conversation.id}/chat",
            json={"message": "生成一个十分钟 set"},
        )
        assert sync.status_code == 200
        body = sync.json()
        assert body["playlist"] is None
        assert body["error_code"] == "constraint_conflict"
        assert body["run_id"] and body["resumed"] is False

        app.state.copilot._chat_model = lambda: _JsonModel(responses=[
            command_response, AIMessage(content="流式冲突说明。")
        ])
        streamed = client.post(
            f"/api/projects/{project.id}/conversations/{stream_conversation.id}/chat/stream",
            json={"message": "生成一个十分钟 set"},
        )
        frames = [json.loads(line[6:]) for line in streamed.text.splitlines()
                  if line.startswith("data: ")]
        assert next(frame for frame in frames if frame["type"] == "error")[
            "error_code"
        ] == "constraint_conflict"
        assert frames[-1]["type"] == "complete"
        assert frames[-1]["playlist"] is None
        assert frames[-1]["error_code"] == "constraint_conflict"

        conflict = client.post(
            f"/api/projects/{project.id}/playlists",
            json={"title": "invalid", "duration_min": 10, "bpm_min": 110,
                  "bpm_max": 140, "energy": "build"},
        )
        assert conflict.status_code == 409
        detail = conflict.json()["detail"]
        assert detail["code"] == "constraint_conflict"
        assert detail["repair_attempts"] == 2
        assert detail["issues"] and detail["suggestions"]
        assert store.list_playlists(project.id) == []
