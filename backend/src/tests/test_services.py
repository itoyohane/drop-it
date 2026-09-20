import asyncio
import io
import sqlite3
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from backend.agent.agent import DropItAgent, MODEL_NOT_CONFIGURED_ERROR
from backend.agent.intent import IntentRecognizer, OllamaIntentFallback, OVERSTEP_RESPONSE
from backend.agent.memory import ShortTermMemory
from backend.agent.prompts import SYSTEM_PROMPT
from backend.config import Settings
from backend.workers.analyze_track import JobRunner
from backend.main import create_app
from backend.models import Brief, MusicFilters, ToolResult, TrackUpdate
from backend.music.text_models import unit_vector
from backend.repositories import GLOBAL_PROJECT_ID, DropItStore
from backend.agent.tools import (
    DropItToolRegistry, generate_playlist, is_camelot_compatible, render_export,
)
from backend.tests.conftest import FakeAnalyzer, FakeDescriptor, FakeEmbedder, add_track


def invoke(registry, project_id, name, **arguments):
    tool = next(tool for tool in registry.tools_for(project_id) if tool.name == name)
    return ToolResult.model_validate_json(tool.invoke(arguments))


def test_music_retrieval_uses_audio_and_enforces_scope(library):
    store, project, other, embedder, rag = library
    add_track(store, project, "dark-techno", [1, 0, 0])
    add_track(store, project, "ambient", [0, 1, 0])
    add_track(store, other, "private-perfect-match", [1, 0, 0])
    matches = rag.search(project.id, "dark techno")
    assert [m.track.id for m in matches] == ["dark-techno", "ambient"]
    assert embedder.text_calls == 1
    assert all(m.track.id != "private-perfect-match" for m in matches)


def test_metadata_lookup_does_not_need_models(library):
    store, project, _, embedder, rag = library
    add_track(store, project, "Signal", bpm=124)
    add_track(store, project, "Slow", bpm=90)
    embedder.fail = True
    matches = rag.search(project.id, filters=MusicFilters(title="sig", bpm_min=120))
    assert [m.track.id for m in matches] == ["Signal"]
    assert embedder.text_calls == 0
    assert matches[0].score is None


def test_similarity_excludes_reference_and_applies_constraints(library):
    store, project, other, _, rag = library
    add_track(store, project, "reference", [1, 0, 0])
    add_track(store, project, "near", [.95, .1, 0], bpm=126)
    add_track(store, project, "too-fast", [1, 0, 0], bpm=150)
    add_track(store, other, "foreign", [1, 0, 0])
    matches = rag.similar(project.id, "reference", MusicFilters(bpm_max=130))
    assert [m.track.id for m in matches] == ["near"]
    assert matches[0].description_similarity > .99
    with pytest.raises(ValueError, match="当前曲库"):
        rag.similar(project.id, "foreign")


def test_removing_membership_hides_existing_vector(library):
    store, project, other, _, rag = library
    track = add_track(store, project, "shared", [1, 0, 0])
    store.upsert_track(track, track.id, other.id)
    store.remove_track_from_project(project.id, track.id)
    assert rag.search(project.id, "techno") == []
    assert [m.track.id for m in rag.search(other.id, "techno")] == [track.id]


def test_metadata_edit_invalidates_and_rebuilds_description_embedding(library):
    store, project, _, embedder, rag = library
    track = add_track(store, project, "original", [1, 0, 0])
    store.update_track(track.id, TrackUpdate(title="Corrected", artist="New artist", bpm=122,
                                            key="A minor", camelot_key="8A", energy=.4))
    with pytest.raises(ValueError, match="索引"):
        rag.search(project.id, "techno")
    descriptor = FakeDescriptor()
    runner = JobRunner(store, embedder, descriptor, FakeAnalyzer())
    try:
        assert runner._analyze(track.id)
    finally:
        runner.close()
    match = rag.search(project.id, "techno")[0]
    assert match.track.title == "Corrected"
    assert match.track.bpm == 122
    assert descriptor.calls == 1 and embedder.text_calls == 2
    assert "path" not in match.context()


def test_missing_or_wrong_model_never_falls_back_to_fake_matches(library):
    store, project, _, _, rag = library
    add_track(store, project, "old-model", [1, 0, 0], model="old-text-model")
    with pytest.raises(ValueError, match="索引"):
        rag.search(project.id, "techno")
    with pytest.raises(ValueError, match="索引"):
        rag.similar(project.id, "old-model")


@pytest.mark.parametrize("values", [[0, 0], [float("nan"), 1], [[1, 2]]])
def test_invalid_vectors_rejected(values):
    with pytest.raises(ValueError):
        unit_vector(values)


def test_dimension_mismatch_is_explicit(library):
    store, project, _, _, rag = library
    add_track(store, project, "bad-dimensions", [1, 0])
    with pytest.raises(ValueError, match="维度"):
        rag.search(project.id, "techno")


def test_tools_have_exactly_requested_schema_and_server_scope(library):
    store, project, _, _, rag = library
    registry = rag
    assert registry.names == ["search_library", "find_similar_tracks", "generate_dj_set"]
    for tool in registry.tools_for(project.id):
        assert "project_id" not in tool.args
    with pytest.raises(ValidationError):
        invoke(registry, project.id, "search_library", limit=-1)


def test_tools_ground_responses_and_report_missing_index(library):
    store, project, _, _, rag = library
    add_track(store, project, "Signal")
    registry = rag
    result = invoke(registry, project.id, "search_library", filters={"title": "Signal"})
    assert result.ok and result.data["tracks"][0]["track_id"] == "Signal"
    result = invoke(registry, project.id, "find_similar_tracks", track_id="Signal")
    assert not result.ok
    result = invoke(registry, project.id, "search_library", query="techno")
    assert not result.ok


def test_set_tool_works_with_features_and_requires_index_for_style(library):
    store, project, other, _, rag = library
    for index, energy in enumerate((.3, .5, .8)):
        add_track(store, project, f"set-{index}", energy=energy)
    add_track(store, other, "foreign", energy=.4)
    registry = rag
    result = invoke(registry, project.id, "generate_dj_set", request="逐步升能量",
                    duration_min=10, bpm_min=120, bpm_max=130)
    assert result.ok
    playlist = store.get_playlist(result.data["playlist_id"])
    assert len(playlist.tracks) == 3
    assert [row.track.energy for row in playlist.tracks] == [.3, .5, .8]
    assert "foreign" not in [row.track.id for row in playlist.tracks]
    assert render_export(playlist, "m3u").startswith("#EXTM3U")
    before = len(store.list_playlists(project.id))
    result = invoke(registry, project.id, "generate_dj_set", request="techno",
                    style_query="techno", duration_min=10)
    assert not result.ok and len(store.list_playlists(project.id)) == before


def test_global_catalog_can_retrieve_but_not_save_set(library):
    store, project, _, _, rag = library
    add_track(store, project, "global-song", [1, 0, 0])
    registry = rag
    assert rag.search(GLOBAL_PROJECT_ID, "techno")[0].track.id == "global-song"
    assert not invoke(registry, GLOBAL_PROJECT_ID, "generate_dj_set", request="set").ok


def test_agent_selected_tracks_constrain_the_set(library):
    store, project, other, _, rag = library
    for name in ("chosen", "unselected"):
        add_track(store, project, name)
    add_track(store, other, "foreign")
    registry = rag
    result = invoke(registry, project.id, "generate_dj_set", request="selected songs", track_ids=["chosen"])
    assert result.ok and [t["track_id"] for t in result.data["tracks"]] == ["chosen"]
    result = invoke(registry, project.id, "generate_dj_set", request="invalid", track_ids=["foreign"])
    assert not result.ok


def test_planner_preserves_camelot_transitions(library):
    store, project, _, _, _ = library
    tracks = [add_track(store, project, f"key-{index}", camelot_key=key, energy=.2 + index * .2)
              for index, key in enumerate(("1A", "12A", "11A", "10A"))]
    playlist = generate_playlist(project.id, Brief(duration_min=10), tracks)
    assert all(is_camelot_compatible(a.track.camelot_key, b.track.camelot_key)
               for a, b in zip(playlist.tracks, playlist.tracks[1:]))


def test_job_retains_features_on_embedding_failure_and_retries_only_text_index(library):
    store, project, _, embedder, _ = library
    track = add_track(store, project, "pending", analysis_status="pending", analyzer="")
    analyzer = FakeAnalyzer()
    descriptor = FakeDescriptor()
    runner = JobRunner(store, embedder, descriptor, analyzer)
    try:
        embedder.fail = True
        job = store.create_job(project.id, "analyze", {"track_ids": [track.id]}, 1)
        runner._run(job.id)
        assert store.get_job(job.id).status == "failed"
        failed = store.get_track(track.id)
        assert failed.analysis_status == "analyzed"
        assert failed.embedding_status == "failed"
        assert failed.analysis_details["beat_positions"] == [0, .5]
        embedder.fail = False
        retry = store.create_job(project.id, "analyze", {"track_ids": [track.id]}, 1)
        runner._run(retry.id)
        assert store.get_job(retry.id).status == "completed"
        assert analyzer.calls == 1
        assert embedder.text_calls == 2
        assert descriptor.calls == 1
        assert runner.ready(store.get_track(track.id))
        cached = store.create_job(project.id, "analyze", {"track_ids": [track.id]}, 1)
        runner._run(cached.id)
        assert embedder.text_calls == 2
    finally:
        runner.close()


def test_failed_reanalysis_hides_old_vector(library):
    store, project, _, embedder, rag = library
    track = add_track(store, project, "broken", [1, 0, 0])

    class BrokenAnalyzer:
        def analyze(self, track):
            raise ValueError("bad audio")

    runner = JobRunner(store, embedder, FakeDescriptor(), BrokenAnalyzer())
    try:
        job = store.create_job(project.id, "analyze", {"track_ids": [track.id], "force": True}, 1)
        runner._run(job.id)
        assert store.get_track(track.id).analysis_status == "failed"
        assert store.music_vectors(project.id, embedder.model_key) == {}
    finally:
        runner.close()


def test_vectors_persist_across_restart(tmp_path):
    path = str(tmp_path / "persistent.db")
    store = DropItStore(path)
    project = store.create_project("Persistent")
    add_track(store, project, "saved", [3, 4, 0])
    store.close()
    reopened = DropItStore(path)
    try:
        vector = reopened.music_vectors(project.id, "test-text@1")["saved"]
        assert np.allclose(vector, [.6, .8, 0])
        assert reopened.connection.execute("SELECT COUNT(*) FROM music_embeddings").fetchone()[0] == 0
        assert reopened.vector_store.path.endswith("chroma")
    finally:
        reopened.close()


def test_legacy_database_migration_preserves_songs(tmp_path):
    path = str(tmp_path / "legacy.db")
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE tracks (id TEXT PRIMARY KEY, title TEXT, artist TEXT, path TEXT UNIQUE,
                  duration_sec INTEGER, bpm REAL, musical_key TEXT, energy REAL, mood TEXT, role TEXT)""")
    db.execute("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
               ("old", "Old Song", "Mira", "/music/old.wav", 240, 124, "8A", .6, "[]", "builder"))
    db.commit()
    db.close()
    store = DropItStore(path)
    try:
        track = store.all_tracks("legacy-project")[0]
        assert track.title == "Old Song"
        assert track.embedding_status == "pending"
        assert store.music_vectors("legacy-project", "test-text@1") == {}
    finally:
        store.close()


def test_api_upload_analysis_and_metadata_update(tmp_path):
    settings = Settings(_env_file=None, DROPIT_DATA_DIR=tmp_path, DEEPSEEK_API_KEY="")
    embedder = FakeEmbedder()
    app = create_app(settings, embedder=embedder, descriptor=FakeDescriptor(), analyzer=FakeAnalyzer())
    audio = io.BytesIO()
    with wave.open(audio, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(np.ones(8000, dtype="<i2").tobytes())
    with TestClient(app) as client:
        project = client.post("/api/projects", json={"name": "Test"}).json()
        source = client.post(f"/api/projects/{project['id']}/sources/resolve",
                             json={"folder_key": "f" * 32, "name": "Folder"}).json()["source"]
        response = client.post(f"/api/projects/{project['id']}/library/import-files",
                               data={"source_id": source["id"]},
                               files={"files": ("Mira - Signal.wav", audio.getvalue(), "audio/wav")})
        assert response.status_code == 202
        job_id = response.json()["job"]["id"]
        for _ in range(100):
            job = client.get("/api/jobs/" + job_id).json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "completed"
        track = client.get(f"/api/projects/{project['id']}/library").json()["tracks"][0]
        assert track["title"] == "Signal" and track["embedding_status"] == "ready"
        patch = {"title": "Edited", "artist": "Mira", "bpm": 122, "key": "A minor",
                 "camelot_key": "8A", "energy": .5}
        assert client.patch(f"/api/projects/{project['id']}/library/{track['id']}", json=patch).status_code == 200
        reindex = client.get(f"/api/projects/{project['id']}/jobs").json()["jobs"][0]
        for _ in range(100):
            reindex = client.get("/api/jobs/" + reindex["id"]).json()
            if reindex["status"] in {"completed", "failed"}:
                break
            time.sleep(.01)
        assert reindex["status"] == "completed" and embedder.text_calls == 2
        assert app.state.registry.search(project["id"], "techno")[0].track.title == "Edited"


def test_remove_track_endpoint_unlinks_only_current_project(tmp_path):
    settings = Settings(_env_file=None, DROPIT_DATA_DIR=tmp_path, DEEPSEEK_API_KEY="")
    app = create_app(settings, embedder=FakeEmbedder(), descriptor=FakeDescriptor(), analyzer=FakeAnalyzer())
    with TestClient(app) as client:
        project = app.state.store.create_project("Delete from project")
        other = app.state.store.create_project("Keep shared track")
        track = add_track(app.state.store, project, "Shared song", [1, 0, 0])
        app.state.store.upsert_track(track, track.id, other.id)

        response = client.delete(f"/api/projects/{project.id}/library/{track.id}")
        assert response.status_code == 204
        assert client.get(f"/api/projects/{project.id}/library").json()["tracks"] == []
        assert client.get(f"/api/projects/{other.id}/library").json()["tracks"][0]["id"] == track.id
        assert client.get("/api/library").json()["tracks"][0]["id"] == track.id

        global_delete = client.delete(f"/api/projects/{GLOBAL_PROJECT_ID}/library/{track.id}")
        assert global_delete.status_code == 400
        assert "总曲库" in global_delete.json()["detail"]


def test_chat_unconfigured_does_not_persist_turn(tmp_path):
    app = create_app(Settings(_env_file=None, DROPIT_DATA_DIR=tmp_path, DEEPSEEK_API_KEY=""))
    with TestClient(app) as client:
        project = client.post("/api/projects", json={"name": "Chat"}).json()
        conversation = client.get(f"/api/projects/{project['id']}/conversations").json()["conversations"][0]
        url = f"/api/projects/{project['id']}/conversations/{conversation['id']}"
        response = client.post(url + "/chat", json={"message": "帮我找歌"})
        assert response.status_code == 503
        assert response.json()["detail"] == MODEL_NOT_CONFIGURED_ERROR
        assert client.get(url + "/messages").json()["messages"] == []
        assert client.get("/api/health").json()["tools"] == DropItToolRegistry.names
        assert client.get("/api/health").json()["agent_framework"] == "langgraph.state_graph"
        assert client.get("/api/chat/conversations").status_code == 200


def test_agent_executes_controlled_graph_without_free_tool_loop(library):
    store, project, _, _, rag = library
    add_track(store, project, "Signal", [1, 0, 0])
    registry = rag
    settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-only")
    agent = DropItAgent(store, registry, settings)

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            raise AssertionError("controlled graph must not bind business tools")

        def with_structured_output(self, schema, **kwargs):
            raise NotImplementedError

    model = Model(responses=[
        AIMessage(content='{"query":"","filters":{"title":"Signal"},"limit":20}'),
        AIMessage(content="曲库中找到 Signal。"),
    ])
    agent._chat_model = lambda: model
    conversation = store.ensure_default_conversation(project.id)

    async def collect():
        return [event async for event in agent.stream_chat(project.id, conversation.id, "有哪些歌？")]

    events = asyncio.run(collect())
    assert any(e["type"] == "tool" and e["name"] == "search_library" and e["status"] == "done" for e in events)
    assert events[-1]["type"] == "complete"
    assert events[-1]["message"]["tool_events"][0]["name"] == "search_library"
    assert events[-1]["message"]["content"] == "曲库中找到 Signal。"
    assert [e["name"] for e in events[-1]["message"]["tool_events"]] == ["search_library"]
    assert events[-1]["playlist"] is None


def test_agent_refuses_overstep_without_calling_chat_model_or_tools():
    store = DropItStore(":memory:")
    try:
        project = store.create_project("Overstep")
        registry = DropItToolRegistry(store, FakeEmbedder())
        settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-only")
        agent = DropItAgent(store, registry, settings, intent=IntentRecognizer())
        agent._chat_model = lambda: (_ for _ in ()).throw(AssertionError("model must not run"))
        conversation = store.ensure_default_conversation(project.id)

        async def collect():
            return [event async for event in agent.stream_chat(
                project.id, conversation.id, "帮我写一段 Python 股票分析代码"
            )]

        events = asyncio.run(collect())
        assert events[-1]["type"] == "complete"
        assert events[-1]["message"]["content"] == OVERSTEP_RESPONSE
        assert events[-1]["message"]["tool_events"] == []
        assert events[-1]["playlist"] is None
    finally:
        store.close()


def test_agent_routes_invalid_intent_to_tool_free_music_chat():
    store = DropItStore(":memory:")
    try:
        project = store.create_project("Music chat")
        registry = DropItToolRegistry(store, FakeEmbedder())
        settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-only")

        class InvalidFallback:
            def classify(self, text):
                return OllamaIntentFallback._parse("not valid JSON")

        agent = DropItAgent(
            store,
            registry,
            settings,
            intent=IntentRecognizer(fallback=InvalidFallback()),
        )

        class Model(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                raise AssertionError("music_chat must not bind business tools")

        model = Model(responses=[AIMessage(content="可以，我在。")])
        agent._chat_model = lambda: model
        conversation = store.ensure_default_conversation(project.id)

        async def collect():
            return [event async for event in agent.stream_chat(
                project.id, conversation.id, "随便聊聊"
            )]

        events = asyncio.run(collect())
        assert events[-1]["type"] == "complete"
        assert events[-1]["message"]["content"] == "可以，我在。"
        assert events[-1]["message"]["tool_events"] == []
        assert "You have exactly three tools" not in SYSTEM_PROMPT
        assert "DSML" in SYSTEM_PROMPT
    finally:
        store.close()


def test_agent_compacts_context_at_configured_threshold():
    store = DropItStore(":memory:")
    try:
        project = store.create_project("Compaction")
        registry = DropItToolRegistry(store, FakeEmbedder())
        settings = Settings(
            _env_file=None,
            DEEPSEEK_API_KEY="test-only",
            DROPIT_AGENT_CONTEXT_WINDOW_TOKENS=4096,
            DROPIT_AGENT_CONTEXT_COMPACTION_RATIO=.5,
            DROPIT_AGENT_CONTEXT_KEEP_MESSAGES=2,
            DROPIT_AGENT_CONTEXT_RESERVED_TOKENS=0,
        )
        memory = ShortTermMemory(max_messages=100)
        agent = DropItAgent(store, registry, settings, memory=memory,
                            intent=IntentRecognizer())

        class Model(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        model = Model(responses=[AIMessage(content="已确认的历史事实"),
                                 AIMessage(content="继续处理。")])
        agent._chat_model = lambda: model
        conversation = store.ensure_default_conversation(project.id)
        memory_key = f"{project.id}:{conversation.id}"
        memory.seed(memory_key, [
            {"role": "user", "content": "a" * 3000},
            {"role": "assistant", "content": "b" * 3000},
            {"role": "user", "content": "最近问题"},
            {"role": "assistant", "content": "最近回答"},
        ])

        async def collect():
            return [event async for event in agent.stream_chat(
                project.id, conversation.id, "继续"
            )]

        events = asyncio.run(collect())
        compacted = memory.messages(memory_key)
        assert any(event.get("label") == "正在压缩上下文" for event in events)
        assert compacted[0]["content"].startswith("[历史摘要")
        assert "已确认的历史事实" in compacted[0]["content"]
        assert events[-1]["message"]["content"] == "继续处理。"
    finally:
        store.close()


def test_invalid_tool_result_is_not_reported_success():
    assert not DropItAgent._parse_tool_result("garbled output").ok
