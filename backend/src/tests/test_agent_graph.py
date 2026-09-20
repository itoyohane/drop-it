import asyncio

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError
import pytest

from backend.agent.agent import DropItAgent
from backend.agent.commands import GenerateSetCommand, SearchCommand
from backend.agent.intent import OVERSTEP_RESPONSE, IntentRecognizer
from backend.agent.tools import DropItToolRegistry
from backend.config import Settings
from backend.tests.conftest import FakeEmbedder, add_track


class JsonSequenceModel(FakeMessagesListChatModel):
    """Test model that makes typed extraction explicit and forbids tool binding."""

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError

    def bind_tools(self, tools, **kwargs):
        raise AssertionError("the controlled graph must not bind business tools")


class DelayedStreamingModel:
    def __init__(self):
        self.first_chunk_started = asyncio.Event()
        self.release_second_chunk = asyncio.Event()

    async def astream(self, messages):
        self.first_chunk_started.set()
        yield AIMessage(content="第一段")
        await self.release_second_chunk.wait()
        yield AIMessage(content="第二段")

    async def ainvoke(self, messages):
        return AIMessage(content="unused")


def collect(agent, project_id, conversation_id, text):
    async def run():
        return [event async for event in agent.stream_chat(project_id, conversation_id, text)]

    return asyncio.run(run())


def configured_agent(store, project, model):
    agent = DropItAgent(
        store,
        DropItToolRegistry(store, FakeEmbedder()),
        Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model
    conversation = store.ensure_default_conversation(project.id)
    return agent, conversation


def test_graph_has_explicit_routes_and_terminal_edges(library):
    store, project, _, _, registry = library
    agent = DropItAgent(store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"))
    graph = agent.graph.get_graph()
    nodes = set(graph.nodes)
    assert {
        "search_library", "respond", "resolve_reference", "find_similar_tracks",
        "retrieve_candidates", "plan_set", "persist_set", "respond_chat", "reject_response",
    } <= nodes
    assert "route" not in nodes
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("search_library", "respond") in edges
    assert ("resolve_reference", "find_similar_tracks") in edges
    assert ("retrieve_candidates", "plan_set") in edges
    assert ("plan_set", "persist_set") in edges
    assert ("persist_set", "respond") in edges
    assert ("respond_chat", "__end__") in edges
    assert ("reject_response", "__end__") in edges


def test_search_route_uses_typed_command_and_preserves_sse_contract(library):
    store, project, _, _, _ = library
    add_track(store, project, "Signal", [1, 0, 0])
    model = JsonSequenceModel(responses=[
        AIMessage(content='{"query":"","filters":{"title":"Signal"},"limit":20}'),
        AIMessage(content="曲库中找到 Signal。"),
    ])
    agent, conversation = configured_agent(store, project, model)

    events = collect(agent, project.id, conversation.id, "曲库里找 Signal")
    assert [event["type"] for event in events][-1] == "complete"
    assert {event["type"] for event in events} >= {
        "user_saved", "status", "tool", "token", "complete"
    }
    assert events[-1]["message"]["content"] == "曲库中找到 Signal。"
    assert events[-1]["message"]["tool_events"][0]["name"] == "search_library"
    assert events[-1]["playlist"] is None


def test_command_schemas_forbid_model_selected_scope():
    with pytest.raises(ValidationError):
        SearchCommand.model_validate({"query": "house", "project_id": "other-project"})


def test_final_response_tokens_stream_before_graph_node_finishes(library):
    store, project, _, _, registry = library
    model = DelayedStreamingModel()
    agent = DropItAgent(
        store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
        intent=IntentRecognizer(),
    )
    agent._chat_model = lambda: model
    conversation = store.ensure_default_conversation(project.id)

    async def run():
        events = []
        first_token_seen = asyncio.Event()

        async def consume():
            async for event in agent.stream_chat(project.id, conversation.id, "解释一下 Camelot wheel"):
                events.append(event)
                if event["type"] == "token":
                    first_token_seen.set()

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(model.first_chunk_started.wait(), 1)
            await asyncio.wait_for(first_token_seen.wait(), .2)
            assert not task.done(), "the graph must still be waiting for the second model chunk"
        finally:
            model.release_second_chunk.set()
            await task
        return events

    events = asyncio.run(run())
    token_contents = [event["content"] for event in events if event["type"] == "token"]
    assert token_contents == ["第一段", "第二段"]
    assert events.index(next(event for event in events if event["type"] == "token")) < len(events) - 1
    assert events[-1]["type"] == "complete"


def test_similar_route_resolves_reference_before_ranking(library):
    store, project, _, _, _ = library
    add_track(store, project, "Signal", [1, 0, 0])
    add_track(store, project, "Similar", [.95, .1, 0])
    model = JsonSequenceModel(responses=[
        AIMessage(content='{"reference":"Signal","limit":3}'),
        AIMessage(content="可以接 Signal 的是 Similar。"),
    ])
    agent, conversation = configured_agent(store, project, model)

    events = collect(agent, project.id, conversation.id, "找几首和 Signal 相似的歌")
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["message"]["tool_events"][0]["name"] == "find_similar_tracks"
    assert complete["message"]["tool_events"][0]["status"] == "done"


def test_similar_route_reports_missing_and_ambiguous_references(library):
    store, project, _, _, _ = library
    add_track(store, project, "one", [1, 0, 0], title="Signal")
    add_track(store, project, "two", [.9, .1, 0], title="Signal")

    ambiguous_model = JsonSequenceModel(responses=[
        AIMessage(content='{"reference":"Signal"}'),
        AIMessage(content="请指定具体版本。"),
    ])
    ambiguous_agent, conversation = configured_agent(store, project, ambiguous_model)
    ambiguous = collect(ambiguous_agent, project.id, conversation.id, "找和 Signal 相似的歌")
    assert any(event["type"] == "error" and "不唯一" in event["detail"] for event in ambiguous)
    assert ambiguous[-1]["message"]["tool_events"][0]["status"] == "failed"

    missing_model = JsonSequenceModel(responses=[
        AIMessage(content='{"reference":"Missing"}'),
        AIMessage(content="请先确认曲库中的参考歌曲。"),
    ])
    missing_agent, missing_conversation = configured_agent(store, project, missing_model)
    missing = collect(missing_agent, project.id, missing_conversation.id, "找 Missing 的相似歌曲")
    assert any(event["type"] == "error" and "没有找到" in event["detail"] for event in missing)
    assert missing[-1]["message"]["tool_events"][0]["status"] == "failed"


def test_generate_set_retrieves_plans_and_persists_once(library):
    store, project, _, _, _ = library
    add_track(store, project, "Signal", [1, 0, 0])
    model = JsonSequenceModel(responses=[
        AIMessage(content=(
            '{"request":"warm-up","duration_min":10,"bpm_min":110,"bpm_max":140,'
            '"energy_curve":"build","track_ids":["Signal"]}'
        )),
        AIMessage(content="已保存 warm-up Set。"),
    ])
    agent, conversation = configured_agent(store, project, model)

    events = collect(agent, project.id, conversation.id, "生成一个 warm-up set")
    assert events[-1]["playlist"] is not None
    assert len(store.list_playlists(project.id)) == 1
    assert events[-1]["message"]["tool_events"] == [{
        "name": "generate_dj_set",
        "status": "done",
        "summary": "已生成 1 首、约 4 分钟的 Set。",
    }]


def test_project_scope_is_server_selected_for_set_candidates(library):
    store, project, other, _, registry = library
    add_track(store, project, "inside", [1, 0, 0])
    add_track(store, other, "outside", [1, 0, 0])
    command = GenerateSetCommand(request="scope", track_ids=["outside"])
    try:
        registry.retrieve_set_candidates(project.id, command)
    except ValueError as exc:
        assert "当前项目之外" in str(exc)
    else:
        raise AssertionError("candidate retrieval escaped the server-selected project scope")


def test_music_chat_is_tool_free_and_reject_uses_no_main_model(library):
    store, project, _, _, registry = library
    settings = Settings(_env_file=None, DEEPSEEK_API_KEY="test-only")
    music_agent = DropItAgent(store, registry, settings, intent=IntentRecognizer())
    music_model = JsonSequenceModel(responses=[AIMessage(content="Camelot wheel 是调性混音工具。")])
    music_agent._chat_model = lambda: music_model
    conversation = store.ensure_default_conversation(project.id)
    music_events = collect(music_agent, project.id, conversation.id, "解释一下 Camelot wheel")
    assert music_events[-1]["message"]["content"] == "Camelot wheel 是调性混音工具。"
    assert music_events[-1]["message"]["tool_events"] == []

    reject_agent = DropItAgent(store, registry, settings, intent=IntentRecognizer())
    reject_agent._chat_model = lambda: (_ for _ in ()).throw(
        AssertionError("reject must not call the main model")
    )
    reject_conversation = store.create_conversation(project.id, "Reject")
    reject_events = collect(
        reject_agent, project.id, reject_conversation.id, "帮我写一段 Python 股票分析代码"
    )
    assert reject_events[-1]["message"]["content"] == OVERSTEP_RESPONSE
    assert reject_events[-1]["message"]["tool_events"] == []
