"""No API calls: run real production graph, checkpoints and repositories."""

import json

from langchain_core.messages import AIMessage
import pytest

from backend.agent.agent import DropItAgent
from backend.agent.graph import IntentRecognizer
from backend.config import Settings
from backend.evals.run_agent_eval import LiveAgentRunner, compute_metrics, evaluate_trace
from backend.tests.conftest import add_track
from backend.tests.test_agent_graph import JsonSequenceModel


def runner_for(library, responses):
    store, project, _, _, registry = library
    model = JsonSequenceModel(responses=[AIMessage(content=response, usage_metadata={
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }) for response in responses])
    agent = DropItAgent(store, registry, Settings(_env_file=None, DEEPSEEK_API_KEY="test-only"),
                        intent=IntentRecognizer())
    agent._chat_model = lambda: model
    tracks = store.all_tracks(project.id)
    return LiveAgentRunner(agent, store, project.id, tracks, tracks)


def case(text, command, tools, nodes):
    return {"id": "integration", "turns": [{"user": text}], "expected_command": command,
            "expected_tool_sequence": tools, "expected_graph_nodes": nodes,
            "tool_argument_assertions": [], "expected_completion": True,
            "expected_interception": False}


def test_live_search_records_real_nodes_arguments_checkpoints_and_usage(library):
    store, project, *_ = library
    add_track(store, project, "Signal", [1, 0, 0])
    runner = runner_for(library, ['{"query":"","filters":{"title":"Signal"},"limit":20}', "找到 Signal。"])
    task = case("曲库里找 Signal", "search_library", ["search_library"], ["search_library", "respond"])
    trace = runner.run(task)
    assert trace["error_code"] is None, trace
    assert [x["name"] for x in trace["graph_steps"]] == ["search_library", "respond"]
    assert trace["tool_calls"][0]["arguments"]["filters"]["title"] == "Signal"
    assert trace["tool_calls"][0]["status"] == "done"
    assert "completed" in trace["turn_traces"][0]["checkpoints"]
    assert trace["token_usage"] == {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    assert evaluate_trace(task, trace)["task_completed"] is True
    assert runner.agent.trace_callback is None


def test_live_similar_resolves_title_without_fabricated_search_call(library):
    store, project, *_ = library
    add_track(store, project, "Signal", [1, 0, 0])
    add_track(store, project, "Echo", [.9, .1, 0])
    runner = runner_for(library, ['{"reference":"Signal","limit":3}', "推荐 Echo。"])
    task = case("找和 Signal 相似的歌", "find_similar_tracks", ["find_similar_tracks"],
                ["resolve_reference", "find_similar_tracks", "respond"])
    trace = runner.run(task)
    assert trace["error_code"] is None, trace
    assert [x["name"] for x in trace["tool_calls"]] == ["find_similar_tracks"]
    assert trace["tool_calls"][0]["arguments"]["reference"] == "Signal"
    assert trace["tool_calls"][0]["arguments"]["track_id"] == "Signal"
    assert evaluate_trace(task, trace)["task_completed"] is True


@pytest.mark.parametrize("required,success", [(None, True), (["missing"], False)])
def test_set_validation_and_bounded_repair_are_observed(library, required, success):
    store, project, *_ = library
    for index, energy in enumerate((.2, .5, .8)):
        add_track(store, project, f"t{index}", [1, .1 * index, 0], duration_sec=200, energy=energy)
    command = {"request": "10 minute set", "duration_min": 10, "bpm_min": 118,
               "bpm_max": 132, "energy_curve": "build", "required_tracks": required}
    runner = runner_for(library, [json.dumps(command), "Set 处理完毕。"])
    task = case("生成一个 10 分钟 DJ Set", "generate_dj_set", ["generate_dj_set"],
                ["retrieve_candidates", "plan_set", "validate_set", "persist_set", "respond"])
    trace = runner.run(task)
    assert trace["constraints_passed"] is success, trace
    assert evaluate_trace(task, trace)["task_completed"] is success
    nodes = [x["name"] for x in trace["graph_steps"]]
    assert nodes.count("repair_set") == (0 if success else 2)
    assert trace["repair_attempts"] == (0 if success else 2)
    if not success:
        assert trace["error_code"] == "constraint_conflict"
        assert trace["failure_node"] == "validate_set"
        assert trace["output"]["playlist"] is None
        metrics = compute_metrics([{"trace": trace, "evaluation": evaluate_trace(task, trace)}])
        assert metrics["repair_success_rate"]["attempted"] == 1
        assert metrics["repair_success_rate"]["value_pct"] == 0
    else:
        assert trace["output"]["playlist"]["agent_run_id"] == trace["turn_traces"][0]["run_id"]


def test_multi_turn_uses_distinct_run_ids_and_keeps_earlier_set_constraints(library):
    store, project, *_ = library
    add_track(store, project, "Signal", [1, 0, 0])
    runner = runner_for(library, ['{"duration_min":10}', "无法生成。", '{"query":"","limit":20}', "找到 Signal。"])
    task = case("生成一个 10 分钟 DJ Set", "search_library", ["generate_dj_set", "search_library"], [])
    task["turns"].append({"user": "重新搜索曲库里的歌"})
    trace = runner.run(task)
    assert len(trace["turn_traces"]) == 2
    assert len({x["run_id"] for x in trace["turn_traces"]}) == 2
    assert trace["command"] == "search_library"
    assert trace["turn_traces"][1]["error_code"] is None
    assert trace["constraints_passed"] is False
    assert evaluate_trace(task, trace)["task_completed"] is False


def test_missing_reference_preserves_actual_failure_node(library):
    store, project, *_ = library
    add_track(store, project, "Signal", [1, 0, 0])
    runner = runner_for(library, ['{"reference":"does-not-exist"}', "未找到参考曲目。"])
    task = case("找和 Missing 相似的歌", "find_similar_tracks", ["find_similar_tracks"], [])
    trace = runner.run(task)
    assert trace["failure_node"] == "resolve_reference"
    assert trace["error_code"] is not None
    assert evaluate_trace(task, trace)["failure_node"] == "resolve_reference"


def test_successful_repair_is_measured_from_actual_graph(library, monkeypatch):
    import backend.agent.graph as graph

    store, project, *_ = library
    for index, energy in enumerate((.2, .5, .8)):
        add_track(store, project, f"t{index}", [1, .1 * index, 0], duration_sec=200, energy=energy)
    original_build = graph.build_set

    def broken_summary(*args, **kwargs):
        return original_build(*args, **kwargs).model_copy(update={"duration_sec": 1})

    monkeypatch.setattr(graph, "build_set", broken_summary)
    runner = runner_for(library, ['{"duration_min":10,"bpm_min":118,"bpm_max":132}', "生成成功。"])
    task = case("生成一个 10 分钟 DJ Set", "generate_dj_set", ["generate_dj_set"],
                ["retrieve_candidates", "plan_set", "validate_set", "persist_set", "respond"])
    trace = runner.run(task)
    assert trace["repair_attempts"] == 1, trace
    assert trace["repair_success"] is True
    assert evaluate_trace(task, trace)["task_completed"] is True
    metrics = compute_metrics([{"trace": trace, "evaluation": evaluate_trace(task, trace)}])
    assert metrics["repair_success_rate"]["value_pct"] == 100
