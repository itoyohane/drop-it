import assert from "node:assert/strict";

import {
  TOOL_LABELS,
  applyChatEvent,
  createSseParser,
  reconcileChatFailure,
} from "../src/chatStream.mjs";

let checks = 0;
function check(name, callback) {
  callback();
  checks += 1;
  console.log(`ok ${checks} - ${name}`);
}

check("SSE parser handles split CRLF frames, multiple data lines, and an unterminated tail", () => {
  const events = [];
  const parser = createSseParser((event) => events.push(event));
  parser.push('data: {"type":"status",\r');
  parser.push('\n' + 'data: "label":"正在思考"}\r\n\r\ndata: {"type":"token","content":"Hi"}');
  parser.finish();

  assert.deepEqual(events, [
    { type: "status", label: "正在思考" },
    { type: "token", content: "Hi" },
  ]);
});

check("stream events reconcile optimistic records and preserve failures on completion", () => {
  const ids = { optimisticId: "temp-1", assistantId: "stream-1" };
  let messages = [
    { id: "temp-1", role: "user", content: "hello", tool_events: [] },
    { id: "stream-1", role: "assistant", content: "", tool_events: [] },
  ];

  messages = applyChatEvent(messages, {
    type: "user_saved",
    message: { id: "user-1", role: "user", content: "hello", tool_events: [] },
  }, ids);
  messages = applyChatEvent(messages, { type: "token", content: "partial " }, ids);
  messages = applyChatEvent(messages, {
    type: "tool", name: "find_similar_tracks", status: "failed", summary: "not found",
  }, ids);
  messages = applyChatEvent(messages, { type: "error", detail: "not found" }, ids);
  messages = applyChatEvent(messages, {
    type: "complete",
    message: { id: "assistant-1", role: "assistant", content: "not found", tool_events: [] },
  }, ids);

  assert.equal(messages.length, 2);
  assert.equal(messages[0].id, "user-1");
  assert.equal(messages[1].id, "assistant-1");
  assert.deepEqual(messages[1].tool_events, [
    { type: "tool", name: "find_similar_tracks", status: "failed", summary: "not found" },
  ]);
  assert.deepEqual(messages[1].stream_errors, ["not found"]);
});

check("request failure keeps the optimistic user and leaves a visible assistant result", () => {
  const messages = reconcileChatFailure([
    { id: "temp-2", role: "user", content: "hello", tool_events: [] },
    { id: "stream-2", role: "assistant", content: "", tool_events: [] },
  ], { assistantId: "stream-2", detail: "模型未配置" });

  assert.equal(messages[0].content, "hello");
  assert.equal(messages[1].content, "模型未配置");
  assert.deepEqual(messages[1].stream_errors, ["模型未配置"]);
});

check("tool labels cover exactly the controlled graph tools", () => {
  assert.deepEqual(Object.keys(TOOL_LABELS).sort(), [
    "find_similar_tracks",
    "generate_dj_set",
    "search_library",
  ]);
});

console.log(`1..${checks}`);
