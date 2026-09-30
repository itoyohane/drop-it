export const TOOL_LABELS = Object.freeze({
  search_library: "检索曲库",
  find_similar_tracks: "查找相似曲目",
  generate_dj_set: "编排 DJ Set",
});

export function toolLabel(name) {
  return TOOL_LABELS[name] || name || "Agent 工具";
}

function decodeFrame(frame) {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n");
  if (!data) return null;
  try {
    return JSON.parse(data);
  } catch (error) {
    throw new Error(`服务端返回了无效的流式事件：${error.message}`);
  }
}

export function createSseParser(onEvent) {
  let buffer = "";

  const drain = (flush = false) => {
    // Preserve a trailing CR until the next chunk so a split CRLF is not mistaken
    // for the blank line that terminates an SSE frame.
    const pendingCarriageReturn = !flush && buffer.endsWith("\r");
    const input = pendingCarriageReturn ? buffer.slice(0, -1) : buffer;
    const normalized = input.replace(/\r\n?/g, "\n");
    const frames = normalized.split("\n\n");
    const tail = frames.pop() || "";
    if (flush && tail.trim()) frames.push(tail);
    buffer = flush ? "" : `${tail}${pendingCarriageReturn ? "\r" : ""}`;
    for (const frame of frames) {
      const event = decodeFrame(frame);
      if (event) onEvent(event);
    }
  };

  return {
    push(chunk) {
      buffer += chunk;
      drain(false);
    },
    finish() {
      drain(true);
    },
  };
}

function updateAssistant(messages, assistantId, update) {
  return messages.map((message) => message.id === assistantId ? update(message) : message);
}

function reconcileSavedUser(messages, optimisticId, savedMessage) {
  let inserted = false;
  const result = [];
  for (const message of messages) {
    if (message.id === optimisticId || message.id === savedMessage.id) {
      if (!inserted) {
        result.push(savedMessage);
        inserted = true;
      }
    } else {
      result.push(message);
    }
  }
  if (!inserted) result.push(savedMessage);
  return result;
}

function mergeToolEvents(current = [], incoming = []) {
  const merged = [...current];
  for (const event of incoming) {
    const index = merged.findIndex((item) => item.name === event.name);
    if (index === -1) merged.push(event);
    else merged[index] = event;
  }
  return merged;
}

export function applyChatEvent(messages, event, { optimisticId, assistantId }) {
  if (!event || typeof event.type !== "string") return messages;
  if (event.type === "user_saved" && event.message) {
    return reconcileSavedUser(messages, optimisticId, event.message);
  }
  if (event.type === "token" && typeof event.content === "string") {
    return updateAssistant(messages, assistantId, (message) => ({
      ...message,
      content: `${message.content || ""}${event.content}`,
    }));
  }
  if (event.type === "tool") {
    return updateAssistant(messages, assistantId, (message) => ({
      ...message,
      tool_events: mergeToolEvents(message.tool_events, [event]),
    }));
  }
  if (event.type === "error" && event.detail) {
    return updateAssistant(messages, assistantId, (message) => ({
      ...message,
      stream_errors: [...new Set([...(message.stream_errors || []), event.detail])],
    }));
  }
  if (event.type === "complete" && event.message) {
    return updateAssistant(messages, assistantId, (message) => ({
      ...event.message,
      tool_events: mergeToolEvents(message.tool_events, event.message.tool_events),
      stream_errors: message.stream_errors || [],
    }));
  }
  return messages;
}

export function reconcileChatFailure(messages, { assistantId, detail }) {
  return updateAssistant(messages, assistantId, (message) => ({
    ...message,
    content: message.content || detail,
    stream_errors: [...new Set([...(message.stream_errors || []), detail])],
  }));
}
