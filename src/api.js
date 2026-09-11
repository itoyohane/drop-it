// Use one same-origin API path in every environment. Vite proxies this path in development,
// while FastAPI serves it directly in production; the browser never has to reach a loopback host.
const API = "/api";

async function request(path, options = {}) {
  // Multipart bodies set their own boundary; forcing JSON here would break audio imports.
  const headers = options.body instanceof FormData
    ? { ...options.headers }
    : { "Content-Type": "application/json", ...options.headers };
  const response = await fetch(`${API}${path}`, { headers, ...options });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const requestId = response.headers.get("x-request-id");
    const detail = payload.detail || `请求失败（${response.status}）`;
    const error = new Error(requestId ? `${detail}（请求编号：${requestId}）` : detail);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

async function streamChat(projectId, conversationId, message, onEvent, signal) {
  const response = await fetch(
    `${API}/projects/${projectId}/conversations/${conversationId}/chat/stream`,
    { method: "POST", signal, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }) },
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `对话请求失败（${response.status}）`);
  }
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    // Network chunks do not align with SSE frames, so retain the incomplete tail for the next read.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
    if (done) break;
  }
}

async function streamGlobalChat(conversationId, message, onEvent, signal) {
  const response = await fetch(`${API}/chat/conversations/${conversationId}/chat/stream`, {
    method: "POST", signal, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `普通对话请求失败（${response.status}）`);
  }
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
    if (done) break;
  }
}

export async function folderIdentity(files) {
  const entries = Array.from(files).map((file) =>
    `${file.webkitRelativePath || file.name}|${file.size}|${file.lastModified}`
  ).sort();
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(entries.join("\n")));
  const folderKey = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  const firstPath = files[0]?.webkitRelativePath || files[0]?.name || "本地文件夹";
  return { folderKey, folderName: firstPath.split("/")[0] || "本地文件夹" };
}

export const api = {
  health: () => request("/health"),
  projects: () => request("/projects"),
  createProject: (body) => request("/projects", { method: "POST", body: JSON.stringify(body) }),
  updateProject: (id, body) => request(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteProject: (id) => request(`/projects/${id}`, { method: "DELETE" }),
  conversations: (id) => request(`/projects/${id}/conversations`),
  createConversation: (id, title = "新对话") => request(`/projects/${id}/conversations`, {
    method: "POST", body: JSON.stringify({ title }),
  }),
  deleteConversation: (projectId, id) => request(`/projects/${projectId}/conversations/${id}`, { method: "DELETE" }),
  globalConversations: () => request("/chat/conversations"),
  createGlobalConversation: (title = "新对话") => request("/chat/conversations", {
    method: "POST", body: JSON.stringify({ title }),
  }),
  deleteGlobalConversation: (id) => request(`/chat/conversations/${id}`, { method: "DELETE" }),
  globalMessages: (id) => request(`/chat/conversations/${id}/messages`),
  globalLibrary: () => request("/library"),
  projectLibrary: (id) => request(`/projects/${id}/library`),
  projectSources: (id) => request(`/projects/${id}/sources`),
  resolveFolder: (projectId, folderKey, name) => request(`/projects/${projectId}/sources/resolve`, {
    method: "POST", body: JSON.stringify({ folder_key: folderKey, name }),
  }),
  messages: (projectId, conversationId) => request(`/projects/${projectId}/conversations/${conversationId}/messages`),
  playlists: (id) => request(`/projects/${id}/playlists`),
  jobs: (id) => request(`/projects/${id}/jobs`),
  importFiles: (projectId, sourceId, files) => {
    const body = new FormData();
    body.append("source_id", sourceId);
    Array.from(files).forEach((file) => body.append("files", file, file.webkitRelativePath || file.name));
    return request(`/projects/${projectId}/library/import-files`, { method: "POST", body });
  },
  analyzeLibrary: (projectId, trackIds = []) => request(`/projects/${projectId}/library/analyze`, {
    method: "POST", body: JSON.stringify({ track_ids: trackIds }),
  }),
  updateTrack: (projectId, trackId, body) => request(`/projects/${projectId}/library/${trackId}`, {
    method: "PATCH", body: JSON.stringify(body),
  }),
  removeTrack: (projectId, trackId) => request(`/projects/${projectId}/library/${trackId}`, { method: "DELETE" }),
  streamChat,
  streamGlobalChat,
  createPlaylist: (projectId, brief) => request(`/projects/${projectId}/playlists`, {
    method: "POST", body: JSON.stringify(brief),
  }),
  reorder: (id, trackIds) => request(`/playlists/${id}/order`, {
    method: "PUT", body: JSON.stringify({ track_ids: trackIds }),
  }),
  approve: (id) => request(`/playlists/${id}/approve`, { method: "POST" }),
  deletePlaylist: (id) => request(`/playlists/${id}`, { method: "DELETE" }),
  export: (id, format) => request(`/playlists/${id}/export`, {
    method: "POST", body: JSON.stringify({ format }),
  }),
};
