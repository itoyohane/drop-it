import { createSseParser } from "./chatStream.mjs";

// Use one same-origin API path in every environment. Vite proxies this path in development,
// while FastAPI serves it directly in production; the browser never has to reach a loopback host.
const API = "/api";

async function responseError(response, fallback) {
  const payload = await response.json().catch(() => ({}));
  const requestId = response.headers.get("x-request-id");
  const detail = payload.detail || `${fallback}（${response.status}）`;
  const error = new Error(requestId ? `${detail}（请求编号：${requestId}）` : detail);
  error.status = response.status;
  return error;
}

async function request(path, options = {}) {
  // Multipart bodies set their own boundary; forcing JSON here would break audio imports.
  const headers = options.body instanceof FormData
    ? { ...options.headers }
    : { "Content-Type": "application/json", ...options.headers };
  const response = await fetch(`${API}${path}`, { headers, ...options });
  if (!response.ok) {
    throw await responseError(response, "请求失败");
  }
  if (response.status === 204) return null;
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

async function stream(path, message, onEvent, signal, fallback) {
  const response = await fetch(`${API}${path}`, {
    method: "POST", signal, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    throw await responseError(response, fallback);
  }
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser(onEvent);
  while (true) {
    const { value, done } = await reader.read();
    if (value) parser.push(decoder.decode(value, { stream: !done }));
    if (done) {
      parser.push(decoder.decode());
      parser.finish();
      break;
    }
  }
}

const streamChat = (projectId, conversationId, message, onEvent, signal) =>
  stream(`/projects/${projectId}/conversations/${conversationId}/chat/stream`, message, onEvent, signal, "对话请求失败");

const streamGlobalChat = (conversationId, message, onEvent, signal) =>
  stream(`/chat/conversations/${conversationId}/chat/stream`, message, onEvent, signal, "普通对话请求失败");

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
