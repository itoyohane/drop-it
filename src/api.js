const API = import.meta.env.DEV ? "http://127.0.0.1:8765/api" : "/api";

async function request(path, options = {}) {
  const headers = options.body instanceof FormData
    ? { ...options.headers }
    : { "Content-Type": "application/json", ...options.headers };
  const response = await fetch(`${API}${path}`, {
    headers,
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "本地服务暂时不可用");
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

export const api = {
  health: () => request("/health"),
  library: () => request("/library"),
  importFiles: (files) => {
    const body = new FormData();
    Array.from(files).forEach((file) => body.append("files", file, file.webkitRelativePath || file.name));
    return request("/library/import-files", { method: "POST", body });
  },
  createPlaylist: (brief) => request("/playlists", { method: "POST", body: JSON.stringify(brief) }),
  reorder: (id, trackIds) => request(`/playlists/${id}/order`, { method: "PUT", body: JSON.stringify({ track_ids: trackIds }) }),
  approve: (id) => request(`/playlists/${id}/approve`, { method: "POST" }),
  export: (id, format) => request(`/playlists/${id}/export`, {
    method: "POST", body: JSON.stringify({ format })
  }),
};
