import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown, ArrowUp, Books, ChatCircleDots, Check, CheckCircle, CircleNotch,
  Clock, Database, Export, FolderOpen, ListChecks, MagnifyingGlass, MusicNotes,
  PaperPlaneRight, Plus, SidebarSimple, SlidersHorizontal, Sparkle, Stack,
  WarningCircle, Waveform, X, Trash, PencilSimple,
} from "@phosphor-icons/react";
import { api, folderIdentity } from "./api";

const initialBrief = {
  title: "Friday warm-up", duration_min: 45, bpm_min: 118, bpm_max: 132,
  energy: "build", style: "deep house, warm, groovy", notes: "先留白，后半段逐步抬升。",
};

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

// Keep assistant output safe while still covering the markdown people commonly use in a set plan:
// headings, lists, quotes, links, tables, fenced code and inline emphasis/code.
function safeMarkdownHref(value) {
  const href = value.trim();
  return /^(https?:|mailto:)/i.test(href) ? href : "";
}

function renderMarkdownInline(value, keyPrefix) {
  const text = String(value || "");
  const tokenPattern = /(`[^`\n]+`|\[[^\]]+\]\([^\)]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_)/g;
  const children = [];
  let cursor = 0;
  let match;
  while ((match = tokenPattern.exec(text))) {
    if (match.index > cursor) children.push(text.slice(cursor, match.index));
    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      children.push(<code key={`${keyPrefix}-code-${match.index}`}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("[") && token.includes("](")) {
      const link = token.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
      const href = link ? safeMarkdownHref(link[2]) : "";
      children.push(href
        ? <a key={`${keyPrefix}-link-${match.index}`} href={href} target="_blank" rel="noreferrer">{link[1]}</a>
        : (link?.[1] || token));
    } else if (token.startsWith("**") || token.startsWith("__")) {
      children.push(<strong key={`${keyPrefix}-strong-${match.index}`}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("~~")) {
      children.push(<del key={`${keyPrefix}-del-${match.index}`}>{token.slice(2, -2)}</del>);
    } else {
      children.push(<em key={`${keyPrefix}-em-${match.index}`}>{token.slice(1, -1)}</em>);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) children.push(text.slice(cursor));
  return children.length ? children : text;
}

function markdownCells(value) {
  return value.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function isMarkdownTableDivider(value) {
  const cells = markdownCells(value);
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function parseMarkdownBlocks(value) {
  const lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let paragraph = [];
  let list = null;
  let quote = [];
  const flushParagraph = () => { if (paragraph.length) { blocks.push({ type: "paragraph", lines: paragraph }); paragraph = []; } };
  const flushList = () => { if (list) { blocks.push(list); list = null; } };
  const flushQuote = () => { if (quote.length) { blocks.push({ type: "quote", lines: quote }); quote = []; } };
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const fence = line.match(/^\s*```\s*([\w+-]*)\s*$/);
    if (fence) {
      flushParagraph(); flushList(); flushQuote();
      const code = []; const language = fence[1]; index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) { code.push(lines[index]); index += 1; }
      blocks.push({ type: "code", language, value: code.join("\n") });
      if (index < lines.length) index += 1;
      continue;
    }
    if (!line.trim()) { flushParagraph(); flushList(); flushQuote(); index += 1; continue; }
    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) { flushParagraph(); flushList(); flushQuote(); blocks.push({ type: "heading", level: heading[1].length, value: heading[2] }); index += 1; continue; }
    if (/^\s*(?:[-*_]\s*){3,}$/.test(line)) { flushParagraph(); flushList(); flushQuote(); blocks.push({ type: "hr" }); index += 1; continue; }
    if (/^\s*>/.test(line)) { flushParagraph(); flushList(); quote.push(line.replace(/^\s*>\s?/, "")); index += 1; continue; }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph(); flushQuote();
      const type = unordered ? "ul" : "ol";
      if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
      list.items.push((unordered || ordered)[1]); index += 1; continue;
    }
    if (line.includes("|") && index + 1 < lines.length && isMarkdownTableDivider(lines[index + 1])) {
      flushParagraph(); flushList(); flushQuote();
      const header = markdownCells(line); const align = markdownCells(lines[index + 1]); const rows = [];
      index += 2;
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) { rows.push(markdownCells(lines[index])); index += 1; }
      blocks.push({ type: "table", header, align, rows }); continue;
    }
    paragraph.push(line); index += 1;
  }
  flushParagraph(); flushList(); flushQuote();
  return blocks;
}

function MarkdownContent({ value }) {
  const blocks = parseMarkdownBlocks(value);
  return <div className="markdown-content">{blocks.map((block, index) => {
    const key = `markdown-${index}`;
    if (block.type === "heading") {
      const Heading = `h${block.level}`;
      return <Heading key={key}>{renderMarkdownInline(block.value, key)}</Heading>;
    }
    if (block.type === "paragraph") return <p key={key}>{block.lines.map((line, lineIndex) => <span key={`${key}-${lineIndex}`}>{lineIndex > 0 && <br />}{renderMarkdownInline(line, `${key}-${lineIndex}`)}</span>)}</p>;
    if (block.type === "quote") return <blockquote key={key}>{block.lines.map((line, lineIndex) => <p key={`${key}-${lineIndex}`}>{renderMarkdownInline(line, `${key}-${lineIndex}`)}</p>)}</blockquote>;
    if (block.type === "code") return <pre key={key}><code className={block.language ? `language-${block.language}` : ""}>{block.value}</code></pre>;
    if (block.type === "hr") return <hr key={key} />;
    if (block.type === "table") return <div className="markdown-table-wrap" key={key}><table><thead><tr>{block.header.map((cell, cellIndex) => <th key={`${key}-h-${cellIndex}`}>{renderMarkdownInline(cell, `${key}-h-${cellIndex}`)}</th>)}</tr></thead><tbody>{block.rows.map((row, rowIndex) => <tr key={`${key}-r-${rowIndex}`}>{block.header.map((_, cellIndex) => <td key={`${key}-r-${rowIndex}-${cellIndex}`}>{renderMarkdownInline(row[cellIndex] || "", `${key}-r-${rowIndex}-${cellIndex}`)}</td>)}</tr>)}</tbody></table></div>;
    const List = block.type === "ol" ? "ol" : "ul";
    return <List key={key}>{block.items.map((item, itemIndex) => <li key={`${key}-${itemIndex}`}>{renderMarkdownInline(item, `${key}-${itemIndex}`)}</li>)}</List>;
  })}</div>;
}

async function retryStartup(task, attempts = 10) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await task();
    } catch (error) {
      lastError = error;
      if ((error.status && error.status < 500) || attempt === attempts - 1) throw error;
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(250 * (attempt + 1), 1000)));
    }
  }
  throw lastError;
}

function App() {
  const [chatMode, setChatMode] = useState("global");
  const [projects, setProjects] = useState([]);
  const [activeId, setActiveId] = useState(localStorage.getItem("dropit-project") || "");
  const [section, setSection] = useState("chat");
  const [projectTracks, setProjectTracks] = useState([]);
  const [globalTracks, setGlobalTracks] = useState([]);
  const [messages, setMessages] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState("");
  const [conversationScope, setConversationScope] = useState("");
  const [jobs, setJobs] = useState([]);
  const [playlists, setPlaylists] = useState([]);
  const [sources, setSources] = useState([]);
  const [playlist, setPlaylist] = useState(null);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [busy, setBusy] = useState("");
  const [chatStatus, setChatStatus] = useState("");
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 820);
  const [projectDialog, setProjectDialog] = useState(false);
  const [trackToRemove, setTrackToRemove] = useState(null);
  const [removingTrackId, setRemovingTrackId] = useState("");
  const folderInput = useRef(null);
  const streamController = useRef(null);

  const activeProject = projects.find((item) => item.id === activeId) || null;
  useEffect(() => {
    let disposed = false;
    // The web process can become ready before the API process. Retry transient proxy/network failures
    // so the normal startup race does not become a persistent error banner.
    Promise.allSettled([
      retryStartup(api.health),
      retryStartup(api.projects),
      retryStartup(api.globalLibrary),
    ])
      .then(([healthResult, projectsResult, libraryResult]) => {
        if (disposed) return;
        const failures = [];
        if (healthResult.status === "rejected") failures.push(`服务状态：${healthResult.reason.message}`);
        if (projectsResult.status === "fulfilled") {
          setProjects(projectsResult.value.projects);
        } else {
          failures.push(`项目列表：${projectsResult.reason.message}`);
        }
        if (libraryResult.status === "fulfilled") setGlobalTracks(libraryResult.value.tracks);
        else failures.push(`总曲库：${libraryResult.reason.message}`);
        setError(failures.length ? `部分初始化失败。${failures.join("；")}` : "");
      })
      .finally(() => { if (!disposed) setBootstrapped(true); });
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    if (!bootstrapped) return;
    let cancelled = false;
    const scope = chatMode === "global" ? "global" : activeId;
    setConversationScope("");
    setConversationId("");
    setMessages([]);
    if (chatMode === "global") {
      api.globalConversations().then((data) => {
        if (cancelled) return;
        setConversations(data.conversations);
        const saved = localStorage.getItem("dropit-conversation-global");
        const next = data.conversations.find((item) => item.id === saved)?.id || data.conversations[0]?.id || "";
        setConversationScope(scope);
        setConversationId(next);
      }).catch((err) => { if (!cancelled) setError(err.message); });
      return () => { cancelled = true; };
    }
    if (!activeId) return () => { cancelled = true; };
    localStorage.setItem("dropit-project", activeId);
    // Treat a project change as a context switch: refresh every project-scoped resource together.
    Promise.all([api.projectLibrary(activeId), api.conversations(activeId), api.playlists(activeId), api.jobs(activeId), api.projectSources(activeId)])
      .then(async ([libraryData, conversationData, playlistData, jobData, sourceData]) => {
        const projectConversations = conversationData.conversations.length
          ? conversationData.conversations
          : [await api.createConversation(activeId, "项目对话")];
        if (cancelled) return;
        setProjectTracks(libraryData.tracks);
        setConversations(projectConversations);
        const saved = localStorage.getItem(`dropit-conversation-${activeId}`);
        const next = projectConversations.find((item) => item.id === saved)?.id || projectConversations[0]?.id || "";
        setConversationScope(scope);
        setConversationId(next);
        setJobs(jobData.jobs);
        setSources(sourceData.sources);
        setPlaylists(playlistData.playlists);
        setPlaylist((current) => current?.project_id === activeId ? current : playlistData.playlists[0] || null);
      })
      .catch((err) => { if (!cancelled) setError(err.message); });
    return () => { cancelled = true; };
  }, [activeId, bootstrapped, chatMode]);

  useEffect(() => {
    const scope = chatMode === "global" ? "global" : activeId;
    if (conversationScope !== scope || !conversationId || !conversations.some((item) => item.id === conversationId)) { setMessages([]); return; }
    const storageKey = chatMode === "global" ? "dropit-conversation-global" : `dropit-conversation-${activeId}`;
    localStorage.setItem(storageKey, conversationId);
    const request = chatMode === "global" ? api.globalMessages(conversationId) : api.messages(activeId, conversationId);
    request.then((data) => setMessages(data.messages)).catch((err) => setError(err.message));
  }, [activeId, chatMode, conversationId, conversationScope, conversations]);

  useEffect(() => {
    if (chatMode !== "project" || !activeId || !jobs.some((job) => ["queued", "running"].includes(job.status))) return undefined;
    // Poll only while background analysis exists, and refresh tracks once their metadata changes.
    const timer = window.setInterval(async () => {
      try {
        const [jobData, libraryData, globalData, sourceData] = await Promise.all([
          api.jobs(activeId), api.projectLibrary(activeId), api.globalLibrary(), api.projectSources(activeId),
        ]);
        setJobs(jobData.jobs); setProjectTracks(libraryData.tracks); setGlobalTracks(globalData.tracks);
        setSources(sourceData.sources);
      } catch (err) { setError(err.message); }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [activeId, chatMode, jobs]);

  const selectProject = (id) => {
    setConversationScope(""); setConversationId("");
    setTrackToRemove(null);
    setActiveId(id); setChatMode("project"); setSection("chat");
    if (window.innerWidth <= 820) setSidebarOpen(false);
  };

  const selectGlobalChat = () => {
    setConversationScope(""); setConversationId("");
    setTrackToRemove(null);
    setChatMode("global"); setSection("chat");
    if (window.innerWidth <= 820) setSidebarOpen(false);
  };

  const createProject = async (name) => {
    setBusy("project"); setError("");
    try {
      const project = await api.createProject({ name, description: "" });
      setProjects((items) => [project, ...items]);
      setConversationScope(""); setConversationId("");
      setActiveId(project.id); setChatMode("project"); setSection("chat"); setProjectDialog(false);
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const deleteProject = async (id) => {
    const project = projects.find((item) => item.id === id);
    if (!project || !window.confirm(`删除项目「${project.name}」？项目中的对话、曲库和 Set 将一并删除。`)) return;
    setBusy(`delete-project-${id}`); setError("");
    try {
      await api.deleteProject(id);
      setProjects((items) => items.filter((item) => item.id !== id));
      if (activeId === id) {
        localStorage.removeItem("dropit-project");
        setConversationScope(""); setConversationId("");
        setActiveId(""); setChatMode("global"); setSection("chat");
        setProjectTracks([]); setMessages([]); setConversations([]); setPlaylist(null);
        setTrackToRemove(null);
      }
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const importFiles = async (event) => {
    const files = event.target.files;
    if (!activeId || !files?.length) return;
    setBusy("import"); setError("");
    try {
      const identity = await folderIdentity(files);
      const resolved = await api.resolveFolder(activeId, identity.folderKey, identity.folderName);
      let result = resolved;
      if (!resolved.reused) result = await api.importFiles(activeId, resolved.source.id, files);
      const [projectData, globalData, sourceData, libraryData] = await Promise.all([
        api.projects(), api.globalLibrary(), api.projectSources(activeId), api.projectLibrary(activeId),
      ]);
      setProjectTracks(libraryData.tracks); setProjects(projectData.projects); setGlobalTracks(globalData.tracks);
      setSources(sourceData.sources);
      if (result.job) setJobs((items) => [result.job, ...items]);
      setSection("project-library");
    } catch (err) { setError(err.message); }
    finally { setBusy(""); event.target.value = ""; }
  };

  const sendChat = async (text) => {
    if ((chatMode === "project" && !activeId) || !conversationId || !text.trim() || busy === "chat") return;
    const scopeId = chatMode === "global" ? "global-chat" : activeId;
    // Render the user's turn immediately; replace it with the server record when SSE confirms persistence.
    const optimistic = {
      id: `temp-${Date.now()}`, project_id: scopeId, conversation_id: conversationId, role: "user", content: text.trim(),
      tool_events: [], created_at: new Date().toISOString(),
    };
    setMessages((items) => [...items, optimistic]); setBusy("chat"); setChatStatus("正在生成回答"); setError("");
    try {
      const assistantId = `stream-${Date.now()}`;
      setMessages((items) => [...items, { id: assistantId, project_id: scopeId, conversation_id: conversationId,
        role: "assistant", content: "", tool_events: [], created_at: new Date().toISOString() }]);
      streamController.current = new AbortController();
      const stream = chatMode === "global" ? api.streamGlobalChat : api.streamChat;
      const streamArgs = chatMode === "global"
        ? [conversationId, text.trim()]
        : [activeId, conversationId, text.trim()];
      await stream(...streamArgs, (event) => {
        // Each event type updates only the small piece of transient UI state it owns.
        if (event.type === "user_saved") setMessages((items) => items.map((item) => item.id === optimistic.id ? event.message : item));
        if (event.type === "token") {
          setChatStatus("正在生成回答");
          setMessages((items) => items.map((item) => item.id === assistantId ? { ...item, content: item.content + event.content } : item));
        }
        if (event.type === "tool") {
          const toolLabels = { search_library: "检索曲库", create_playlist: "编排 Set", validate_playlist: "检查衔接" };
          setChatStatus(event.status === "running" ? `正在${toolLabels[event.name] || "调用工具"}` : "正在整理结果");
          setMessages((items) => items.map((item) => item.id === assistantId ? {
            ...item, tool_events: [...item.tool_events.filter((tool) => tool.name !== event.name || tool.status !== "running"), event],
          } : item));
        }
        if (event.type === "status") setChatStatus(event.label || "正在生成回答");
        if (event.type === "complete") {
          setMessages((items) => items.map((item) => item.id === assistantId ? event.message : item));
          if (event.playlist) {
            setPlaylist(event.playlist);
            setPlaylists((items) => [event.playlist, ...items.filter((item) => item.id !== event.playlist.id)]);
          }
        }
        if (event.type === "error") setError(event.detail);
      }, streamController.current.signal);
      const data = await (chatMode === "global" ? api.globalConversations() : api.conversations(activeId));
      setConversations(data.conversations);
    } catch (err) {
      setMessages((items) => items.filter((item) => item.id !== optimistic.id && !item.id.startsWith("stream-")));
      setError(err.message);
    } finally { setBusy(""); setChatStatus(""); streamController.current = null; }
  };

  const analyzeLibrary = async () => {
    if (!activeId) return;
    setBusy("analyze"); setError("");
    try {
      const result = await api.analyzeLibrary(activeId);
      if (result.job) setJobs((items) => [result.job, ...items]);
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const saveTrack = async (track) => {
    try { const updated = await api.updateTrack(activeId, track.id, track); setProjectTracks((items) => items.map((item) => item.id === updated.id ? updated : item)); }
    catch (err) { setError(err.message); throw err; }
  };

  const requestRemoveTrack = (track) => setTrackToRemove(track);

  const removeTrack = async () => {
    if (!trackToRemove || !activeId || removingTrackId) return;
    const track = trackToRemove;
    setRemovingTrackId(track.id); setError("");
    try {
      await api.removeTrack(activeId, track.id);
      setProjectTracks((items) => items.filter((item) => item.id !== track.id));
      setProjects((items) => items.map((project) => project.id === activeId
        ? { ...project, track_count: Math.max(0, project.track_count - 1) } : project));
      setTrackToRemove(null);
    } catch (err) { setError(err.message); }
    finally { setRemovingTrackId(""); }
  };

  const generatePlaylist = async (brief) => {
    setBusy("generate"); setError("");
    try {
      const result = await api.createPlaylist(activeId, brief);
      setPlaylist(result);
      setPlaylists((items) => [result, ...items.filter((item) => item.id !== result.id)]);
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const moveTrack = async (index, direction) => {
    const rows = [...playlist.tracks];
    const next = index + direction;
    if (next < 0 || next >= rows.length) return;
    [rows[index], rows[next]] = [rows[next], rows[index]];
    // Optimistically move the row, then reconcile with the server's incremented revision.
    setPlaylist({ ...playlist, tracks: rows });
    try {
      const updated = await api.reorder(playlist.id, rows.map((row) => row.track.id));
      setPlaylist(updated);
      setPlaylists((items) => items.map((item) => item.id === updated.id ? updated : item));
    }
    catch (err) { setError(err.message); }
  };

  const approve = async () => {
    setBusy("approve");
    try {
      const updated = await api.approve(playlist.id);
      setPlaylist(updated);
      setPlaylists((items) => items.map((item) => item.id === updated.id ? updated : item));
    }
    catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const exportSet = async (format) => {
    try {
      const content = await api.export(playlist.id, format);
      const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url; link.download = `${playlist.brief.title.replace(/\s+/g, "-").toLowerCase()}.${format}`;
      link.click(); URL.revokeObjectURL(url);
    } catch (err) { setError(err.message); }
  };

  const title = section === "chat" ? (chatMode === "global" ? "普通对话" : activeProject?.name || "项目内对话") :
    section === "project-library" ? "项目曲库" : section === "global-library" ? "总曲库" : "排 Set";
  const scopeLabel = section === "chat"
    ? (chatMode === "global" ? "总曲库上下文" : "项目文件夹上下文")
    : section === "project-library" ? `${sources.length} 个绑定文件夹`
    : section === "global-library" ? "全部导入曲目"
    : "当前项目曲库";

  return <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
    <input ref={folderInput} hidden type="file" accept=".mp3,.wav,.flac,.aiff,.aif,.m4a,audio/*"
      multiple webkitdirectory="" directory="" onChange={importFiles} />
    <Sidebar projects={projects} activeId={activeId} chatMode={chatMode} section={section}
      selectProject={selectProject} selectGlobalChat={selectGlobalChat} setSection={setSection}
      openProjectDialog={() => setProjectDialog(true)} projectTracks={projectTracks} deleteProject={deleteProject} busy={busy} />
    {sidebarOpen && <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="关闭侧边栏" />}
    <main className="main-pane">
      <header className="topbar">
        <button className="icon-button" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="切换侧边栏"><SidebarSimple /></button>
        <strong>{title}</strong>
        <span className={`scope-pill ${chatMode}`}>{scopeLabel}</span>
      </header>
      {error && <div className="error-banner" role="alert"><WarningCircle weight="fill" />{error}<button onClick={() => setError("")}><X /></button></div>}
      {section === "chat" && (chatMode === "global" || activeProject) ? <ChatView mode={chatMode}
          project={activeProject} tracks={chatMode === "global" ? globalTracks : projectTracks} messages={messages}
          busy={busy} chatStatus={chatStatus} sendChat={sendChat} importFolder={() => folderInput.current?.click()}
          openSet={() => setSection("set-builder")} playlist={playlist} /> :
        section === "chat" ? <Welcome onCreate={() => setProjectDialog(true)} /> : null}
      {section !== "chat" && <>
        {section === "project-library" && <LibraryView scope="project" tracks={projectTracks} busy={busy}
          importFolder={() => folderInput.current?.click()} analyzeLibrary={analyzeLibrary} jobs={jobs}
          saveTrack={saveTrack} requestRemoveTrack={requestRemoveTrack} removingTrackId={removingTrackId} sources={sources} />}
        {section === "global-library" && <LibraryView scope="global" tracks={globalTracks} />}
        {section === "set-builder" && <SetWorkspace tracks={projectTracks} playlist={playlist} setPlaylist={setPlaylist}
          playlists={playlists} generate={generatePlaylist} moveTrack={moveTrack} approve={approve}
          exportSet={exportSet} busy={busy} />}
      </>}
    </main>
    {projectDialog && <ProjectDialog busy={busy} close={() => setProjectDialog(false)} create={createProject} />}
    {trackToRemove && <RemoveTrackDialog track={trackToRemove} busy={removingTrackId === trackToRemove.id}
      close={() => { if (!removingTrackId) setTrackToRemove(null); }} confirm={removeTrack} />}
  </div>;
}

function Sidebar({ projects, activeId, chatMode, section, selectProject, selectGlobalChat, setSection, openProjectDialog, projectTracks, deleteProject, busy }) {
  return <aside className="sidebar">
    <div className="brand"><span className="brand-mark"><Waveform weight="bold" /></span><span>DropIt</span></div>
    <button className="new-project" onClick={openProjectDialog}><Plus weight="bold" />新建项目</button>
    <nav className="global-chat-nav"><button className={chatMode === "global" && section === "chat" ? "active" : ""}
      onClick={selectGlobalChat}><ChatCircleDots />普通对话<span>总曲库</span></button></nav>
    <div className="sidebar-label">项目</div>
    <div className="project-list">
      {projects.map((project) => <div key={project.id} className={`project-row ${project.id === activeId ? "active" : ""}`}>
        <button className="project-select" onClick={() => selectProject(project.id)}><span>{project.name.slice(0, 1).toUpperCase()}</span><div><strong>{project.name}</strong><small>{project.track_count} 首曲目</small></div></button>
        <button className="project-delete" onClick={() => deleteProject(project.id)} disabled={busy === `delete-project-${project.id}`}
          aria-label={`删除项目 ${project.name}`} title="删除项目"><Trash weight="bold" /></button>
      </div>)}
      {!projects.length && <p>还没有项目</p>}
    </div>
    {activeId && chatMode === "project" && <nav>
      <button className={section === "chat" ? "active" : ""} onClick={() => setSection("chat")}><ChatCircleDots />项目内对话</button>
      <button className={section === "project-library" ? "active" : ""} onClick={() => setSection("project-library")}><Books />项目曲库<span>{projectTracks.length}</span></button>
      <button className={section === "set-builder" ? "active" : ""} onClick={() => setSection("set-builder")}><ListChecks />排 Set</button>
    </nav>}
    <div className="sidebar-bottom">
      <button className={section === "global-library" ? "active" : ""} onClick={() => setSection("global-library")}><Database />总曲库</button>
    </div>
  </aside>;
}

function Welcome({ onCreate }) {
  return <section className="welcome"><div className="welcome-mark"><Waveform /></div><h1>从一个项目开始</h1>
    <p>每个项目绑定一组独立曲库与对话记录；所有导入曲目也会汇总到总曲库。</p>
    <button className="primary-button" onClick={onCreate}><Plus />创建项目</button></section>;
}

function ChatView({ mode, project, tracks, messages, busy, chatStatus, sendChat, importFolder, openSet, playlist }) {
  const [draft, setDraft] = useState("");
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);
  const submit = (event) => { event.preventDefault(); if (draft.trim()) { sendChat(draft); setDraft(""); } };
  const suggestions = mode === "global"
    ? ["介绍一下我的总曲库", "推荐一些适合 warm-up 的选曲思路", "解释 Camelot 调性轮怎么使用"]
    : ["排一个 45 分钟、逐步升温的 warm-up set", "找出 124 BPM 附近、能量适中的曲目", "做一套调性衔接平滑的 closing set"];
  return <section className="chat-view">
    <div className="chat-scroll">
      {!messages.length ? <div className="chat-empty"><span className="eyebrow">{mode === "global" ? "普通对话" : project.name}</span><h1>{mode === "global" ? "想聊点什么？" : "今晚想怎么走？"}</h1>
        <p>{mode === "global" ? `无需创建项目即可对话，涉及歌曲时会参考总曲库的 ${tracks.length} 首曲目。` : tracks.length ? `已连接 ${tracks.length} 首曲目。描述场地、时长、BPM 和能量走向。` : "项目还没有绑定文件夹，但你仍然可以先描述需求。"}</p>
        <div className="suggestions">{suggestions.map((text) => <button key={text} onClick={() => setDraft(text)}>{text}<PaperPlaneRight /></button>)}</div></div>
      : <div className="message-list">{messages.map((message) => <Message key={message.id} message={message} />)}
          {busy === "chat" && <ModelThinking status={chatStatus} />}
          {mode === "project" && playlist && messages.at(-1)?.role === "assistant" && <button className="open-set-card" onClick={openSet}><ListChecks /><span><strong>{playlist.brief.title}</strong><small>{playlist.tracks.length} 首 · 打开排 Set 工作区</small></span></button>}
          <div ref={endRef} /></div>}
    </div>
    <form className="composer" onSubmit={submit}>
      <textarea rows="1" value={draft} onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing || event.keyCode === 229) return;
          if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(event); }
        }}
        placeholder={mode === "global" ? "随心输入你的想法" : "描述需求，或询问当前项目曲库…"} />
      <div className="composer-actions">{mode === "project" && <button type="button" className="attach-button" onClick={importFolder} disabled={busy === "import"}>
        {busy === "import" ? <CircleNotch className="spin" /> : <FolderOpen />}<span>{busy === "import" ? "正在分析音频" : "导入曲库"}</span></button>
        }
        <button className="send-button" disabled={!draft.trim() || busy === "chat"} aria-label="发送"><PaperPlaneRight weight="fill" /></button></div>
    </form>
  </section>;
}

function ModelThinking({ status }) {
  const phase = status || "正在生成回答";
  const hint = phase.includes("压缩") ? "整理历史上下文" : phase.includes("思考") ? "分析需求与曲库" : "准备一份清晰的回复";
  return <div className="model-thinking" role="status" aria-live="polite">
    <div className="thinking-orb" aria-hidden="true"><span /><i /><b /></div>
    <div className="thinking-copy"><strong>{phase}</strong><span>{hint}</span></div>
    <div className="thinking-meter" aria-hidden="true">{[0, 1, 2, 3, 4].map((index) => <i key={index} style={{ "--delay": `${index * 100}ms` }} />)}</div>
  </div>;
}

function Message({ message }) {
  if (message.role === "assistant" && !message.content && !message.tool_events?.length) return null;
  return <article className={`message ${message.role}`}>
    {message.role === "assistant" && <div className="assistant-mark"><Waveform /></div>}
    <div><MarkdownContent value={message.content} />
      {!!message.tool_events?.length && <div className="tool-events">{message.tool_events.map((event, index) => <div key={`${event.name}-${index}`}>
        {event.status === "failed" ? <WarningCircle /> : <CheckCircle weight="fill" />}<span><strong>{event.name}</strong>{event.summary}</span></div>)}</div>}
    </div>
  </article>;
}

function LibraryView({ scope, tracks, importFolder, analyzeLibrary, busy, jobs = [], sources = [], saveTrack, requestRemoveTrack, removingTrackId }) {
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState(null);
  const visible = useMemo(() => tracks.filter((track) => `${track.title} ${track.artist} ${track.key} ${track.camelot_key}`.toLowerCase().includes(query.toLowerCase())), [tracks, query]);
  return <section className="workspace library-view">
    <div className="page-heading"><div><span className="eyebrow">{scope === "global" ? "ALL TRACKS" : "PROJECT SOURCE"}</span>
      <h1>{scope === "global" ? "总曲库" : "项目曲库"}</h1><p>{scope === "global" ? "所有项目导入过的曲目，按音频内容去重。" : "Agent 只会从当前项目的这些曲目中检索与排 Set。"}</p></div>
      {scope === "project" && <div className="library-actions">{tracks.some((track) => track.analysis_status !== "analyzed") && <button className="secondary-button" onClick={analyzeLibrary} disabled={busy === "analyze"}>{busy === "analyze" ? <CircleNotch className="spin" /> : <Waveform />}分析待处理曲目</button>}<button className="secondary-button" onClick={importFolder}>{busy === "import" ? <CircleNotch className="spin" /> : <FolderOpen />}导入音频</button></div>}</div>
    {scope === "project" && <div className="source-strip"><div><strong>已绑定文件夹</strong><span>{sources.length} 个，可继续添加</span></div>
      <div className="source-list">{sources.map((source) => <span key={source.id} title="同一文件夹可被多个项目复用"><FolderOpen /><b>{source.name}</b><small>{source.track_count} 首 · {source.status === "ready" ? "索引就绪" : source.status === "failed" ? "处理失败" : "处理中"}</small></span>)}
        {!sources.length && <em>尚未绑定文件夹</em>}</div></div>}
    {scope === "project" && jobs.some((job) => ["queued", "running", "failed"].includes(job.status)) && <JobStatus job={jobs.find((job) => ["queued", "running"].includes(job.status)) || jobs.find((job) => job.status === "failed")} tracks={tracks} />}
    {!tracks.length ? <div className="empty-state"><MusicNotes /><h2>{scope === "global" ? "总曲库还是空的" : "还没有项目曲库"}</h2><p>导入音频后会真实分析 BPM、调性和能量，并写入向量索引。</p>
      {scope === "project" && <button className="primary-button" onClick={importFolder}><FolderOpen />选择文件夹</button>}</div>
    : <><div className="library-toolbar"><MagnifyingGlass /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索歌名、艺人、调性…" /><span>{visible.length} 首</span></div>
      <TrackTable tracks={visible} editable={scope === "project"} edit={setEditing}
        requestRemoveTrack={requestRemoveTrack} removingTrackId={removingTrackId} /></>}
    {editing && <TrackDialog track={editing} close={() => setEditing(null)} save={async (value) => { await saveTrack(value); setEditing(null); }} />}
  </section>;
}

function TrackTable({ tracks, editable, edit, requestRemoveTrack, removingTrackId }) {
  return <div className={`track-table ${editable ? "editable" : ""}`}><div className="table-head"><span>#</span><span>曲目</span><span>BPM</span><span>调性</span><span>能量</span><span>状态</span>{editable && <span>操作</span>}</div>
    {tracks.map((track, index) => <div className="track-row" key={track.id}><span>{String(index + 1).padStart(2, "0")}</span>
      <div><strong>{track.title}</strong><small>{track.artist} · {formatTime(track.duration_sec)}</small></div>
      <span>{track.bpm ? track.bpm.toFixed(1) : "未知"}<small>{track.bpm_confidence ? `${Math.round(track.bpm_confidence * 100)}%` : ""}</small></span>
      <span>{track.camelot_key}<small>{track.key}</small></span><span>{track.energy ? Math.round(track.energy * 100) : "未知"}</span>
      <span className={`analysis-state ${track.analysis_status}`} title={track.analysis_error || ""}>{track.analysis_status === "analyzed" ? "已分析" : track.analysis_status === "failed" ? "失败" : track.analysis_status === "analyzing" ? "分析中" : "待分析"}</span>
      {editable && <span className="track-actions"><button onClick={() => edit(track)} title="编辑"><PencilSimple /></button><button className="danger-action"
        onClick={() => requestRemoveTrack(track)} disabled={removingTrackId === track.id} title="从当前项目移除" aria-label={`移除 ${track.title}`}>
        {removingTrackId === track.id ? <CircleNotch className="spin" /> : <Trash />}</button></span>}
    </div>)}</div>;
}

function JobStatus({ job, tracks = [] }) {
  const total = Math.max(Number(job.total) || 0, 0);
  const progress = Math.min(Math.max(Number(job.progress) || 0, 0), total || Number.MAX_SAFE_INTEGER);
  const percent = total ? Math.min(100, Math.round((progress / total) * 100)) : null;
  const trackStats = tracks.reduce((stats, track) => {
    stats[track.analysis_status] = (stats[track.analysis_status] || 0) + 1;
    return stats;
  }, {});
  const statusLabel = job.status === "failed" ? "音频处理失败" : job.status === "queued" ? "等待开始分析" : "正在分析曲库";
  const detail = job.error || (job.status === "queued"
    ? `队列中 · ${total ? `${total} 首待处理` : "正在准备任务"}`
    : job.message || (total ? `已处理 ${progress} / ${total} 首` : "正在读取音频文件"));
  const countLabel = total ? `${progress} / ${total} 首` : trackStats.analyzing ? `${trackStats.analyzing} 首分析中` : "准备中";
  return <div className={`job-status ${job.status}`}>
    {job.status === "failed" ? <WarningCircle /> : <CircleNotch className={job.status === "running" ? "spin" : ""} />}
    <div className="job-status-copy"><strong>{statusLabel}</strong><span>{detail}</span>
      {job.status !== "failed" && <div className={`job-progress ${percent === null ? "indeterminate" : ""}`} role="progressbar" aria-label="曲库分析进度" aria-valuemin="0" aria-valuemax="100" {...(percent === null ? {} : { "aria-valuenow": percent })}><b style={percent === null ? undefined : { width: `${percent}%` }} /></div>}</div>
    <small>{job.status === "failed" ? "需重试" : countLabel}</small>
  </div>;
}

function TrackDialog({ track, close, save }) {
  const [value, setValue] = useState({ ...track });
  const [saving, setSaving] = useState(false);
  const update = (key) => (event) => setValue({ ...value, [key]: ["bpm", "energy"].includes(key) ? Number(event.target.value) : event.target.value });
  return <div className="dialog-backdrop" onMouseDown={close}><form className="dialog track-dialog" onMouseDown={(event) => event.stopPropagation()} onSubmit={async (event) => { event.preventDefault(); setSaving(true); try { await save(value); } finally { setSaving(false); } }}>
    <button type="button" className="dialog-close" onClick={close}><X /></button><span className="dialog-icon"><MusicNotes /></span><h2>编辑曲目信息</h2><p>手动校正后会重新写入项目向量索引。</p>
    <label><span>标题</span><input value={value.title} onChange={update("title")} /></label><label><span>艺人</span><input value={value.artist} onChange={update("artist")} /></label>
    <div className="dialog-grid"><label><span>BPM</span><input type="number" min="40" max="260" step="0.1" value={value.bpm} onChange={update("bpm")} /></label><label><span>能量 0-1</span><input type="number" min="0" max="1" step="0.01" value={value.energy} onChange={update("energy")} /></label><label><span>调性</span><input value={value.key} onChange={update("key")} /></label><label><span>Camelot</span><input value={value.camelot_key} onChange={update("camelot_key")} /></label></div>
    <button className="primary-button" disabled={saving}>{saving ? <CircleNotch className="spin" /> : <Check />}保存修改</button>
  </form></div>;
}

function RemoveTrackDialog({ track, busy, close, confirm }) {
  return <div className="dialog-backdrop" onMouseDown={close}>
    <div className="dialog remove-dialog" role="dialog" aria-modal="true" aria-labelledby="remove-track-title"
      onMouseDown={(event) => event.stopPropagation()}>
      <button type="button" className="dialog-close" onClick={close} disabled={busy} aria-label="关闭"><X /></button>
      <span className="dialog-icon remove-dialog-icon"><Trash /></span>
      <h2 id="remove-track-title">移除这首歌？</h2>
      <p>这首歌将从当前项目曲库中移除，但不会删除总曲库中的原文件，也不会影响其他项目。</p>
      <div className="remove-track-preview"><MusicNotes /><div><strong>{track.title}</strong><span>{track.artist} · {formatTime(track.duration_sec)}</span></div></div>
      <div className="dialog-actions"><button type="button" className="secondary-button" onClick={close} disabled={busy}>取消</button>
        <button type="button" className="danger-button" onClick={confirm} disabled={busy}>{busy ? <CircleNotch className="spin" /> : <Trash />}移除歌曲</button></div>
    </div>
  </div>;
}

function SetWorkspace({ tracks, playlist, setPlaylist, playlists, generate, moveTrack, approve, exportSet, busy }) {
  const [brief, setBrief] = useState(initialBrief);
  const update = (key) => (event) => setBrief({ ...brief, [key]: event.target.value });
  const submit = (event) => { event.preventDefault(); generate({ ...brief, duration_min: Number(brief.duration_min), bpm_min: Number(brief.bpm_min), bpm_max: Number(brief.bpm_max) }); };
  return <section className="workspace set-workspace">
    <div className="page-heading"><div><span className="eyebrow">SET BUILDER</span><h1>排 Set</h1><p>独立的结构化编排与人工审核工作区。</p></div>
      <div className="set-history">{playlists.map((item) => <button key={item.id} className={playlist?.id === item.id ? "active" : ""} onClick={() => setPlaylist(item)}>{item.brief.title}</button>)}
        {playlist && <button onClick={() => setPlaylist(null)}><Plus />新建</button>}</div></div>
    {!playlist ? <form className="brief-panel" onSubmit={submit}>
      <label className="title-field"><span>SET 名称</span><input value={brief.title} onChange={update("title")} /></label>
      <div className="form-grid"><label><span><Clock />时长</span><div className="input-unit"><input type="number" min="10" max="240" value={brief.duration_min} onChange={update("duration_min")} /><b>分钟</b></div></label>
        <label><span><SlidersHorizontal />BPM 范围</span><div className="range-input"><input type="number" value={brief.bpm_min} onChange={update("bpm_min")} /><i>至</i><input type="number" value={brief.bpm_max} onChange={update("bpm_max")} /></div></label>
        <label><span><Waveform />能量曲线</span><select value={brief.energy} onChange={update("energy")}><option value="build">逐步抬升</option><option value="steady">保持稳定</option><option value="peak">快速进入高峰</option><option value="wave">波浪起伏</option></select></label>
        <label><span><MusicNotes />风格与情绪</span><input value={brief.style} onChange={update("style")} /></label></div>
      <label className="notes-field"><span>补充说明</span><textarea rows="3" value={brief.notes} onChange={update("notes")} /></label>
      <div className="form-actions"><span>{tracks.length ? `使用当前项目的 ${tracks.length} 首曲目` : "请先导入项目曲库"}</span><button className="primary-button" disabled={!tracks.length || busy === "generate"}>{busy === "generate" ? <CircleNotch className="spin" /> : <Sparkle weight="fill" />}生成 Set</button></div>
    </form> : <PlaylistReview playlist={playlist} moveTrack={moveTrack} approve={approve} exportSet={exportSet} busy={busy} />}
  </section>;
}

function PlaylistReview({ playlist, moveTrack, approve, exportSet, busy }) {
  return <div className="review-grid"><div className="set-list"><div className="set-summary"><div><span className={`approval ${playlist.status}`}><CheckCircle weight="fill" />{playlist.status === "approved" ? "已确认" : "待审核"}</span><h2>{playlist.brief.title}</h2><p>{playlist.tracks.length} 首 · {formatTime(playlist.duration_sec)} · 修订 {playlist.revision}</p></div>
    <div>{playlist.status !== "approved" && <button className="primary-button" onClick={approve} disabled={busy === "approve"}><Check />确认</button>}{["m3u", "json", "csv"].map((format) => <button className="icon-button export-button" key={format} onClick={() => exportSet(format)} title={`导出 ${format}`}><Export /><small>{format}</small></button>)}</div></div>
    <div className="set-list-head"><span>顺序</span><span>曲目与理由</span><span>特征</span><span>调整</span></div>
    {playlist.tracks.map((row, index) => <div className="set-row" key={row.track.id}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{row.track.title}</strong><small>{row.track.artist}</small><p>{row.reason}</p></div><div className="set-metrics"><strong>{row.track.bpm.toFixed(1)}</strong><span>{row.track.camelot_key}</span><small>{Math.round(row.track.energy * 100)} EN</small></div><div className="row-controls"><button disabled={!index} onClick={() => moveTrack(index, -1)}><ArrowUp /></button><button disabled={index === playlist.tracks.length - 1} onClick={() => moveTrack(index, 1)}><ArrowDown /></button></div></div>)}</div>
    <aside className="audit-panel"><h3>生成轨迹</h3>{playlist.trace.map((event, index) => <div className="trace-item" key={`${event.agent}-${index}`}><span><Check /></span><div><strong>{event.agent}</strong><p>{event.message}</p></div></div>)}
      <div className="rule-report"><h3>约束报告</h3>{playlist.report.map((item) => <p key={item}><CheckCircle weight="fill" />{item}</p>)}</div></aside></div>;
}

function ProjectDialog({ busy, close, create }) {
  const [name, setName] = useState("");
  return <div className="dialog-backdrop" onMouseDown={close}><form className="dialog" onSubmit={(event) => { event.preventDefault(); if (name.trim()) create(name.trim()); }} onMouseDown={(event) => event.stopPropagation()}>
    <button type="button" className="dialog-close" onClick={close}><X /></button><span className="dialog-icon"><Stack /></span><h2>新建项目</h2><p>项目可绑定多个文件夹；相同文件夹会复用已有音频分析和向量库。</p>
    <label><span>项目名称</span><input autoFocus value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：Basement Friday" /></label>
    <button className="primary-button" disabled={!name.trim() || busy === "project"}>{busy === "project" ? <CircleNotch className="spin" /> : <Plus />}创建项目</button></form></div>;
}

export default App;
