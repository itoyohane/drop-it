import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown, ArrowUp, Check, CheckCircle, CircleNotch, Clock, Export,
  FolderOpen, Headphones, ListChecks, MusicNotes, Plus, SidebarSimple,
  SlidersHorizontal, Sparkle, SquaresFour, WarningCircle, Waveform
} from "@phosphor-icons/react";
import { api } from "./api";

const initialBrief = {
  title: "Friday warm-up", duration_min: 45, bpm_min: 118, bpm_max: 132,
  energy: "build", style: "deep house, warm, groovy", notes: "先留白，后半段逐步抬升。"
};

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${String(secs).padStart(2, "0")}`;
}

function App() {
  const [section, setSection] = useState("create");
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 820);
  const [tracks, setTracks] = useState([]);
  const [playlist, setPlaylist] = useState(null);
  const [brief, setBrief] = useState(initialBrief);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [serviceReady, setServiceReady] = useState(false);
  const mainRef = useRef(null);
  const folderInputRef = useRef(null);

  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0, behavior: "auto" });
  }, [section]);

  useEffect(() => {
    let cancelled = false;
    const connect = async () => {
      for (let attempt = 0; attempt < 12 && !cancelled; attempt += 1) {
        try {
          await api.health();
          const data = await api.library();
          if (!cancelled) { setTracks(data.tracks); setServiceReady(true); }
          return;
        } catch { await new Promise((resolve) => setTimeout(resolve, 500)); }
      }
      if (!cancelled) setError("无法连接本地服务，请确认 FastAPI 已启动。");
    };
    connect();
    return () => { cancelled = true; };
  }, []);

  const importFolder = () => folderInputRef.current?.click();

  const importFiles = async (event) => {
    const files = event.target.files;
    if (!files?.length) return;
    setError("");
    setBusy("import");
    try {
      const data = await api.importFiles(files);
      setTracks(data.tracks);
      setSection("library");
    } catch (err) { setError(err.message); }
    finally { setBusy(""); event.target.value = ""; }
  };

  const generate = async (event) => {
    event.preventDefault(); setError(""); setBusy("generate");
    try {
      const result = await api.createPlaylist({
        ...brief, duration_min: Number(brief.duration_min), bpm_min: Number(brief.bpm_min), bpm_max: Number(brief.bpm_max)
      });
      setPlaylist(result); setSection("review");
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const moveTrack = async (index, direction) => {
    const rows = [...playlist.tracks];
    const next = index + direction;
    if (next < 0 || next >= rows.length) return;
    [rows[index], rows[next]] = [rows[next], rows[index]];
    setPlaylist({ ...playlist, tracks: rows });
    try { setPlaylist(await api.reorder(playlist.id, rows.map((row) => row.track.id))); }
    catch (err) { setError(err.message); }
  };

  const approve = async () => {
    setBusy("approve"); setError("");
    try { setPlaylist(await api.approve(playlist.id)); }
    catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  const exportSet = async (format) => {
    const extension = format === "m3u" ? "m3u" : format;
    const defaultPath = `${brief.title.replace(/\s+/g, "-").toLowerCase()}.${extension}`;
    setBusy(`export-${format}`);
    try {
      const content = await api.export(playlist.id, format);
      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = defaultPath; link.click();
      URL.revokeObjectURL(link.href);
    } catch (err) { setError(err.message); }
    finally { setBusy(""); }
  };

  return (
    <div className={`app-shell ${sidebarOpen ? "" : "sidebar-collapsed"}`}>
      <input
        ref={folderInputRef}
        hidden
        type="file"
        accept=".mp3,.wav,.flac,.aiff,.aif,.m4a,audio/*"
        multiple
        webkitdirectory=""
        directory=""
        onChange={importFiles}
      />
      <Sidebar section={section} setSection={setSection} playlist={playlist} tracks={tracks} />
      <main className="main-pane" ref={mainRef}>
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="切换侧边栏"><SidebarSimple /></button>
          <div className="breadcrumbs"><span>DropIt</span><span>/</span><strong>{section === "create" ? "新建 Set" : section === "library" ? "曲库" : "审核 Set"}</strong></div>
          <div className={`service-state ${serviceReady ? "ready" : ""}`}><span />{serviceReady ? "服务已连接" : "正在连接"}</div>
        </header>

        {error && <div className="error-banner" role="alert"><WarningCircle weight="fill" />{error}<button onClick={() => setError("")}>关闭</button></div>}

        {section === "create" && <CreateView tracks={tracks} brief={brief} setBrief={setBrief} importFolder={importFolder} generate={generate} busy={busy} />}
        {section === "library" && <LibraryView tracks={tracks} importFolder={importFolder} busy={busy} setSection={setSection} />}
        {section === "review" && <ReviewView playlist={playlist} moveTrack={moveTrack} approve={approve} exportSet={exportSet} busy={busy} />}
      </main>
    </div>
  );
}

function Sidebar({ section, setSection, playlist, tracks }) {
  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark"><Waveform weight="bold" /></div><span>DropIt</span></div>
    <button className="new-set" onClick={() => setSection("create")}><Plus weight="bold" />新建 Set</button>
    <nav>
      <button className={section === "create" ? "active" : ""} onClick={() => setSection("create")}><Sparkle />生成</button>
      <button className={section === "library" ? "active" : ""} onClick={() => setSection("library")}><SquaresFour />曲库 <span className="nav-count">{tracks.length}</span></button>
      <button disabled={!playlist} className={section === "review" ? "active" : ""} onClick={() => setSection("review")}><ListChecks />审核</button>
    </nav>
    <div className="sidebar-spacer" />
    <div className="privacy-note"><Headphones /><div><strong>私有曲库</strong><span>文件只发送到当前服务</span></div></div>
  </aside>;
}

function CreateView({ tracks, brief, setBrief, importFolder, generate, busy }) {
  const update = (key) => (event) => setBrief({ ...brief, [key]: event.target.value });
  return <section className="workspace create-view">
    <div className="page-heading"><div><h1>编排下一段 Set</h1><p>描述现场，DropIt 会从本地曲库中检索、排序并检查每一次衔接。</p></div><LibraryStatus tracks={tracks} importFolder={importFolder} busy={busy} /></div>
    <form className="brief-panel" onSubmit={generate}>
      <label className="title-field"><span>SET 名称</span><input value={brief.title} onChange={update("title")} /></label>
      <div className="form-grid">
        <label><span><Clock />时长</span><div className="input-unit"><input type="number" min="10" max="240" value={brief.duration_min} onChange={update("duration_min")} /><b>分钟</b></div></label>
        <label><span><SlidersHorizontal />BPM 范围</span><div className="range-input"><input type="number" value={brief.bpm_min} onChange={update("bpm_min")} /><i>至</i><input type="number" value={brief.bpm_max} onChange={update("bpm_max")} /></div></label>
        <label><span><Waveform />能量曲线</span><select value={brief.energy} onChange={update("energy")}><option value="build">逐步抬升</option><option value="steady">保持稳定</option><option value="peak">快速进入高峰</option><option value="wave">波浪起伏</option></select></label>
        <label><span><MusicNotes />风格与情绪</span><input value={brief.style} onChange={update("style")} placeholder="deep house, warm" /></label>
      </div>
      <label className="notes-field"><span>补充说明</span><textarea rows="3" value={brief.notes} onChange={update("notes")} placeholder="例如：前 15 分钟不要出现人声，结尾保留一首明亮的歌。" /></label>
      <div className="agent-strip">
        <div><AgentGlyph label="C" /><span><strong>Curator</strong>理解曲库</span></div><i />
        <div><AgentGlyph label="P" /><span><strong>Planner</strong>编排顺序</span></div><i />
        <div><AgentGlyph label="C" /><span><strong>Critic</strong>检查约束</span></div>
      </div>
      <div className="form-actions"><span>{tracks.length ? `将从 ${tracks.length} 首本地歌曲中生成` : "请先导入至少一个音频文件夹"}</span><button className="primary-button" disabled={!tracks.length || busy === "generate"}>{busy === "generate" ? <><CircleNotch className="spin" />正在协作</> : <><Sparkle weight="fill" />生成 Set</>}</button></div>
    </form>
  </section>;
}

function LibraryStatus({ tracks, importFolder, busy }) {
  return <button className="library-status" onClick={importFolder} disabled={busy === "import"}>
    <div className="status-icon">{busy === "import" ? <CircleNotch className="spin" /> : <FolderOpen />}</div>
    <span><strong>{tracks.length ? `${tracks.length} 首歌曲已就绪` : "导入本地曲库"}</strong><small>{tracks.length ? "点击更换文件夹" : "MP3、WAV、FLAC、AIFF、M4A"}</small></span>
  </button>;
}

function LibraryView({ tracks, importFolder, busy, setSection }) {
  const [query, setQuery] = useState("");
  const visible = useMemo(() => tracks.filter((t) => `${t.title} ${t.artist} ${t.mood.join(" ")}`.toLowerCase().includes(query.toLowerCase())), [tracks, query]);
  return <section className="workspace library-view">
    <div className="page-heading"><div><h1>本地曲库</h1><p>检索只基于已索引的歌曲，Agent 不会编造 track_id。</p></div><button className="secondary-button" onClick={importFolder}>{busy === "import" ? <CircleNotch className="spin" /> : <FolderOpen />}更换文件夹</button></div>
    {!tracks.length ? <EmptyLibrary importFolder={importFolder} /> : <>
      <div className="library-toolbar"><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="按歌名、艺人或情绪搜索" /><span>{visible.length} 首</span></div>
      <div className="track-table"><div className="table-head"><span>#</span><span>歌曲</span><span>BPM</span><span>KEY</span><span>ENERGY</span><span>ROLE</span></div>
      {visible.map((track, index) => <div className="track-row" key={track.id}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{track.title}</strong><small>{track.artist} · {track.mood.join(", ")}</small></div><span>{track.bpm.toFixed(1)}</span><span>{track.key}</span><span>{Math.round(track.energy * 100)}</span><span className="role-label">{track.role}</span></div>)}</div>
      <div className="library-footer"><span>标签与基础特征已索引</span><button className="primary-button" onClick={() => setSection("create")}>创建 Set</button></div>
    </>}
  </section>;
}

function EmptyLibrary({ importFolder }) {
  return <div className="empty-state"><div className="empty-icon"><MusicNotes /></div><h2>还没有可检索的歌曲</h2><p>选择一个本地音频文件夹，DropIt 会读取文件并建立轻量索引。</p><button className="primary-button" onClick={importFolder}><FolderOpen />选择文件夹</button></div>;
}

function ReviewView({ playlist, moveTrack, approve, exportSet, busy }) {
  if (!playlist) return <section className="workspace"><div className="empty-state"><ListChecks /><h2>还没有待审核的 Set</h2><p>完成一次生成后，Agent 轨迹和曲目顺序会显示在这里。</p></div></section>;
  return <section className="workspace review-view">
    <div className="review-header"><div><span className={`approval-badge ${playlist.status}`}>{playlist.status === "approved" ? <CheckCircle weight="fill" /> : <Sparkle weight="fill" />}{playlist.status === "approved" ? "已确认" : "待人工审核"}</span><h1>{playlist.brief.title}</h1><p>{playlist.tracks.length} 首歌曲 · {formatTime(playlist.duration_sec)} · 修订 {playlist.revision}</p></div><div className="review-actions">{playlist.status !== "approved" && <button className="primary-button" onClick={approve} disabled={busy === "approve"}>{busy === "approve" ? <CircleNotch className="spin" /> : <Check />}确认 Set</button>}<ExportMenu exportSet={exportSet} busy={busy} /></div></div>
    <div className="review-grid">
      <div className="set-list"><div className="set-list-head"><span>顺序</span><span>曲目与选择理由</span><span>特征</span><span>调整</span></div>
        {playlist.tracks.map((row, index) => <div className="set-row" key={row.track.id}><span className="position">{String(index + 1).padStart(2, "0")}</span><div className="track-main"><strong>{row.track.title}</strong><small>{row.track.artist}</small><p>{row.reason}</p></div><div className="track-metrics"><strong>{row.track.bpm.toFixed(1)}</strong><span>{row.track.key}</span><span>{Math.round(row.track.energy * 100)} EN</span></div><div className="row-controls"><button disabled={index === 0} onClick={() => moveTrack(index, -1)} aria-label="上移"><ArrowUp /></button><button disabled={index === playlist.tracks.length - 1} onClick={() => moveTrack(index, 1)} aria-label="下移"><ArrowDown /></button></div></div>)}
      </div>
      <aside className="audit-panel"><h2>Agent 轨迹</h2><div className="trace-list">{playlist.trace.map((event, index) => <div className="trace-item" key={`${event.agent}-${index}`}><div className="trace-rail"><span className={event.status}><Check weight="bold" /></span>{index < playlist.trace.length - 1 && <i />}</div><div><strong>{event.agent}<em>{event.status === "revised" ? "已修订" : event.status === "approved" ? "通过" : "完成"}</em></strong><p>{event.message}</p></div></div>)}</div>
        <div className="rule-report"><h3>约束报告</h3>{playlist.report.map((item) => <p key={item}><CheckCircle weight="fill" />{item}</p>)}</div></aside>
    </div>
  </section>;
}

function ExportMenu({ exportSet, busy }) {
  const [open, setOpen] = useState(false);
  return <div className="export-wrap"><button className="secondary-button" onClick={() => setOpen(!open)}><Export />导出</button>{open && <div className="export-menu">{["m3u", "json", "csv"].map((fmt) => <button key={fmt} onClick={() => { setOpen(false); exportSet(fmt); }}>{busy === `export-${fmt}` ? <CircleNotch className="spin" /> : <Export />}<span><strong>{fmt.toUpperCase()}</strong><small>{fmt === "m3u" ? "播放器兼容列表" : fmt === "json" ? "完整 Agent 数据" : "表格与分析"}</small></span></button>)}</div>}</div>;
}

function AgentGlyph({ label }) { return <span className="agent-glyph">{label}</span>; }

export default App;
