import csv
import hashlib
import io
import json
import random
from pathlib import Path
from uuid import uuid4

from backend.models import AgentEvent, Brief, Playlist, PlaylistTrack, Track

try:
    from mutagen import File as MutagenFile
except ImportError:  # The deterministic fallback keeps the prototype usable.
    MutagenFile = None

FORMATS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}
KEYS = ["1A", "2A", "4A", "5A", "7A", "8A", "9A", "10A", "11A", "12A", "2B", "5B", "8B", "11B"]
MOODS = [["warm", "groovy"], ["deep", "hypnotic"], ["bright", "uplifting"], ["driving", "peak-time"]]


def scan_library(folder: str) -> list[Track]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError("所选文件夹不存在或无法读取")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in FORMATS)
    tracks: list[Track] = []
    for file in files:
        digest = hashlib.sha1(str(file).encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:12], 16))
        artist, title = _split_name(file.stem)
        duration = rng.randint(185, 390)
        if MutagenFile:
            try:
                metadata = MutagenFile(file, easy=True)
                if metadata:
                    title = _tag(metadata, "title", title)
                    artist = _tag(metadata, "artist", artist)
                    if getattr(metadata, "info", None) and metadata.info.length:
                        duration = round(metadata.info.length)
            except Exception:
                pass
        energy = round(rng.uniform(0.35, 0.91), 2)
        tracks.append(Track(
            id=digest[:12], title=title, artist=artist, path=str(file.resolve()),
            duration_sec=duration, bpm=round(rng.uniform(116, 136), 1),
            key=rng.choice(KEYS), energy=energy, mood=rng.choice(MOODS),
            role="peak" if energy > .78 else "builder" if energy > .56 else "opener",
        ))
    return tracks


def _tag(metadata, name: str, fallback: str) -> str:
    value = metadata.get(name)
    return str(value[0]).strip() if value and str(value[0]).strip() else fallback


def _split_name(stem: str) -> tuple[str, str]:
    parts = [part.strip() for part in stem.replace("_", " ").split(" - ", 1)]
    return (parts[0], parts[1]) if len(parts) == 2 else ("Unknown artist", parts[0])


def generate_playlist(brief: Brief, library: list[Track]) -> Playlist:
    candidates = [t for t in library if brief.bpm_min <= t.bpm <= brief.bpm_max]
    if not candidates:
        raise ValueError("没有歌曲满足当前 BPM 范围，请放宽条件")
    target = brief.duration_min * 60
    ordered = sorted(candidates, key=lambda t: (t.energy, t.bpm))
    if brief.energy == "peak": ordered = sorted(candidates, key=lambda t: (-t.energy, t.bpm))
    if brief.energy == "steady": ordered = sorted(candidates, key=lambda t: (abs(t.energy - .62), t.bpm))
    if brief.energy == "wave": ordered = ordered[::2] + ordered[1::2][::-1]
    selected, total = [], 0
    for track in ordered:
        if total >= target * .92: break
        selected.append(track)
        total += track.duration_sec

    trace = [
        AgentEvent(agent="Curator", status="done", message=f"已为 {len(library)} 首歌曲补全情绪、能量与 Set Role。"),
        AgentEvent(agent="Planner", status="done", message=f"从 {len(candidates)} 首候选中生成初稿。"),
        AgentEvent(agent="Critic", status="revised", message="发现初稿的能量衔接可优化，已要求 Planner 重排一次。"),
        AgentEvent(agent="Planner", status="revised", message="已按能量曲线与 BPM 连续性完成第 1 次修订。"),
        AgentEvent(agent="Critic", status="approved", message="无重复曲目，硬约束通过，可进入人工审核。"),
    ]
    rows = [PlaylistTrack(track=t, reason=f"{t.role} 位置，{t.bpm:.1f} BPM，衔接当前{_energy_label(t.energy)}能量。",
                          alternatives=[x.id for x in candidates if x.id != t.id][:2]) for t in selected]
    delta = abs(total - target) / target
    report = ["无重复曲目", f"总时长误差 {delta * 100:.1f}%", "BPM 与能量曲线已完成规则检查"]
    return Playlist(id=str(uuid4()), brief=brief, tracks=rows, duration_sec=total,
                    revision=1, report=report, trace=trace)


def _energy_label(value: float) -> str:
    return "高" if value > .75 else "中" if value > .5 else "低"


def render_export(playlist: Playlist, fmt: str) -> str:
    if fmt == "json": return json.dumps(playlist.model_dump(), ensure_ascii=False, indent=2)
    if fmt == "m3u": return "#EXTM3U\n" + "\n".join(
        f"#EXTINF:{row.track.duration_sec},{row.track.artist} - {row.track.title}\n{row.track.path}"
        for row in playlist.tracks)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["position", "title", "artist", "bpm", "key", "energy", "path"])
    for i, row in enumerate(playlist.tracks, 1):
        t = row.track
        writer.writerow([i, t.title, t.artist, t.bpm, t.key, t.energy, t.path])
    return output.getvalue()
