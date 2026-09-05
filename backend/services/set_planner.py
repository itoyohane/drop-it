import csv
import io
import json
import math
import re
from uuid import uuid4

from backend.models import AgentEvent, Brief, Playlist, PlaylistTrack, Track


def generate_playlist(project_id: str, brief: Brief, library: list[Track]) -> Playlist:
    if brief.bpm_min > brief.bpm_max:
        raise ValueError("BPM 下限不能高于上限")
    analyzed = [track for track in library if track.analysis_status == "analyzed"]
    candidates = [track for track in analyzed if brief.bpm_min <= track.bpm <= brief.bpm_max]
    if not candidates:
        raise ValueError("项目曲库里没有满足 BPM 范围且已完成分析的歌曲")

    target = brief.duration_min * 60
    # Order the full candidate pool first, then cut close to the requested duration.
    ordered = _order_for_curve(candidates, brief.energy, target)
    selected: list[Track] = []
    total = 0
    for track in ordered:
        if total >= target * .94:
            break
        selected.append(track)
        total += track.duration_sec

    rows: list[PlaylistTrack] = []
    for index, track in enumerate(selected):
        previous = selected[index - 1] if index else None
        reason = _selection_reason(track, previous, brief.energy)
        alternatives = [candidate.id for candidate in candidates
                        if candidate.id != track.id and abs(candidate.bpm - track.bpm) <= 3][:2]
        rows.append(PlaylistTrack(track=track, reason=reason, alternatives=alternatives))

    duration_error = abs(total - target) / target if target else 0
    bpm_jumps = sum(abs(a.bpm - b.bpm) > 8 for a, b in zip(selected, selected[1:]))
    harmonic_breaks = sum(
        not is_camelot_compatible(first.camelot_key, second.camelot_key)
        for first, second in zip(selected, selected[1:])
    )
    trace = [
        AgentEvent(agent="Library", status="done", message=f"从项目曲库取得 {len(candidates)} 首候选曲目。"),
        AgentEvent(agent="Set Planner", status="done", message=f"按 {brief.energy} 能量曲线编排 {len(rows)} 首曲目。"),
        AgentEvent(agent="Transition Rules", status="failed" if bpm_jumps or harmonic_breaks else "approved",
                   message=f"检查重复、时长和 BPM 跳跃；发现 {bpm_jumps} 处大于 8 BPM 的跳跃。"),
    ]
    report = [
        f"Camelot 相邻兼容 {len(selected) - 1 - harmonic_breaks}/{max(0, len(selected) - 1)} 处",
        "无重复曲目",
        f"目标 {brief.duration_min} 分钟，当前误差 {duration_error * 100:.1f}%",
        f"相邻 BPM 大跳跃 {bpm_jumps} 处",
        "所有 track_id 均来自当前项目曲库",
    ]
    return Playlist(id=uuid4().hex, project_id=project_id, brief=brief, tracks=rows,
                    duration_sec=total, report=report, trace=trace)


def _order_for_curve(candidates: list[Track], curve: str, target_seconds: int) -> list[Track]:
    """Greedy transition planner balancing energy shape, tempo and Camelot compatibility."""
    approximate_count = max(1, min(len(candidates), math.ceil(
        target_seconds / max(1, sum(track.duration_sec for track in candidates) / len(candidates))
    )))
    energies = sorted(track.energy for track in candidates)
    low, high = energies[max(0, len(energies) // 5)], energies[min(len(energies) - 1, len(energies) * 4 // 5)]

    def target_energy(position: int) -> float:
        # Translate the requested curve into an energy target for each expected set position.
        ratio = position / max(1, approximate_count - 1)
        if curve == "steady":
            return energies[len(energies) // 2]
        if curve == "wave":
            return low + (high - low) * (.5 + .5 * math.sin(ratio * math.pi * 2 - math.pi / 2))
        if curve == "peak":
            # Peak-time sets reach their high-energy section immediately.
            return high
        return low + (high - low) * ratio

    remaining = candidates[:]
    ordered: list[Track] = []
    while remaining:
        position = len(ordered)
        desired = target_energy(position)
        previous = ordered[-1] if ordered else None
        # A valid Camelot transition is a hard constraint for every adjacent pair.
        # For example, 1A may move to 1A, 12A, 2A, or 1B.
        compatible = [track for track in remaining if previous and is_camelot_compatible(
            previous.camelot_key, track.camelot_key
        )]
        if previous and not compatible:
            # Do not trade harmonic continuity for a longer but incompatible set.
            break
        pool = compatible if previous else remaining

        def score(track: Track) -> float:
            # Lower is better: honor energy shape while allowing half/double-time tempo relations.
            energy_cost = abs(track.energy - desired) * 8
            if previous is None:
                return energy_cost + track.bpm / 1000
            bpm_cost = min(abs(track.bpm - previous.bpm), abs(track.bpm * 2 - previous.bpm),
                           abs(track.bpm - previous.bpm * 2)) / 5
            key_cost = camelot_distance(previous.camelot_key, track.camelot_key)
            return energy_cost + bpm_cost + key_cost

        selected = min(pool, key=score)
        ordered.append(selected)
        remaining.remove(selected)
    return ordered


def camelot_distance(first: str, second: str) -> float:
    match_a = re.fullmatch(r"(\d{1,2})([AB])", (first or "").strip().upper())
    match_b = re.fullmatch(r"(\d{1,2})([AB])", (second or "").strip().upper())
    if not match_a or not match_b:
        return 1.5
    number_a, mode_a = int(match_a.group(1)), match_a.group(2)
    number_b, mode_b = int(match_b.group(1)), match_b.group(2)
    ring = min((number_a - number_b) % 12, (number_b - number_a) % 12)
    if ring == 0 and mode_a == mode_b:
        return 0
    if ring == 0 or (ring == 1 and mode_a == mode_b):
        return .2
    return 1 + ring * .25 + (mode_a != mode_b) * .3


def is_camelot_compatible(first: str, second: str) -> bool:
    """Return whether a transition stays on the Camelot wheel's safe DJ moves."""
    match_a = re.fullmatch(r"(\d{1,2})([AB])", (first or "").strip().upper())
    match_b = re.fullmatch(r"(\d{1,2})([AB])", (second or "").strip().upper())
    if not match_a or not match_b:
        return False
    number_a, mode_a = int(match_a.group(1)), match_a.group(2)
    number_b, mode_b = int(match_b.group(1)), match_b.group(2)
    same_number = number_a == number_b
    adjacent_number = min((number_a - number_b) % 12, (number_b - number_a) % 12) == 1
    return same_number or (adjacent_number and mode_a == mode_b)


def _selection_reason(track: Track, previous: Track | None, curve: str) -> str:
    energy = "高" if track.energy >= .72 else "中" if track.energy >= .45 else "低"
    if previous:
        transition = f"与上一首相差 {track.bpm - previous.bpm:+.1f} BPM"
        if previous.camelot_key == track.camelot_key:
            transition += "，同 Camelot 调性"
    else:
        transition = "作为开场锚点"
    return f"{transition}；{energy}能量，符合 {curve} 曲线。"


def render_export(playlist: Playlist, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(playlist.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if fmt == "m3u":
        return "#EXTM3U\n" + "\n".join(
            f"#EXTINF:{row.track.duration_sec},{row.track.artist} - {row.track.title}\n{row.track.path}"
            for row in playlist.tracks
        )
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["position", "title", "artist", "bpm", "key", "camelot", "energy", "path"])
    for index, row in enumerate(playlist.tracks, 1):
        track = row.track
        writer.writerow([index, track.title, track.artist, track.bpm, track.key,
                         track.camelot_key, track.energy, track.path])
    return output.getvalue()
