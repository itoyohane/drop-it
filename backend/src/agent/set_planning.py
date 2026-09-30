"""Deterministic DJ Set planning, validation, repair, and export."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.models import (
    AgentEvent,
    Brief,
    Playlist,
    PlaylistTrack,
    Track,
    derive_agent_playlist_id,
)

if TYPE_CHECKING:
    from backend.repositories import DropItStore


DEFAULT_DURATION_TOLERANCE = 0.10
DEFAULT_MAX_BPM_TRANSITION = 8.0


def target_energy_for_curve(curve: str, position: int, count: int,
                            low: float, high: float) -> float:
    """Return the target energy for one position in a named curve."""

    ratio = position / max(1, count - 1)
    if curve == "steady":
        return (low + high) / 2
    if curve == "wave":
        return low + (high - low) * (.5 + .5 * math.sin(
            ratio * math.pi * 2 - math.pi / 2
        ))
    if curve == "peak":
        peak_ratio = .7
        if ratio <= peak_ratio:
            return low + (high - low) * (ratio / peak_ratio)
        return high - (high - low) * .35 * ((ratio - peak_ratio) / (1 - peak_ratio))
    return low + (high - low) * ratio


def _camelot_parts(value: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"(\d{1,2})([AB])", (value or "").strip().upper())
    if not match:
        return None
    number = int(match.group(1))
    return (number, match.group(2)) if 1 <= number <= 12 else None


def camelot_distance(first: str, second: str) -> float:
    """Return a stable distance on the Camelot wheel."""

    first_parts, second_parts = _camelot_parts(first), _camelot_parts(second)
    if not first_parts or not second_parts:
        return 1.5
    number_a, mode_a = first_parts
    number_b, mode_b = second_parts
    ring = min((number_a - number_b) % 12, (number_b - number_a) % 12)
    if ring == 0 and mode_a == mode_b:
        return 0.0
    if ring == 0 or (ring == 1 and mode_a == mode_b):
        return 0.2
    return 1 + ring * .25 + (mode_a != mode_b) * .3


def is_camelot_compatible(first: str, second: str) -> bool:
    first_parts, second_parts = _camelot_parts(first), _camelot_parts(second)
    if not first_parts or not second_parts:
        return False
    number_a, mode_a = first_parts
    number_b, mode_b = second_parts
    same_number = number_a == number_b
    adjacent_number = min((number_a - number_b) % 12, (number_b - number_a) % 12) == 1
    return same_number or (adjacent_number and mode_a == mode_b)


class SetIssue(BaseModel):
    """One actionable Set constraint violation."""

    code: str
    position: int | None = None
    message: str
    track_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SetValidationResult(BaseModel):
    """Structured validation output used by graph state and API evidence."""

    valid: bool
    issues: list[SetIssue] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)

    @property
    def failed_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


def _track_ids(value: Iterable[Track | str] | None) -> set[str]:
    if not value:
        return set()
    return {item.id if isinstance(item, Track) else str(item) for item in value}


class SetValidator:
    """Validate every hard Set constraint without changing the Playlist."""

    def __init__(self, *, duration_tolerance: float = DEFAULT_DURATION_TOLERANCE,
                 max_bpm_transition: float = DEFAULT_MAX_BPM_TRANSITION,
                 energy_tolerance: float = 0.20) -> None:
        if duration_tolerance < 0 or max_bpm_transition < 0 or energy_tolerance < 0:
            raise ValueError("Set 校验阈值不能为负数")
        self.duration_tolerance = duration_tolerance
        self.max_bpm_transition = max_bpm_transition
        self.energy_tolerance = energy_tolerance

    def validate(self, playlist: Playlist, allowed_track_ids: Iterable[str] | None = None,
                 required_tracks: Iterable[Track | str] | None = None) -> SetValidationResult:
        tracks = [row.track for row in playlist.tracks]
        allowed = {str(track_id) for track_id in allowed_track_ids} if allowed_track_ids is not None else None
        required = _track_ids(required_tracks)
        issues: list[SetIssue] = []

        if allowed is not None:
            for position, track in enumerate(tracks, 1):
                if track.id not in allowed:
                    issues.append(SetIssue(
                        code="track_membership", position=position,
                        message=f"第 {position} 首曲目不属于当前项目曲库或候选范围",
                        track_ids=[track.id],
                    ))

        seen: dict[str, int] = {}
        for position, track in enumerate(tracks, 1):
            if track.id in seen:
                issues.append(SetIssue(
                    code="duplicate_tracks", position=position,
                    message=f"第 {position} 首曲目与第 {seen[track.id]} 首重复",
                    track_ids=[track.id],
                ))
            else:
                seen[track.id] = position

        for position, track in enumerate(tracks, 1):
            if not playlist.brief.bpm_min <= track.bpm <= playlist.brief.bpm_max:
                issues.append(SetIssue(
                    code="bpm_range", position=position,
                    message=(f"第 {position} 首 BPM {track.bpm:.1f} 不在 "
                             f"{playlist.brief.bpm_min}-{playlist.brief.bpm_max} 范围内"),
                    track_ids=[track.id],
                    details={"bpm": track.bpm, "bpm_min": playlist.brief.bpm_min,
                             "bpm_max": playlist.brief.bpm_max},
                ))

        target = playlist.brief.duration_min * 60
        actual_duration = sum(max(0, track.duration_sec) for track in tracks)
        if playlist.duration_sec != actual_duration:
            issues.append(SetIssue(
                code="duration_summary_mismatch",
                message=(f"Playlist 时长摘要为 {playlist.duration_sec} 秒，"
                         f"曲目实际合计为 {actual_duration} 秒"),
                track_ids=[track.id for track in tracks],
                details={"summary_duration_sec": playlist.duration_sec,
                         "actual_duration_sec": actual_duration},
            ))
        duration_error = abs(actual_duration - target) / target if target else 0.0
        if duration_error > self.duration_tolerance:
            issues.append(SetIssue(
                code="duration_tolerance",
                message=(f"Set 实际时长 {actual_duration // 60} 分钟与目标 "
                         f"{playlist.brief.duration_min} 分钟相差 {duration_error * 100:.1f}%"),
                details={"duration_sec": actual_duration, "target_sec": target,
                         "summary_duration_sec": playlist.duration_sec,
                         "relative_error": duration_error,
                         "tolerance": self.duration_tolerance},
            ))

        for position, (previous, current) in enumerate(zip(tracks, tracks[1:]), 1):
            bpm_delta = abs(previous.bpm - current.bpm)
            if bpm_delta > self.max_bpm_transition:
                issues.append(SetIssue(
                    code="bpm_transition", position=position,
                    message=(f"第 {position} 首到第 {position + 1} 首 BPM 相差 "
                             f"{bpm_delta:.1f}，超过 {self.max_bpm_transition:.1f}"),
                    track_ids=[previous.id, current.id],
                    details={"bpm_delta": bpm_delta,
                             "max_bpm_transition": self.max_bpm_transition},
                ))
            if not is_camelot_compatible(previous.camelot_key, current.camelot_key):
                issues.append(SetIssue(
                    code="camelot_compatibility", position=position,
                    message=f"第 {position} 首到第 {position + 1} 首调性不兼容",
                    track_ids=[previous.id, current.id],
                    details={"from": previous.camelot_key, "to": current.camelot_key},
                ))

        issues.extend(self._energy_issues(tracks, playlist.brief.energy))
        missing_required = sorted(required - {track.id for track in tracks})
        if missing_required:
            issues.append(SetIssue(
                code="required_tracks", message=f"缺少用户要求的曲目：{', '.join(missing_required)}",
                track_ids=missing_required,
            ))

        failed_codes = {issue.code for issue in issues}
        checks = {
            "track_membership": "track_membership" not in failed_codes,
            "duplicate_tracks": "duplicate_tracks" not in failed_codes,
            "bpm_range": "bpm_range" not in failed_codes,
            "duration_tolerance": "duration_tolerance" not in failed_codes,
            "duration_summary": "duration_summary_mismatch" not in failed_codes,
            "bpm_transition": "bpm_transition" not in failed_codes,
            "camelot_compatibility": "camelot_compatibility" not in failed_codes,
            "energy_curve": "energy_curve" not in failed_codes,
            "required_tracks": "required_tracks" not in failed_codes,
        }
        return SetValidationResult(valid=not issues, issues=issues, checks=checks)

    def _energy_issues(self, tracks: list[Track], curve: str) -> list[SetIssue]:
        if len(tracks) < 2:
            return []
        energies = [float(track.energy) for track in tracks]
        violations: list[int] = []
        tolerance = self.energy_tolerance / 2
        if curve == "steady":
            if max(energies) - min(energies) > self.energy_tolerance:
                violations.append(1)
        elif curve == "build":
            violations = [index for index, (a, b) in enumerate(zip(energies, energies[1:]), 1)
                          if b + tolerance < a]
            if energies[-1] + tolerance < energies[0]:
                violations.append(len(energies) - 1)
        elif curve == "peak":
            peak_index = max(range(len(energies)), key=lambda index: (energies[index], index))
            rises = all(b + tolerance >= a for a, b in zip(
                energies[:peak_index], energies[1:peak_index + 1]
            ))
            falls = all(b <= a + tolerance for a, b in zip(
                energies[peak_index:], energies[peak_index + 1:]
            ))
            has_shape = energies[peak_index] - min(energies) >= tolerance
            if peak_index < len(energies) // 2 or not rises or not falls or not has_shape:
                violations.append(max(1, peak_index + 1))
        elif curve == "wave":
            changes = [b - a for a, b in zip(energies, energies[1:])]
            if not (any(change > tolerance for change in changes)
                    and any(change < -tolerance for change in changes)):
                violations.append(1)
        if not violations:
            return []
        return [SetIssue(
            code="energy_curve", position=min(violations),
            message=f"能量曲线 {curve} 不满足确定性曲线约束",
            track_ids=[track.id for track in tracks],
            details={"curve": curve, "energies": energies},
        )]


@dataclass(frozen=True)
class SetRepairResult:
    playlist: Playlist
    changed: bool
    repaired_codes: tuple[str, ...]


class SetRepairer:
    """Repair local deterministic constraints; never relax a command."""

    def __init__(self, *, max_rounds: int = 2, validator: SetValidator | None = None) -> None:
        if max_rounds < 1:
            raise ValueError("Set 修复轮数必须为正数")
        self.max_rounds = min(max_rounds, 2)
        self.validator = validator or SetValidator()

    def repair(self, playlist: Playlist, issues: Iterable[SetIssue],
               candidates: Iterable[Track], *, required_tracks: Iterable[str] | None = None,
               attempt: int = 1) -> SetRepairResult:
        if attempt < 1 or attempt > self.max_rounds:
            return SetRepairResult(playlist=playlist, changed=False, repaired_codes=())

        issue_list = list(issues)
        issue_codes = tuple(dict.fromkeys(issue.code for issue in issue_list))
        candidate_list = self._stable_candidates(candidates)
        allowed = {track.id: track for track in candidate_list}
        required = {str(track_id) for track_id in (required_tracks or [])}

        selected: list[Track] = []
        seen: set[str] = set()
        for row in playlist.tracks:
            track = allowed.get(row.track.id)
            if track is None or track.id in seen:
                continue
            if not playlist.brief.bpm_min <= track.bpm <= playlist.brief.bpm_max:
                continue
            selected.append(track)
            seen.add(track.id)

        for track_id in sorted(required):
            track = allowed.get(track_id)
            if track is not None and track.id not in seen:
                selected.append(track)
                seen.add(track.id)

        target = playlist.brief.duration_min * 60
        while sum(track.duration_sec for track in selected) < target * (1 - self.validator.duration_tolerance):
            additions = [track for track in candidate_list if track.id not in seen]
            if not additions:
                break
            next_track = min(additions, key=lambda track: self._addition_score(selected, track))
            selected.append(next_track)
            seen.add(next_track.id)

        while (sum(track.duration_sec for track in selected) > target * (1 + self.validator.duration_tolerance)
               and len(selected) > len(required)):
            removable = [track for track in selected if track.id not in required]
            if not removable:
                break
            before = abs(sum(track.duration_sec for track in selected) - target)
            best = min(removable, key=lambda track: (abs(before - track.duration_sec), track.id))
            selected.remove(best)

        ordered = self._order(selected, playlist.brief.energy)
        repaired = self._rebuild(playlist, ordered)
        changed = [row.track.id for row in repaired.tracks] != [row.track.id for row in playlist.tracks]
        return SetRepairResult(playlist=repaired, changed=changed, repaired_codes=issue_codes)

    @staticmethod
    def _stable_candidates(candidates: Iterable[Track]) -> list[Track]:
        unique: dict[str, Track] = {}
        for track in candidates:
            unique.setdefault(track.id, track)
        return sorted(unique.values(), key=lambda track: (track.id, track.title, track.artist))

    def _addition_score(self, selected: list[Track], candidate: Track) -> tuple[float, str]:
        if not selected:
            return (abs(candidate.energy - .5), candidate.id)
        previous = selected[-1]
        key_cost = 0 if is_camelot_compatible(previous.camelot_key, candidate.camelot_key) else 100
        return (key_cost + abs(previous.bpm - candidate.bpm)
                + abs(previous.energy - candidate.energy), candidate.id)

    def _order(self, tracks: list[Track], curve: str) -> list[Track]:
        if len(tracks) < 2:
            return tracks
        remaining = tracks[:]
        ordered: list[Track] = []
        energies = [track.energy for track in tracks]
        low, high = min(energies), max(energies)
        while remaining:
            previous = ordered[-1] if ordered else None
            compatible = [track for track in remaining if previous is None or (
                abs(previous.bpm - track.bpm) <= self.validator.max_bpm_transition
                and is_camelot_compatible(previous.camelot_key, track.camelot_key)
            )]
            pool = compatible or remaining
            desired = target_energy_for_curve(curve, len(ordered), len(tracks), low, high)
            selected = min(pool, key=lambda track: (
                abs(track.energy - desired),
                abs(previous.bpm - track.bpm) if previous else 0,
                track.id,
            ))
            ordered.append(selected)
            remaining.remove(selected)
        return ordered

    @staticmethod
    def _rebuild(playlist: Playlist, tracks: list[Track]) -> Playlist:
        old_rows = {row.track.id: row for row in playlist.tracks}
        rows: list[PlaylistTrack] = []
        for position, track in enumerate(tracks):
            previous = tracks[position - 1] if position else None
            old = old_rows.get(track.id)
            reason = old.reason if old else (
                "作为开场锚点" if previous is None
                else f"与上一首相差 {track.bpm - previous.bpm:+.1f} BPM；确定性修复排序。"
            )
            rows.append(PlaylistTrack(track=track, reason=reason,
                                      alternatives=old.alternatives if old else []))
        trace = [*playlist.trace, AgentEvent(
            agent="Set Repairer", status="revised", message="按确定性约束修复了曲目顺序或候选。"
        )]
        return playlist.model_copy(update={
            "tracks": rows,
            "duration_sec": sum(track.duration_sec for track in tracks),
            "trace": trace,
        })


class SetConstraintConflictError(ValueError):
    """Raised when bounded deterministic repair cannot satisfy a Set."""

    def __init__(self, result: SetValidationResult, attempts: int):
        self.result = result
        self.attempts = attempts
        suggestions_by_code = {
            "track_membership": "仅使用当前项目曲库中的候选曲目",
            "duplicate_tracks": "允许增加候选曲目数量",
            "bpm_range": "放宽 BPM 范围",
            "duration_tolerance": "放宽目标时长或时长误差",
            "duration_summary_mismatch": "重新生成 Playlist 时长摘要",
            "bpm_transition": "放宽相邻 BPM 转场阈值",
            "camelot_compatibility": "放宽 Camelot 调性兼容要求",
            "energy_curve": "放宽能量曲线要求",
            "required_tracks": "减少或更换必选曲目",
        }
        self.issues = [issue.model_dump(mode="json") for issue in result.issues]
        self.suggestions = list(dict.fromkeys(
            suggestions_by_code.get(issue.code, f"放宽 {issue.code} 条件")
            for issue in result.issues
        ))
        codes = ", ".join(dict.fromkeys(issue.code for issue in result.issues))
        super().__init__(
            f"Set 约束冲突（最多修复 {attempts} 轮仍失败）：{codes}。"
            f"建议：{'；'.join(self.suggestions)}。"
        )

    def api_detail(self) -> dict[str, object]:
        return {
            "code": "constraint_conflict",
            "issues": self.issues,
            "suggestions": self.suggestions,
            "repair_attempts": self.attempts,
        }


def generate_playlist(project_id: str, brief: Brief, library: list[Track],
                      *, playlist_id: str | None = None) -> Playlist:
    if brief.bpm_min > brief.bpm_max:
        raise ValueError("BPM 下限不能高于上限")
    candidates = [track for track in library if track.analysis_status == "analyzed"
                  and brief.bpm_min <= track.bpm <= brief.bpm_max]
    if not candidates:
        raise ValueError("项目曲库里没有满足 BPM 范围且已完成分析的歌曲")
    target = brief.duration_min * 60
    ordered = _order_for_curve(candidates, brief.energy, target)
    selected: list[Track] = []
    total = 0
    for track in ordered:
        if total >= target * .94:
            break
        selected.append(track)
        total += track.duration_sec
    rows = []
    for index, track in enumerate(selected):
        previous = selected[index - 1] if index else None
        alternatives = [item.id for item in candidates
                        if item.id != track.id and abs(item.bpm - track.bpm) <= 3][:2]
        rows.append(PlaylistTrack(track=track, reason=_selection_reason(track, previous, brief.energy),
                                  alternatives=alternatives))
    duration_error = abs(total - target) / target if target else 0
    bpm_jumps = sum(abs(a.bpm - b.bpm) > 8 for a, b in zip(selected, selected[1:]))
    harmonic_breaks = sum(not is_camelot_compatible(a.camelot_key, b.camelot_key)
                          for a, b in zip(selected, selected[1:]))
    trace = [
        AgentEvent(agent="Library", status="done", message=f"从项目曲库取得 {len(candidates)} 首候选曲目。"),
        AgentEvent(agent="Set Planner", status="done", message=f"按 {brief.energy} 能量曲线编排 {len(rows)} 首曲目。"),
        AgentEvent(agent="Transition Rules", status="failed" if bpm_jumps or harmonic_breaks else "approved",
                   message=f"检查重复、时长和 BPM 跳跃；发现 {bpm_jumps} 处大于 8 BPM 的跳跃。"),
    ]
    report = [
        f"Camelot 相邻兼容 {len(selected) - 1 - harmonic_breaks}/{max(0, len(selected) - 1)} 处",
        "无重复曲目", f"目标 {brief.duration_min} 分钟，当前误差 {duration_error * 100:.1f}%",
        f"相邻 BPM 大跳跃 {bpm_jumps} 处", "所有 track_id 均来自当前项目曲库",
    ]
    return Playlist(id=playlist_id or uuid4().hex, project_id=project_id, brief=brief,
                    tracks=rows, duration_sec=total, report=report, trace=trace)


def plan_set(
    project_id: str,
    candidates: list[Track],
    *,
    request: str,
    duration_min: int,
    bpm_min: float,
    bpm_max: float,
    energy_curve: str,
    style_query: str = "",
    playlist_id: str | None = None,
) -> Playlist:
    """Build one deterministic Set from explicit command fields."""

    brief = Brief(
        title=request[:60] or "Untitled set",
        duration_min=duration_min,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        energy=energy_curve,
        style=style_query,
        notes=request,
    )
    return generate_playlist(project_id, brief, candidates, playlist_id=playlist_id)


def validate_and_repair_set(
    playlist: Playlist,
    candidates: list[Track],
    *,
    required_tracks: Iterable[str] | None = None,
    validator: SetValidator | None = None,
    repairer: SetRepairer | None = None,
) -> tuple[Playlist, SetValidationResult, int]:
    """Validate and perform at most two deterministic repair rounds."""

    validator = validator or SetValidator()
    repairer = repairer or SetRepairer(validator=validator)
    required = list(required_tracks or [])
    allowed_ids = [track.id for track in candidates]
    result = validator.validate(playlist, allowed_ids, required_tracks=required)
    attempts = 0
    while not result.valid and attempts < repairer.max_rounds:
        attempts += 1
        playlist = repairer.repair(
            playlist,
            result.issues,
            candidates,
            required_tracks=required,
            attempt=attempts,
        ).playlist
        result = validator.validate(playlist, allowed_ids, required_tracks=required)
    if not result.valid:
        raise SetConstraintConflictError(result, attempts)
    return playlist, result, attempts


def persist_set(
    store: DropItStore,
    project_id: str,
    playlist: Playlist,
    *,
    run_id: str | None = None,
    owner: str | None = None,
    token: int | None = None,
) -> Playlist:
    """Validate project scope and persist a Set with idempotent agent-run semantics."""

    if playlist.project_id != project_id:
        raise ValueError("Set 项目范围与当前对话不一致。")
    result = SetValidator().validate(
        playlist, [track.id for track in store.all_tracks(project_id)]
    )
    if not result.valid:
        raise SetConstraintConflictError(result, 0)
    if run_id is not None:
        expected_id = derive_agent_playlist_id(project_id, run_id)
        if playlist.id != expected_id:
            raise ValueError("Agent Playlist ID 不是由 project_id+run_id 规范派生")
        if playlist.agent_run_id != run_id:
            raise ValueError("Agent Playlist payload 缺少匹配的 agent_run_id")
        if owner is None or token is None:
            raise ValueError("Agent Playlist 写入需要当前 claim owner/token")
        return store.save_agent_playlist(run_id, project_id, playlist, owner, token)
    existing = store.get_playlist(playlist.id)
    return existing or store.save_playlist(playlist)


def _order_for_curve(candidates: list[Track], curve: str, target_seconds: int) -> list[Track]:
    approximate_count = max(1, min(len(candidates), math.ceil(
        target_seconds / max(1, sum(track.duration_sec for track in candidates) / len(candidates))
    )))
    energies = sorted(track.energy for track in candidates)
    low = energies[max(0, len(energies) // 5)]
    high = energies[min(len(energies) - 1, len(energies) * 4 // 5)]
    remaining = candidates[:]
    ordered: list[Track] = []
    while remaining:
        previous = ordered[-1] if ordered else None
        compatible = [track for track in remaining if previous and
                      is_camelot_compatible(previous.camelot_key, track.camelot_key)]
        if previous and not compatible:
            break
        pool = compatible if previous else remaining
        desired = target_energy_for_curve(curve, len(ordered), approximate_count, low, high)

        def score(track: Track) -> float:
            energy_cost = abs(track.energy - desired) * 8
            if previous is None:
                return energy_cost + track.bpm / 1000
            bpm_cost = min(abs(track.bpm - previous.bpm), abs(track.bpm * 2 - previous.bpm),
                           abs(track.bpm - previous.bpm * 2)) / 5
            return energy_cost + bpm_cost + camelot_distance(previous.camelot_key, track.camelot_key)

        selected = min(pool, key=score)
        ordered.append(selected)
        remaining.remove(selected)
    return ordered


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


__all__ = [
    "DEFAULT_DURATION_TOLERANCE", "DEFAULT_MAX_BPM_TRANSITION", "SetConstraintConflictError",
    "SetIssue", "SetRepairResult", "SetRepairer", "SetValidationResult", "SetValidator",
    "camelot_distance", "generate_playlist", "is_camelot_compatible", "persist_set",
    "plan_set", "render_export", "target_energy_for_curve", "validate_and_repair_set",
]
