"""Deterministic validation for generated DJ sets.

The planner is intentionally heuristic.  This module is the final authority
before a Playlist can be persisted.  It has no model or database dependency so
the same Playlist and constraints always produce the same result.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from backend.models import Playlist, Track


DEFAULT_DURATION_TOLERANCE = 0.10
DEFAULT_MAX_BPM_TRANSITION = 8.0


def target_energy_for_curve(curve: str, position: int, count: int,
                            low: float, high: float) -> float:
    """Shared planner/repair definition for Set energy shapes."""

    ratio = position / max(1, count - 1)
    if curve == "steady":
        return (low + high) / 2
    if curve == "wave":
        import math
        return low + (high - low) * (.5 + .5 * math.sin(ratio * math.pi * 2 - math.pi / 2))
    if curve == "peak":
        peak_ratio = .7
        if ratio <= peak_ratio:
            return low + (high - low) * (ratio / peak_ratio)
        return high - (high - low) * .35 * ((ratio - peak_ratio) / (1 - peak_ratio))
    return low + (high - low) * ratio


class SetIssue(BaseModel):
    """One actionable constraint violation."""

    code: str
    position: int | None = None
    message: str
    track_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SetValidationResult(BaseModel):
    """Structured validation output used by Graph state and API evidence."""

    valid: bool
    issues: list[SetIssue] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)

    @property
    def failed_codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


def _camelot_parts(value: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"(\d{1,2})([AB])", (value or "").strip().upper())
    if not match:
        return None
    number = int(match.group(1))
    if not 1 <= number <= 12:
        return None
    return number, match.group(2)


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
    return 1 + ring * 0.25 + (mode_a != mode_b) * 0.3


def is_camelot_compatible(first: str, second: str) -> bool:
    first_parts, second_parts = _camelot_parts(first), _camelot_parts(second)
    if not first_parts or not second_parts:
        return False
    number_a, mode_a = first_parts
    number_b, mode_b = second_parts
    same_number = number_a == number_b
    adjacent_number = min((number_a - number_b) % 12, (number_b - number_a) % 12) == 1
    return same_number or (adjacent_number and mode_a == mode_b)


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
        rows = playlist.tracks
        tracks = [row.track for row in rows]
        allowed = {str(track_id) for track_id in allowed_track_ids} if allowed_track_ids is not None else None
        required = _track_ids(required_tracks)
        issues: list[SetIssue] = []

        # Membership is checked against the server-selected candidate/catalog set.
        if allowed is not None:
            for position, track in enumerate(tracks, 1):
                if track.id not in allowed:
                    issues.append(SetIssue(
                        code="track_membership",
                        position=position,
                        message=f"第 {position} 首曲目不属于当前项目曲库或候选范围",
                        track_ids=[track.id],
                    ))

        seen: dict[str, int] = {}
        for position, track in enumerate(tracks, 1):
            if track.id in seen:
                issues.append(SetIssue(
                    code="duplicate_tracks",
                    position=position,
                    message=f"第 {position} 首曲目与第 {seen[track.id]} 首重复",
                    track_ids=[track.id],
                ))
            else:
                seen[track.id] = position

        for position, track in enumerate(tracks, 1):
            if not playlist.brief.bpm_min <= track.bpm <= playlist.brief.bpm_max:
                issues.append(SetIssue(
                    code="bpm_range",
                    position=position,
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
                    code="bpm_transition",
                    position=position,
                    message=(f"第 {position} 首到第 {position + 1} 首 BPM 相差 "
                             f"{bpm_delta:.1f}，超过 {self.max_bpm_transition:.1f}"),
                    track_ids=[previous.id, current.id],
                    details={"bpm_delta": bpm_delta, "max_bpm_transition": self.max_bpm_transition},
                ))
            if not is_camelot_compatible(previous.camelot_key, current.camelot_key):
                issues.append(SetIssue(
                    code="camelot_compatibility",
                    position=position,
                    message=f"第 {position} 首到第 {position + 1} 首调性不兼容",
                    track_ids=[previous.id, current.id],
                    details={"from": previous.camelot_key, "to": current.camelot_key},
                ))

        energy_issues = self._energy_issues(tracks, playlist.brief.energy)
        issues.extend(energy_issues)

        present = {track.id for track in tracks}
        missing_required = sorted(required - present)
        if missing_required:
            issues.append(SetIssue(
                code="required_tracks",
                message=f"缺少用户要求的曲目：{', '.join(missing_required)}",
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
        if curve == "steady":
            if max(energies) - min(energies) > self.energy_tolerance:
                violations.append(1)
        elif curve == "build":
            violations = [index for index, (a, b) in enumerate(zip(energies, energies[1:]), 1)
                          if b + self.energy_tolerance / 2 < a]
            if energies[-1] + self.energy_tolerance / 2 < energies[0]:
                violations.append(len(energies) - 1)
        elif curve == "peak":
            peak_index = max(range(len(energies)), key=lambda index: (energies[index], index))
            rises = all(b + self.energy_tolerance / 2 >= a
                        for a, b in zip(energies[:peak_index], energies[1:peak_index + 1]))
            falls = all(b <= a + self.energy_tolerance / 2
                        for a, b in zip(energies[peak_index:], energies[peak_index + 1:]))
            has_shape = energies[peak_index] - min(energies) >= self.energy_tolerance / 2
            if peak_index < len(energies) // 2 or not rises or not falls or not has_shape:
                violations.append(max(1, peak_index + 1))
        elif curve == "wave":
            changes = [b - a for a, b in zip(energies, energies[1:])]
            if not (any(change > self.energy_tolerance / 2 for change in changes)
                    and any(change < -self.energy_tolerance / 2 for change in changes)):
                violations.append(1)
        if not violations:
            return []
        return [SetIssue(
            code="energy_curve",
            position=min(violations),
            message=f"能量曲线 {curve} 不满足确定性曲线约束",
            track_ids=[track.id for track in tracks],
            details={"curve": curve, "energies": energies},
        )]


__all__ = [
    "DEFAULT_DURATION_TOLERANCE", "DEFAULT_MAX_BPM_TRANSITION", "SetIssue",
    "SetValidationResult", "SetValidator", "camelot_distance", "is_camelot_compatible",
    "target_energy_for_curve",
]
