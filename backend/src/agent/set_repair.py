"""Deterministic, bounded repair for invalid DJ Sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from backend.agent.set_validation import (
    SetIssue, SetValidator, is_camelot_compatible, target_energy_for_curve,
)
from backend.models import AgentEvent, Playlist, PlaylistTrack, Track


@dataclass(frozen=True)
class SetRepairResult:
    playlist: Playlist
    changed: bool
    repaired_codes: tuple[str, ...]


class SetRepairer:
    """Repair only local, deterministic constraints; never relax a command."""

    def __init__(self, *, max_rounds: int = 2, validator: SetValidator | None = None) -> None:
        if max_rounds < 1:
            raise ValueError("Set 修复轮数必须为正数")
        self.max_rounds = min(max_rounds, 2)
        self.validator = validator or SetValidator()

    def repair(self, playlist: Playlist, issues: Iterable[SetIssue],
               candidates: Iterable[Track], *, required_tracks: Iterable[str] | None = None,
               attempt: int = 1) -> SetRepairResult:
        """Produce one repaired candidate.  The caller owns the retry loop."""

        if attempt < 1 or attempt > self.max_rounds:
            return SetRepairResult(playlist=playlist, changed=False, repaired_codes=())

        issue_list = list(issues)
        issue_codes = tuple(dict.fromkeys(issue.code for issue in issue_list))
        candidate_list = self._stable_candidates(candidates, playlist)
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

        # Required tracks are never dropped.  If one is outside the allowed
        # candidate set, the next validation round reports the conflict.
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
            next_track = min(additions, key=lambda track: self._addition_score(selected, track, playlist))
            selected.append(next_track)
            seen.add(next_track.id)

        # Remove optional tail tracks when the target is exceeded.  A removal
        # is accepted only when it gets closer to the requested duration.
        while (sum(track.duration_sec for track in selected) > target * (1 + self.validator.duration_tolerance)
               and len(selected) > len(required)):
            removable = [track for track in selected if track.id not in required]
            if not removable:
                break
            before = abs(sum(track.duration_sec for track in selected) - target)
            best = min(removable, key=lambda track: (abs(before - track.duration_sec), track.id))
            selected.remove(best)

        ordered = self._order(selected, playlist.brief.energy, required)
        repaired = self._rebuild(playlist, ordered)
        changed = [row.track.id for row in repaired.tracks] != [row.track.id for row in playlist.tracks]
        return SetRepairResult(playlist=repaired, changed=changed, repaired_codes=issue_codes)

    @staticmethod
    def _stable_candidates(candidates: Iterable[Track], playlist: Playlist) -> list[Track]:
        # Preserve planner order as a tie breaker while making the result
        # independent of database row ordering.
        unique: dict[str, Track] = {}
        for track in candidates:
            unique.setdefault(track.id, track)
        return sorted(unique.values(), key=lambda track: (track.id, track.title, track.artist))

    def _addition_score(self, selected: list[Track], candidate: Track, playlist: Playlist) -> tuple[float, str]:
        if not selected:
            return (abs(candidate.energy - .5), candidate.id)
        previous = selected[-1]
        bpm_cost = abs(previous.bpm - candidate.bpm)
        key_cost = 0 if is_camelot_compatible(previous.camelot_key, candidate.camelot_key) else 100
        return (key_cost + bpm_cost + abs(previous.energy - candidate.energy), candidate.id)

    def _order(self, tracks: list[Track], curve: str, required: set[str]) -> list[Track]:
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
            alternatives = old.alternatives if old else []
            rows.append(PlaylistTrack(track=track, reason=reason, alternatives=alternatives))
        total = sum(track.duration_sec for track in tracks)
        trace = [*playlist.trace]
        trace.append(AgentEvent(
            agent="Set Repairer", status="revised", message="按确定性约束修复了曲目顺序或候选。"
        ))
        return playlist.model_copy(update={
            "tracks": rows,
            "duration_sec": total,
            "trace": trace,
        })


__all__ = ["SetRepairResult", "SetRepairer"]
