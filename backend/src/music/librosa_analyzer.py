"""Librosa-based audio feature extraction for DJ metadata and text description input."""

from importlib.metadata import version

import numpy as np

from backend.models import Track


SAMPLE_RATE = 22050
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_CAMELOT = ("8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B")
MINOR_CAMELOT = ("5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_audio(path: str, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa 未安装；请安装 requirements-audio.txt。") from exc
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < sr or not np.isfinite(audio).all():
        raise ValueError("音频不足一秒或包含无效采样")
    if float(np.max(np.abs(audio))) < 1e-7:
        raise ValueError("静音音频无法进行音乐分析")
    return audio, sr


def _key_from_chroma(chroma: np.ndarray) -> tuple[str, str, float]:
    values = np.mean(chroma, axis=1)
    norm = float(np.linalg.norm(values))
    if norm < 1e-8:
        return "Unknown", "unknown", 0.0
    values = values / norm
    candidates: list[tuple[float, int, str]] = []
    for tonic in range(12):
        candidates.append((float(values @ (np.roll(MAJOR_PROFILE, tonic) / np.linalg.norm(MAJOR_PROFILE))),
                           tonic, "major"))
        candidates.append((float(values @ (np.roll(MINOR_PROFILE, tonic) / np.linalg.norm(MINOR_PROFILE))),
                           tonic, "minor"))
    candidates.sort(reverse=True)
    best, second = candidates[0], candidates[1]
    confidence = float(np.clip((best[0] - second[0]) * 4, 0, 1))
    return PITCHES[best[1]], best[2], confidence


class LibrosaAnalyzer:
    def analyze(self, track: Track) -> Track:
        import librosa

        audio, sr = load_audio(track.path)
        onset = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        beat_frames = np.asarray(beat_frames)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        key, scale, key_confidence = _key_from_chroma(chroma)
        tonic = PITCHES.index(key) if key in PITCHES else 0
        camelot = (MAJOR_CAMELOT if scale == "major" else MINOR_CAMELOT)[tonic]
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = float(np.mean(rms))
        rms_db = float(20 * np.log10(max(rms_mean, 1e-8)))
        energy = float(np.clip((rms_db + 38) / 30, 0, 1))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        onset_mean = float(np.mean(onset)) if onset.size else 0.0
        onset_std = float(np.std(onset)) if onset.size else 0.0
        tempo_confidence = onset_mean / max(onset_mean + onset_std, 1e-8)
        return track.model_copy(update={
            "duration_sec": round(float(librosa.get_duration(y=audio, sr=sr))),
            "bpm": round(bpm, 1), "key": f"{key} {scale}", "camelot_key": camelot,
            "energy": round(energy, 3), "bpm_confidence": round(tempo_confidence, 4),
            "key_confidence": round(key_confidence, 4),
            "analyzer": f"librosa:{version('librosa')}:dj-v1",
            "analysis_details": {
                "beat_positions": np.asarray(librosa.frames_to_time(beat_frames, sr=sr)).round(3).tolist(),
                "rms_dbfs": round(rms_db, 2), "spectral_centroid_hz": round(centroid, 2),
                "spectral_bandwidth_hz": round(bandwidth, 2), "spectral_rolloff_hz": round(rolloff, 2),
                "zero_crossing_rate": round(zcr, 5), "onset_strength": round(onset_mean, 4),
                "energy_method": "rms-dbfs-v1",
                "confidence_scale": "librosa-derived heuristics, not calibrated probabilities",
            },
            "analysis_status": "analyzed", "analysis_error": None,
        })
