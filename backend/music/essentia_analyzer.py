"""Essentia DJ descriptors. Semantic tags require separate, explicitly selected models."""

from importlib.metadata import version

import numpy as np

from backend.models import Track


PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR = ("8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B")
MINOR = ("5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A")
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    try:
        import essentia.standard as es
    except ImportError as exc:
        raise RuntimeError("Essentia 未安装；请使用 Python 3.11 Linux/Docker 音频环境。") from exc
    audio = es.MonoLoader(filename=path, sampleRate=sample_rate)()
    if len(audio) < sample_rate or not np.isfinite(audio).all():
        raise ValueError("音频不足一秒或包含无效采样")
    if np.max(np.abs(audio)) < 1e-7:
        raise ValueError("静音音频无法进行音乐分析")
    return audio


class EssentiaAnalyzer:
    def analyze(self, track: Track) -> Track:
        audio = load_audio(track.path, 44100)
        import essentia.standard as es

        bpm, beats, confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
        key, scale, strength = es.KeyExtractor(sampleRate=44100)(audio)
        key = FLATS.get(key, key)
        camelot = (MAJOR if scale == "major" else MINOR)[PITCHES.index(key)]
        rms = float(es.RMS()(audio))
        rms_db = float(20 * np.log10(max(rms, 1e-8)))
        # A transparent loudness proxy, not a learned mood/danceability score.
        energy = float(np.clip((rms_db + 38) / 30, 0, 1))
        return track.model_copy(update={
            "duration_sec": round(len(audio) / 44100), "bpm": round(float(bpm), 1),
            "key": f"{key} {scale}", "camelot_key": camelot, "energy": round(energy, 3),
            "bpm_confidence": float(confidence), "key_confidence": float(strength),
            "analyzer": f"essentia:{version('essentia')}:dj-v1",
            "analysis_details": {
                "beat_positions": np.asarray(beats).round(3).tolist(),
                "rms_dbfs": round(rms_db, 2), "energy_method": "rms-dbfs-v1",
                "confidence_scale": "raw Essentia strengths, not probabilities",
            },
            "analysis_status": "analyzed", "analysis_error": None,
        })
