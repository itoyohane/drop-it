"""Contracts for librosa analysis, small description generation, and text embeddings."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import Settings
from backend.models import Track
from backend.music.librosa_analyzer import LibrosaAnalyzer, _key_from_chroma
from backend.music.text_models import (
    DashScopeTextEmbedder, DeepSeekTrackDescriptor, SentenceTransformerEmbedder, SmallTextDescriptor,
)


def test_key_detection_maps_pitch_to_camelot(monkeypatch):
    import backend.music.librosa_analyzer as adapter

    chroma = np.zeros((12, 4), dtype=np.float32)
    chroma[10] = 1
    monkeypatch.setattr(adapter, "load_audio", lambda path: (np.ones(44100, dtype=np.float32) * .1, 22050))
    monkeypatch.setattr(adapter, "version", lambda name: "test")
    fake_librosa = SimpleNamespace(
        onset=SimpleNamespace(onset_strength=lambda **kwargs: np.array([1., 2., 1.])),
        beat=SimpleNamespace(beat_track=lambda **kwargs: (np.array([124.]), np.array([0, 10]))),
        feature=SimpleNamespace(
            chroma_cqt=lambda **kwargs: chroma,
            rms=lambda **kwargs: np.array([[.1, .1]]),
            spectral_centroid=lambda **kwargs: np.array([[1800.]]),
            spectral_bandwidth=lambda **kwargs: np.array([[900.]]),
            spectral_rolloff=lambda **kwargs: np.array([[3000.]]),
            zero_crossing_rate=lambda audio: np.array([[.05]]),
        ),
        get_duration=lambda **kwargs: 2.,
        frames_to_time=lambda frames, **kwargs: np.array([0., .5]),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    result = LibrosaAnalyzer().analyze(
        Track(id="a", title="A", artist="B", filename="a.wav", path="a.wav")
    )
    key, scale, _ = _key_from_chroma(chroma)
    assert result.key == f"{key} {scale}"
    assert result.camelot_key.endswith("A" if scale == "minor" else "B")
    assert result.bpm == 124 and result.duration_sec == 2
    assert result.analysis_details["spectral_centroid_hz"] == 1800
    assert result.analyzer == "librosa:test:dj-v1"


def test_small_descriptor_keeps_measured_facts(monkeypatch):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            assert "124.0 BPM" in prompt and "do not invent" in prompt
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, **kwargs):
            return "steady mid-energy tonal track"

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[3, 4]])

    descriptor = SmallTextDescriptor(Settings(_env_file=None))
    descriptor._tokenizer, descriptor._model = Tokenizer(), Model()
    track = Track(id="a", title="A", artist="B", filename="a.wav", path="a.wav", bpm=124,
                  key="A minor", camelot_key="8A", energy=.6,
                  analysis_details={"spectral_centroid_hz": 1800, "onset_strength": 1.2})
    text = descriptor.describe(track)
    assert "124.0 BPM" in text and "energy 0.60" in text
    assert text.endswith("steady mid-energy tonal track")


def test_sentence_transformer_adapter_normalizes_and_checks_dimension(monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            assert kwargs["local_files_only"] is True

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, value, **kwargs):
            assert value == "dark electronic music"
            return np.array([3, 4, 0], dtype=np.float32)

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=Model))
    embedder = SentenceTransformerEmbedder(
        Settings(_env_file=None, DROPIT_TEXT_EMBEDDING_DIMENSIONS=3)
    )
    result = embedder.text("dark electronic music")
    assert np.allclose(result, [.6, .8, 0])


def test_deepseek_descriptor_uses_measured_features(monkeypatch):
    import httpx

    calls = []

    class Response:
        status_code = 200
        is_error = False

        def json(self):
            return {"choices": [{"message": {"content": "steady tonal texture"}}]}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    settings = Settings(_env_file=None, DEEPSEEK_DESCRIPTION_API_KEY="secret")
    descriptor = DeepSeekTrackDescriptor(settings)
    track = Track(id="a", title="A", artist="B", filename="a.wav", path="a.wav", bpm=124,
                  key="A minor", camelot_key="8A", energy=.6,
                  analysis_details={"spectral_centroid_hz": 1800, "onset_strength": 1.2})
    result = descriptor.describe(track)
    assert result.endswith("steady tonal texture")
    assert calls[0][0].endswith("/chat/completions")
    assert calls[0][1]["json"]["model"] == settings.description_model
    assert "124.0 BPM" in calls[0][1]["json"]["messages"][1]["content"]


def test_dashscope_embedder_parses_and_normalizes_openai_response(monkeypatch):
    import httpx

    class Response:
        status_code = 200
        is_error = False

        def json(self):
            return {"data": [{"index": 0, "embedding": [3, 4, 0]}]}

    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        return Response()

    monkeypatch.setattr(httpx, "post", post)
    settings = Settings(_env_file=None, DASHSCOPE_API_KEY="secret",
                        DROPIT_TEXT_EMBEDDING_DIMENSIONS=3)
    vector = DashScopeTextEmbedder(settings).text("dark electronic music")
    assert np.allclose(vector, [.6, .8, 0])
    assert captured["url"].endswith("/embeddings")
    assert captured["kwargs"]["json"]["model"] == "qwen3.7-text-embedding"


@pytest.mark.skipif(os.getenv("DROPIT_TEST_AUDIO_MODELS") != "1",
                    reason="requires cached pretrained description and embedding models")
def test_real_audio_models(tmp_path):
    import soundfile as sf

    sr = 22050
    t = np.arange(sr * 20) / sr
    audio = .05 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 261.63 * t))
    for start in range(0, len(audio), sr // 2):
        audio[start:start + 512] += .8 * np.hanning(512)
    path = tmp_path / "clicks.wav"
    sf.write(path, audio, sr)
    settings = Settings(DROPIT_DATA_DIR=os.getenv("DROPIT_TEST_MODEL_DATA_DIR", "data"))
    track = LibrosaAnalyzer().analyze(
        Track(id="clicks", title="Clicks", artist="Test", filename=path.name, path=str(path))
    )
    description = SmallTextDescriptor(settings).describe(track)
    vector = SentenceTransformerEmbedder(settings).text(description)
    assert track.analysis_status == "analyzed" and description
    assert vector.shape == (settings.text_embedding_dimensions,)
