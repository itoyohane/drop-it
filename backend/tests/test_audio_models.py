"""Model adapter contracts plus an opt-in real Essentia/CLAP smoke test."""

import os
import sys
from types import ModuleType

import numpy as np
import pytest

from backend.config import Settings
from backend.models import Track
from backend.music.clap_embedder import ClapEmbedder
from backend.music.essentia_analyzer import EssentiaAnalyzer


def test_essentia_maps_flat_keys_and_preserves_raw_confidence(monkeypatch):
    import backend.music.essentia_analyzer as adapter

    standard = ModuleType("essentia.standard")
    standard.RhythmExtractor2013 = lambda **kwargs: lambda y: (124, [0, .5], 3.2, [], [])
    standard.KeyExtractor = lambda **kwargs: lambda y: ("Bb", "minor", .7)
    standard.RMS = lambda: lambda y: .1
    root = ModuleType("essentia")
    root.standard = standard
    monkeypatch.setitem(sys.modules, "essentia", root)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard)
    monkeypatch.setattr(adapter, "version", lambda name: "test")
    monkeypatch.setattr(adapter, "load_audio", lambda path, sr: np.ones(sr * 2, dtype=np.float32))
    result = EssentiaAnalyzer().analyze(Track(id="a", title="A", artist="B", filename="a.wav", path="a.wav"))
    assert result.key == "A# minor" and result.camelot_key == "3A"
    assert result.bpm_confidence == 3.2
    assert result.analysis_details["rms_dbfs"] == -20
    assert result.duration_sec == 2
    assert result.analyzer == "essentia:test:dj-v1"


def test_clap_adapter_with_real_transformers_tensor_contract(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if int(transformers.__version__.split(".")[0]) != 4:
        pytest.skip("requires the project's transformers<5 dependency")
    torch = pytest.importorskip("torch")
    from transformers import ClapConfig, ClapFeatureExtractor, ClapModel
    from transformers.feature_extraction_utils import BatchFeature

    # Small random model validates the real inference API, not retrieval quality.
    model = ClapModel(ClapConfig(
        audio_config={"depths": [1, 1, 1, 1], "num_attention_heads": [1, 2, 4, 8],
                      "patch_embeds_hidden_size": 8, "hidden_size": 64},
        text_config={"vocab_size": 20, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4},
        projection_dim=512,
    )).eval()
    extractor = ClapFeatureExtractor(truncation="rand_trunc")

    class Processor:
        def __call__(self, *, audios=None, text=None, **kwargs):
            if audios is not None:
                assert len(audios) == 3
                assert all(len(clip) <= 480000 for clip in audios)
                return extractor(audios, **kwargs)
            return BatchFeature({"input_ids": torch.tensor([[0, 5, 2]]),
                                 "attention_mask": torch.ones(1, 3, dtype=torch.long)})

    embedder = ClapEmbedder(Settings(_env_file=None))
    embedder._model, embedder._processor = model, Processor()
    samples = np.sin(np.arange(48000 * 30, dtype=np.float32) / 50) * .1
    monkeypatch.setattr("backend.music.clap_embedder.load_audio", lambda path, sr: samples)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        audio, text = embedder.audio("test.wav"), embedder.text("dark electronic music")
    finally:
        torch.set_num_threads(previous_threads)
    assert audio.shape == text.shape == (512,)
    assert np.isclose(np.linalg.norm(audio), 1) and np.isclose(np.linalg.norm(text), 1)


@pytest.mark.skipif(os.getenv("DROPIT_TEST_AUDIO_MODELS") != "1",
                    reason="requires Essentia and cached pretrained CLAP weights")
def test_real_audio_models(tmp_path):
    import soundfile as sf

    sr = 44100
    t = np.arange(sr * 20) / sr
    audio = .05 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 261.63 * t))
    for start in range(0, len(audio), sr // 2):
        audio[start:start + 512] += .8 * np.hanning(512)
    path = tmp_path / "clicks.wav"
    sf.write(path, audio, sr)
    track = Track(id="clicks", title="Clicks", artist="Test", filename=path.name, path=str(path))
    result = EssentiaAnalyzer().analyze(track)
    assert 110 <= result.bpm <= 130
    assert result.analysis_status == "analyzed"
    # Use configured data_dir so this test sees explicitly downloaded weights.
    embedder = ClapEmbedder(Settings(DROPIT_DATA_DIR=os.getenv("DROPIT_TEST_MODEL_DATA_DIR", "data")))
    assert embedder.audio(str(path)).shape == (512,)
    assert embedder.text("rhythmic music").shape == (512,)
