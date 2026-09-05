import os
from pathlib import Path

import numpy as np
import pytest

# Prevent the module-level ASGI app from opening a developer's real database.
os.environ["DROPIT_ENV"] = "test"
os.environ["DROPIT_DATA_DIR"] = str(Path.cwd() / "data" / "test-bootstrap")
os.environ["DEEPSEEK_API_KEY"] = ""

from backend.models import Track
from backend.agent.retriever import RagLibrary
from backend.repositories import DropItStore


class FakeEmbedder:
    """Only for tests; production has no fake vector fallback."""
    model_key = "test-clap@1"
    dimensions = 3

    def __init__(self):
        self.audio_calls = 0
        self.text_calls = 0
        self.fail = False

    def audio(self, path):
        self.audio_calls += 1
        if self.fail:
            raise RuntimeError("test model unavailable")
        return np.array([1, 0, 0], dtype=np.float32)

    def text(self, query):
        self.text_calls += 1
        if self.fail:
            raise RuntimeError("test model unavailable")
        return np.array([1, 0, 0], dtype=np.float32)


class FakeAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, track):
        self.calls += 1
        return track.model_copy(update={
            "bpm": 124, "key": "A minor", "camelot_key": "8A", "energy": .6,
            "duration_sec": 240, "analysis_status": "analyzed", "analyzer": "essentia:test",
            "analysis_details": {"beat_positions": [0.0, .5], "energy_method": "test"},
        })


@pytest.fixture
def library(tmp_path):
    store = DropItStore(str(tmp_path / "test.db"))
    project = store.create_project("Local")
    other = store.create_project("Other")
    embedder = FakeEmbedder()
    rag = RagLibrary(store, embedder)
    yield store, project, other, embedder, rag
    store.close()


def add_track(store, project, name, vector=None, model="test-clap@1", **values):
    fields = dict(
        id=name, title=name, artist="Mira", filename=f"{name}.wav", path=f"/music/{name}.wav",
        bpm=124, key="A minor", camelot_key="8A", energy=.6, duration_sec=240,
        analysis_status="analyzed", analyzer="essentia:test",
    )
    fields.update(values)
    track = Track(**fields)
    store.upsert_track(track, name, project.id)
    store.update_track_analysis(track)
    if vector is not None:
        store.save_embedding(name, model, np.asarray(vector, dtype=np.float32))
    return store.get_track(name)
