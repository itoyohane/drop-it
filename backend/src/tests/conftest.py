import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

# Prevent the module-level ASGI app from opening a developer's real database.
os.environ["DROPIT_ENV"] = "test"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
_bootstrap_dir = Path.cwd() / f".dropit-test-bootstrap-{os.getpid()}"
_bootstrap_dir.mkdir(exist_ok=True)
os.environ["DROPIT_DATA_DIR"] = str(_bootstrap_dir)
os.environ["DEEPSEEK_API_KEY"] = ""

from backend.models import Track
from backend.agent.tools import DropItToolRegistry
from backend.repositories import DropItStore


def pytest_sessionfinish(session, exitstatus):
    main_module = sys.modules.get("backend.main")
    global_app = getattr(main_module, "app", None)
    if global_app is not None:
        global_app.state.jobs.close()
        global_app.state.store.close()
    shutil.rmtree(_bootstrap_dir, ignore_errors=True)


class FakeEmbedder:
    """Only for tests; production has no fake vector fallback."""
    model_key = "test-text@1"
    dimensions = 3

    def __init__(self):
        self.text_calls = 0
        self.values: list[str] = []
        self.fail = False

    def text(self, value):
        self.text_calls += 1
        self.values.append(value)
        if self.fail:
            raise RuntimeError("test model unavailable")
        return np.array([1, 0, 0], dtype=np.float32)


class FakeDescriptor:
    model_key = "test-descriptor@1"

    def __init__(self):
        self.calls = 0
        self.fail = False

    def describe(self, track):
        self.calls += 1
        if self.fail:
            raise RuntimeError("test descriptor unavailable")
        return f"{track.bpm:.1f} BPM {track.key} energy {track.energy:.2f}"


class FakeAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, track):
        self.calls += 1
        return track.model_copy(update={
            "bpm": 124, "key": "A minor", "camelot_key": "8A", "energy": .6,
            "duration_sec": 240, "analysis_status": "analyzed", "analyzer": "librosa:test",
            "analysis_details": {"beat_positions": [0.0, .5], "energy_method": "test",
                                 "spectral_centroid_hz": 1800, "onset_strength": 1.2},
        })


@pytest.fixture
def library(tmp_path):
    store = DropItStore(str(tmp_path / "test.db"))
    project = store.create_project("Local")
    other = store.create_project("Other")
    embedder = FakeEmbedder()
    registry = DropItToolRegistry(store, embedder)
    yield store, project, other, embedder, registry
    store.close()


def add_track(store, project, name, vector=None, model="test-text@1", **values):
    fields = dict(
        id=name, title=name, artist="Mira", filename=f"{name}.wav", path=f"/music/{name}.wav",
        bpm=124, key="A minor", camelot_key="8A", energy=.6, duration_sec=240,
        analysis_status="analyzed", analyzer="librosa:test", description=f"description for {name}",
        description_model="test-descriptor@1",
    )
    fields.update(values)
    track = Track(**fields)
    store.upsert_track(track, name, project.id)
    store.update_track_analysis(track)
    if vector is not None:
        store.save_embedding(name, model, np.asarray(vector, dtype=np.float32))
    return store.get_track(name)
