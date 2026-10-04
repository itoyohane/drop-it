"""Audio cache lifecycle, persisted retry data and upload deduplication."""

from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from backend.audio import pending_track, sha256_file
from backend.config import Settings
from backend.main import create_app
from backend.music.audio_cache import AudioCache
from backend.music.cleanup_audio import cleanup_candidates
from backend.workers.analyze_track import JobRunner
from conftest import FakeAnalyzer, FakeDescriptor, FakeEmbedder


def cached_track(store, project, root, name="song"):
    path = root / f"{name}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"audio-{name}".encode())
    file_hash = sha256_file(path)
    track = store.upsert_track(pending_track(path, file_hash), file_hash, project.id)
    return track, path


@pytest.mark.parametrize("failure", [None, "description", "embedding", "analysis"])
def test_features_commit_before_release_and_api_failures_retry_without_audio(library, tmp_path, failure):
    store, project, _, embedder, _ = library
    root = tmp_path / "imports" / "sources"
    track, path = cached_track(store, project, root)
    analyzer, descriptor = FakeAnalyzer(), FakeDescriptor()
    if failure == "description":
        descriptor.fail = True
    if failure == "embedding":
        embedder.fail = True
    if failure == "analysis":
        class BrokenAnalyzer:
            def analyze(self, track):
                raise ValueError("bad audio")
        analyzer = BrokenAnalyzer()
    runner = JobRunner(store, embedder, descriptor, analyzer, AudioCache(root))
    try:
        job = store.create_job(project.id, "analyze", {"track_ids": [track.id]}, 1)
        runner._run(job.id)
        assert store.get_job(job.id).status == ("completed" if failure is None else "failed")
        assert path.exists() == (failure == "analysis")
        if failure == "analysis":
            assert store.get_track(track.id).analysis_status == "failed"
            return
        saved = store.get_track(track.id)
        assert saved.analysis_status == "analyzed"
        assert saved.analysis_details["beat_positions"] == [0, .5]
        descriptor.fail = embedder.fail = False
        retry = store.create_job(project.id, "analyze", {"track_ids": [track.id]}, 1)
        runner._run(retry.id)
        assert store.get_job(retry.id).status == "completed"
        assert analyzer.calls == 1
        assert runner.ready(store.get_track(track.id))
    finally:
        runner.close()


def test_release_waits_for_successful_database_commit(library, tmp_path, monkeypatch):
    store, project, _, embedder, _ = library
    root = tmp_path / "imports" / "sources"
    track, path = cached_track(store, project, root)
    runner = JobRunner(store, embedder, FakeDescriptor(), FakeAnalyzer(), AudioCache(root))
    monkeypatch.setattr(store, "update_track_analysis", lambda track: (_ for _ in ()).throw(RuntimeError("disk full")))
    try:
        job = store.create_job(project.id, "analyze", {"track_ids": [track.id]}, 1)
        runner._run(job.id)
        assert store.get_job(job.id).status == "failed"
        assert path.exists()
    finally:
        runner.close()


@pytest.mark.parametrize("mode", ["retain", "external", "pending", "failed", "locked"])
def test_cleanup_boundaries_and_locked_files(library, tmp_path, monkeypatch, mode):
    store, project, _, _, _ = library
    root = tmp_path / "imports" / "sources"
    track, path = cached_track(store, project, tmp_path / "originals" if mode == "external" else root)
    track = FakeAnalyzer().analyze(track)
    if mode in {"pending", "failed"}:
        track = track.model_copy(update={"analysis_status": mode})
    if mode == "locked":
        monkeypatch.setattr(Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked")))
    AudioCache(root, retain=mode == "retain").release(track)
    assert path.exists()


def test_startup_cleanup_protects_recovered_force_jobs(library, tmp_path, monkeypatch):
    store, project, _, embedder, _ = library
    root = tmp_path / "imports" / "sources"
    complete, complete_path = cached_track(store, project, root, "complete")
    forced, forced_path = cached_track(store, project, root, "forced")
    pending, pending_path = cached_track(store, project, root, "pending")
    for track in [complete, forced]:
        store.update_track_analysis(FakeAnalyzer().analyze(track))
    job = store.create_job(project.id, "analyze", {"track_ids": [forced.id], "force": True}, 1)
    store.update_job(job.id, status="running")
    runner = JobRunner(store, embedder, FakeDescriptor(), FakeAnalyzer(), AudioCache(root))
    submitted = []
    monkeypatch.setattr(runner, "submit", submitted.append)
    try:
        runner.start()
        assert not complete_path.exists()
        assert forced_path.exists() and pending_path.exists()
        assert submitted == [job.id]
        assert store.get_job(job.id).status == "queued"
    finally:
        runner.close()


def test_legacy_cleanup_verifies_duplicate_hashes_and_preserves_force_sources(library, tmp_path):
    store, project, _, _, _ = library
    root = tmp_path / "imports" / "sources"
    analyzed, path = cached_track(store, project, root)
    store.update_track_analysis(FakeAnalyzer().analyze(analyzed))
    duplicate = root / "duplicate.mp3"
    duplicate.write_bytes(path.read_bytes())
    _, pending_path = cached_track(store, project, root, "pending")
    unknown = root / "unknown.mp3"
    unknown.write_bytes(b"unknown audio")
    assert set(cleanup_candidates(store, root)) == {path, duplicate}
    assert all(p.exists() for p in [path, duplicate, pending_path, unknown])
    store.create_job(project.id, "analyze", {"track_ids": [analyzed.id], "force": True}, 1)
    assert cleanup_candidates(store, root) == []


def wait_job(client, response):
    assert response.status_code == 202, response.text
    job_id = response.json()["job"]["id"]
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(.01)
    pytest.fail("job did not finish")


@pytest.mark.parametrize("retain", [False, True])
def test_upload_dedup_across_projects_reupload_force_and_reindex(tmp_path, retain):
    settings = Settings(_env_file=None, DROPIT_DATA_DIR=tmp_path,
                        DROPIT_RETAIN_AUDIO_FILES=retain, DEEPSEEK_API_KEY="")
    analyzer, descriptor, embedder = FakeAnalyzer(), FakeDescriptor(), FakeEmbedder()
    app = create_app(settings, analyzer=analyzer, descriptor=descriptor, embedder=embedder)
    with TestClient(app) as client:
        projects = [client.post("/api/projects", json={"name": name}).json()["id"] for name in ["A", "B"]]
        sources = [client.post(f"/api/projects/{pid}/sources/resolve", json={
            "folder_key": str(i) * 32, "name": "Folder"
        }).json()["source"]["id"] for i, pid in enumerate(projects)]

        def upload(index=0, reanalyze=False):
            return client.post(f"/api/projects/{projects[index]}/library/import-files",
                               data={"source_id": sources[index], "reanalyze": str(reanalyze).lower()},
                               files={"files": ("Artist - Song.mp3", b"audio content", "audio/mpeg")})

        assert wait_job(client, upload())["status"] == "completed"
        assert wait_job(client, upload(1))["status"] == "completed"
        assert analyzer.calls == 1 and embedder.text_calls == 1
        cached = list((tmp_path / "imports" / "sources").rglob("*.mp3"))
        assert len(cached) == int(retain)
        first = client.get(f"/api/projects/{projects[0]}/library").json()["tracks"][0]
        second = client.get(f"/api/projects/{projects[1]}/library").json()["tracks"][0]
        assert first["id"] == second["id"]
        assert first["analysis_status"] == "analyzed" and first["embedding_status"] == "ready"
        if not retain:
            response = client.post(f"/api/projects/{projects[0]}/library/analyze", json={"track_ids": [first["id"]]})
            assert response.status_code == 409 and "重新上传" in response.json()["detail"]
            assert app.state.store.get_track(first["id"]).embedding_status == "ready"
        assert wait_job(client, upload(reanalyze=True))["status"] == "completed"
        assert analyzer.calls == 2
        patch = {"title": "Edited", "artist": "Artist", "bpm": 122, "key": "A minor",
                 "camelot_key": "8A", "energy": .5}
        assert client.patch(f"/api/projects/{projects[0]}/library/{first['id']}", json=patch).status_code == 200
        jobs = client.get(f"/api/projects/{projects[0]}/jobs").json()["jobs"]
        job_id = next(job["id"] for job in jobs if job["kind"] == "reindex")
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(.01)
        assert job["status"] == "completed" and analyzer.calls == 2
        assert app.state.registry.search(projects[1], "test")[0].track.title == "Edited"
        assert len(list((tmp_path / "imports" / "sources").rglob("*.mp3"))) == int(retain)


def test_oversized_upload_removes_partial_file_on_windows(tmp_path):
    settings = Settings(_env_file=None, DROPIT_DATA_DIR=tmp_path, DROPIT_MAX_UPLOAD_MB=10,
                        DEEPSEEK_API_KEY="")
    app = create_app(settings, analyzer=FakeAnalyzer(), descriptor=FakeDescriptor(), embedder=FakeEmbedder())
    with TestClient(app) as client:
        project = app.state.store.create_project("Large upload")
        source, _ = app.state.store.get_or_create_source("x" * 32, "Folder")
        app.state.store.bind_source(project.id, source.id)
        response = client.post(f"/api/projects/{project.id}/library/import-files",
                               data={"source_id": source.id},
                               files={"files": ("large.mp3", b"a" * (10 * 1024 * 1024 + 1), "audio/mpeg")})
        assert response.status_code == 413
        assert not list((tmp_path / "imports" / "sources").rglob("*.mp3"))
        assert app.state.store.all_tracks() == []


def test_shared_cache_survives_until_last_queued_force_job(library, tmp_path):
    store, project, _, embedder, _ = library
    root = tmp_path / "imports" / "sources"
    track, path = cached_track(store, project, root)
    analyzer = FakeAnalyzer()
    runner = JobRunner(store, embedder, FakeDescriptor(), analyzer, AudioCache(root))
    try:
        jobs = [store.create_job(project.id, "analyze", {"track_ids": [track.id], "force": True}, 1)
                for _ in range(2)]
        runner._run(jobs[0].id)
        assert store.get_job(jobs[0].id).status == "completed" and path.exists()
        runner._run(jobs[1].id)
        assert store.get_job(jobs[1].id).status == "completed" and not path.exists()
        assert analyzer.calls == 2
    finally:
        runner.close()
