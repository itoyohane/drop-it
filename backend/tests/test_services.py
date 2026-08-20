from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.models import Brief
from backend.services import generate_playlist, render_export, scan_library


def test_scan_and_generate(tmp_path: Path):
    for name in ["Mira - First Light.mp3", "Mira - Night Drive.wav", "Sol - Open Air.flac"]:
        (tmp_path / name).touch()
    tracks = scan_library(str(tmp_path))
    assert len(tracks) == 3
    low, high = min(t.bpm for t in tracks), max(t.bpm for t in tracks)
    playlist = generate_playlist(Brief(duration_min=10, bpm_min=int(low), bpm_max=int(high) + 1), tracks)
    assert len({row.track.id for row in playlist.tracks}) == len(playlist.tracks)
    assert any(event.status == "revised" for event in playlist.trace)
    assert render_export(playlist, "m3u").startswith("#EXTM3U")


def test_web_upload_endpoint():
    client = TestClient(app)
    response = client.post(
        "/api/library/import-files",
        files=[
            ("files", ("Mira - First Light.mp3", b"demo", "audio/mpeg")),
            ("files", ("Sol - Open Air.wav", b"demo", "audio/wav")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert {track["artist"] for track in payload["tracks"]} == {"Mira", "Sol"}
