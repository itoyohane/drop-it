import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.models import Brief, ExportRequest, Playlist, ReorderRequest
from backend.services import generate_playlist, render_export, scan_library
from backend.store import LibraryStore

app = FastAPI(title="DropIt Local API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "null"],
                   allow_methods=["*"], allow_headers=["*"])
data_dir = Path(os.environ.get("DROPIT_DATA_DIR", "data"))
store = LibraryStore(str(data_dir / "dropit.db"))
playlists: dict[str, Playlist] = {}


@app.get("/api/health")
def health(): return {"status": "ok"}


@app.get("/api/library")
def get_library(): return {"tracks": store.all()}


@app.post("/api/library/import-files")
async def import_files(files: list[UploadFile] = File(...)):
    import_dir = data_dir / "imports" / uuid4().hex
    import_dir.mkdir(parents=True, exist_ok=True)
    supported = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}
    saved = 0
    for index, upload in enumerate(files):
        original_name = Path((upload.filename or "audio").replace("\\", "/")).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in supported:
            await upload.close()
            continue
        destination = import_dir / original_name
        if destination.exists():
            destination = import_dir / f"{Path(original_name).stem}-{index:04d}{suffix}"
        with destination.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                target.write(chunk)
        await upload.close()
        saved += 1
    if not saved:
        raise HTTPException(400, "所选文件夹中没有支持的音频文件")
    tracks = scan_library(str(import_dir))
    if not tracks: raise HTTPException(400, "文件夹中没有支持的音频文件")
    store.replace(tracks)
    return {"tracks": tracks, "count": len(tracks)}


@app.post("/api/playlists")
def create_playlist(brief: Brief):
    try: playlist = generate_playlist(brief, store.all())
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc
    playlists[playlist.id] = playlist
    return playlist


@app.put("/api/playlists/{playlist_id}/order")
def reorder_playlist(playlist_id: str, body: ReorderRequest):
    playlist = _playlist(playlist_id)
    rows = {row.track.id: row for row in playlist.tracks}
    if set(rows) != set(body.track_ids): raise HTTPException(400, "曲目顺序与当前 Set 不匹配")
    playlist.tracks = [rows[track_id] for track_id in body.track_ids]
    playlist.revision += 1
    return playlist


@app.post("/api/playlists/{playlist_id}/approve")
def approve_playlist(playlist_id: str):
    playlist = _playlist(playlist_id)
    playlist.status = "approved"
    return playlist


@app.post("/api/playlists/{playlist_id}/export", response_class=PlainTextResponse)
def export_playlist(playlist_id: str, body: ExportRequest):
    playlist = _playlist(playlist_id)
    return render_export(playlist, body.format)


def _playlist(playlist_id: str) -> Playlist:
    if playlist_id not in playlists: raise HTTPException(404, "Set 不存在")
    return playlists[playlist_id]


frontend_dir = Path(__file__).resolve().parent.parent / "dist"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
