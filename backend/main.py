import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.agent import DropItAgent, MODEL_NOT_CONFIGURED_ERROR
from backend.audio import pending_track, sha256_file
from backend.config import Settings
from backend.workers.analyze_track import JobRunner
from backend.models import (
    AnalyzeRequest, Brief, ChatRequest, ChatResponse, ConversationCreate, ExportRequest,
    FolderBindRequest, Playlist, ProjectCreate, ProjectUpdate, ReorderRequest, TrackUpdate,
)
from backend.music.clap_embedder import ClapEmbedder, MusicEmbedder
from backend.music.essentia_analyzer import EssentiaAnalyzer
from backend.agent.retriever import RagLibrary
from backend.services.set_planner import generate_playlist, render_export
from backend.repositories import GLOBAL_PROJECT_ID, DropItStore
from backend.agent.tools import DropItToolRegistry


SUPPORTED_AUDIO = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}
logger = logging.getLogger("dropit")


def create_app(settings: Settings | None = None, *, embedder: MusicEmbedder | None = None,
               analyzer: EssentiaAnalyzer | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.configure_langsmith()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Keep long-lived dependencies on the app so every route uses one SQLite/RAG/job context.
    store = DropItStore(str(settings.data_dir / "dropit.db"))
    embedder = embedder or ClapEmbedder(settings)
    rag = RagLibrary(store, embedder)
    registry = DropItToolRegistry(store, rag)
    copilot = DropItAgent(store, registry, settings)
    jobs = JobRunner(store, embedder, analyzer)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Requeue interrupted work before requests arrive; close resources only after serving stops.
        jobs.start()
        try:
            yield
        finally:
            jobs.close()
            store.close()

    app = FastAPI(title="DropIt DJ Agent API", version="1.0.0", lifespan=lifespan,
                  docs_url="/api/docs" if settings.environment != "production" else None,
                  redoc_url=None)
    app.state.settings = settings
    app.state.store = store
    app.state.rag = rag
    app.state.registry = registry
    app.state.copilot = copilot
    app.state.jobs = jobs
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def security(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid4().hex)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", extra={"request_id": request_id})
            # Do not expose stack traces to the browser. The request ID connects this response to server logs.
            response = JSONResponse(
                status_code=500,
                content={"detail": f"服务端请求失败，请提供请求编号 {request_id}", "request_id": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "1.0.0", "agent_framework": "langchain.create_agent",
                "tools": registry.names, "model_configured": copilot.model_configured,
                # This is display-only metadata. Credentials remain server-side.
                "model_name": settings.model_name,
                "model_source": "system",
                "embedding_provider": rag.provider}

    @app.get("/api/projects")
    def list_projects():
        return {"projects": store.list_projects()}

    @app.post("/api/projects", status_code=201)
    def create_project(body: ProjectCreate):
        return store.create_project(body.name, body.description)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        return project_or_404(store, project_id)

    @app.patch("/api/projects/{project_id}")
    def update_project(project_id: str, body: ProjectUpdate):
        project = store.update_project(project_id, body.name, body.description)
        if not project:
            raise HTTPException(404, "项目不存在")
        return project

    @app.delete("/api/projects/{project_id}", status_code=204)
    def delete_project(project_id: str):
        if not store.delete_project(project_id):
            raise HTTPException(404, "项目不存在")
        return Response(status_code=204)

    @app.get("/api/projects/{project_id}/conversations")
    def list_conversations(project_id: str):
        project_or_404(store, project_id)
        return {"conversations": store.list_conversations(project_id)}

    @app.post("/api/projects/{project_id}/conversations", status_code=201)
    def create_conversation(project_id: str, body: ConversationCreate):
        project_or_404(store, project_id)
        return store.create_conversation(project_id, body.title)

    @app.delete("/api/projects/{project_id}/conversations/{conversation_id}", status_code=204)
    def delete_conversation(project_id: str, conversation_id: str):
        if not store.delete_conversation(project_id, conversation_id):
            raise HTTPException(409, "项目至少需要保留一个对话")
        return Response(status_code=204)

    @app.get("/api/chat/conversations")
    def list_global_conversations():
        return {"conversations": store.list_conversations(GLOBAL_PROJECT_ID)}

    @app.post("/api/chat/conversations", status_code=201)
    def create_global_conversation(body: ConversationCreate):
        return store.create_conversation(GLOBAL_PROJECT_ID, body.title)

    @app.delete("/api/chat/conversations/{conversation_id}", status_code=204)
    def delete_global_conversation(conversation_id: str):
        if not store.delete_conversation(GLOBAL_PROJECT_ID, conversation_id):
            raise HTTPException(409, "普通对话至少需要保留一个会话")
        return Response(status_code=204)

    @app.get("/api/chat/conversations/{conversation_id}/messages")
    def get_global_messages(conversation_id: str):
        conversation_or_404(store, GLOBAL_PROJECT_ID, conversation_id)
        return {"messages": store.list_messages(GLOBAL_PROJECT_ID, conversation_id)}

    @app.post("/api/chat/conversations/{conversation_id}/chat/stream")
    async def global_chat_stream(conversation_id: str, body: ChatRequest):
        conversation_or_404(store, GLOBAL_PROJECT_ID, conversation_id)
        model_or_503(copilot)

        async def events():
            async for event in copilot.stream_chat(GLOBAL_PROJECT_ID, conversation_id, body.message):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/library")
    def get_global_library():
        return {"tracks": store.all_tracks(), "scope": "global"}

    @app.get("/api/projects/{project_id}/library")
    def get_project_library(project_id: str):
        project_or_404(store, project_id)
        return {"tracks": store.all_tracks(project_id), "scope": "project"}

    @app.get("/api/projects/{project_id}/sources")
    def list_project_sources(project_id: str):
        project_or_404(store, project_id)
        return {"sources": store.list_sources(project_id)}

    @app.post("/api/projects/{project_id}/sources/resolve")
    def resolve_project_source(project_id: str, body: FolderBindRequest):
        project_or_404(store, project_id)
        source, existed = store.get_or_create_source(body.folder_key, body.name)
        store.bind_source(project_id, source.id)
        source = store.get_source(source.id)
        return {"source": source, "reused": bool(existed and source and source.track_count),
                "tracks": store.all_tracks(project_id)}

    @app.post("/api/projects/{project_id}/library/import-files", status_code=202)
    async def import_files(project_id: str, source_id: str = Form(...),
                           files: list[UploadFile] = File(...)):
        project_or_404(store, project_id)
        source = store.get_source(source_id)
        if not source or source_id not in store.project_source_ids(project_id):
            raise HTTPException(400, "文件夹资源未绑定到当前项目")
        if len(files) > settings.max_upload_files:
            raise HTTPException(413, f"单次最多导入 {settings.max_upload_files} 个文件")
        import_dir = settings.data_dir / "imports" / "sources" / source_id
        import_dir.mkdir(parents=True, exist_ok=True)
        track_ids: list[str] = []
        skipped = 0
        max_bytes = settings.max_upload_mb * 1024 * 1024

        for index, upload in enumerate(files):
            original_name = Path((upload.filename or "audio").replace("\\", "/")).name
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_AUDIO:
                skipped += 1
                await upload.close()
                continue
            destination = import_dir / f"{Path(original_name).stem}-{uuid4().hex}{suffix}"
            written = 0
            with destination.open("wb") as target:
                # Stream uploads to disk: neither a single file nor a batch needs to fit in memory.
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        destination.unlink(missing_ok=True)
                        await upload.close()
                        raise HTTPException(413, f"文件 {original_name} 超过 {settings.max_upload_mb} MB")
                    target.write(chunk)
            await upload.close()
            file_hash = sha256_file(destination)
            track = store.upsert_track(
                pending_track(destination, file_hash, original_name), file_hash, project_id, source_id=source_id
            )
            track_ids.append(track.id)

        track_ids = list(dict.fromkeys(track_ids))
        if not track_ids:
            raise HTTPException(400, "没有找到支持的音频文件")
        job = store.create_job(
            project_id, "analyze", {"track_ids": track_ids, "source_ids": [source_id]}, len(track_ids)
        )
        jobs.submit(job.id)
        return {"job": job, "tracks": store.all_tracks(project_id),
                "count": len(track_ids), "skipped": skipped}

    @app.post("/api/projects/{project_id}/library/analyze", status_code=202)
    def analyze_project_library(project_id: str, body: AnalyzeRequest):
        project_or_404(store, project_id)
        track_ids = body.track_ids or [track.id for track in store.all_tracks(project_id)
                                       if not jobs.ready(track)]
        if not track_ids:
            return {"job": None, "count": 0}
        allowed = [track.id for track in store.tracks_by_ids(track_ids, project_id)]
        source_ids = list(store.source_ids_for_tracks(allowed, project_id))
        job = store.create_job(
            project_id, "analyze", {"track_ids": allowed, "source_ids": source_ids,
                                   "force": bool(body.track_ids)}, len(allowed)
        )
        jobs.submit(job.id)
        return {"job": job, "count": len(allowed)}

    @app.patch("/api/projects/{project_id}/library/{track_id}")
    def update_track(project_id: str, track_id: str, body: TrackUpdate):
        project_or_404(store, project_id)
        if not store.tracks_by_ids([track_id], project_id):
            raise HTTPException(404, "曲目不存在")
        # Audio vectors do not depend on edited tags; RAG reads current database facts.
        return store.update_track(track_id, body)

    @app.delete("/api/projects/{project_id}/library/{track_id}", status_code=204)
    def remove_track(project_id: str, track_id: str):
        if not store.remove_track_from_project(project_id, track_id):
            raise HTTPException(404, "曲目不在当前项目")
        return Response(status_code=204)

    @app.get("/api/projects/{project_id}/jobs")
    def list_jobs(project_id: str):
        project_or_404(store, project_id)
        return {"jobs": store.list_jobs(project_id)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return job

    @app.get("/api/projects/{project_id}/conversations/{conversation_id}/messages")
    def get_messages(project_id: str, conversation_id: str):
        conversation_or_404(store, project_id, conversation_id)
        return {"messages": store.list_messages(project_id, conversation_id)}

    @app.post("/api/projects/{project_id}/conversations/{conversation_id}/chat",
              response_model=ChatResponse)
    async def chat(project_id: str, conversation_id: str, body: ChatRequest):
        conversation_or_404(store, project_id, conversation_id)
        model_or_503(copilot)
        message, playlist = await copilot.chat(project_id, conversation_id, body.message)
        return ChatResponse(message=message, playlist=playlist,
                            model_configured=copilot.model_configured)

    @app.post("/api/projects/{project_id}/conversations/{conversation_id}/chat/stream")
    async def chat_stream(project_id: str, conversation_id: str, body: ChatRequest):
        conversation_or_404(store, project_id, conversation_id)
        model_or_503(copilot)

        async def events():
            try:
                # Convert internal agent events into SSE frames so the UI can render tokens/tools live.
                async for event in copilot.stream_chat(project_id, conversation_id, body.message):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                logger.exception("agent_stream_failed")
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)[:300]}, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/projects/{project_id}/playlists")
    def list_playlists(project_id: str):
        project_or_404(store, project_id)
        return {"playlists": store.list_playlists(project_id)}

    @app.post("/api/projects/{project_id}/playlists", response_model=Playlist)
    def create_playlist(project_id: str, brief: Brief):
        project_or_404(store, project_id)
        try:
            return store.save_playlist(generate_playlist(project_id, brief, store.all_tracks(project_id)))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/api/playlists/{playlist_id}/order", response_model=Playlist)
    def reorder_playlist(playlist_id: str, body: ReorderRequest):
        playlist = playlist_or_404(store, playlist_id)
        rows = {row.track.id: row for row in playlist.tracks}
        if set(rows) != set(body.track_ids) or len(rows) != len(body.track_ids):
            raise HTTPException(400, "曲目顺序与当前 Set 不匹配")
        playlist.tracks = [rows[track_id] for track_id in body.track_ids]
        playlist.revision += 1
        return store.save_playlist(playlist)

    @app.post("/api/playlists/{playlist_id}/approve", response_model=Playlist)
    def approve_playlist(playlist_id: str):
        playlist = playlist_or_404(store, playlist_id)
        playlist.status = "approved"
        return store.save_playlist(playlist)

    @app.delete("/api/playlists/{playlist_id}", status_code=204)
    def delete_playlist(playlist_id: str):
        if not store.delete_playlist(playlist_id):
            raise HTTPException(404, "Set 不存在")
        return Response(status_code=204)

    @app.post("/api/playlists/{playlist_id}/export", response_class=PlainTextResponse)
    def export_playlist(playlist_id: str, body: ExportRequest):
        return render_export(playlist_or_404(store, playlist_id), body.format)

    frontend_dir = Path(__file__).resolve().parent.parent / "dist"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


def project_or_404(store: DropItStore, project_id: str):
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(404, "项目不存在")
    return project


def model_or_503(copilot: DropItAgent) -> None:
    """Return an HTTP error before creating an SSE response with no usable model."""
    if not copilot.model_configured:
        raise HTTPException(status_code=503, detail=MODEL_NOT_CONFIGURED_ERROR)


def conversation_or_404(store: DropItStore, project_id: str, conversation_id: str):
    conversation = store.get_conversation(conversation_id)
    if not conversation or conversation.project_id != project_id:
        raise HTTPException(404, "对话不存在")
    return conversation


def playlist_or_404(store: DropItStore, playlist_id: str) -> Playlist:
    playlist = store.get_playlist(playlist_id)
    if not playlist:
        raise HTTPException(404, "Set 不存在")
    return playlist


app = create_app()
