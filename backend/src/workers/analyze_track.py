"""Durable jobs for librosa analysis, song description generation, and text indexing."""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from backend.music.indexer import MusicIndexer
from backend.music.librosa_analyzer import LibrosaAnalyzer
from backend.music.text_models import TextEmbedder, TrackDescriptor
from backend.repositories import DropItStore

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, store: DropItStore, embedder: TextEmbedder,
                 descriptor: TrackDescriptor, analyzer: LibrosaAnalyzer | None = None):
        self.store, self.embedder, self.descriptor = store, embedder, descriptor
        self.analyzer = analyzer or LibrosaAnalyzer()
        self.indexer = MusicIndexer(store, embedder)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dropit-analysis")
        self._submitted: set[str] = set()
        self._lock = Lock()

    def start(self) -> None:
        for job in self.store.recover_jobs():
            self.submit(job.id)

    def close(self) -> None:
        # Finish database writes before FastAPI closes the connection.
        self.executor.shutdown(wait=True, cancel_futures=False)

    def submit(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._submitted:
                return
            self._submitted.add(job_id)
        self.executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        try:
            job = self.store.get_job(job_id)
            if not job or job.status == "cancelled":
                return
            tracks = self.store.tracks_by_ids(job.payload.get("track_ids", []), job.project_id)
            sources = list(self.store.source_ids_for_tracks([t.id for t in tracks], job.project_id))
            for source_id in sources:
                self.store.update_source_status(source_id, "analyzing")
            self.store.update_job(job_id, status="running", total=len(tracks), message="正在分析音乐")
            failed = 0
            for index, track in enumerate(tracks, 1):
                current = self.store.get_job(job_id)
                if not current or current.status == "cancelled":
                    return
                failed += not self._analyze(track.id, force=bool(job.payload.get("force")))
                self.store.update_job(job_id, progress=index, message=f"已处理 {index}/{len(tracks)} 首")
            for source_id in sources:
                ready = all(self.ready(t) for t in self.store.source_tracks(source_id))
                self.store.update_source_status(source_id, "ready" if ready else "failed")
            self.store.update_job(
                job_id, status="failed" if failed else "completed", progress=len(tracks),
                message=(f"已处理 {len(tracks)} 首，{failed} 首分析、描述或索引失败。"
                         if failed else "librosa 分析和歌曲描述索引完成。"),
                error=f"{failed} 首失败；详情见曲目的分析/描述/索引错误。" if failed else None,
            )
        except Exception as exc:
            logger.exception("music_job_failed")
            self.store.update_job(job_id, status="failed", message="处理失败", error=str(exc)[:500])
        finally:
            with self._lock:
                self._submitted.discard(job_id)

    def ready(self, track) -> bool:
        return (track.analysis_status == "analyzed" and track.analyzer.startswith("librosa:")
                and bool(track.description) and track.description_model == self.descriptor.model_key
                and track.embedding_status == "ready"
                and track.embedding_model == self.embedder.model_key)

    def _analyze(self, track_id: str, force: bool = False) -> bool:
        track = self.store.get_track(track_id)
        if track is None:
            return False
        if not force and self.ready(track):
            return True
        needs_features = force or track.analysis_status != "analyzed" or not track.analyzer.startswith("librosa:")
        needs_description = (needs_features or not track.description
                             or track.description_model != self.descriptor.model_key)
        if needs_features:
            self.store.mark_track_analyzing(track_id)
            try:
                track = self.analyzer.analyze(track)
            except Exception as exc:
                logger.exception("librosa_analysis_failed")
                self.store.update_track_analysis(track.model_copy(update={
                    "analysis_status": "failed", "analysis_error": str(exc)[:500],
                    "description": "", "description_model": "",
                }))
                return False
        if needs_description:
            try:
                description = self.descriptor.describe(track)
                track = track.model_copy(update={
                    "description": description, "description_model": self.descriptor.model_key,
                })
                track = self.store.update_track_analysis(track)
            except Exception as exc:
                logger.exception("song_description_failed")
                self.store.update_track_analysis(track.model_copy(update={
                    "description": "", "description_model": "",
                }))
                self.store.fail_embedding(track.id, str(exc))
                return False
        try:
            self.indexer.index(track.id, track.description)
        except Exception as exc:
            logger.exception("description_index_failed")
            self.store.fail_embedding(track.id, str(exc))
            return False
        return True
