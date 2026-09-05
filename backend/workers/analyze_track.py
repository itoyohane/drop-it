"""Durable local jobs; one worker shares the audio models and serializes track updates."""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from backend.music.clap_embedder import MusicEmbedder
from backend.music.essentia_analyzer import EssentiaAnalyzer
from backend.repositories import DropItStore
from backend.services.analysis_service import AnalysisService

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, store: DropItStore, embedder: MusicEmbedder,
                 analyzer: EssentiaAnalyzer | None = None):
        self.store, self.embedder = store, embedder
        self.analysis = AnalysisService(store, store, store, embedder, analyzer)
        self.analyzer = self.analysis.analyzer
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
                message=f"已处理 {len(tracks)} 首，{failed} 首分析或索引失败。" if failed else "音乐分析和 CLAP 索引完成。",
                error=f"{failed} 首失败；详情见曲目的分析/索引错误。" if failed else None,
            )
        except Exception as exc:
            logger.exception("music_job_failed")
            self.store.update_job(job_id, status="failed", message="处理失败", error=str(exc)[:500])
        finally:
            with self._lock:
                self._submitted.discard(job_id)

    def ready(self, track) -> bool:
        return self.analysis.ready(track)

    def _analyze(self, track_id: str, force: bool = False) -> bool:
        return self.analysis.analyze(track_id, force)
