"""Orchestrate Essentia analysis and CLAP indexing for one track."""

import logging

from backend.music.clap_embedder import MusicEmbedder
from backend.music.essentia_analyzer import EssentiaAnalyzer
from backend.music.indexer import MusicIndexer
from backend.repositories.analysis import AnalysisRepository
from backend.repositories.embeddings import EmbeddingsRepository
from backend.repositories.tracks import TracksRepository

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, tracks: TracksRepository, analysis: AnalysisRepository,
                 embeddings: EmbeddingsRepository, embedder: MusicEmbedder,
                 analyzer: EssentiaAnalyzer | None = None):
        self.tracks = tracks
        self.analysis = analysis
        self.embeddings = embeddings
        self.embedder = embedder
        self.analyzer = analyzer or EssentiaAnalyzer()
        self.indexer = MusicIndexer(embeddings, embedder)

    def ready(self, track) -> bool:
        return (track.analysis_status == "analyzed" and track.analyzer.startswith("essentia:")
                and track.embedding_status == "ready" and track.embedding_model == self.embedder.model_key)

    def analyze(self, track_id: str, force: bool = False) -> bool:
        track = self.tracks.get_track(track_id)
        if track is None:
            return False
        if not force and self.ready(track):
            return True
        if force or track.analysis_status != "analyzed" or not track.analyzer.startswith("essentia:"):
            self.analysis.mark_track_analyzing(track_id)
            try:
                track = self.analyzer.analyze(track)
                track = self.analysis.update_track_analysis(track)
            except Exception as exc:
                logger.exception("essentia_analysis_failed")
                self.analysis.update_track_analysis(track.model_copy(update={
                    "analysis_status": "failed", "analysis_error": str(exc)[:500],
                }))
                return False
        try:
            self.indexer.index(track.id, track.path)
        except Exception as exc:
            logger.exception("clap_index_failed")
            self.embeddings.fail_embedding(track.id, str(exc))
            return False
        return True
