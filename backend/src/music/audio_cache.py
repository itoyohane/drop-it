"""Release only app-owned upload copies whose audio features are durable."""

import logging
from pathlib import Path

from backend.models import Track

logger = logging.getLogger(__name__)


class AudioCache:
    def __init__(self, root: Path, retain: bool = False):
        self.root = root.resolve()
        self.retain = retain

    @staticmethod
    def analyzed(track: Track) -> bool:
        return track.analysis_status == "analyzed" and track.analyzer.startswith("librosa:")

    def release(self, track: Track) -> None:
        if self.retain or not self.analyzed(track):
            return
        path = Path(track.path)
        # Resolve before deleting: legacy/external music and symlink targets outside
        # the upload directory must never be removed.
        if not path.resolve().is_relative_to(self.root):
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A file locked by another process must not invalidate saved features.
            # Startup (or another analysis job) retries the cleanup.
            logger.warning("audio_cache_cleanup_failed: %s", path, exc_info=True)
