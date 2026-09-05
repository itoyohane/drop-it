"""Music-only RAG facade used by the Agent tools."""

from backend.models import MusicFilters, MusicMatch, Track
from backend.music.clap_embedder import MusicEmbedder
from backend.repositories import DropItStore
from backend.services.catalog_service import CatalogService
from backend.services.similarity_service import SimilarityService


class RagLibrary:
    provider = "clap"

    def __init__(self, store: DropItStore, embedder: MusicEmbedder):
        self.catalog_service = CatalogService(store)
        self.similarity_service = SimilarityService(store, embedder)

    def catalog(self, project_id: str, filters: MusicFilters | None = None) -> list[Track]:
        return self.catalog_service.list(project_id, filters)

    def search(self, project_id: str, query: str = "", filters: MusicFilters | None = None,
               k: int = 20) -> list[MusicMatch]:
        """Empty descriptions list metadata; nonempty ones search audio with CLAP text vectors."""
        candidates = self.catalog(project_id, filters)
        if not query.strip():
            return [MusicMatch(track=track) for track in candidates[:k]]
        if not candidates:
            return []
        return self.similarity_service.search(project_id, candidates, query)[:k]

    def similar(self, project_id: str, track_id: str, filters: MusicFilters | None = None,
                k: int = 3) -> list[MusicMatch]:
        catalog = self.catalog(project_id)
        reference = next((track for track in catalog if track.id == track_id), None)
        if reference is None:
            raise ValueError("当前曲库没有这个 track_id，请先调用 search_library 确认曲目。")
        constraints = filters or MusicFilters()
        candidates = [track for track in catalog if track.id != track_id and constraints.matches(track)]
        return self.similarity_service.similar(project_id, reference, candidates)[:k]
