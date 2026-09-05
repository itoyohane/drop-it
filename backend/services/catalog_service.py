"""Read the current library and apply exact metadata constraints."""

from backend.models import MusicFilters, Track
from backend.repositories import GLOBAL_PROJECT_ID
from backend.repositories.tracks import TracksRepository


class CatalogService:
    def __init__(self, tracks: TracksRepository):
        self.tracks = tracks

    def list(self, project_id: str, filters: MusicFilters | None = None) -> list[Track]:
        constraints = filters or MusicFilters()
        scope = None if project_id == GLOBAL_PROJECT_ID else project_id
        return [track for track in self.tracks.all_tracks(scope) if constraints.matches(track)]
