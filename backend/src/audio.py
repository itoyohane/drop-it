"""File identity and embedded tags; audio inference lives in backend.music."""

import hashlib
from pathlib import Path

from mutagen import File as MutagenFile

from backend.models import Track


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pending_track(path: Path, file_hash: str, filename: str | None = None) -> Track:
    filename = filename or path.name
    parts = [part.strip() for part in Path(filename).stem.replace("_", " ").split(" - ", 1)]
    artist, title = parts if len(parts) == 2 else ("Unknown artist", parts[0])
    duration = 0
    try:
        metadata = MutagenFile(path, easy=True)
        if metadata:
            title = str(metadata.get("title", [title])[0])
            artist = str(metadata.get("artist", [artist])[0])
            duration = round(metadata.info.length)
    except Exception:
        # Missing/corrupt tags should not prevent the audio decoder from trying.
        pass
    return Track(id=file_hash[:20], title=title, artist=artist, filename=filename,
                 path=str(path.resolve()), duration_sec=duration)
