from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Track(BaseModel):
    id: str
    title: str
    artist: str
    filename: str
    path: str
    duration_sec: int = 0
    bpm: float = 0
    key: str = "Unknown"
    camelot_key: str = "—"
    energy: float = 0
    bpm_confidence: float = 0
    key_confidence: float = 0
    analysis_status: Literal["pending", "analyzing", "analyzed", "failed"] = "pending"
    analysis_error: str | None = None
    analyzer: str = ""
    analysis_details: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    description_model: str = ""
    embedding_status: Literal["pending", "ready", "failed"] = "pending"
    embedding_model: str = ""
    embedding_error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Project(BaseModel):
    id: str
    name: str
    description: str = ""
    track_count: int = 0
    scope: Literal["project", "global"] = "project"
    created_at: datetime
    updated_at: datetime


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field("", max_length=300)


class ProjectUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field("", max_length=300)


class Conversation(BaseModel):
    id: str
    project_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationCreate(BaseModel):
    title: str = Field("新对话", min_length=1, max_length=80)


class LibrarySource(BaseModel):
    id: str
    folder_key: str
    name: str
    status: Literal["pending", "analyzing", "ready", "failed"] = "pending"
    track_count: int = 0
    created_at: datetime
    updated_at: datetime


class FolderBindRequest(BaseModel):
    folder_key: str = Field(min_length=16, max_length=128)
    name: str = Field(min_length=1, max_length=200)


class Brief(BaseModel):
    title: str = "Untitled set"
    duration_min: int = Field(45, ge=10, le=240)
    bpm_min: int = Field(118, ge=60, le=220)
    bpm_max: int = Field(132, ge=60, le=220)
    energy: Literal["steady", "build", "peak", "wave"] = "build"
    style: str = "house, warm-up"
    notes: str = ""


class PlaylistTrack(BaseModel):
    track: Track
    reason: str
    alternatives: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    agent: str
    status: Literal["done", "revised", "approved", "failed"]
    message: str


class Playlist(BaseModel):
    id: str
    project_id: str
    brief: Brief
    tracks: list[PlaylistTrack]
    duration_sec: int
    status: Literal["draft", "approved"] = "draft"
    revision: int = 1
    report: list[str] = Field(default_factory=list)
    trace: list[AgentEvent] = Field(default_factory=list)
    created_at: datetime | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class AnalyzeRequest(BaseModel):
    track_ids: list[str] = Field(default_factory=list)


class TrackUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    artist: str = Field(min_length=1, max_length=200)
    bpm: float = Field(ge=40, le=260)
    key: str = Field(min_length=1, max_length=32)
    camelot_key: str = Field(min_length=1, max_length=4)
    energy: float = Field(ge=0, le=1)


class ToolEvent(BaseModel):
    name: str
    status: Literal["running", "done", "failed"] = "done"
    summary: str


class ChatMessage(BaseModel):
    id: str
    project_id: str
    conversation_id: str | None = None
    role: Literal["user", "assistant"]
    content: str
    tool_events: list[ToolEvent] = Field(default_factory=list)
    created_at: datetime


class ChatResponse(BaseModel):
    message: ChatMessage
    playlist: Playlist | None = None
    model_configured: bool


class Job(BaseModel):
    id: str
    project_id: str
    kind: Literal["analyze", "reindex"]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    progress: int = 0
    total: int = 0
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ReorderRequest(BaseModel):
    track_ids: list[str]


class ExportRequest(BaseModel):
    format: Literal["json", "csv", "m3u"]


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class MusicFilters(BaseModel):
    """Exact catalog constraints, kept separate from the semantic sound description."""

    title: str = ""
    artist: str = ""
    bpm_min: float = Field(0, ge=0, le=300)
    bpm_max: float = Field(300, ge=0, le=300)
    energy_min: float = Field(0, ge=0, le=1)
    energy_max: float = Field(1, ge=0, le=1)
    camelot_key: str = Field("", pattern=r"^(?:[1-9]|1[0-2])[AB]$|^$")

    @model_validator(mode="after")
    def ordered_ranges(self):
        if self.bpm_min > self.bpm_max or self.energy_min > self.energy_max:
            raise ValueError("筛选下限不能高于上限")
        return self

    def matches(self, track: Track) -> bool:
        return (
            self.title.casefold() in track.title.casefold()
            and self.artist.casefold() in track.artist.casefold()
            and self.bpm_min <= track.bpm <= self.bpm_max
            and self.energy_min <= track.energy <= self.energy_max
            and (not self.camelot_key or self.camelot_key == track.camelot_key)
        )


class MusicMatch(BaseModel):
    track: Track
    score: float | None = None
    description_similarity: float | None = None

    def context(self) -> dict[str, Any]:
        """Only music evidence enters the model context; never local file paths."""
        return {
            "track_id": self.track.id, "title": self.track.title, "artist": self.track.artist,
            "bpm": self.track.bpm, "key": self.track.key, "camelot_key": self.track.camelot_key,
            "energy": self.track.energy, "duration_sec": self.track.duration_sec,
            "analysis_status": self.track.analysis_status,
            "description": self.track.description,
            "description_model": self.track.description_model,
            "embedding_status": self.track.embedding_status,
            "score": self.score, "description_similarity": self.description_similarity,
        }
