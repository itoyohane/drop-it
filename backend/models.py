from typing import Literal

from pydantic import BaseModel, Field


class Track(BaseModel):
    id: str
    title: str
    artist: str
    path: str
    duration_sec: int
    bpm: float
    key: str
    energy: float
    mood: list[str]
    role: str


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
    alternatives: list[str] = []


class AgentEvent(BaseModel):
    agent: Literal["Curator", "Planner", "Critic"]
    status: Literal["done", "revised", "approved"]
    message: str


class Playlist(BaseModel):
    id: str
    brief: Brief
    tracks: list[PlaylistTrack]
    duration_sec: int
    status: Literal["draft", "approved"] = "draft"
    revision: int = 1
    report: list[str]
    trace: list[AgentEvent]


class ReorderRequest(BaseModel):
    track_ids: list[str]


class ExportRequest(BaseModel):
    format: Literal["json", "csv", "m3u"]
