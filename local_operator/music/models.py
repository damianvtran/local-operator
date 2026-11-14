"""Strict neutral music controls; unsupported provider operations fail upstream.

Durations are explicit milliseconds because Radient's fixed quote is distinct
from the decoded playback duration. Owned references use opaque request IDs,
never arbitrary URLs that could turn a paid generation into a server-side fetch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MusicSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1)
    track_id: str | None = None


class MusicRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> MusicRange:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be after start_ms")
        return self


class MusicSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["generate", "keep"]
    duration_ms: int | None = Field(default=None, ge=3000, le=120000)
    lyrics: str | None = None
    styles: list[str] | None = None
    negative_styles: list[str] | None = None
    source: MusicSource | None = None
    range: MusicRange | None = None

    @model_validator(mode="after")
    def representable(self) -> MusicSection:
        if self.kind == "keep":
            if self.source is None or self.range is None:
                raise ValueError("keep requires an owned source and range")
            if any(
                value is not None
                for value in (self.duration_ms, self.lyrics, self.styles, self.negative_styles)
            ):
                raise ValueError("keep accepts only source and range")
        elif self.duration_ms is None or self.source is not None or self.range is not None:
            raise ValueError("generate requires duration_ms and no retained source")
        return self


class GenerateMusicParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str | None = None
    duration_ms: int = Field(ge=3000, le=300000)
    instrumental: bool | None = None
    lyrics: str | None = None
    styles: list[str] | None = None
    negative_styles: list[str] | None = None
    sections: list[MusicSection] | None = None
    title: str | None = None
    variations: int = Field(default=1, ge=1, le=2)
    model: Literal["radient/music"] = "radient/music"

    @model_validator(mode="after")
    def intent(self) -> GenerateMusicParams:
        if self.sections:
            if any(
                value is not None
                for value in (self.prompt, self.lyrics, self.styles, self.negative_styles)
            ):
                raise ValueError("sections cannot be combined with prompt, lyrics or global styles")
            duration = sum(
                section.duration_ms
                or (section.range.end_ms - section.range.start_ms if section.range else 0)
                for section in self.sections
            )
            if duration != self.duration_ms:
                raise ValueError("section durations must sum to duration_ms")
        elif not self.prompt and self.lyrics is None:
            raise ValueError("prompt or exact lyrics is required")
        if self.instrumental and (self.lyrics is not None or self.sections):
            raise ValueError("instrumental cannot be combined with lyrics or a composition plan")
        return self

    def body(self) -> dict[str, object]:
        return {"operation": "compose", "format": "mp3", **self.model_dump(exclude_none=True)}


class MusicStatusParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str | None = None
    action: Literal["status", "cancel", "models"] = "status"

    @model_validator(mode="after")
    def job_required(self) -> MusicStatusParams:
        if self.action != "models" and not self.job_id:
            raise ValueError("job_id is required for status or cancel")
        return self
