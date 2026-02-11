from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Base64Media(BaseModel):
  encoding: Literal["base64"] = "base64"
  media_type: Optional[str] = None
  data: str = Field(..., min_length=1)


class Base64Audio(BaseModel):
  encoding: Literal["base64"] = "base64"
  format: str = Field("wav")
  sample_rate: int = Field(24_000, ge=8_000, le=96_000)
  data: str = Field(..., min_length=1)


class GenerateRequest(BaseModel):
  model: Optional[str] = None
  session_id: str = Field("deckard-session")
  persona: Optional[str] = None
  source_image: Base64Media
  audio: Base64Audio
  options: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
  # Deckard's `LipSyncService` understands these keys (and base64 variants).
  video_url: Optional[str] = None
  result_url: Optional[str] = None
  url: Optional[str] = None
  video_base64: Optional[str] = None
  error: Optional[str] = None

