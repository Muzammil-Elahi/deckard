from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class InferenceInputs:
  """Normalized inputs for a lip-sync backend."""

  image_bytes: bytes
  image_media_type: str
  audio_wav_bytes: bytes
  sample_rate: int


class LipSyncBackend(Protocol):
  """Backend protocol: implement generate() to produce an MP4 payload."""

  async def generate(self, *, inputs: InferenceInputs) -> bytes:  # mp4 bytes
    raise NotImplementedError

