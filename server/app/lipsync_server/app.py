from __future__ import annotations

import base64
import os
from typing import Any

from fastapi import FastAPI, HTTPException

from app.lipsync_server.backends.base import InferenceInputs, LipSyncBackend
from app.lipsync_server.backends.noop import NoopBackend
from app.lipsync_server.backends.wav2lip_subprocess import Wav2LipSubprocessBackend
from app.lipsync_server.models import GenerateRequest, GenerateResponse


def _select_backend() -> LipSyncBackend:
  backend = (os.getenv("LIPSYNC_SERVER_BACKEND") or "").strip().lower() or "noop"
  if backend == "wav2lip":
    return Wav2LipSubprocessBackend()
  if backend == "noop":
    return NoopBackend()
  raise RuntimeError(f"Unknown LIPSYNC_SERVER_BACKEND: {backend}")


def create_app() -> FastAPI:
  app = FastAPI(title="Deckard LipSync Server", version="0.1.0")
  try:
    backend = _select_backend()
  except Exception as exc:
    # Server should still start so the user can hit /health and see what's wrong.
    backend = NoopBackend()
    app.state.backend_init_error = str(exc)
  else:
    app.state.backend_init_error = None

  app.state.backend = backend

  @app.get("/health")
  async def health() -> dict[str, Any]:
    return {
      "service": "deckard-lipsync",
      "status": "ok",
      "backend": (os.getenv("LIPSYNC_SERVER_BACKEND") or "noop"),
      "backend_init_error": app.state.backend_init_error,
    }

  @app.post("/generate", response_model=GenerateResponse)
  async def generate(req: GenerateRequest) -> GenerateResponse:
    if app.state.backend_init_error:
      raise HTTPException(status_code=503, detail=app.state.backend_init_error)

    try:
      image_bytes = base64.b64decode(req.source_image.data)
      audio_wav_bytes = base64.b64decode(req.audio.data)
    except Exception as exc:
      raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}")

    media_type = (req.source_image.media_type or "image/png").strip().lower()
    inputs = InferenceInputs(
      image_bytes=image_bytes,
      image_media_type=media_type,
      audio_wav_bytes=audio_wav_bytes,
      sample_rate=req.audio.sample_rate,
    )

    try:
      mp4_bytes = await app.state.backend.generate(inputs=inputs)
    except Exception as exc:
      return GenerateResponse(error=str(exc))

    # Return base64 so Deckard can emit a data: URL without requiring public ports or object storage.
    return GenerateResponse(video_base64=base64.b64encode(mp4_bytes).decode("ascii"))

  return app

