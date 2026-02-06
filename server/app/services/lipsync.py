from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.services.audio_utils import pcm16le_to_wav
from app.services.runpod import (
    RunPodClient,
    RunPodClientConfigError,
    RunPodClientError,
    RunPodJobState,
)


class LipSyncServiceError(RuntimeError):
    """Raised when lip-sync inference fails."""


class LipSyncServiceConfigError(LipSyncServiceError):
    """Raised when lip-sync service configuration is invalid."""


@dataclass(slots=True)
class LipSyncResult:
    """Unified response model for lip-sync generation requests."""

    talk_id: str
    status: str
    result_url: Optional[str]
    error: Optional[str] = None


def resolve_persona_image(persona: str) -> Path:
    """Map persona key -> local reference image path used for lip-sync."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    public = repo_root / "web" / "public"
    mapping = {
        "joi": public / "joi.png",
        "officer_k": public / "officer_k.png",
        "officer_j": public / "officer_j.png",
    }
    key = (persona or "joi").lower().strip()
    if key not in mapping:
        key = "joi"
    path = mapping[key]
    if not path.exists():
        raise FileNotFoundError(f"Persona image not found: {path}")
    return path


class LipSyncService:
    """RunPod-backed lip-sync service for self-hosted MuseTalk-style jobs."""

    def __init__(
        self,
        *,
        client: Optional[RunPodClient] = None,
        poll_interval_seconds: float = 0.75,
        timeout_seconds: float = 120.0,
        model_name: str = "musetalk",
        max_pcm_bytes: int = 8 * 1024 * 1024,
        max_image_bytes: int = 6 * 1024 * 1024,
    ) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._model_name = model_name
        self._max_pcm_bytes = max_pcm_bytes
        self._max_image_bytes = max_image_bytes
        self._client = client or self._build_default_client()

    @staticmethod
    def _build_default_client() -> RunPodClient:
        try:
            return RunPodClient(
                api_key=settings.runpod_api_key or "",
                endpoint_id=settings.runpod_endpoint_id,
                api_base=settings.runpod_api_base,
                run_url=settings.runpod_run_url,
                status_url_template=settings.runpod_status_url_template,
                request_timeout_seconds=settings.runpod_request_timeout_seconds,
            )
        except RunPodClientConfigError as exc:
            raise LipSyncServiceConfigError(str(exc)) from exc

    def _extract_result_url(self, output: Any) -> Optional[str]:
        """Best-effort URL extraction across common RunPod handler schemas."""
        if isinstance(output, str) and output.strip().startswith(("http://", "https://")):
            return output.strip()

        if isinstance(output, list):
            for item in output:
                url = self._extract_result_url(item)
                if url:
                    return url
            return None

        if isinstance(output, dict):
            direct_keys = (
                "result_url",
                "url",
                "video_url",
                "output_url",
                "mp4_url",
            )
            for key in direct_keys:
                value = output.get(key)
                if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                    return value.strip()
            for nested_key in ("output", "result", "data"):
                if nested_key in output:
                    url = self._extract_result_url(output[nested_key])
                    if url:
                        return url
        return None

    async def generate_talk_from_pcm(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate: int,
        persona_image_path: Path,
        session_id: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> LipSyncResult:
        if not pcm_bytes:
            raise LipSyncServiceError("Assistant audio buffer is empty")
        if len(pcm_bytes) > self._max_pcm_bytes:
            raise LipSyncServiceError(
                f"Assistant audio exceeds max size ({len(pcm_bytes)} > {self._max_pcm_bytes} bytes)"
            )
        if not persona_image_path.exists():
            raise LipSyncServiceError(f"Persona image missing: {persona_image_path}")

        image_bytes = persona_image_path.read_bytes()
        if len(image_bytes) > self._max_image_bytes:
            raise LipSyncServiceError(
                f"Persona image exceeds max size ({len(image_bytes)} > {self._max_image_bytes} bytes)"
            )

        wav_bytes = pcm16le_to_wav(pcm_bytes, sample_rate=sample_rate, channels=1)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        mime_type = "image/png" if persona_image_path.suffix.lower() == ".png" else "image/jpeg"
        payload: dict[str, Any] = {
            "model": self._model_name,
            "session_id": session_id or "deckard-session",
            "persona": (persona or "joi").lower(),
            "source_image": {
                "encoding": "base64",
                "media_type": mime_type,
                "data": image_b64,
            },
            "audio": {
                "encoding": "base64",
                "format": "wav",
                "sample_rate": sample_rate,
                "data": audio_b64,
            },
            # Handler hint: many MuseTalk handlers accept runtime options under
            # dedicated options/config keys. Passing these avoids hard-coding
            # values in deployment scripts.
            "options": {
                "engine": "musetalk",
                "return_format": "url",
            },
        }

        try:
            job_id = await self._client.submit_job(payload)
            job = await self._client.wait_for_completion(
                job_id,
                poll_interval_seconds=self._poll_interval_seconds,
                timeout_seconds=self._timeout_seconds,
            )
        except RunPodClientError as exc:
            raise LipSyncServiceError(f"RunPod lip-sync request failed: {exc}") from exc

        if job.state == RunPodJobState.COMPLETED:
            url = self._extract_result_url(job.output)
            if not url:
                return LipSyncResult(
                    talk_id=job.job_id,
                    status="failed",
                    result_url=None,
                    error="RunPod completed without a video URL in output payload",
                )
            return LipSyncResult(
                talk_id=job.job_id,
                status="done",
                result_url=url,
            )

        terminal_error = job.error or f"RunPod job ended in state {job.state.value}"
        status = "timeout" if job.state == RunPodJobState.TIMED_OUT else "failed"
        return LipSyncResult(
            talk_id=job.job_id,
            status=status,
            result_url=self._extract_result_url(job.output),
            error=terminal_error,
        )

