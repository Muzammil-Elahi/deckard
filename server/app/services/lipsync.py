from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import settings
from app.services.audio_utils import pcm16le_to_wav
from app.services.runpod import (
    RunPodClient,
    RunPodClientConfigError,
    RunPodClientError,
    RunPodJobState,
    is_runpod_configured,
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
    """Lip-sync service: direct HTTP URL (your GPU server) or RunPod Serverless."""

    def __init__(
        self,
        *,
        client: Optional[RunPodClient] = None,
        direct_url: Optional[str] = None,
        poll_interval_seconds: float = 0.75,
        timeout_seconds: float = 120.0,
        direct_timeout_seconds: Optional[float] = None,
        model_name: str = "musetalk",
        max_pcm_bytes: int = 8 * 1024 * 1024,
        max_image_bytes: int = 6 * 1024 * 1024,
    ) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._direct_timeout_seconds = (
            direct_timeout_seconds
            if direct_timeout_seconds is not None
            else getattr(settings, "lipsync_direct_timeout_seconds", None)
            or timeout_seconds
        )
        self._model_name = model_name
        self._max_pcm_bytes = max_pcm_bytes
        self._max_image_bytes = max_image_bytes
        # Explicit injected client is highest precedence (e.g. tests/fakes).
        if client is not None:
            self._direct_url = None
            self._client = client
            return

        # Otherwise prefer direct URL (your own GPU pod HTTP server), then RunPod API.
        url = (direct_url or "").strip() or (getattr(settings, "lipsync_direct_url", None) or "").strip() or None
        if url:
            self._direct_url = url
            self._client = None
            return

        self._direct_url = None
        self._client = self._build_default_client()

    @staticmethod
    def _build_default_client() -> RunPodClient | None:
        """Return a RunPod client if configured; otherwise None (lip-sync disabled)."""
        if not is_runpod_configured(
            api_key=settings.runpod_api_key,
            endpoint_id=settings.runpod_endpoint_id,
            run_url=settings.runpod_run_url,
            status_url_template=settings.runpod_status_url_template,
        ):
            return None
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
        if isinstance(output, str) and output.strip().startswith(("http://", "https://", "data:video/")):
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
                "video",
                "videoUrl",
                "output_url",
                "mp4_url",
                "mp4",
                "file_url",
                "link",
                "file",
            )
            for key in direct_keys:
                value = output.get(key)
                if isinstance(value, str) and value.strip().startswith(("http://", "https://", "data:video/")):
                    return value.strip()
            for b64_key in ("video_base64", "base64_video", "video_b64"):
                b64_value = output.get(b64_key)
                if isinstance(b64_value, str) and b64_value.strip():
                    return f"data:video/mp4;base64,{b64_value.strip()}"
            for nested_key in ("output", "result", "data", "response"):
                if nested_key in output:
                    url = self._extract_result_url(output[nested_key])
                    if url:
                        return url
        return None

    @staticmethod
    def _extract_error_message(payload: Any) -> Optional[str]:
        """Best-effort extraction of human-readable error details from JSON payloads."""
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for nested_key in ("output", "result", "data", "response"):
                if nested_key in payload:
                    nested = LipSyncService._extract_error_message(payload[nested_key])
                    if nested:
                        return nested
        if isinstance(payload, list):
            for item in payload:
                nested = LipSyncService._extract_error_message(item)
                if nested:
                    return nested
        return None

    def _direct_request_urls(self) -> list[str]:
        """Return URL candidates for direct mode.

        If LIPSYNC_DIRECT_URL is a bare Runpod proxy root, try common handler paths.
        """
        if not self._direct_url:
            return []
        base_url = self._direct_url.strip()
        if not base_url:
            return []

        parsed = urlsplit(base_url)
        path = (parsed.path or "").strip()
        candidates: list[str] = [base_url]
        if path not in {"", "/"}:
            return candidates

        for route in ("/run", "/generate", "/predict", "/invocations"):
            candidate = urlunsplit((parsed.scheme, parsed.netloc, route, parsed.query, ""))
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    async def _generate_via_direct_url(
        self, payload: dict[str, Any], timeout_seconds: float
    ) -> LipSyncResult:
        """POST payload to LIPSYNC_DIRECT_URL and parse JSON for video URL."""
        attempted_errors: list[str] = []
        target_urls = self._direct_request_urls()
        if not target_urls:
            return LipSyncResult(
                talk_id="",
                status="failed",
                result_url=None,
                error="LIPSYNC_DIRECT_URL is not configured",
            )

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for target_url in target_urls:
                try:
                    response = await client.post(
                        target_url,
                        json=payload,
                        headers={"Content-Type": "application/json", "Accept": "application/json"},
                    )
                except httpx.HTTPError as exc:
                    attempted_errors.append(f"{target_url}: request error ({exc})")
                    continue

                try:
                    response.raise_for_status()
                except httpx.HTTPError:
                    status = response.status_code
                    reason = response.reason_phrase or "HTTP error"
                    attempted_errors.append(f"{target_url}: {status} {reason}")
                    continue

                try:
                    data = response.json()
                except Exception as exc:
                    attempted_errors.append(f"{target_url}: invalid JSON ({exc})")
                    continue

                url = self._extract_result_url(data)
                if url:
                    return LipSyncResult(talk_id="direct", status="done", result_url=url)

                error_msg = self._extract_error_message(data)
                if error_msg:
                    attempted_errors.append(f"{target_url}: {error_msg}")
                    continue
                attempted_errors.append(f"{target_url}: response had no video URL")

        max_entries = 4
        summary = " | ".join(attempted_errors[:max_entries])
        if len(attempted_errors) > max_entries:
            summary = f"{summary} | ... ({len(attempted_errors) - max_entries} more)"
        return LipSyncResult(
            talk_id="",
            status="failed",
            result_url=None,
            error=(
                "Direct lip-sync request failed. "
                f"Tried {len(target_urls)} URL(s): {summary}"
            ),
        )

    async def generate_talk_from_pcm(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate: int,
        persona_image_path: Path,
        session_id: Optional[str] = None,
        persona: Optional[str] = None,
    ) -> LipSyncResult:
        if self._direct_url:
            # Your own server on GPU pod: same payload, single POST, no RunPod API.
            if not pcm_bytes:
                return LipSyncResult(talk_id="", status="failed", result_url=None, error="No audio")
            if not persona_image_path.exists():
                return LipSyncResult(
                    talk_id="", status="failed", result_url=None, error="Persona image missing"
                )
            wav_bytes = pcm16le_to_wav(pcm_bytes, sample_rate=sample_rate, channels=1)
            image_b64 = base64.b64encode(persona_image_path.read_bytes()).decode("ascii")
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
            mime = "image/png" if persona_image_path.suffix.lower() == ".png" else "image/jpeg"
            payload = {
                "model": self._model_name,
                "session_id": session_id or "deckard-session",
                "persona": (persona or "joi").lower(),
                "source_image": {"encoding": "base64", "media_type": mime, "data": image_b64},
                "audio": {
                    "encoding": "base64",
                    "format": "wav",
                    "sample_rate": sample_rate,
                    "data": audio_b64,
                },
                "options": {"engine": "musetalk", "return_format": "url"},
            }
            timeout = getattr(settings, "lipsync_direct_timeout_seconds", self._direct_timeout_seconds)
            return await self._generate_via_direct_url(payload, float(timeout))

        if self._client is None:
            return LipSyncResult(
                talk_id="",
                status="unavailable",
                result_url=None,
                error=(
                    "Lip-sync provider unavailable. Configure LIPSYNC_DIRECT_URL "
                    "or RunPod (RUNPOD_ENDPOINT_ID, or RUNPOD_RUN_URL + RUNPOD_STATUS_URL_TEMPLATE)."
                ),
            )
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
