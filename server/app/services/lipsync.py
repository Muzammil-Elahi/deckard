from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import settings
from app.services.audio_utils import pcm16le_to_wav, trim_pcm16le_silence
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
    """Provider-agnostic lip-sync orchestrator.

    Resolution order:
    1. Direct mode (`LIPSYNC_DIRECT_URL`) for colocated GPU inference
    2. RunPod serverless mode

    Performance-sensitive behavior in this class:
    - Reuses persistent HTTP clients (keep-alive)
    - Caches avatar image bytes
    - Trims leading/trailing silence before inference
    - Remembers the last successful direct route to avoid route probing on every turn
    """

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
        self._trim_silence = settings.lipsync_trim_silence
        self._silence_threshold = max(0, settings.lipsync_silence_threshold)
        self._silence_pad_ms = max(0, settings.lipsync_silence_pad_ms)
        self._max_audio_seconds = max(0.0, settings.lipsync_max_audio_seconds)
        self._direct_preferred_path = (
            settings.lipsync_direct_preferred_path.strip()
            if settings.lipsync_direct_preferred_path
            else "/generate"
        )
        self._last_successful_direct_url: Optional[str] = None
        self._direct_http_client: Optional[httpx.AsyncClient] = None
        self._persona_image_cache: dict[Path, tuple[bytes, str]] = {}
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

    def _optimized_pcm(self, pcm_bytes: bytes, *, sample_rate: int) -> bytes:
        """Apply low-cost audio optimizations before lip-sync inference.

        The goal is to reduce payload and model compute without changing user-visible
        content quality for normal conversational turns.
        """
        optimized = pcm_bytes
        if self._trim_silence and sample_rate > 0:
            pad_samples = int(sample_rate * (self._silence_pad_ms / 1000.0))
            optimized = trim_pcm16le_silence(
                optimized,
                threshold=self._silence_threshold,
                pad_samples=pad_samples,
            )
        if self._max_audio_seconds > 0 and sample_rate > 0:
            max_bytes = int(sample_rate * self._max_audio_seconds * 2)
            if max_bytes > 0 and len(optimized) > max_bytes:
                optimized = optimized[:max_bytes]
        return optimized

    def _load_persona_image(self, path: Path) -> tuple[bytes, str]:
        """Load and cache avatar image bytes for repeated requests."""
        resolved = path.resolve()
        cached = self._persona_image_cache.get(resolved)
        if cached is not None:
            return cached
        image_bytes = resolved.read_bytes()
        mime = "image/png" if resolved.suffix.lower() == ".png" else "image/jpeg"
        value = (image_bytes, mime)
        self._persona_image_cache[resolved] = value
        return value

    def _direct_client(self, timeout_seconds: float) -> httpx.AsyncClient:
        """Return a shared direct-mode HTTP client with keep-alive enabled."""
        if self._direct_http_client is None:
            self._direct_http_client = httpx.AsyncClient(
                timeout=timeout_seconds,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                http2=True,
            )
        return self._direct_http_client

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
        The last-known working route is always prioritized first.
        """
        if not self._direct_url:
            return []
        base_url = self._direct_url.strip()
        if not base_url:
            return []

        parsed = urlsplit(base_url)
        path = (parsed.path or "").strip()
        candidates: list[str] = []
        if self._last_successful_direct_url:
            candidates.append(self._last_successful_direct_url)
        if path not in {"", "/"}:
            if base_url not in candidates:
                candidates.append(base_url)
            return candidates

        preferred_route = self._direct_preferred_path
        if preferred_route and not preferred_route.startswith("/"):
            preferred_route = f"/{preferred_route}"
        routes: list[str] = []
        if preferred_route:
            routes.append(preferred_route)
        routes.extend(["/generate", "/run", "/predict", "/invocations", "/"])

        for route in routes:
            candidate = urlunsplit((parsed.scheme, parsed.netloc, route, parsed.query, ""))
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    async def _generate_via_direct_url(
        self, payload: dict[str, Any], timeout_seconds: float
    ) -> LipSyncResult:
        """POST payload to direct mode endpoint and parse a video URL from response.

        We intentionally try multiple handler-compatible routes in sequence so one
        deployment can tolerate image/handler differences without code changes.
        """
        attempted_errors: list[str] = []
        target_urls = self._direct_request_urls()
        if not target_urls:
            return LipSyncResult(
                talk_id="",
                status="failed",
                result_url=None,
                error="LIPSYNC_DIRECT_URL is not configured",
            )

        client = self._direct_client(timeout_seconds)
        for target_url in target_urls:
            try:
                response = await client.post(target_url, json=payload)
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
                # Stick to the first known-good route on subsequent calls.
                self._last_successful_direct_url = target_url
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
        """Generate avatar video from assistant PCM bytes.

        Both direct and RunPod modes expect the same logical payload:
        source image + WAV audio + options metadata.
        """
        if self._direct_url:
            # Your own server on GPU pod: same payload, single POST, no RunPod API.
            if not pcm_bytes:
                return LipSyncResult(talk_id="", status="failed", result_url=None, error="No audio")
            if not persona_image_path.exists():
                return LipSyncResult(
                    talk_id="", status="failed", result_url=None, error="Persona image missing"
                )
            # Apply silence trimming/capping before serialization to reduce inference cost.
            optimized_pcm = self._optimized_pcm(pcm_bytes, sample_rate=sample_rate)
            wav_bytes = pcm16le_to_wav(optimized_pcm, sample_rate=sample_rate, channels=1)
            image_bytes, mime = self._load_persona_image(persona_image_path)
            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
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
        # Shared optimization path for serverless mode.
        optimized_pcm = self._optimized_pcm(pcm_bytes, sample_rate=sample_rate)
        if len(optimized_pcm) > self._max_pcm_bytes:
            raise LipSyncServiceError(
                f"Assistant audio exceeds max size ({len(optimized_pcm)} > {self._max_pcm_bytes} bytes)"
            )
        if not persona_image_path.exists():
            raise LipSyncServiceError(f"Persona image missing: {persona_image_path}")

        image_bytes, mime_type = self._load_persona_image(persona_image_path)
        if len(image_bytes) > self._max_image_bytes:
            raise LipSyncServiceError(
                f"Persona image exceeds max size ({len(image_bytes)} > {self._max_image_bytes} bytes)"
            )

        wav_bytes = pcm16le_to_wav(optimized_pcm, sample_rate=sample_rate, channels=1)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
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

    async def aclose(self) -> None:
        """Close shared HTTP clients used by lip-sync providers."""
        if self._direct_http_client is not None:
            await self._direct_http_client.aclose()
            self._direct_http_client = None
        if self._client is not None and hasattr(self._client, "aclose"):
            await self._client.aclose()  # type: ignore[attr-defined]
