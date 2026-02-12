from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

import httpx


class RunPodClientError(RuntimeError):
    """Raised when RunPod request/response handling fails."""


class RunPodClientConfigError(RunPodClientError):
    """Raised when required RunPod client configuration is missing."""


class RunPodJobState(str, Enum):
    """Canonical RunPod serverless job states."""

    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunPodJobState.COMPLETED,
            RunPodJobState.FAILED,
            RunPodJobState.CANCELLED,
            RunPodJobState.TIMED_OUT,
        }


@dataclass(slots=True)
class RunPodJob:
    """Normalized representation of a RunPod job status payload."""

    job_id: str
    state: RunPodJobState
    output: Any = None
    error: Optional[str] = None
    raw: Optional[dict[str, Any]] = None


def is_runpod_configured(
    *,
    api_key: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    run_url: Optional[str] = None,
    status_url_template: Optional[str] = None,
) -> bool:
    """Return True if enough env is set to create a RunPodClient."""
    key = (api_key or "").strip()
    if not key:
        return False
    eid = (endpoint_id or "").strip() or None
    run = (run_url or "").strip() or None
    status = (status_url_template or "").strip() or None
    if run and not status:
        return False
    return bool(eid or (run and status))


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
            if isinstance(dumped, dict):
                return dumped
        except Exception:  # pragma: no cover - defensive conversion fallback
            pass
    raise RunPodClientError(f"Expected dict payload from RunPod, got {type(value)!r}")


class RunPodClient:
    """HTTP client for RunPod Serverless v2 endpoints.

    Supports both:
    - Canonical endpoint-id mode using `api_base` + `endpoint_id`
    - Fully custom URL mode using explicit `run_url` and `status_url_template`
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint_id: Optional[str] = None,
        api_base: str = "https://api.runpod.ai/v2",
        run_url: Optional[str] = None,
        status_url_template: Optional[str] = None,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        normalized_key = (api_key or "").strip()
        if not normalized_key:
            raise RunPodClientConfigError("RUNPOD_API_KEY is required")

        explicit_run_url = (run_url or "").strip() or None
        explicit_status_template = (status_url_template or "").strip() or None
        normalized_endpoint_id = (endpoint_id or "").strip() or None

        if explicit_run_url and not explicit_status_template:
            raise RunPodClientConfigError(
                "RUNPOD_STATUS_URL_TEMPLATE is required when RUNPOD_RUN_URL is provided"
            )
        if not explicit_run_url and not normalized_endpoint_id:
            raise RunPodClientConfigError(
                "Set RUNPOD_ENDPOINT_ID or both RUNPOD_RUN_URL and RUNPOD_STATUS_URL_TEMPLATE"
            )

        self._api_key = normalized_key
        self._request_timeout_seconds = request_timeout_seconds

        if explicit_run_url:
            self._run_url = explicit_run_url
            self._status_url_template = explicit_status_template or ""
        else:
            normalized_base = api_base.rstrip("/")
            assert normalized_endpoint_id is not None
            self._run_url = f"{normalized_base}/{normalized_endpoint_id}/run"
            self._status_url_template = f"{normalized_base}/{normalized_endpoint_id}/status/{{job_id}}"
        # Keep-alive connections reduce TLS/session setup overhead on hot paths.
        self._http = httpx.AsyncClient(
            timeout=self._request_timeout_seconds,
            headers=self.headers,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            http2=True,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _status_url(self, job_id: str) -> str:
        return self._status_url_template.format(job_id=job_id)

    async def submit_job(self, payload: Mapping[str, Any]) -> str:
        """Submit an inference job and return the provider job id."""
        body = {"input": dict(payload)}
        response = await self._http.post(self._run_url, json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RunPodClientError(
                f"RunPod submit failed with status {response.status_code}"
            ) from exc
        data = _coerce_dict(response.json())
        job_id = data.get("id") or data.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RunPodClientError(f"RunPod submit response missing job id: {data}")
        return job_id

    async def get_job(self, job_id: str) -> RunPodJob:
        """Fetch and normalize current job status for a previously submitted job."""
        status_url = self._status_url(job_id)
        response = await self._http.get(status_url)
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RunPodClientError(
                f"RunPod status failed with status {response.status_code}"
            ) from exc
        data = _coerce_dict(response.json())

        raw_status = str(data.get("status") or "UNKNOWN").upper()
        try:
            state = RunPodJobState(raw_status)
        except ValueError:
            state = RunPodJobState.UNKNOWN

        error = data.get("error")
        if error is None:
            # Different handlers may surface terminal reasons via "message".
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                error = message.strip()

        return RunPodJob(
            job_id=job_id,
            state=state,
            output=data.get("output"),
            error=str(error) if error else None,
            raw=data,
        )

    async def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval_seconds: float = 0.75,
        timeout_seconds: float = 120.0,
    ) -> RunPodJob:
        """Poll job status until terminal state or timeout."""
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be > 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        latest: Optional[RunPodJob] = None

        while loop.time() < deadline:
            latest = await self.get_job(job_id)
            if latest.state.is_terminal:
                return latest
            await asyncio.sleep(poll_interval_seconds)

        if latest is None:
            return RunPodJob(
                job_id=job_id,
                state=RunPodJobState.TIMED_OUT,
                error="Timed out waiting for RunPod job status",
            )
        return RunPodJob(
            job_id=job_id,
            state=RunPodJobState.TIMED_OUT,
            output=latest.output,
            error="Timed out waiting for RunPod job completion",
            raw=latest.raw,
        )

    async def aclose(self) -> None:
        """Close persistent HTTP resources used by this client."""
        await self._http.aclose()
