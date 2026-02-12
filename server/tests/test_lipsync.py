from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from app.services.audio_utils import trim_pcm16le_silence
from app.services.lipsync import LipSyncService, LipSyncServiceError
from app.services.runpod import (
    RunPodClientConfigError,
    RunPodJob,
    RunPodJobState,
)


class FakeRunPodClient:
    def __init__(self, job: RunPodJob) -> None:
        self._job = job
        self.last_payload: dict | None = None

    async def submit_job(self, payload: dict) -> str:
        self.last_payload = payload
        return self._job.job_id

    async def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval_seconds: float,
        timeout_seconds: float,
    ) -> RunPodJob:
        _ = job_id, poll_interval_seconds, timeout_seconds
        return self._job


def _run(coro):  # noqa: ANN001
    return asyncio.run(coro)


def _write_png(tmp_path: Path) -> Path:
    image_path = tmp_path / "persona.png"
    # Tiny valid PNG header bytes are enough for payload construction tests.
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return image_path


def test_lipsync_service_maps_completed_job_to_video_url(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path)
    fake_job = RunPodJob(
        job_id="job_123",
        state=RunPodJobState.COMPLETED,
        output={"video_url": "https://cdn.example.com/out.mp4"},
    )
    fake_client = FakeRunPodClient(fake_job)
    service = LipSyncService(client=fake_client, model_name="musetalk")

    result = _run(
        service.generate_talk_from_pcm(
            pcm_bytes=b"\x00\x00" * 4800,
            sample_rate=24_000,
            persona_image_path=image_path,
            session_id="session_1",
            persona="joi",
        )
    )

    assert result.status == "done"
    assert result.talk_id == "job_123"
    assert result.result_url == "https://cdn.example.com/out.mp4"
    assert fake_client.last_payload is not None
    assert fake_client.last_payload["model"] == "musetalk"
    assert fake_client.last_payload["source_image"]["encoding"] == "base64"


def test_lipsync_service_maps_alternate_video_key(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path)
    fake_job = RunPodJob(
        job_id="job_alt_key",
        state=RunPodJobState.COMPLETED,
        output={"video": "https://cdn.example.com/alt.mp4"},
    )
    service = LipSyncService(client=FakeRunPodClient(fake_job))

    result = _run(
        service.generate_talk_from_pcm(
            pcm_bytes=b"\x00\x00" * 2400,
            sample_rate=24_000,
            persona_image_path=image_path,
        )
    )

    assert result.status == "done"
    assert result.result_url == "https://cdn.example.com/alt.mp4"


def test_lipsync_service_handles_completed_job_without_url(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path)
    fake_job = RunPodJob(
        job_id="job_missing_url",
        state=RunPodJobState.COMPLETED,
        output={"result": "ok"},
    )
    service = LipSyncService(client=FakeRunPodClient(fake_job))

    result = _run(
        service.generate_talk_from_pcm(
            pcm_bytes=b"\x00\x00" * 2400,
            sample_rate=24_000,
            persona_image_path=image_path,
        )
    )

    assert result.status == "failed"
    assert result.result_url is None
    assert "video URL" in (result.error or "")


def test_lipsync_service_rejects_empty_audio(tmp_path: Path) -> None:
    image_path = _write_png(tmp_path)
    fake_job = RunPodJob(job_id="unused", state=RunPodJobState.COMPLETED, output={})
    service = LipSyncService(client=FakeRunPodClient(fake_job))

    with pytest.raises(LipSyncServiceError):
        _run(
            service.generate_talk_from_pcm(
                pcm_bytes=b"",
                sample_rate=24_000,
                persona_image_path=image_path,
            )
        )


def test_runpod_client_requires_endpoint_or_explicit_urls() -> None:
    from app.services.runpod import RunPodClient

    with pytest.raises(RunPodClientConfigError):
        RunPodClient(api_key="token", endpoint_id=None, run_url=None, status_url_template=None)


def test_lipsync_service_reports_unavailable_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = _write_png(tmp_path)
    monkeypatch.setattr("app.services.lipsync.settings.lipsync_direct_url", None, raising=False)
    monkeypatch.setattr(LipSyncService, "_build_default_client", staticmethod(lambda: None))
    service = LipSyncService(client=None, direct_url=None)

    result = _run(
        service.generate_talk_from_pcm(
            pcm_bytes=b"\x00\x00" * 2400,
            sample_rate=24_000,
            persona_image_path=image_path,
        )
    )

    assert result.status == "unavailable"
    assert result.result_url is None
    assert "Lip-sync provider unavailable" in (result.error or "")


def test_direct_url_expands_root_to_common_handler_paths() -> None:
    service = LipSyncService(client=None, direct_url="https://abc123-8000.proxy.runpod.net/")

    assert service._direct_request_urls() == [
        "https://abc123-8000.proxy.runpod.net/generate",
        "https://abc123-8000.proxy.runpod.net/run",
        "https://abc123-8000.proxy.runpod.net/predict",
        "https://abc123-8000.proxy.runpod.net/invocations",
        "https://abc123-8000.proxy.runpod.net/",
    ]


def test_direct_url_retries_common_paths_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = _write_png(tmp_path)
    requested_urls: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            _ = args, kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            _ = exc_type, exc, tb
            return False

        async def post(self, url: str, json: dict, headers: dict | None = None) -> httpx.Response:
            _ = json, headers
            requested_urls.append(url)
            request = httpx.Request("POST", url)
            if url.endswith("/predict"):
                return httpx.Response(
                    200,
                    json={"video_url": "https://cdn.example.com/direct.mp4"},
                    request=request,
                )
            return httpx.Response(
                502,
                json={"error": "bad gateway"},
                request=request,
            )

    monkeypatch.setattr("app.services.lipsync.httpx.AsyncClient", FakeAsyncClient)
    service = LipSyncService(client=None, direct_url="https://abc123-8000.proxy.runpod.net/")

    result = _run(
        service.generate_talk_from_pcm(
            pcm_bytes=b"\x00\x00" * 2400,
            sample_rate=24_000,
            persona_image_path=image_path,
            persona="joi",
        )
    )

    assert result.status == "done"
    assert result.result_url == "https://cdn.example.com/direct.mp4"
    assert requested_urls == [
        "https://abc123-8000.proxy.runpod.net/generate",
        "https://abc123-8000.proxy.runpod.net/run",
        "https://abc123-8000.proxy.runpod.net/predict",
    ]


def test_trim_pcm16le_silence_removes_leading_and_trailing_quiet() -> None:
    # 4 silent samples + 3 voiced samples + 4 silent samples
    samples = [0, 0, 0, 0, 1600, -1700, 1800, 0, 0, 0, 0]
    pcm = b"".join(int(s).to_bytes(2, byteorder="little", signed=True) for s in samples)
    trimmed = trim_pcm16le_silence(pcm, threshold=500, pad_samples=0)
    trimmed_samples = [
        int.from_bytes(trimmed[i : i + 2], byteorder="little", signed=True)
        for i in range(0, len(trimmed), 2)
    ]
    assert trimmed_samples == [1600, -1700, 1800]

