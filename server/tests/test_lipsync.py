from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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

