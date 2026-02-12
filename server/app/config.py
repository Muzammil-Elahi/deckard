"""Configuration helpers for the orchestrator service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass
class Settings:
    """Centralized environment-driven configuration.

    Replace this lightweight dataclass with a pydantic BaseSettings implementation
    once dependencies are wired up. For now it documents the knobs the orchestrator
    expects without enforcing them.
    """

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    # RunPod serverless settings for the self-hosted lip-sync model.
    runpod_api_key: Optional[str] = os.getenv("RUNPOD_API_KEY")
    runpod_api_base: str = os.getenv("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
    runpod_endpoint_id: Optional[str] = os.getenv("RUNPOD_ENDPOINT_ID")
    # Optional overrides for custom proxy routes or non-standard handlers.
    runpod_run_url: Optional[str] = os.getenv("RUNPOD_RUN_URL")
    runpod_status_url_template: Optional[str] = os.getenv("RUNPOD_STATUS_URL_TEMPLATE")
    runpod_request_timeout_seconds: float = float(os.getenv("RUNPOD_REQUEST_TIMEOUT_SECONDS", "30"))
    runpod_poll_interval_seconds: float = float(os.getenv("RUNPOD_POLL_INTERVAL_SECONDS", "0.4"))
    runpod_job_timeout_seconds: float = float(os.getenv("RUNPOD_JOB_TIMEOUT_SECONDS", "120"))
    lipsync_model_name: str = os.getenv("LIPSYNC_MODEL_NAME", "musetalk")
    # Direct lip-sync URL (e.g. your own server on a GPU pod). No RunPod API needed.
    # POST same payload as RunPod; response JSON with video_url/result_url/url.
    lipsync_direct_url: Optional[str] = os.getenv("LIPSYNC_DIRECT_URL")
    lipsync_direct_timeout_seconds: float = float(os.getenv("LIPSYNC_DIRECT_TIMEOUT_SECONDS", "120"))
    # Preferred route tried first when LIPSYNC_DIRECT_URL is a proxy root.
    lipsync_direct_preferred_path: str = os.getenv("LIPSYNC_DIRECT_PREFERRED_PATH", "/generate")
    # Audio payload optimizations for lip-sync requests.
    lipsync_trim_silence: bool = _env_bool("LIPSYNC_TRIM_SILENCE", True)
    lipsync_silence_threshold: int = int(os.getenv("LIPSYNC_SILENCE_THRESHOLD", "500"))
    lipsync_silence_pad_ms: int = int(os.getenv("LIPSYNC_SILENCE_PAD_MS", "120"))
    # Optional hard cap for long assistant turns; 0 disables truncation.
    lipsync_max_audio_seconds: float = float(os.getenv("LIPSYNC_MAX_AUDIO_SECONDS", "0"))
    # Default delivery strategy for assistant output:
    # - synced: coordinated audio + avatar video
    # - fast: low-latency audio first, avatar video follows opportunistically
    response_mode_default: str = os.getenv("RESPONSE_MODE_DEFAULT", "synced").strip().lower() or "synced"

    # Legacy runpod URL retained for backward compatibility with old docs.
    runpod_base_url: str = os.getenv("RUNPOD_BASE_URL", "https://runpod.example.com")
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_service_role_key: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    # Memory guardrails
    memory_max_local_entries: int = int(os.getenv("MEMORY_MAX_LOCAL_ENTRIES", "80"))
    memory_max_recall_items: int = int(os.getenv("MEMORY_MAX_RECALL_ITEMS", "5"))
    memory_max_summary_chars: int = int(os.getenv("MEMORY_MAX_SUMMARY_CHARS", "650"))
    memory_local_ttl_seconds: float = float(os.getenv("MEMORY_LOCAL_TTL_SECONDS", "86400"))
    memory_dedupe_window_seconds: float = float(os.getenv("MEMORY_DEDUPE_WINDOW_SECONDS", "900"))
    memory_remote_timeout_seconds: float = float(os.getenv("MEMORY_REMOTE_TIMEOUT_SECONDS", "0.45"))
    memory_remote_cache_ttl_seconds: float = float(os.getenv("MEMORY_REMOTE_CACHE_TTL_SECONDS", "30"))

    # Safety checks
    enable_secret_safety_checks: bool = _env_bool("ENABLE_SECRET_SAFETY_CHECKS", True)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Legacy D-ID settings retained for compatibility while migrating.
    did_api_key: Optional[str] = os.getenv("DID_API_KEY")
    # Optional webhook endpoint invoked by D-ID once a talk is ready.
    did_webhook_url: Optional[str] = os.getenv("DID_WEBHOOK_URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance to avoid repeated environment reads."""

    return Settings()


settings = get_settings()
