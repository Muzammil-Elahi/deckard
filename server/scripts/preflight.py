"""Preflight checks for Deckard backend configuration.

Run this before starting the realtime service to catch common misconfiguration:
  uv run python scripts/preflight.py

Optional network checks:
  uv run python scripts/preflight.py --check-http
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv


VALID_RESPONSE_MODES = {"synced", "fast"}


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)


def _load_environment() -> None:
    """Load local env files in precedence order without overwriting existing vars."""
    script_dir = Path(__file__).resolve().parent
    server_dir = script_dir.parent
    repo_dir = server_dir.parent
    candidates = [
        server_dir / ".env.local",
        server_dir / ".env",
        repo_dir / ".env",
    ]
    for env_file in candidates:
        if env_file.exists():
            load_dotenv(env_file, override=False)


def _is_valid_http_url(value: str) -> bool:
    """Return True when value is an absolute HTTP(S) URL."""
    parsed = urlparse((value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _env_int(name: str, default: int, report: Report, *, minimum: int = 1) -> int:
    """Parse int env var and emit validation failures into the report."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        report.fail(f"{name} must be an integer. Got: {raw!r}")
        return default
    if value < minimum:
        report.fail(f"{name} must be >= {minimum}. Got: {value}")
    return value


def _env_float(name: str, default: float, report: Report, *, minimum: float = 0.0) -> float:
    """Parse float env var and emit validation failures into the report."""
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        report.fail(f"{name} must be a number. Got: {raw!r}")
        return default
    if value < minimum:
        report.fail(f"{name} must be >= {minimum}. Got: {value}")
    return value


def _mask(value: str) -> str:
    """Mask secret values for safe console output."""
    trimmed = value.strip()
    if len(trimmed) < 8:
        return "***"
    return f"{trimmed[:4]}...{trimmed[-4:]}"


def check_core_env(report: Report) -> None:
    """Validate core runtime requirements shared by all deployment modes."""
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not openai_key:
        report.fail("OPENAI_API_KEY is required.")
    else:
        if len(openai_key) < 20:
            report.warn("OPENAI_API_KEY looks unusually short; verify it is correct.")
        report.ok(f"OPENAI_API_KEY detected ({_mask(openai_key)}).")

    mode = (os.getenv("RESPONSE_MODE_DEFAULT", "synced") or "synced").strip().lower()
    if mode not in VALID_RESPONSE_MODES:
        report.fail(
            f"RESPONSE_MODE_DEFAULT must be one of {sorted(VALID_RESPONSE_MODES)}. Got: {mode!r}"
        )
    else:
        report.ok(f"RESPONSE_MODE_DEFAULT={mode}")


def check_lipsync_provider(report: Report) -> tuple[str, str | None]:
    """Validate that at least one lip-sync provider is correctly configured."""
    direct_url = (os.getenv("LIPSYNC_DIRECT_URL") or "").strip()
    runpod_key = (os.getenv("RUNPOD_API_KEY") or "").strip()
    endpoint_id = (os.getenv("RUNPOD_ENDPOINT_ID") or "").strip()
    run_url = (os.getenv("RUNPOD_RUN_URL") or "").strip()
    status_template = (os.getenv("RUNPOD_STATUS_URL_TEMPLATE") or "").strip()

    if direct_url:
        if not _is_valid_http_url(direct_url):
            report.fail(f"LIPSYNC_DIRECT_URL is not a valid HTTP(S) URL: {direct_url!r}")
        else:
            report.ok(f"Direct lip-sync mode enabled via LIPSYNC_DIRECT_URL ({direct_url}).")
        return ("direct", direct_url)

    if not runpod_key:
        report.fail(
            "No lip-sync provider configured. Set LIPSYNC_DIRECT_URL or RunPod settings."
        )
        return ("none", None)

    if endpoint_id:
        report.ok("RunPod configured with RUNPOD_ENDPOINT_ID.")
        return ("runpod_endpoint_id", None)

    if not run_url or not status_template:
        report.fail(
            "RunPod URL mode requires both RUNPOD_RUN_URL and RUNPOD_STATUS_URL_TEMPLATE."
        )
        return ("runpod_invalid", None)

    if not _is_valid_http_url(run_url):
        report.fail(f"RUNPOD_RUN_URL is not a valid HTTP(S) URL: {run_url!r}")
    else:
        report.ok("RUNPOD_RUN_URL is set.")

    if "{job_id}" not in status_template:
        report.fail("RUNPOD_STATUS_URL_TEMPLATE must include '{job_id}'.")
    elif not _is_valid_http_url(status_template.replace("{job_id}", "example")):
        report.fail(f"RUNPOD_STATUS_URL_TEMPLATE is not a valid HTTP(S) URL: {status_template!r}")
    else:
        report.ok("RUNPOD_STATUS_URL_TEMPLATE is set.")

    return ("runpod_urls", run_url)


def check_memory_guardrails(report: Report) -> None:
    """Validate memory-related guardrail values and flag suspicious combinations."""
    max_local = _env_int("MEMORY_MAX_LOCAL_ENTRIES", 80, report)
    max_recall = _env_int("MEMORY_MAX_RECALL_ITEMS", 5, report)
    _env_int("MEMORY_MAX_SUMMARY_CHARS", 650, report, minimum=120)
    ttl_seconds = _env_float("MEMORY_LOCAL_TTL_SECONDS", 86_400, report, minimum=1.0)
    dedupe_seconds = _env_float("MEMORY_DEDUPE_WINDOW_SECONDS", 900, report, minimum=1.0)
    _env_float("MEMORY_REMOTE_TIMEOUT_SECONDS", 0.45, report, minimum=0.05)
    _env_float("MEMORY_REMOTE_CACHE_TTL_SECONDS", 30, report, minimum=0.0)

    if max_recall > max_local:
        report.warn(
            "MEMORY_MAX_RECALL_ITEMS is greater than MEMORY_MAX_LOCAL_ENTRIES; recall will cap to available entries."
        )
    if dedupe_seconds > ttl_seconds:
        report.warn(
            "MEMORY_DEDUPE_WINDOW_SECONDS exceeds MEMORY_LOCAL_TTL_SECONDS; dedupe may outlive local memory entries."
        )
    report.ok("Memory guardrail settings parsed successfully.")


def check_lipsync_optimization(report: Report) -> None:
    """Validate performance tuning knobs for the lip-sync inference path."""
    poll = _env_float("RUNPOD_POLL_INTERVAL_SECONDS", 0.4, report, minimum=0.1)
    if poll > 1.0:
        report.warn(
            "RUNPOD_POLL_INTERVAL_SECONDS is high; completed jobs may take longer to surface."
        )

    preferred_path = (os.getenv("LIPSYNC_DIRECT_PREFERRED_PATH", "/generate") or "").strip()
    if preferred_path and not preferred_path.startswith("/"):
        report.fail("LIPSYNC_DIRECT_PREFERRED_PATH must start with '/'.")
    else:
        report.ok(f"LIPSYNC_DIRECT_PREFERRED_PATH={preferred_path or '/generate'}")

    trim_silence = (os.getenv("LIPSYNC_TRIM_SILENCE", "true") or "").strip().lower()
    trim_enabled = trim_silence not in {"0", "false", "no", "off"}
    if not trim_enabled:
        report.warn("LIPSYNC_TRIM_SILENCE is disabled; payloads may be larger and slower.")
    else:
        report.ok("LIPSYNC_TRIM_SILENCE enabled.")

    threshold = _env_int("LIPSYNC_SILENCE_THRESHOLD", 500, report, minimum=0)
    if threshold > 3000:
        report.warn(
            "LIPSYNC_SILENCE_THRESHOLD is very high; quiet speech may be trimmed too aggressively."
        )

    pad_ms = _env_int("LIPSYNC_SILENCE_PAD_MS", 120, report, minimum=0)
    if pad_ms > 500:
        report.warn(
            "LIPSYNC_SILENCE_PAD_MS is high; optimization impact will be reduced."
        )

    max_audio_seconds = _env_float("LIPSYNC_MAX_AUDIO_SECONDS", 0, report, minimum=0.0)
    if 0 < max_audio_seconds < 2.0:
        report.warn(
            "LIPSYNC_MAX_AUDIO_SECONDS is very low; responses may be truncated."
        )
    report.ok("Lip-sync optimization settings parsed successfully.")


def check_secret_hygiene(report: Report) -> None:
    """Run lightweight secret safety checks for common local misconfigurations."""
    repo_dir = Path(__file__).resolve().parents[2]
    if (repo_dir / ".env").exists():
        report.warn("Root .env detected. Ensure it is local-only and gitignored.")

    runpod_key = (os.getenv("RUNPOD_API_KEY") or "").strip()
    if runpod_key:
        if not runpod_key.startswith("rpa_"):
            report.warn("RUNPOD_API_KEY does not start with 'rpa_'; verify key value.")
        report.ok(f"RUNPOD_API_KEY detected ({_mask(runpod_key)}).")

    enable_checks = (os.getenv("ENABLE_SECRET_SAFETY_CHECKS", "true") or "").strip().lower()
    if enable_checks in {"0", "false", "no", "off"}:
        report.warn(
            "ENABLE_SECRET_SAFETY_CHECKS is disabled; startup secret warnings in app.main.py will not run."
        )
    else:
        report.ok("Secret safety checks are enabled.")


def _probe(url: str, *, timeout_seconds: float) -> tuple[bool, str]:
    """Probe URL reachability and return (ok, message)."""
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.get(url)
        if response.status_code >= 500:
            return (False, f"{url} responded with HTTP {response.status_code}.")
        return (True, f"{url} reachable (HTTP {response.status_code}).")
    except Exception as exc:
        return (False, f"{url} not reachable ({exc}).")


def _health_candidates(direct_url: str) -> Iterable[str]:
    """Generate health probe candidates for direct lip-sync URLs."""
    trimmed = direct_url.rstrip("/")
    parsed = urlparse(trimmed)
    if parsed.path in {"", "/"}:
        yield f"{trimmed}/health"
    yield trimmed


def check_http_health(report: Report, *, provider: str, provider_url: str | None, timeout_seconds: float) -> None:
    """Optionally run live HTTP checks against direct-mode lip-sync endpoints."""
    if provider != "direct" or not provider_url:
        report.warn("HTTP health check currently validates only LIPSYNC_DIRECT_URL mode.")
        return

    outcomes = [_probe(candidate, timeout_seconds=timeout_seconds) for candidate in _health_candidates(provider_url)]
    if any(ok for ok, _ in outcomes):
        details = "; ".join(message for ok, message in outcomes if ok)
        report.ok(f"Lip-sync endpoint health check passed: {details}")
        return

    detail = "; ".join(message for _, message in outcomes)
    report.fail(f"Lip-sync endpoint health check failed: {detail}")


def print_report(report: Report) -> None:
    """Render a human-readable summary report to stdout."""
    for message in report.passed:
        print(f"[PASS] {message}")
    for message in report.warnings:
        print(f"[WARN] {message}")
    for message in report.failures:
        print(f"[FAIL] {message}")
    print(
        f"\nSummary: {len(report.passed)} passed, {len(report.warnings)} warnings, {len(report.failures)} failures."
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for optional network checks and probe timeouts."""
    parser = argparse.ArgumentParser(description="Deckard backend preflight checks")
    parser.add_argument(
        "--check-http",
        action="store_true",
        help="Probe configured lip-sync HTTP endpoints before booting the backend.",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=3.0,
        help="Timeout (seconds) for preflight HTTP probes (default: 3.0).",
    )
    return parser.parse_args()


def main() -> int:
    """Run preflight suite and return process exit code."""
    args = parse_args()
    _load_environment()
    report = Report()

    check_core_env(report)
    provider, provider_url = check_lipsync_provider(report)
    check_memory_guardrails(report)
    check_lipsync_optimization(report)
    check_secret_hygiene(report)
    if args.check_http:
        check_http_health(
            report,
            provider=provider,
            provider_url=provider_url,
            timeout_seconds=max(args.http_timeout, 0.1),
        )

    print_report(report)
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
