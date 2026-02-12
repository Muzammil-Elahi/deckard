from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from app.config import settings

try:
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional import in constrained envs
    Client = object  # type: ignore[assignment,misc]
    create_client = None  # type: ignore[assignment]


_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(slots=True)
class MemoryEntry:
    role: str
    text: str
    created_at: float


def _tokenize(text: str) -> set[str]:
    normalized = (text or "").lower()
    return {token for token in _WORD_RE.findall(normalized) if len(token) > 2}


def _safe_memory_key(memory_key: str) -> str:
    raw = (memory_key or "").strip()
    if not raw:
        return "anonymous"
    # Keep keys query-safe and SQL-filter-safe for label prefix matching.
    return re.sub(r"[^a-zA-Z0-9:_-]+", "_", raw)[:120]


class ConversationMemoryService:
    """Low-latency memory service with local-first recall and optional Supabase sync."""

    def __init__(
        self,
        *,
        max_local_entries: int = 80,
        max_recall_items: int = 5,
        max_summary_chars: int = 650,
        local_ttl_seconds: float = 86_400,
        dedupe_window_seconds: float = 900,
        remote_timeout_seconds: float = 0.45,
        remote_cache_ttl_seconds: float = 30.0,
    ) -> None:
        self._max_local_entries = max_local_entries
        self._max_recall_items = max_recall_items
        self._max_summary_chars = max_summary_chars
        self._local_ttl_seconds = local_ttl_seconds
        self._dedupe_window_seconds = dedupe_window_seconds
        self._remote_timeout_seconds = remote_timeout_seconds
        self._remote_cache_ttl_seconds = remote_cache_ttl_seconds
        # In-memory rolling window per memory identity; this keeps recall fast and local.
        self._local: dict[str, deque[MemoryEntry]] = defaultdict(
            lambda: deque(maxlen=self._max_local_entries)
        )
        # Signature timestamps used to skip repetitive writes in a short window.
        self._recent_signatures: dict[str, dict[str, float]] = defaultdict(dict)
        # Tiny TTL cache to avoid repeated remote reads on rapid consecutive turns.
        self._remote_cache: dict[str, tuple[float, list[MemoryEntry]]] = {}
        self._client: Optional[Client] = None
        self._client_init_error: Optional[str] = None
        self._init_client()

    def _init_client(self) -> None:
        url = (settings.supabase_url or "").strip()
        key = (settings.supabase_service_role_key or "").strip()
        if not url or not key:
            return
        if create_client is None:
            self._client_init_error = "supabase client import unavailable"
            return
        try:
            self._client = create_client(url, key)
        except Exception as exc:  # pragma: no cover - network/env specific
            self._client_init_error = str(exc)
            self._client = None

    @property
    def has_remote_store(self) -> bool:
        return self._client is not None

    async def remember(
        self,
        *,
        memory_key: str,
        role: str,
        text: str,
        persona: str,
    ) -> None:
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return
        key = _safe_memory_key(memory_key)
        now = time.time()
        self._prune_local(key, now=now)
        signature = cleaned.lower()
        last_seen = self._recent_signatures[key].get(signature)
        # Skip near-duplicate writes in a short window to avoid noisy memory drift.
        if last_seen is not None and (now - last_seen) < self._dedupe_window_seconds:
            return
        self._recent_signatures[key][signature] = now
        self._local[key].append(
            MemoryEntry(role=(role or "unknown").lower(), text=cleaned, created_at=now)
        )

        if self._client is None:
            return

        label = f"mem:{key}|role:{(role or 'unknown').lower()}|persona:{(persona or 'joi').lower()}"
        payload = {
            "label": label,
            "content": cleaned[:3000],
            "importance": self._importance_score(cleaned, role=role),
        }
        asyncio.create_task(self._insert_remote(payload))

    async def _insert_remote(self, payload: dict[str, object]) -> None:
        if self._client is None:
            return

        def _insert() -> None:
            assert self._client is not None
            self._client.table("memories").insert(payload).execute()

        try:
            await asyncio.wait_for(
                asyncio.to_thread(_insert),
                timeout=self._remote_timeout_seconds,
            )
        except Exception:
            # Best-effort persistence; never block realtime flow on memory writes.
            return

    def _importance_score(self, text: str, *, role: str) -> float:
        lowered = text.lower()
        score = 0.55
        if role == "user":
            score += 0.1
        if any(token in lowered for token in ("my name", "i am", "i like", "i prefer", "remember")):
            score += 0.2
        if len(lowered) > 180:
            score += 0.05
        return min(score, 1.0)

    async def recall_summary(
        self,
        *,
        memory_key: str,
        prompt: str = "",
        max_items: Optional[int] = None,
        max_chars: int = 700,
        include_remote: bool = True,
    ) -> str:
        key = _safe_memory_key(memory_key)
        self._prune_local(key)
        local_entries = list(self._local.get(key, ()))
        remote_entries = await self._recall_remote(key) if include_remote else []

        all_entries: list[MemoryEntry] = []
        seen: set[str] = set()
        # Local first for freshest context, then remote for cross-session continuity.
        for item in local_entries + remote_entries:
            signature = item.text.strip().lower()
            if not signature or signature in seen:
                continue
            seen.add(signature)
            all_entries.append(item)

        if not all_entries:
            return ""

        prompt_tokens = _tokenize(prompt)

        def _score(entry: MemoryEntry) -> float:
            # Recency weighted toward fresh context and short factual snippets.
            recency = 1.0 / max(1.0, (time.time() - entry.created_at) / 600.0)
            if prompt_tokens:
                overlap = len(prompt_tokens & _tokenize(entry.text))
                prompt_score = overlap / max(1, len(prompt_tokens))
            else:
                prompt_score = 0.0
            role_bias = 0.1 if entry.role == "user" else 0.0
            return recency + prompt_score + role_bias

        ranked = sorted(all_entries, key=_score, reverse=True)
        top_n = max_items if max_items is not None else self._max_recall_items
        selected = ranked[: max(1, top_n)]

        # Bound recall output so memory injection cannot balloon prompt size/latency.
        summary_limit = max(120, min(max_chars, self._max_summary_chars))
        lines: list[str] = []
        total_chars = 0
        for entry in selected:
            role_tag = "User" if entry.role == "user" else "Assistant"
            line = f"- {role_tag}: {entry.text}"
            if total_chars + len(line) > summary_limit:
                break
            lines.append(line)
            total_chars += len(line)
        return "\n".join(lines)

    async def _recall_remote(self, key: str) -> list[MemoryEntry]:
        if self._client is None:
            return []
        now = time.time()
        cached = self._remote_cache.get(key)
        if cached and (now - cached[0]) < self._remote_cache_ttl_seconds:
            return cached[1]

        prefix = f"mem:{key}|%"

        def _select() -> list[dict]:
            assert self._client is not None
            response = (
                self._client.table("memories")
                .select("label, content, created_at")
                .ilike("label", prefix)
                .order("created_at", desc=True)
                .limit(24)
                .execute()
            )
            data = getattr(response, "data", None)
            return data if isinstance(data, list) else []

        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(_select),
                timeout=self._remote_timeout_seconds,
            )
        except Exception:
            return []

        parsed: list[MemoryEntry] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("content") or "").strip()
            label = str(row.get("label") or "")
            if not text:
                continue
            role = "assistant"
            if "|role:user|" in label:
                role = "user"
            created_at = _parse_created_at(row.get("created_at")) or now
            parsed.append(MemoryEntry(role=role, text=text, created_at=created_at))

        self._remote_cache[key] = (now, parsed)
        return parsed

    def _prune_local(self, key: str, *, now: Optional[float] = None) -> None:
        """Drop expired local entries and stale dedupe signatures."""
        ts = now if now is not None else time.time()
        entries = self._local.get(key)
        if entries:
            while entries and (ts - entries[0].created_at) > self._local_ttl_seconds:
                entries.popleft()

        signatures = self._recent_signatures.get(key)
        if signatures:
            stale = [
                sig
                for sig, last_seen in signatures.items()
                if (ts - last_seen) > self._dedupe_window_seconds
            ]
            for sig in stale:
                signatures.pop(sig, None)


def _parse_created_at(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
        except ValueError:
            return None
    return None
