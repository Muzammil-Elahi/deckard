from __future__ import annotations

import asyncio

from app.services.memory import ConversationMemoryService


def _run(coro):  # noqa: ANN001
    return asyncio.run(coro)


def test_memory_service_local_recall_without_remote() -> None:
    service = ConversationMemoryService(max_local_entries=20, max_recall_items=3)

    _run(
        service.remember(
            memory_key="local-user",
            role="user",
            text="My name is Sam and I prefer concise answers.",
            persona="joi",
        )
    )
    _run(
        service.remember(
            memory_key="local-user",
            role="assistant",
            text="You prefer concise answers and your name is Sam.",
            persona="joi",
        )
    )

    summary = _run(
        service.recall_summary(
            memory_key="local-user",
            prompt="Can you keep responses concise?",
            include_remote=False,
            max_items=3,
        )
    )

    assert "Sam" in summary
    assert "concise" in summary.lower()


def test_memory_service_empty_prompt_still_returns_recent_items() -> None:
    service = ConversationMemoryService(max_local_entries=20, max_recall_items=2)
    _run(
        service.remember(
            memory_key="another-user",
            role="user",
            text="I work in healthcare analytics.",
            persona="officer_k",
        )
    )

    summary = _run(
        service.recall_summary(
            memory_key="another-user",
            prompt="",
            include_remote=False,
            max_items=2,
        )
    )

    assert "healthcare analytics" in summary.lower()
