# Developer Guide

This guide is for new contributors who need to become productive quickly.

## Repository Layout

- `server/` - FastAPI realtime orchestrator and backend services
- `web/` - Next.js frontend client
- `web/supabase/` - database schema and migrations
- `docs/` - architecture, plans, and runbooks

Key backend files:
- `server/app/main.py` - websocket lifecycle, realtime event loop, sentiment, lip-sync triggers
- `server/app/ai_agents/realtime_conversation.py` - OpenAI Agents SDK graph and tools
- `server/app/services/lipsync.py` - lip-sync provider orchestration (direct URL or RunPod)
- `server/app/services/memory.py` - low-latency memory/personalization
- `server/app/lipsync_server/` - optional self-hosted lip-sync inference API

Key frontend file:
- `web/src/app/page.tsx` - websocket client, audio capture/playback, avatar rendering

## Core Runtime Path

1. Frontend opens websocket to backend (`/ws/{session_id}`).
2. Browser streams microphone PCM chunks to backend.
3. Backend streams user audio into OpenAI Realtime session.
4. Assistant audio is buffered.
5. At turn end, backend requests lip-sync video from provider.
6. Backend emits coordinated audio + `talk_video` to frontend.
7. Frontend plays synced audio/video clip.

## Local Development

## Backend

```bash
cd server
uv sync
cp .env.example .env.local
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

```bash
cd web
npm install
npm run dev
```

## Testing and Quality Gates

```bash
cd server
uv run pytest

cd web
npm run lint
npm run build
```

## Configuration Strategy

Environment loading order for backend:
1. repo root `.env`
2. `server/.env`
3. `server/.env.local` (overrides prior files)

Important backend variables:
- `OPENAI_API_KEY`
- `LIPSYNC_DIRECT_URL` (preferred for same-pod local inference)
- `LIPSYNC_DIRECT_TIMEOUT_SECONDS`
- RunPod serverless variables (if not using direct mode)
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (optional memory persistence)

## Memory and Personalization

- Frontend maintains a stable `memory_key` in browser localStorage.
- Websocket URL includes `?memory_key=...`.
- Backend stores user/assistant utterances asynchronously.
- Memory recall is injected as compact system context before user turn commits.
- Memory writes are best effort and should never block conversation flow.

## Contribution Checklist

1. Read `docs/ARCHITECTURE.md` before changing realtime flow.
2. Keep websocket event contracts backward compatible when possible.
3. Add/adjust tests in `server/tests/` for backend behavior changes.
4. Run `pytest`, `lint`, and `build` before opening PR.
5. Update `CHANGELOG.md`.
