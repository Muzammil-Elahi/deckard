# Architecture

This document describes the current runtime architecture (MVP path).

## High-Level Components

- Frontend: Next.js app (`web/src/app/page.tsx`)
  - Captures microphone audio
  - Sends audio/image events over websocket
  - Renders persona state and lip-sync video output
- Backend: FastAPI websocket service (`server/app/main.py`)
  - Runs OpenAI Realtime session
  - Handles tool calls and sentiment updates
  - Buffers assistant audio and triggers lip-sync generation
  - Emits synchronized playback events to client
- Lip-sync provider (`server/app/services/lipsync.py`)
  - Direct URL mode for self-hosted inference service
  - RunPod Serverless mode as alternate path
- Optional local inference server (`server/app/lipsync_server/`)
  - Exposes `/generate` for image+audio payloads
  - Supports pluggable inference backends
- Memory service (`server/app/services/memory.py`)
  - Local-first recall for low latency
  - Optional Supabase persistence for continuity

## Event Flow

1. `web` connects to `ws://.../ws/{session_id}?memory_key=...`.
2. `server` initializes OpenAI Realtime session.
3. `web` sends `audio` events (PCM int16 arrays).
4. `server` forwards audio to OpenAI and receives assistant audio events.
5. `server` buffers assistant audio until `audio_end`.
6. `server` calls lip-sync provider with:
   - persona source image
   - assistant audio (WAV/base64)
7. If lip-sync succeeds:
   - `server` sends buffered `audio` chunks
   - `server` sends `talk_video` with URL/data URL
   - `server` sends `audio_end`
8. If lip-sync fails:
   - `server` falls back to audio-only playback
   - `server` sends `talk_error`

## Websocket Message Types

Client -> Server:
- `audio`
- `image`, `image_start`, `image_chunk`, `image_end`
- `commit_audio`
- `interrupt`
- `set_persona`

Server -> Client:
- Realtime stream events (`audio`, `audio_end`, etc.)
- `client_info` status/mood updates
- `talk_video`
- `talk_error`

## Agent and Tool Orchestration

- Agent graph is defined in `server/app/ai_agents/realtime_conversation.py`.
- Uses OpenAI Agents SDK realtime agents.
- Tool functions include web search, sentiment tool, and browser automation utilities.

## Memory Path

- Frontend includes stable `memory_key` in websocket URL.
- Backend saves user and assistant turns asynchronously.
- Before turn commit, backend injects a compact summary as a system message.
- Memory injection is deduplicated and timeout-capped to avoid latency spikes.

## Deployment Topology (Same GPU Pod)

- Port `8000`: Deckard backend websocket/API
- Port `8001`: lip-sync inference server
- Recommended backend env:
  - `LIPSYNC_DIRECT_URL=http://127.0.0.1:8001/generate`

This avoids external proxy hops for inference calls and reduces response latency.
