# Deckard Orchestrator Service

FastAPI backend for realtime conversation orchestration:
- OpenAI Realtime + Agents SDK for voice/text/tool flows
- RunPod-hosted lip-sync inference (MuseTalk-style handlers)
- WebSocket bridge to the Next.js client (`web/src/app/page.tsx`)
- Low-latency memory and personalization hooks via `memory_key`

## Local setup
1. Create and activate a Python environment (Python 3.12+ recommended).
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Copy `server/.env.example` to `server/.env.local` and set:
   - `OPENAI_API_KEY`
   - `RUNPOD_API_KEY`
   - `RUNPOD_ENDPOINT_ID` (or `RUNPOD_RUN_URL` + `RUNPOD_STATUS_URL_TEMPLATE`)
   - Optional direct mode:
     - `LIPSYNC_DIRECT_URL` (example: `http://127.0.0.1:8001/generate`)
4. Start the API:
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Preflight checks
Run quick configuration checks before booting services:
```bash
uv run python scripts/preflight.py
```

Optional network validation for direct lip-sync mode:
```bash
uv run python scripts/preflight.py --check-http
```

## Websocket memory key
- Endpoint: `/ws/{session_id}?memory_key={stable_user_key}`
- If `memory_key` is provided, server memory recall/persistence uses it as the identity key.
- If omitted, `session_id` is used as fallback.
- Memory writes are asynchronous and should not block the realtime loop.

## Websocket control messages
Client -> server control events:
- `text` with plain user text payload
- `set_persona` with `persona` in `joi | officer_k | officer_j`
- `set_response_mode` with `mode` in `synced | fast`

Server -> client confirmations:
- `client_info` with `info=response_mode_set` and the selected mode
- `client_info` with `info=persona_set` and the selected persona

## Co-Located Lip-Sync Server (Same GPU Pod)
If you are running Deckard on a RunPod GPU pod and want the lip-sync model on the same machine:

1. Start Deckard backend (WebSocket + orchestration) on `:8000`:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. Start the lip-sync server on `:8001`:
   ```bash
   # Backend selection:
   # - noop (default): returns a clear error until you configure a model
   # - wav2lip: shells out to a local Wav2Lip checkout
   export LIPSYNC_SERVER_BACKEND=wav2lip
   export WAV2LIP_REPO_DIR=/workspace/Wav2Lip
   export WAV2LIP_CHECKPOINT_PATH=/workspace/wav2lip_gan.pth
   uv run uvicorn app.lipsync_server.main:app --host 0.0.0.0 --port 8001
   ```
3. Point Deckard to the local server (lowest latency):
   - `LIPSYNC_DIRECT_URL=http://127.0.0.1:8001/generate`

## RunPod integration
- The backend submits async jobs to RunPod Serverless v2 `/run`, then polls `/status/{job_id}`.
- Expected handler output must include a video URL in one of:
  - `output.video_url`
  - `output.result_url`
  - `output.url`
- If no URL is returned, the backend emits `talk_error` to the client.

## Latency tuning
For lower lip-sync turnaround on GPU pods:
- Set `RESPONSE_MODE_DEFAULT=fast` to stream assistant audio immediately while video is generated.
- Use direct mode when possible:
  - `LIPSYNC_DIRECT_URL=http://127.0.0.1:8001/generate`
  - `LIPSYNC_DIRECT_PREFERRED_PATH=/generate`
- Keep silence trimming enabled:
  - `LIPSYNC_TRIM_SILENCE=true`
  - `LIPSYNC_SILENCE_THRESHOLD=500`
  - `LIPSYNC_SILENCE_PAD_MS=120`
- Reduce serverless polling latency:
  - `RUNPOD_POLL_INTERVAL_SECONDS=0.4`
- Optional hard cap for long turns:
  - `LIPSYNC_MAX_AUDIO_SECONDS=0` (disabled by default; set >0 only if you need strict latency caps).

## Directory guide
- `app/main.py` - realtime websocket manager and event serialization
- `app/config.py` - environment-driven settings
- `app/services/lipsync.py` - provider-agnostic lip-sync orchestration
- `app/services/runpod.py` - RunPod HTTP client + status normalization
- `app/services/memory.py` - local-first conversational memory + optional Supabase sync
- `app/services/audio_utils.py` - PCM/WAV conversion utilities
- `app/ai_agents/` - OpenAI Agents SDK agent definitions
- `app/lipsync_server/` - optional local inference API for same-pod deployments
- `tests/` - pytest coverage for service and agent behavior
