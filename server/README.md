# Deckard Orchestrator Service

FastAPI backend for realtime conversation orchestration:
- OpenAI Realtime + Agents SDK for voice/text/tool flows
- RunPod-hosted lip-sync inference (MuseTalk-style handlers)
- WebSocket bridge to the Next.js client (`web/src/app/page.tsx`)

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
4. Start the API:
   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## RunPod integration
- The backend submits async jobs to RunPod Serverless v2 `/run`, then polls `/status/{job_id}`.
- Expected handler output must include a video URL in one of:
  - `output.video_url`
  - `output.result_url`
  - `output.url`
- If no URL is returned, the backend emits `talk_error` to the client.

## Directory guide
- `app/main.py` - realtime websocket manager and event serialization
- `app/config.py` - environment-driven settings
- `app/services/lipsync.py` - provider-agnostic lip-sync orchestration
- `app/services/runpod.py` - RunPod HTTP client + status normalization
- `app/services/audio_utils.py` - PCM/WAV conversion utilities
- `app/ai_agents/` - OpenAI Agents SDK agent definitions
- `tests/` - pytest coverage for service and agent behavior
