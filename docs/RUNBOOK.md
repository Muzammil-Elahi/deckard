# Operations Runbook

Use this for environment bring-up and incident debugging.

## Runpod Bring-Up (Single Pod, Two Services)

## Start backend (port 8000)

```bash
cd /workspace/deckard/server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Start lip-sync server (port 8001)

```bash
cd /workspace/deckard/server
uv run uvicorn app.lipsync_server.main:app --host 0.0.0.0 --port 8001
```

If using Wav2Lip backend:

```bash
export LIPSYNC_SERVER_BACKEND=wav2lip
export WAV2LIP_REPO_DIR=/workspace/Wav2Lip
export WAV2LIP_CHECKPOINT_PATH=/workspace/wav2lip_gan.pth
```

## Backend env for same-pod inference

```bash
export LIPSYNC_DIRECT_URL=http://127.0.0.1:8001/generate
export LIPSYNC_DIRECT_TIMEOUT_SECONDS=30
```

## Health Checks

Inside pod:

```bash
ss -ltnp | egrep ':8000|:8001'
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8001/health
```

Through Runpod proxy:
- `https://<pod>-8000.proxy.runpod.net/`
- `https://<pod>-8001.proxy.runpod.net/health`

## Common Failures

## `502 Bad Gateway` from proxy

Cause:
- process not running
- service bound to wrong host/port

Fix:
- verify with `ss -ltnp`
- restart service with `--host 0.0.0.0`

## No avatar video, chat text still works

Cause:
- lip-sync endpoint invalid, returns 404/5xx, or response missing video URL key

Fix:
- verify `LIPSYNC_DIRECT_URL`
- test `POST /generate`
- check backend logs for `talk_error`

## Audio appears delayed or silent

Cause:
- coordinated buffering waits for lip-sync completion

Fix:
- reduce `LIPSYNC_DIRECT_TIMEOUT_SECONDS`
- inspect lip-sync server latency/errors

## Memory not persisting across reconnects

Cause:
- missing/stale `memory_key`
- Supabase not configured (remote persistence disabled)

Fix:
- ensure websocket URL has stable `memory_key`
- set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` for persistence

## Minimum Logs to Capture for Bugs

1. Backend logs around websocket session id.
2. Lip-sync server logs for `/generate`.
3. Frontend event log entries for `talk_video` and `talk_error`.
4. Exact env vars used (names only, no secret values).
