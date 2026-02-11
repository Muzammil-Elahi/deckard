# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]
### Added
- Added a modular lip-sync pipeline in `server/app/services/lipsync.py` with RunPod-backed job orchestration, strict validation, and normalized result handling.
- Added reusable audio conversion helpers in `server/app/services/audio_utils.py`.
- Added a RunPod API client in `server/app/services/runpod.py` with config validation, timeout handling, and status normalization.
- Added tests for lip-sync service and RunPod config behavior in `server/tests/test_lipsync.py`.
- Added a co-located lip-sync HTTP server in `server/app/lipsync_server/` (port `8001`) with a pluggable backend and a Wav2Lip subprocess option for GPU pods.

### Changed
- Replaced active D-ID generation calls in `server/app/main.py` with RunPod lip-sync generation while preserving frontend event compatibility.
- Updated `server/.env.example` and `server/README.md` with RunPod serverless configuration details.
- Updated `server/app/ai_agents/transcribe_agent.py` and `server/tests/test_did_talks.py` to use shared `pcm16le_to_wav` utilities.
- Updated Studio event labels in `web/src/app/page.tsx` from D-ID wording to provider-neutral lip-sync messaging.
- Fixed Supabase typing for inserts/updates by adding `Relationships` metadata to table definitions in `web/src/lib/supabase/types.ts`.
- Updated Supabase server client creation to support async `cookies()` in Next.js 15 and updated all callsites to await the client.
- Fixed `Image` constructor shadowing in `web/src/app/page.tsx` by using `window.Image` for client-side image preprocessing.
- Made browser Supabase client initialization non-fatal during build/prerender when env vars are missing, while preserving runtime guardrails.
- Added frontend playback for realtime assistant `audio` chunks in `web/src/app/page.tsx`, including queueing and cleanup on interrupt/disconnect.
- Improved lip-sync result parsing to accept additional output keys (`video`, `videoUrl`, base64 variants) and explicit unavailable-provider errors in `server/app/services/lipsync.py`.
- Updated backend talk delivery to emit `talk_error` when lip-sync returns no video URL instead of silent empty `talk_video` payloads.
