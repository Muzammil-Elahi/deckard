# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]
### Added
- Added a modular lip-sync pipeline in `server/app/services/lipsync.py` with RunPod-backed job orchestration, strict validation, and normalized result handling.
- Added reusable audio conversion helpers in `server/app/services/audio_utils.py`.
- Added a RunPod API client in `server/app/services/runpod.py` with config validation, timeout handling, and status normalization.
- Added tests for lip-sync service and RunPod config behavior in `server/tests/test_lipsync.py`.

### Changed
- Replaced active D-ID generation calls in `server/app/main.py` with RunPod lip-sync generation while preserving frontend event compatibility.
- Updated `server/.env.example` and `server/README.md` with RunPod serverless configuration details.
- Updated `server/app/ai_agents/transcribe_agent.py` and `server/tests/test_did_talks.py` to use shared `pcm16le_to_wav` utilities.
- Updated Studio event labels in `web/src/app/page.tsx` from D-ID wording to provider-neutral lip-sync messaging.
