from __future__ import annotations

from app.lipsync_server.backends.base import InferenceInputs


class NoopBackend:
  """Backend that always fails with a clear configuration error."""

  async def generate(self, *, inputs: InferenceInputs) -> bytes:
    _ = inputs
    raise RuntimeError(
      "Lip-sync backend not configured. "
      "Set LIPSYNC_SERVER_BACKEND=wav2lip and configure WAV2LIP_REPO_DIR + WAV2LIP_CHECKPOINT_PATH, "
      "or replace the backend with your preferred model server."
    )

