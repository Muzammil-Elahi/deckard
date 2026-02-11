from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import wave
import io
from pathlib import Path

from app.lipsync_server.backends.base import InferenceInputs


def _require_env(name: str) -> str:
  value = (os.getenv(name) or "").strip()
  if not value:
    raise RuntimeError(f"Missing required env var: {name}")
  return value


def _wav_duration_seconds(wav_bytes: bytes) -> float:
  with wave.open(io.BytesIO(wav_bytes), "rb") as wf:  # type: ignore[arg-type]
    frames = wf.getnframes()
    rate = wf.getframerate()
    if rate <= 0:
      return 0.0
    return frames / float(rate)


class Wav2LipSubprocessBackend:
  """Wav2Lip backend that shells out to the upstream inference script.

  Requirements on the pod:
  - ffmpeg installed and on PATH
  - Wav2Lip repo available on disk (env WAV2LIP_REPO_DIR)
  - Wav2Lip checkpoint file available (env WAV2LIP_CHECKPOINT_PATH)
  """

  def __init__(self) -> None:
    self._repo_dir = Path(_require_env("WAV2LIP_REPO_DIR")).expanduser().resolve()
    self._checkpoint = Path(_require_env("WAV2LIP_CHECKPOINT_PATH")).expanduser().resolve()
    if not self._repo_dir.exists():
      raise RuntimeError(f"WAV2LIP_REPO_DIR does not exist: {self._repo_dir}")
    if not self._checkpoint.exists():
      raise RuntimeError(f"WAV2LIP_CHECKPOINT_PATH does not exist: {self._checkpoint}")
    if shutil.which("ffmpeg") is None:
      raise RuntimeError("ffmpeg not found on PATH. Install it (apt-get install -y ffmpeg).")

    self._python = (os.getenv("WAV2LIP_PYTHON") or "python").strip()
    self._inference_script = self._repo_dir / "inference.py"
    if not self._inference_script.exists():
      raise RuntimeError(f"Wav2Lip inference.py not found in repo dir: {self._inference_script}")

  async def generate(self, *, inputs: InferenceInputs) -> bytes:
    # Avoid importing heavy deps in-process; Wav2Lip will import in its own process.
    with tempfile.TemporaryDirectory(prefix="deckard-wav2lip-") as td:
      tmp = Path(td)
      image_path = tmp / ("source.png" if inputs.image_media_type.endswith("png") else "source.jpg")
      audio_path = tmp / "audio.wav"
      face_video_path = tmp / "face.mp4"
      out_path = tmp / "out.mp4"

      image_path.write_bytes(inputs.image_bytes)
      audio_path.write_bytes(inputs.audio_wav_bytes)

      # Build a "face video" from a single frame + audio so Wav2Lip can animate it.
      # Keep resolution modest to reduce latency and payload size.
      ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-vf",
        "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2",
        "-t",
        "30",  # hard cap; Wav2Lip will stop at audio end when it merges
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(face_video_path),
      ]

      proc = await asyncio.create_subprocess_exec(
        *ffmpeg_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )
      _stdout, stderr = await proc.communicate()
      if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building face video: {stderr.decode('utf-8', 'ignore')[:800]}")

      wav2lip_cmd = [
        self._python,
        str(self._inference_script),
        "--checkpoint_path",
        str(self._checkpoint),
        "--face",
        str(face_video_path),
        "--audio",
        str(audio_path),
        "--outfile",
        str(out_path),
      ]

      # Run from repo dir so relative imports/workdir assumptions match upstream.
      completed = await asyncio.to_thread(
        subprocess.run,
        wav2lip_cmd,
        cwd=str(self._repo_dir),
        capture_output=True,
        text=True,
        check=False,
      )
      if completed.returncode != 0:
        raise RuntimeError(
          "Wav2Lip inference failed: "
          f"{(completed.stderr or completed.stdout or '').strip()[:1200]}"
        )
      if not out_path.exists():
        raise RuntimeError("Wav2Lip reported success but output file is missing.")
      return out_path.read_bytes()
