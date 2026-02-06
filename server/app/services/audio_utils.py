from __future__ import annotations

import struct


def pcm16le_to_wav(
    pcm: bytes,
    *,
    sample_rate: int = 24_000,
    channels: int = 1,
) -> bytes:
    """Wrap raw PCM16 little-endian audio in a minimal WAV container.

    The realtime stack emits mono 24 kHz PCM16 bytes. A WAV container is
    required by many downstream inference services (including RunPod handlers)
    so they can reliably infer encoding and sample-rate metadata.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if channels <= 0:
        raise ValueError("channels must be > 0")

    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    data_size = len(pcm)
    fmt_chunk_size = 16
    riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_size)

    header = b"RIFF" + struct.pack("<I", riff_chunk_size) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH",
        fmt_chunk_size,
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,  # bits per sample
    )
    header += b"data" + struct.pack("<I", data_size)
    return header + pcm

