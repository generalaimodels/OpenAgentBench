"""Deterministic fixtures used by realtime dry and live tests."""

from __future__ import annotations

import base64
import math
import struct
import subprocess
import zlib
from hashlib import sha256
from typing import Any

from openagentbench.agent_data.types import ContentPart


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return len(payload).to_bytes(4, "big") + tag + payload + crc.to_bytes(4, "big")


def tiny_png_bytes(*, width: int = 32, height: int = 32) -> bytes:
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            red = 240 if (x + y) % 2 == 0 else 64
            green = 96 if (x // 4) % 2 == 0 else 200
            blue = 48 if (y // 4) % 2 == 0 else 220
            row.extend((red, green, blue))
        rows.append(bytes(row))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    compressed = zlib.compress(b"".join(rows), level=9)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        )
    )


def tiny_png_base64() -> str:
    return base64.b64encode(tiny_png_bytes()).decode("ascii")


def tiny_png_data_url() -> str:
    return f"data:image/png;base64,{tiny_png_base64()}"


def _ffmpeg_tts_pcm16(text: str, *, sample_rate_hz: int) -> bytes | None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"flite=text={text}:voice=slt",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _tone_pcm16(*, sample_rate_hz: int, duration_seconds: float, frequency_hz: float) -> bytes:
    sample_count = int(sample_rate_hz * duration_seconds)
    amplitude = 0.2 * 32767.0
    frames = bytearray()
    for index in range(sample_count):
        angle = 2.0 * math.pi * frequency_hz * (index / sample_rate_hz)
        sample = int(math.sin(angle) * amplitude)
        frames.extend(struct.pack("<h", sample))
    return bytes(frames)


def tiny_pcm16_audio_bytes(
    *,
    text: str = "please answer with the exact word audio",
    sample_rate_hz: int = 16_000,
) -> bytes:
    quoted = text.replace("\\", "\\\\").replace(":", "\\:")
    spoken = _ffmpeg_tts_pcm16(quoted, sample_rate_hz=sample_rate_hz)
    if spoken is not None:
        return spoken
    return _tone_pcm16(sample_rate_hz=sample_rate_hz, duration_seconds=1.2, frequency_hz=440.0)


def tiny_pcm16_base64() -> str:
    return base64.b64encode(tiny_pcm16_audio_bytes()).decode("ascii")


def sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def build_text_content_part(text: str) -> ContentPart:
    return {"type": "input_text", "text": text}


def build_image_content_part() -> ContentPart:
    return {
        "type": "input_image",
        "image_url": tiny_png_data_url(),
    }


def fixed_tool_declaration() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "lookup_test_clock",
        "description": "Return the fixed UTC timestamp used by the realtime test harness.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA or UTC timezone label for the returned timestamp.",
                }
            },
            "required": ["timezone"],
        },
    }


def fixed_tool_response(*, timezone_label: str = "UTC") -> dict[str, Any]:
    return {
        "utc_timestamp": "2026-03-22T00:00:00Z",
        "timezone": timezone_label,
        "source": "realtime_test_fixture",
    }
