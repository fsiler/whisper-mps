"""Bounded audio decoding and transcript assembly for long media files."""

from __future__ import annotations

import math
import subprocess
import tempfile
from typing import BinaryIO, Callable, Iterator

import numpy as np


SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
DEFAULT_CHUNK_SECONDS = 30 * 60
DEFAULT_HOP_LENGTH = 160


def should_chunk_audio(duration_seconds: float) -> bool:
    """Keep long inputs below MLX's large one-dimensional array limits."""
    return duration_seconds > DEFAULT_CHUNK_SECONDS


def probe_audio_duration(path: str) -> float:
    """Return the container duration in seconds, or raise a useful error."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or f"ffprobe exited with {result.returncode}"
        raise RuntimeError(f"Could not read audio duration for {path!r}: {detail}")

    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid duration for {path!r}: {result.stdout!r}"
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Audio duration must be positive for {path!r}")
    return duration


def _read_up_to(stream: BinaryIO, size: int) -> bytes:
    """Read up to size bytes, tolerating short reads from pipes."""
    result = bytearray()
    while len(result) < size:
        block = stream.read(size - len(result))
        if not block:
            break
        result.extend(block)
    return bytes(result)


def iter_audio_chunks(
    path: str,
    *,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[tuple[float, np.ndarray]]:
    """Decode a media file once and yield bounded float32 PCM chunks.

    Offsets are relative to the start of the source. FFmpeg streams PCM through
    a pipe, so the file length never determines the size of one in-memory array.
    """
    if chunk_seconds <= 0 or sample_rate <= 0:
        raise ValueError("chunk_seconds and sample_rate must be positive")

    chunk_samples = chunk_seconds * sample_rate
    chunk_bytes = chunk_samples * BYTES_PER_SAMPLE
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]

    stderr_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
    except BaseException:
        stderr_file.close()
        raise
    offset_samples = 0
    try:
        if process.stdout is None:
            raise RuntimeError("FFmpeg did not provide an audio output pipe")

        while True:
            raw = _read_up_to(process.stdout, chunk_bytes)
            if not raw:
                break
            if len(raw) % BYTES_PER_SAMPLE:
                raise RuntimeError("FFmpeg returned an incomplete PCM sample")

            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            samples /= 32768.0
            offset = offset_samples / sample_rate
            offset_samples += samples.size
            yield offset, samples

        return_code = process.wait()
        if return_code:
            stderr_file.seek(0)
            detail = stderr_file.read().decode("utf-8", errors="replace").strip()
            detail = detail or f"FFmpeg exited with {return_code}"
            raise RuntimeError(f"Failed to decode {path!r}: {detail}")
    finally:
        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stderr_file.close()


def transcribe_chunked_audio(
    path: str,
    transcribe_fn: Callable,
    *,
    duration_seconds: float,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = DEFAULT_HOP_LENGTH,
    progress_callback: Callable[[float, float], None] | None = None,
    chunk_iterator: Callable[..., Iterator[tuple[float, np.ndarray]]] = iter_audio_chunks,
    **transcribe_options,
) -> dict:
    """Transcribe bounded chunks and merge their text, segments, and timestamps."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if chunk_seconds <= 0 or sample_rate <= 0 or hop_length <= 0:
        raise ValueError("chunk_seconds, sample_rate, and hop_length must be positive")

    if progress_callback is not None:
        progress_callback(0.0, duration_seconds)

    segments = []
    text_parts = []
    language = transcribe_options.get("language")
    chunks_seen = 0
    completed_seconds = 0.0

    for offset, audio_chunk in chunk_iterator(
        path,
        chunk_seconds=chunk_seconds,
        sample_rate=sample_rate,
    ):
        chunks_seen += 1
        chunk_duration = audio_chunk.size / sample_rate
        if chunk_duration <= 0:
            continue

        options = dict(transcribe_options)
        if language is not None:
            options["language"] = language

        last_reported = completed_seconds

        def report_chunk_progress(processed: float, _chunk_total: float) -> None:
            nonlocal last_reported
            absolute = min(duration_seconds, offset + max(0.0, processed))
            last_reported = max(last_reported, absolute)
            if progress_callback is not None:
                progress_callback(last_reported, duration_seconds)

        result = transcribe_fn(
            audio_chunk,
            progress_callback=report_chunk_progress,
            **options,
        )
        if language is None:
            language = result.get("language")

        chunk_text = (result.get("text") or "").strip()
        if chunk_text:
            text_parts.append(chunk_text)

        seek_offset = round(offset * sample_rate / hop_length)
        for segment in result.get("segments", []):
            merged = dict(segment)
            merged["id"] = len(segments)
            local_start = float(merged.get("start", 0.0))
            local_end = float(merged.get("end", local_start))
            merged["start"] = local_start + offset
            merged["end"] = local_end + offset
            if "seek" in merged:
                merged["seek"] = int(merged["seek"]) + seek_offset
            segments.append(merged)

        completed_seconds = min(duration_seconds, offset + chunk_duration)
        if progress_callback is not None:
            progress_callback(completed_seconds, duration_seconds)

    if not chunks_seen:
        raise RuntimeError(f"FFmpeg produced no audio samples for {path!r}")

    if progress_callback is not None:
        progress_callback(duration_seconds, duration_seconds)

    return {
        "text": " ".join(text_parts),
        "segments": segments,
        "language": language,
    }
