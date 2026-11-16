import os
import shutil
import tempfile
import unittest
import wave

import numpy as np

from whisper_mps.long_audio import (
    DEFAULT_CHUNK_SECONDS,
    iter_audio_chunks,
    probe_audio_duration,
    should_chunk_audio,
    transcribe_chunked_audio,
)


class LongAudioTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_ffmpeg_streams_pcm_in_bounded_chunks_with_exact_offsets(self):
        sample_rate = 16_000
        source_samples = (np.arange(sample_rate * 2 + sample_rate // 4) % 1200 - 600).astype(
            np.int16
        )

        with tempfile.TemporaryDirectory() as directory:
            wav_path = os.path.join(directory, "sample.wav")
            with wave.open(wav_path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(source_samples.astype("<i2").tobytes())

            duration = probe_audio_duration(wav_path)
            chunks = list(
                iter_audio_chunks(
                    wav_path,
                    chunk_seconds=1,
                    sample_rate=sample_rate,
                )
            )

        self.assertAlmostEqual(duration, 2.25, places=2)
        self.assertEqual([offset for offset, _audio in chunks], [0.0, 1.0, 2.0])
        self.assertEqual([audio.size for _offset, audio in chunks], [16_000, 16_000, 4_000])
        decoded = np.concatenate([audio for _offset, audio in chunks])
        expected = source_samples.astype(np.float32) / 32768.0
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=0)

    def test_chunked_transcription_merges_absolute_timestamps_and_progress(self):
        fake_chunks = [
            (0.0, np.zeros(4, dtype=np.float32)),
            (4.0, np.ones(2, dtype=np.float32)),
        ]
        calls = []
        progress = []

        def chunk_iterator(_path, *, chunk_seconds, sample_rate):
            self.assertEqual(chunk_seconds, 4)
            self.assertEqual(sample_rate, 1)
            yield from fake_chunks

        def fake_transcribe(audio, *, progress_callback, language=None):
            calls.append((audio.size, language))
            chunk_duration = float(audio.size)
            progress_callback(0.0, chunk_duration)
            progress_callback(chunk_duration, chunk_duration)
            return {
                "text": f"part {len(calls)}",
                "language": "en",
                "segments": [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.5,
                        "end": 1.5,
                        "text": f"part {len(calls)}",
                    }
                ],
            }

        result = transcribe_chunked_audio(
            "long.mka",
            fake_transcribe,
            duration_seconds=6.0,
            chunk_seconds=4,
            sample_rate=1,
            hop_length=1,
            progress_callback=lambda processed, total: progress.append((processed, total)),
            chunk_iterator=chunk_iterator,
        )

        self.assertEqual(calls, [(4, None), (2, "en")])
        self.assertEqual(result["text"], "part 1 part 2")
        self.assertEqual(result["language"], "en")
        self.assertEqual([segment["id"] for segment in result["segments"]], [0, 1])
        self.assertEqual(
            [(segment["start"], segment["end"], segment["seek"]) for segment in result["segments"]],
            [(0.5, 1.5, 0), (4.5, 5.5, 4)],
        )
        self.assertEqual(progress[-1], (6.0, 6.0))
        self.assertTrue(all(a <= b for (a, _), (b, _) in zip(progress, progress[1:])))

    def test_audio_longer_than_one_chunk_uses_chunked_path(self):
        self.assertFalse(should_chunk_audio(DEFAULT_CHUNK_SECONDS))
        self.assertTrue(should_chunk_audio(DEFAULT_CHUNK_SECONDS + 1))
        self.assertTrue(should_chunk_audio(44 * 60 * 60 + 36 * 60))

    def test_44_hour_timeline_keeps_each_transcribe_input_bounded(self):
        duration = 44 * 60 * 60 + 36 * 60 + 1
        chunks = [
            (0.0, np.zeros(1, dtype=np.float32)),
            (DEFAULT_CHUNK_SECONDS, np.zeros(1, dtype=np.float32)),
            (duration - 1, np.zeros(1, dtype=np.float32)),
        ]
        input_sizes = []

        def chunk_iterator(_path, **_kwargs):
            yield from chunks

        def fake_transcribe(audio, *, progress_callback, **_kwargs):
            input_sizes.append(audio.size)
            progress_callback(audio.size, float(audio.size))
            return {
                "text": "word",
                "language": "en",
                "segments": [{"start": 0.25, "end": 0.75, "seek": 0, "text": "word"}],
            }

        result = transcribe_chunked_audio(
            "44-hour.mka",
            fake_transcribe,
            duration_seconds=duration,
            sample_rate=1,
            hop_length=1,
            chunk_iterator=chunk_iterator,
        )

        self.assertEqual(input_sizes, [1, 1, 1])
        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual(result["segments"][-1]["start"], duration - 0.75)
        self.assertEqual(result["segments"][-1]["end"], duration - 0.25)


if __name__ == "__main__":
    unittest.main()
