from . import whisper
from .utils.ytdownloader import download_and_convert_to_mp3
import argparse
import os
import time
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TaskProgressColumn,
    TextColumn,
)
import json
import logging


parser = argparse.ArgumentParser(
    description="whisper-mps: Blazingly fast transcriptions via your terminal! ⚡️\n\n"
                "An opinionated CLI to transcribe Audio files (or YouTube videos) w/ Whisper on-device!\n"
                "Powered by MLX, Whisper & Apple M series.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Transcribe an audio file with default settings
  whisper-mps --file-name audio.mp3
  
  # Transcribe with a specific model size
  whisper-mps --file-name audio.wav --model-name base
  
  # Transcribe and save to a custom output file
  whisper-mps --file-name audio.mp3 --output-file-name transcript.json
  
  # Control logging verbosity
  whisper-mps --file-name audio.mp3 --log-level WARNING

Available Models:
  tiny, base, small, medium, large
  (Larger models require more loading time but provide better accuracy)

Supported Audio Formats:
  wav, mp3, mp4, m4a, flac, and more (via ffmpeg)

Note: This CLI is optimized for Apple MPS (Metal Performance Shaders).
"""
)

parser.add_argument(
    "--file-name",
    required=False,
    type=str,
    metavar="FILE",
    help="Path or URL to the audio file to be transcribed. Supports various formats: wav, mp3, mp4, etc.",
)

parser.add_argument(
    "--model-name",
    required=False,
    default="tiny",
    type=str,
    metavar="MODEL",
    help='Size of the OpenAI Whisper model. Options: "tiny" (default), "base", "small", "medium", "large". '
         'Larger models provide better accuracy but take longer to load.',
)

parser.add_argument(
    "--youtube-url",
    required=False,
    default=None,
    type=str,
    metavar="URL",
    help="[DEPRECATED] The YouTube video URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID). "
         "This feature is deprecated and may be removed in future releases. "
         "Please use yt-dlp to download YouTube videos instead.",
)

parser.add_argument(
    "--output-file-name",
    required=False,
    default=None,
    type=str,
    metavar="FILE",
    help="The output file name for the transcribed text JSON. "
         "If omitted, it is derived from the input audio file name. "
         "Will automatically append .json extension if not provided. "
         "An .srt file with the same base name is also written.",
)

parser.add_argument(
    "--language",
    required=False,
    default=None,
    type=str,
    metavar="LANGUAGE",
    help='Language spoken in the audio. Supplying this skips automatic language detection. '
         'Use the two-letter ISO 639-1 code (e.g. "en", "fr", "de", "zh") or the full language '
         'name (e.g. "english", "french").',
)

parser.add_argument(
    "--log-level",
    required=False,
    default="INFO",
    type=str,
    metavar="LEVEL",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help='Set the logging level to control output verbosity. '
         'Options: DEBUG (most verbose), INFO (default), WARNING, ERROR, CRITICAL (least verbose).',
)

def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _write_srt(segments, srt_path: str) -> None:
    lines = []
    index = 1
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        lines.append(str(index))
        lines.append(
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}"
        )
        lines.append(text)
        lines.append("")
        index += 1

    with open(srt_path, "w", encoding="utf8") as fp:
        fp.write("\n".join(lines).rstrip() + "\n")


def worker(file_name, model_name, output_file_name, language=None):
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[speed]}"),
    ) as progress:
        task_id = progress.add_task(
            "[yellow]Preparing model/audio",
            total=None,
            speed="waiting for audio",
        )
        transcription_started = None
        last_progress_log_at = None
        total_duration = 0.0

        def update_progress(processed_seconds, duration_seconds):
            nonlocal transcription_started, last_progress_log_at, total_duration
            now = time.perf_counter()
            if transcription_started is None:
                transcription_started = now
                last_progress_log_at = now
            total_duration = duration_seconds
            elapsed = max(now - transcription_started, 1e-9)
            speed = processed_seconds / elapsed
            progress.update(
                task_id,
                total=max(duration_seconds, 0.001),
                completed=processed_seconds,
                description="[cyan]Transcribing",
                speed=f"{speed:.1f}× realtime",
            )
            if now - last_progress_log_at >= 60:
                percent = 100.0 * processed_seconds / max(duration_seconds, 1e-9)
                logging.info(
                    "Transcription progress: %.1f%% (%.1f / %.1f audio minutes, %.1f× realtime)",
                    percent,
                    processed_seconds / 60,
                    duration_seconds / 60,
                    speed,
                )
                last_progress_log_at = now

        decode_options = {}
        if language is not None:
            decode_options["language"] = language
        try:
            text = whisper.transcribe(
                file_name,
                model=model_name,
                progress_callback=update_progress,
                **decode_options,
            )
            transcription_elapsed = max(
                time.perf_counter() - transcription_started,
                1e-9,
            )
            logging.debug(f"Transcription result: {text}")
            with open(output_file_name, "w", encoding="utf8") as fp:
                json.dump(text, fp, ensure_ascii=False)
            srt_path = os.path.splitext(output_file_name)[0] + ".srt"
            _write_srt(text.get("segments", []), srt_path)
        except KeyboardInterrupt:
            progress.update(
                task_id,
                description="[yellow]Interrupted",
                speed="cancelled",
            )
            raise
        except Exception:
            progress.update(
                task_id,
                description="[red]Transcription failed",
                speed="failed",
            )
            raise
        speed = total_duration / transcription_elapsed
        progress.update(
            task_id,
            total=max(total_duration, 0.001),
            completed=max(total_duration, 0.001),
            description="[green]Complete",
            speed=f"done · {speed:.1f}× realtime",
        )
        logging.info(
            f"Voila!✨ Your file has been transcribed go check it out over here 👉 {output_file_name}"
        )
        logging.info(f"SRT subtitles written to 👉 {srt_path}")

def main():
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_name = args.file_name
    model_name = args.model_name
    youtube_url = args.youtube_url
    output_file_name = args.output_file_name

    def _derive_output_name(source_path: str) -> str:
        base, _ = os.path.splitext(os.path.basename(source_path))
        directory = os.path.dirname(source_path)
        return os.path.join(directory, base + ".json")

    language = args.language
    if output_file_name is not None:
        lower = output_file_name.lower()
        if lower.endswith(".srt") or lower.endswith(".json"):
            output_file_name = output_file_name[:-4] + ".json"
        else:
            output_file_name = output_file_name + ".json"
    if youtube_url is not None:
        logging.info(f'start downloading audios: {args.youtube_url}')
        audio_path = download_and_convert_to_mp3(youtube_url)
        if output_file_name is None:
            output_file_name = _derive_output_name(audio_path)
        worker(audio_path, model_name, output_file_name, language=language)
    else:
        if file_name is None:
            logging.error("local file_name should not be none!")
            return None
        if output_file_name is None:
            output_file_name = _derive_output_name(file_name)
        worker(file_name, model_name, output_file_name, language=language)
