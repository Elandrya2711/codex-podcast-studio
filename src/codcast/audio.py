from __future__ import annotations

import subprocess
from pathlib import Path

from .config import AudioConfig
from .models import RenderedSegment
from .progress import CancellationToken, ProgressEvent, ProgressReporter, report_progress
from .util import run_checked


def ffprobe_duration(path: Path) -> float:
    result = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def make_silence(path: Path, duration_sec: float, config: AudioConfig) -> None:
    run_checked(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={config.sample_rate}:cl=mono",
            "-t",
            f"{duration_sec:.3f}",
            "-ac",
            str(config.channels),
            str(path),
        ]
    )


def normalize_wav(src: Path, dest: Path, config: AudioConfig) -> None:
    filters = []
    if config.wav_loudnorm:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.extend(["-ar", str(config.sample_rate), "-ac", str(config.channels), str(dest)])
    run_checked(cmd)


def concat_wavs(inputs: list[Path], dest: Path) -> None:
    concat_file = dest.with_suffix(".ffconcat.txt")
    lines = ["ffconcat version 1.0"]
    for path in inputs:
        safe_path = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{safe_path}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_checked(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(dest)])


def export_mp3(wav_path: Path, mp3_path: Path, bitrate: str) -> None:
    run_checked(["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)])


def assemble_episode(
    segments: list[RenderedSegment],
    run_dir: Path,
    audio_config: AudioConfig,
    pause_between_lines_sec: float,
    output_stem: str = "final",
    progress: ProgressReporter | None = None,
    cancellation: CancellationToken | None = None,
) -> tuple[Path, Path, float]:
    assembly_dir = run_dir / "assembly"
    assembly_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    pause_path = assembly_dir / "pause.wav"
    if cancellation:
        cancellation.raise_if_cancelled()
    report_progress(progress, ProgressEvent("progress", "assembly", "Pause-Datei erzeugen", 0, len(segments) + 3))
    make_silence(pause_path, pause_between_lines_sec, audio_config)

    for segment in segments:
        if cancellation:
            cancellation.raise_if_cancelled()
        src = Path(segment.wav_path)
        dest = assembly_dir / f"{segment.index:04d}_{segment.speaker_id}.wav"
        report_progress(
            progress,
            ProgressEvent(
                "progress",
                "assembly",
                f"Segment {segment.index}/{len(segments)} normalisieren",
                segment.index,
                len(segments) + 3,
            ),
        )
        normalize_wav(src, dest, audio_config)
        normalized.append(dest)
        if segment.index != len(segments):
            normalized.append(pause_path)

    final_wav = run_dir / f"{output_stem}.wav"
    final_mp3 = run_dir / f"{output_stem}.mp3"
    if cancellation:
        cancellation.raise_if_cancelled()
    report_progress(progress, ProgressEvent("progress", "assembly", "WAV zusammensetzen", len(segments) + 1, len(segments) + 3))
    concat_wavs(normalized, final_wav)
    if cancellation:
        cancellation.raise_if_cancelled()
    report_progress(progress, ProgressEvent("progress", "assembly", "MP3 exportieren", len(segments) + 2, len(segments) + 3))
    export_mp3(final_wav, final_mp3, audio_config.mp3_bitrate)
    report_progress(progress, ProgressEvent("progress", "assembly", "Dauer pruefen", len(segments) + 3, len(segments) + 3))
    return final_wav, final_mp3, ffprobe_duration(final_wav)


def command_available(command: str) -> bool:
    return subprocess.run(["sh", "-lc", f"command -v {command}"], capture_output=True, text=True).returncode == 0
