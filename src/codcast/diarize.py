"""Sprechertrennung fuer Referenzstimmen (`codcast voices extract`).

Zieht aus Material mit mehreren Sprechern pro Person eine Auswahl sauberer
Ausschnitte, in denen nachweislich nur diese eine Person zu hoeren ist.

Die Trennung selbst macht der Worker in `.venv-diarize` (VAD, Sprecher-Embeddings,
Clustering). Hier laeuft alles ohne torch: I/O, ffmpeg und vor allem die
Auswahl-Logik. Die steckt bewusst in reinen Funktionen, weil sie und nicht das
Clustering darueber entscheidet, ob im Ergebnis wirklich nur eine Stimme liegt:

* Randerosion an jedem Sprecherwechsel, denn genau dort sitzt die Ueberlappung
* Margin-Score pro Chunk (Aehnlichkeit zum eigenen minus zum fremden Zentroid);
  Ueberlappung, Musikbett und Stoergeraeusche druecken ihn automatisch
* Auswahl ueber Zeit-Bins, damit das Material nicht aus einem Themenblock stammt
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import command_available
from .config import AppConfig, DiarizeConfig
from .util import run_checked, write_json

Span = tuple[float, float]


@dataclass
class Chunk:
    """Ein Kandidat: zusammenhaengender Ausschnitt, der einem Sprecher gehoert."""

    start: float
    end: float
    speaker: int
    speech_sec: float
    own: float = 0.0
    other: float = 0.0
    margin: float = 0.0
    peak: float = 0.0
    rms_dbfs: float = 0.0
    rejected: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_ratio(self) -> float:
        return self.speech_sec / self.duration if self.duration > 0 else 0.0


@dataclass
class SpeakerResult:
    speaker: int
    label: str
    chunks: list[Chunk] = field(default_factory=list)
    seconds: float = 0.0
    master_wav: Path | None = None
    export_wav: Path | None = None
    preview_wav: Path | None = None


# --------------------------------------------------------------------------
# Reine Auswahl-Logik. Kein I/O, keine Fremdabhaengigkeiten, direkt testbar.
# --------------------------------------------------------------------------


def smooth_labels(labels: list[int], width: int = 3) -> list[int]:
    """Mehrheitsfilter gegen einzelne fehlgeclusterte Fenster."""
    if width <= 1 or len(labels) <= width:
        return list(labels)
    half = width // 2
    smoothed = []
    for index in range(len(labels)):
        lo = max(0, index - half)
        hi = min(len(labels), index + half + 1)
        counts = Counter(labels[lo:hi])
        best = max(counts.values())
        # Bei Gleichstand das eigene Label behalten, nicht willkuerlich umkippen.
        winners = [label for label, count in counts.items() if count == best]
        smoothed.append(labels[index] if labels[index] in winners else winners[0])
    return smoothed


def build_runs(windows: list[tuple[float, float, int]], max_gap_sec: float = 1.0) -> list[Chunk]:
    """Fasst Fenster gleichen Sprechers zu Laeufen zusammen.

    Ein Wechsel des Labels bricht immer, eine Pause laenger als `max_gap_sec`
    zusaetzlich: bei kuerzeren Pausen ist es dieselbe Person, die Luft holt.
    """
    runs: list[Chunk] = []
    for start, end, label in windows:
        current = runs[-1] if runs else None
        if current is not None and current.speaker == label and start - current.end <= max_gap_sec:
            current.end = max(current.end, end)
            continue
        runs.append(Chunk(start=start, end=end, speaker=int(label), speech_sec=0.0))
    return runs


def erode_runs(runs: list[Chunk], erosion_sec: float, min_run_sec: float) -> list[Chunk]:
    """Schneidet beide Enden jedes Laufs ab und verwirft, was zu kurz wird.

    Die Enden sind die Uebergangszonen zum anderen Sprecher; dort liegt die
    Ueberlappung, die im Klonmaterial nichts zu suchen hat.
    """
    eroded = []
    for run in runs:
        start = run.start + erosion_sec
        end = run.end - erosion_sec
        if end - start >= min_run_sec:
            eroded.append(Chunk(start=start, end=end, speaker=run.speaker, speech_sec=0.0))
    return eroded


def _clip_speech(speech: list[Span], start: float, end: float) -> list[Span]:
    inside = []
    for region_start, region_end in speech:
        lo = max(region_start, start)
        hi = min(region_end, end)
        if hi > lo:
            inside.append((lo, hi))
    return inside


def split_run(run: Chunk, speech: list[Span], min_chunk_sec: float, max_chunk_sec: float) -> list[Chunk]:
    """Zerlegt einen Lauf an Sprechpausen in Chunks von min..max Sekunden.

    Geschnitten wird nur in der Stille zwischen VAD-Regionen, damit kein Wort
    angeschnitten wird. `speech_sec` faellt dabei ab und macht den Sprachanteil
    spaeter ohne Audioanalyse pruefbar.
    """
    regions = _clip_speech(speech, run.start, run.end)
    if not regions:
        return []

    chunks: list[Chunk] = []
    chunk_start = regions[0][0]
    speech_sec = 0.0
    for index, (region_start, region_end) in enumerate(regions):
        # Eine einzelne Region laenger als max: hart durchschneiden.
        cursor = region_start
        while region_end - cursor > max_chunk_sec:
            cut = cursor + max_chunk_sec
            chunks.append(
                Chunk(
                    start=chunk_start,
                    end=cut,
                    speaker=run.speaker,
                    speech_sec=speech_sec + (cut - cursor),
                )
            )
            cursor = cut
            chunk_start = cursor
            speech_sec = 0.0
        speech_sec += region_end - cursor

        is_last = index == len(regions) - 1
        next_end = regions[index + 1][1] if not is_last else None
        would_overflow = next_end is not None and next_end - chunk_start > max_chunk_sec
        if (is_last or would_overflow) and region_end - chunk_start >= min_chunk_sec:
            chunks.append(
                Chunk(start=chunk_start, end=region_end, speaker=run.speaker, speech_sec=speech_sec)
            )
            if not is_last:
                chunk_start = regions[index + 1][0]
                speech_sec = 0.0
            else:
                chunk_start = None
        elif would_overflow:
            # Zu kurz zum Behalten: verwerfen und beim naechsten Sprachstueck neu ansetzen.
            chunk_start = regions[index + 1][0]
            speech_sec = 0.0
    return chunks


def apply_gates(chunk: Chunk, config: DiarizeConfig) -> Chunk:
    """Setzt `rejected` auf den ersten verletzten Grenzwert, sonst None."""
    if chunk.speech_ratio < config.min_speech_ratio:
        chunk.rejected = f"speech_ratio<{config.min_speech_ratio}"
    elif chunk.margin < config.min_margin:
        chunk.rejected = f"margin<{config.min_margin}"
    elif chunk.peak > config.max_peak:
        chunk.rejected = f"peak>{config.max_peak}"
    elif not config.min_rms_dbfs <= chunk.rms_dbfs <= config.max_rms_dbfs:
        chunk.rejected = f"rms_dbfs outside [{config.min_rms_dbfs},{config.max_rms_dbfs}]"
    else:
        chunk.rejected = None
    return chunk


def select_chunks(
    chunks: list[Chunk], target_sec: float, time_bins: int, total_duration: float
) -> list[Chunk]:
    """Waehlt die besten Chunks bis `target_sec`, ueber die Laufzeit verteilt.

    Rein nach Margin zu sortieren wuerde alles aus der einen Passage ziehen, in
    der die Person am gleichmaessigsten spricht. Round-Robin ueber Zeit-Bins
    kostet etwas Margin und bringt dafuer Prosodie-Varianz, die beim Klonen mehr
    wiegt. Reicht das Material nicht, kommt zurueck, was da ist.
    """
    if not chunks:
        return []
    bins: list[list[Chunk]] = [[] for _ in range(max(1, time_bins))]
    span = max(total_duration, 1e-6)
    for chunk in chunks:
        index = min(len(bins) - 1, int(chunk.start / span * len(bins)))
        bins[index].append(chunk)
    for bucket in bins:
        bucket.sort(key=lambda item: item.margin, reverse=True)

    selected: list[Chunk] = []
    total = 0.0
    cursors = [0] * len(bins)
    while total < target_sec:
        progressed = False
        for index, bucket in enumerate(bins):
            if cursors[index] >= len(bucket):
                continue
            chunk = bucket[cursors[index]]
            cursors[index] += 1
            selected.append(chunk)
            total += chunk.duration
            progressed = True
            if total >= target_sec:
                break
        if not progressed:
            break
    selected.sort(key=lambda item: item.start)
    return selected


def speaker_label(index: int) -> str:
    return f"speaker_{chr(ord('A') + index)}"


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


class DiarizeWorker:
    """Duenner Wrapper um den Subprozess in `.venv-diarize`."""

    WORKER = Path(__file__).resolve().parent / "workers" / "diarize_worker.py"

    def __init__(self, config: DiarizeConfig, savedir: Path, device: str | None = None) -> None:
        self.config = config
        self.savedir = savedir
        self.device = device or config.device
        self._process: subprocess.Popen[str] | None = None
        self._events: queue.Queue[dict | None] = queue.Queue()
        self._stderr_tail: list[str] = []

    def __enter__(self) -> DiarizeWorker:
        python = self.config.python_executable
        if not python.exists():
            raise RuntimeError(
                f"Diarize-Python nicht gefunden: {python}. "
                "Einmalig einrichten mit `uv run codcast setup-diarize`, "
                "oder diarize.python_executable in podcast.yml anpassen."
            )
        if not self.WORKER.exists():
            raise RuntimeError(f"Diarize-Worker fehlt: {self.WORKER}")
        self._process = subprocess.Popen(
            [str(python), "-u", str(self.WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_events, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._send(
            {
                "device": self.device,
                "embedding_model": self.config.embedding_model,
                "savedir": str(self.savedir),
            }
        )
        ready = self._await("Modellstart")
        if ready.get("event") != "ready":
            raise RuntimeError(f"Diarize-Worker meldete kein ready: {ready}. {self._log_hint()}")
        self.device = ready.get("device", self.device)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _read_events(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._events.put(json.loads(line))
            except json.JSONDecodeError:
                self._stderr_tail.append(f"stdout: {line[:200]}")
        self._events.put(None)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_tail.append(line.rstrip())
            del self._stderr_tail[:-80]

    def _log_hint(self) -> str:
        if not self._stderr_tail:
            return ""
        return "Letzte Worker-Ausgabe:\n" + "\n".join(self._stderr_tail[-15:])

    def _send(self, payload: dict) -> None:
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()

    def _await(self, what: str) -> dict:
        try:
            event = self._events.get(timeout=self.config.timeout_sec)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError(
                f"Diarize-Worker antwortete nicht ({what}, {self.config.timeout_sec}s). {self._log_hint()}"
            ) from exc
        if event is None:
            code = self._process.poll() if self._process else None
            raise RuntimeError(f"Diarize-Worker beendet (exit {code}) bei {what}. {self._log_hint()}")
        return event

    def request(self, payload: dict) -> dict:
        self._send(payload)
        event = self._await(payload.get("cmd", "?"))
        if event.get("event") == "error":
            raise RuntimeError(f"Diarize-Worker: {event.get('message')}. {self._log_hint()}")
        return event

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=30)
        except Exception:
            process.kill()


# --------------------------------------------------------------------------
# Audio-I/O
# --------------------------------------------------------------------------


def _decode(source: Path, dest: Path, rate: int) -> None:
    run_checked(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(dest),
        ]
    )


def measure(master: Path, chunks: list[Chunk]) -> None:
    """Peak und RMS pro Chunk, gelesen mit gezieltem Seek statt ffmpeg pro Chunk."""
    with sf.SoundFile(str(master)) as handle:
        rate = handle.samplerate
        for chunk in chunks:
            handle.seek(int(chunk.start * rate))
            data = handle.read(int(chunk.duration * rate), dtype="float32", always_2d=False)
            if data.size == 0:
                chunk.peak, chunk.rms_dbfs = 0.0, -120.0
                continue
            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
            chunk.peak = peak
            chunk.rms_dbfs = 20.0 * float(np.log10(rms)) if rms > 0 else -120.0


def _write_chunks(master: Path, chunks: list[Chunk], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with sf.SoundFile(str(master)) as handle:
        rate = handle.samplerate
        for index, chunk in enumerate(chunks):
            handle.seek(int(chunk.start * rate))
            data = handle.read(int(chunk.duration * rate), dtype="float32", always_2d=False)
            stamp = f"{int(chunk.start // 60):02d}-{int(chunk.start % 60):02d}"
            path = dest_dir / f"{index:03d}_{stamp}.wav"
            sf.write(str(path), data, rate, subtype="PCM_16")
            paths.append(path)
    return paths


def _concat(master: Path, chunks: list[Chunk], dest: Path, gap_sec: float) -> None:
    with sf.SoundFile(str(master)) as handle:
        rate = handle.samplerate
        gap = np.zeros(int(gap_sec * rate), dtype="float32")
        parts: list[np.ndarray] = []
        for chunk in chunks:
            handle.seek(int(chunk.start * rate))
            parts.append(handle.read(int(chunk.duration * rate), dtype="float32", always_2d=False))
            parts.append(gap)
    joined = np.concatenate(parts[:-1]) if parts else np.zeros(0, dtype="float32")
    sf.write(str(dest), joined, rate, subtype="PCM_16")


def _loudnorm(src: Path, dest: Path, rate: int, config: DiarizeConfig) -> None:
    run_checked(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-af", f"loudnorm=I={config.loudnorm_i}:TP={config.loudnorm_tp}:LRA=11",
            "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", str(dest),
        ]
    )


def _trim(src: Path, dest: Path, seconds: float) -> None:
    run_checked(["ffmpeg", "-y", "-i", str(src), "-t", f"{seconds:.3f}", "-c", "copy", str(dest)])


# --------------------------------------------------------------------------
# Ablauf
# --------------------------------------------------------------------------


def extract_voices(
    app_config: AppConfig,
    source: Path,
    out_dir: Path,
    *,
    speakers: int = 2,
    minutes: float = 5.0,
    device: str | None = None,
    skip_head: float = 0.0,
    skip_tail: float = 0.0,
    log=print,
) -> dict:
    config = app_config.diarize
    if not command_available("ffmpeg"):
        raise RuntimeError("ffmpeg wird gebraucht, ist aber nicht installiert.")
    source = source.expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"Quelldatei nicht gefunden: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    analysis = work / "analysis_16k.wav"
    master = work / f"master_{config.master_sample_rate}.wav"

    log(f"Dekodiere {source.name} ...")
    _decode(source, analysis, config.analysis_sample_rate)
    _decode(source, master, config.master_sample_rate)

    with DiarizeWorker(config, config.model_dir, device=device) as worker:
        used_device = worker.device
        log(f"Modell geladen (device={used_device}). Analysiere ...")
        analysis_result = worker.request(
            {
                "cmd": "analyze",
                "audio": str(analysis),
                "speakers": speakers,
                "sample_rate": config.analysis_sample_rate,
                "window_sec": config.window_sec,
                "hop_sec": config.hop_sec,
            }
        )
        duration = float(analysis_result["duration"])
        speech: list[Span] = [(float(a), float(b)) for a, b in analysis_result["speech"]]
        windows = [(float(a), float(b), int(c)) for a, b, c in analysis_result["windows"]]
        similarity = analysis_result["centroid_similarity"]
        cross = max(
            similarity[i][j] for i in range(speakers) for j in range(speakers) if i != j
        )
        log(
            f"{len(windows)} Fenster, Sprachanteil "
            f"{sum(e - s for s, e in speech) / duration * 100:.0f}%, "
            f"Zentroid-Aehnlichkeit zwischen den Sprechern {cross:.3f}"
        )

        labels = smooth_labels([label for _, _, label in windows])
        runs = build_runs(
            [(start, end, label) for (start, end, _), label in zip(windows, labels)],
            config.max_gap_sec,
        )
        runs = erode_runs(runs, config.erosion_sec, config.min_run_sec)
        candidates: list[Chunk] = []
        head_limit = skip_head
        tail_limit = duration - skip_tail
        for run in runs:
            if run.end <= head_limit or run.start >= tail_limit:
                continue
            candidates.extend(
                split_run(run, speech, config.min_chunk_sec, config.max_chunk_sec)
            )
        candidates = [
            chunk for chunk in candidates if chunk.start >= head_limit and chunk.end <= tail_limit
        ]
        if not candidates:
            raise RuntimeError("Keine Kandidaten nach Erosion und Segmentierung uebrig.")
        log(f"{len(runs)} Laeufe -> {len(candidates)} Kandidaten-Chunks, bewerte ...")

        scores = worker.request(
            {
                "cmd": "score",
                "audio": str(analysis),
                "sample_rate": config.analysis_sample_rate,
                "chunks": [[c.start, c.end, c.speaker] for c in candidates],
            }
        )["scores"]
        for chunk, (own, other, margin) in zip(candidates, scores):
            chunk.own, chunk.other, chunk.margin = own, other, margin

        measure(master, candidates)
        for chunk in candidates:
            apply_gates(chunk, config)
        survivors = [chunk for chunk in candidates if chunk.rejected is None]
        rejected = Counter(chunk.rejected for chunk in candidates if chunk.rejected)
        log(f"{len(survivors)} Chunks bestehen die Gates. Verworfen: {dict(rejected)}")

        target_sec = minutes * 60.0
        results: list[SpeakerResult] = []
        for speaker in range(speakers):
            pool = [chunk for chunk in survivors if chunk.speaker == speaker]
            picked = select_chunks(pool, target_sec, config.time_bins, duration)
            result = SpeakerResult(
                speaker=speaker,
                label=speaker_label(speaker),
                chunks=picked,
                seconds=sum(chunk.duration for chunk in picked),
            )
            results.append(result)
            if result.seconds < target_sec:
                log(
                    f"  {result.label}: nur {result.seconds:.0f}s von {target_sec:.0f}s "
                    f"({len(pool)} Chunks im Pool) - das Material gibt nicht mehr her."
                )
            else:
                log(f"  {result.label}: {result.seconds:.0f}s aus {len(picked)} Chunks")

        for result in results:
            speaker_dir = out_dir / result.label
            speaker_dir.mkdir(parents=True, exist_ok=True)
            _write_chunks(master, result.chunks, speaker_dir / "chunks")
            raw = speaker_dir / f"{result.label}_raw.wav"
            _concat(master, result.chunks, raw, config.gap_between_chunks_sec)
            result.master_wav = speaker_dir / f"{result.label}.wav"
            _loudnorm(raw, result.master_wav, config.master_sample_rate, config)
            result.export_wav = speaker_dir / f"{result.label}_{config.export_sample_rate // 1000}k.wav"
            _loudnorm(raw, result.export_wav, config.export_sample_rate, config)
            result.preview_wav = speaker_dir / f"{result.label}_preview.wav"
            _trim(result.master_wav, result.preview_wav, config.preview_sec)
            raw.unlink()

        log("Verifiziere die fertigen Dateien ...")
        verification = worker.request(
            {
                "cmd": "verify",
                "files": [str(result.master_wav) for result in results],
                "window_sec": 10.0,
            }
        )

    report = {
        "source": str(source),
        "duration_sec": duration,
        "device": used_device,
        "speakers": speakers,
        "target_sec": target_sec,
        "windows": len(windows),
        "candidates": len(candidates),
        "survivors": len(survivors),
        "rejected": dict(rejected),
        "centroid_similarity": similarity,
        "verification": {
            "between_speakers": verification["between"],
            "within_speaker": verification["within"],
        },
        "speakers_detail": [
            {
                "label": result.label,
                "seconds": result.seconds,
                "chunk_count": len(result.chunks),
                "margin_mean": (
                    sum(chunk.margin for chunk in result.chunks) / len(result.chunks)
                    if result.chunks
                    else 0.0
                ),
                "wav": str(result.master_wav),
                "export_wav": str(result.export_wav),
                "preview_wav": str(result.preview_wav),
                "chunks": [asdict(chunk) for chunk in result.chunks],
            }
            for result in results
        ],
    }
    write_json(out_dir / "report.json", report)
    return report
