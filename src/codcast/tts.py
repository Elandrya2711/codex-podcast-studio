from __future__ import annotations

import atexit
import builtins
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import soundfile as sf

from .config import AppConfig, VoiceProfile
from .models import PodcastScript, RenderedSegment, ScriptLine
from .progress import CancellationToken, ProgressEvent, ProgressReporter, report_progress
from .text_normalization import normalize_for_tts
from .util import run_checked


@dataclass(frozen=True)
class ScriptRenderChunk:
    speaker_id: str
    text: str
    source_line_indexes: tuple[int, ...]


@dataclass(frozen=True)
class PreparedRenderChunk:
    index: int
    chunk: ScriptRenderChunk
    voice: VoiceProfile
    wav_path: Path


class TTSBackend:
    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        raise NotImplementedError


def _write_bounded_response(response: object, output_path: Path, *, max_bytes: int, label: str) -> None:
    content_length = getattr(response, "headers", {}).get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > max_bytes:
            raise RuntimeError(f"{label} response too large: {declared_size} bytes (limit {max_bytes})")

    written = 0
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                written += len(chunk)
                if written > max_bytes:
                    raise RuntimeError(f"{label} response exceeded {max_bytes} bytes")
                handle.write(chunk)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


class KokoroBackend(TTSBackend):
    sample_rate = 24000

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._pipelines: dict[str, object] = {}

    def _pipeline_for(self, voice: VoiceProfile):
        if not voice.kokoro_model:
            raise ValueError(f"voice {voice.id} has no kokoro_model")
        if not voice.kokoro_voice:
            raise ValueError(f"voice {voice.id} has no kokoro_voice")
        if not voice.kokoro_model.exists():
            raise FileNotFoundError(f"Kokoro model not found: {voice.kokoro_model}")
        if not voice.kokoro_voice.exists():
            raise FileNotFoundError(f"Kokoro voice pack not found: {voice.kokoro_voice}")
        key = str(voice.kokoro_model.resolve())
        if key not in self._pipelines:
            from kokoro import KPipeline
            from kokoro.model import KModel
            import kokoro.pipeline as kokoro_pipeline

            lang_code = self.config.tts.kokoro.lang_code
            kokoro_pipeline.LANG_CODES.setdefault(lang_code, lang_code)
            config_path = self.config.tts.kokoro.config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Kokoro config not found: {config_path}")
            model = _load_kokoro_model(KModel, config_path, voice.kokoro_model)
            model = model.to(self.config.tts.kokoro.device).eval()
            self._pipelines[key] = KPipeline(
                lang_code=lang_code,
                model=model,
                repo_id="hexgrad/Kokoro-82M",
                device=self.config.tts.kokoro.device,
            )
        return self._pipelines[key]

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        pipeline = self._pipeline_for(voice)
        chunks: list[np.ndarray] = []
        for result in pipeline(text, voice=str(voice.kokoro_voice), speed=voice.speed):  # type: ignore[operator]
            if result.audio is None:
                continue
            chunks.append(result.audio.detach().cpu().numpy())
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for voice {voice.id}")
        audio = np.concatenate(chunks)
        sf.write(output_path, audio, self.sample_rate)


class ChatterboxBackend(TTSBackend):
    """Chatterbox Multilingual ueber einen persistenten Worker-Prozess.

    Der Worker laeuft in einer eigenen venv und haelt das Modell im VRAM, damit
    nicht jedes Segment die 3 GB Gewichte neu laedt. Kommuniziert wird mit einer
    JSON-Zeile pro Job (siehe workers/chatterbox_worker.py).
    """

    WORKER = Path(__file__).parent / "workers" / "chatterbox_worker.py"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._events: queue.Queue[dict | None] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._sample_rate: int | None = None
        self._next_id = 0
        self._lock = threading.Lock()
        # Segmente, die auch nach den Wiederholungen auffaellig blieben.
        self.warnings: list[str] = []
        atexit.register(self.close)

    def _start_worker(self) -> None:
        chatterbox = self.config.tts.chatterbox
        python = chatterbox.python_executable
        if not python.exists():
            raise RuntimeError(
                f"Chatterbox-Python nicht gefunden: {python}. "
                "Einmalig einrichten mit `uv run codcast setup-chatterbox`, "
                "oder tts.chatterbox.python_executable in podcast.yml anpassen."
            )
        if not self.WORKER.exists():
            raise RuntimeError(f"Chatterbox-Worker fehlt: {self.WORKER}")

        self._process = subprocess.Popen(
            [str(python), "-u", str(self.WORKER), chatterbox.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_events, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        ready = self._await_event(chatterbox.startup_timeout_sec, "Modellstart")
        if ready.get("event") != "ready":
            raise RuntimeError(f"Chatterbox-Worker meldete kein ready: {ready}. {self._log_hint()}")
        self._sample_rate = ready.get("sample_rate")

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
            del self._stderr_tail[:-60]

    def _log_hint(self) -> str:
        if not self._stderr_tail:
            return ""
        return "Letzte Worker-Ausgabe:\n" + "\n".join(self._stderr_tail[-15:])

    def _await_event(self, timeout_sec: int, what: str) -> dict:
        try:
            event = self._events.get(timeout=timeout_sec)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError(f"Chatterbox-Worker antwortete nicht ({what}, {timeout_sec}s). {self._log_hint()}") from exc
        if event is None:
            code = self._process.poll() if self._process else None
            self.close()
            raise RuntimeError(f"Chatterbox-Worker beendet (exit {code}) waehrend {what}. {self._log_hint()}")
        return event

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        if not voice.speaker_wav:
            raise ValueError(
                f"Chatterbox-Stimme {voice.id} braucht speaker_wav. "
                "Referenz importieren mit `codcast voices import --backend chatterbox ...`."
            )
        if not voice.speaker_wav.exists():
            raise FileNotFoundError(f"Chatterbox reference not found: {voice.speaker_wav}")

        chatterbox = self.config.tts.chatterbox
        spoken = normalize_for_tts(text) if chatterbox.normalize_text else text
        for attempt in range(chatterbox.max_retries + 1):
            seconds = self._render_once(spoken, voice, output_path)
            problem = _implausible_duration(spoken, seconds, chatterbox)
            if problem is None:
                break
            if attempt == chatterbox.max_retries:
                warning = f"Segment bleibt nach {attempt + 1} Versuchen auffaellig ({problem}): {spoken[:70]}"
                self.warnings.append(warning)
                print(f"[chatterbox] {warning}", file=sys.stderr)
                break

        if voice.speed != 1.0:
            _apply_tempo(output_path, voice.speed)

    def _render_once(self, spoken: str, voice: VoiceProfile, output_path: Path) -> float:
        chatterbox = self.config.tts.chatterbox
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start_worker()
            assert self._process is not None and self._process.stdin is not None
            self._next_id += 1
            job_id = self._next_id
            job = {
                "id": job_id,
                "text": spoken,
                "reference": str(voice.speaker_wav),
                "output_path": str(output_path),
                "language": voice.language.split("-")[0] or chatterbox.language,
                "exaggeration": _first_set(voice.chatterbox_exaggeration, chatterbox.exaggeration),
                "cfg_weight": _first_set(voice.chatterbox_cfg_weight, chatterbox.cfg_weight),
                "temperature": _first_set(voice.chatterbox_temperature, chatterbox.temperature),
                "repetition_penalty": chatterbox.repetition_penalty,
            }
            self._process.stdin.write(json.dumps(job) + "\n")
            self._process.stdin.flush()

            while True:
                event = self._await_event(chatterbox.timeout_sec, f"Segment {job_id}")
                if event.get("id") == job_id:
                    break
            if event.get("event") != "done":
                raise RuntimeError(
                    f"Chatterbox konnte Segment nicht rendern: {event.get('message')}. {self._log_hint()}"
                )
            return float(event.get("seconds") or 0.0)

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=10)
        except Exception:
            process.kill()


def _first_set(voice_value: float | None, config_value: float) -> float:
    return config_value if voice_value is None else voice_value


def _implausible_duration(text: str, seconds: float, chatterbox) -> str | None:
    """Grobe Plausibilitaetspruefung der Segmentdauer.

    Ein Wiederholungs-Loop verdoppelt die Dauer, ein Abbruch halbiert sie. Beides
    laesst sich an der erwarteten Sprechdauer erkennen, ohne das Audio zu verstehen.
    """
    if seconds <= 0:
        return "keine Dauer gemeldet"
    expected = max(len(text) / chatterbox.chars_per_second, 1.0)
    if seconds > expected * chatterbox.max_duration_ratio:
        return f"{seconds:.1f}s statt erwarteter {expected:.1f}s, moeglicher Wiederholungs-Loop"
    if seconds < expected * chatterbox.min_duration_ratio:
        return f"{seconds:.1f}s statt erwarteter {expected:.1f}s, moeglicher Abbruch"
    return None


def _apply_tempo(path: Path, speed: float) -> None:
    if not 0.5 <= speed <= 2.0:
        raise ValueError(f"speed must be between 0.5 and 2.0 for tempo adjustment, got {speed}")
    adjusted = path.with_name(f"{path.stem}.tempo{path.suffix}")
    run_checked(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-filter:a", f"atempo={speed:.4f}", str(adjusted)])
    adjusted.replace(path)


class XttsBackend(TTSBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._tts = None

    def _load(self):
        if self._tts is None:
            try:
                from TTS.api import TTS  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "XTTS dependencies are not installed. Run `uv sync --extra xtts` in the project "
                    "or use Fish S2 Pro with `--quality best`."
                ) from exc
            if self.config.tts.xtts.accept_cpml_noncommercial:
                with _auto_accept_coqui_cpml():
                    self._tts = TTS(self.config.tts.xtts.model_name, gpu=self.config.tts.xtts.gpu)
            else:
                self._tts = TTS(self.config.tts.xtts.model_name, gpu=self.config.tts.xtts.gpu)
        return self._tts

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        if not voice.speaker_wav:
            raise ValueError(f"voice {voice.id} has no speaker_wav")
        if not voice.speaker_wav.exists():
            raise FileNotFoundError(f"XTTS speaker reference not found: {voice.speaker_wav}")
        tts = self._load()
        tts.tts_to_file(
            text=text,
            speaker_wav=str(voice.speaker_wav),
            language=self.config.tts.language,
            file_path=str(output_path),
        )


class OpenAIBackend(TTSBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("OpenAI TTS requires `requests`, which is not installed.") from exc

        openai_config = self.config.tts.openai
        api_key = _api_key_from_env_or_file(openai_config.api_key_env, openai_config.env_file)
        if not api_key:
            raise RuntimeError(
                "OpenAI TTS API key not found. "
                f"Set {openai_config.api_key_env}=... in {openai_config.env_file}."
            )

        payload = {
            "model": openai_config.model,
            "voice": voice.openai_voice or openai_config.voice,
            "input": text,
            "instructions": voice.openai_instructions or openai_config.instructions,
            "response_format": openai_config.response_format,
        }
        response = requests.post(
            f"{openai_config.base_url.rstrip('/')}/audio/speech",
            json=payload,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            timeout=openai_config.timeout_sec,
            stream=True,
        )
        if response.status_code != 200:
            detail = response.text[:1000]
            response.close()
            raise RuntimeError(f"OpenAI TTS request failed with HTTP {response.status_code}: {detail}")
        _write_bounded_response(response, output_path, max_bytes=openai_config.max_output_bytes, label="OpenAI TTS")


class FishBackend(TTSBackend):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        if self.config.tts.fish.output_format != "wav":
            raise ValueError("Fish output_format must be 'wav' for podcast segment assembly")
        try:
            import ormsgpack
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Fish dependencies are not installed. Run `uv sync --extra dev` or install "
                "`requests` and `ormsgpack`."
            ) from exc

        payload = fish_payload_for_voice(text, voice, self.config)
        params = urlencode({"format": "msgpack"})
        url = f"{self.config.tts.fish.server_url}?{params}"
        headers = {"content-type": "application/msgpack"}
        api_key = self.config.tts.fish.api_key
        if api_key is None and self.config.tts.fish.api_key_env:
            api_key = os.environ.get(self.config.tts.fish.api_key_env)
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        response = requests.post(
            url,
            data=ormsgpack.packb(payload),
            headers=headers,
            timeout=self.config.tts.fish.timeout_sec,
            stream=True,
        )
        if response.status_code != 200:
            detail = response.text[:1000]
            response.close()
            raise RuntimeError(f"Fish TTS request failed with HTTP {response.status_code}: {detail}")
        _write_bounded_response(response, output_path, max_bytes=self.config.tts.fish.max_output_bytes, label="Fish TTS")


def backend_for_voice(config: AppConfig, voice: VoiceProfile) -> TTSBackend:
    if voice.backend == "chatterbox":
        return ChatterboxBackend(config)
    if voice.backend == "fish":
        return FishBackend(config)
    if voice.backend == "kokoro":
        return KokoroBackend(config)
    if voice.backend == "openai":
        return OpenAIBackend(config)
    if voice.backend == "xtts":
        return XttsBackend(config)
    raise ValueError(f"unsupported TTS backend: {voice.backend}")


class ScriptRenderer:
    def __init__(self, config: AppConfig, voices: dict[str, VoiceProfile]) -> None:
        self.config = config
        self.voices = voices
        self.backends: dict[str, TTSBackend] = {}
        # Wird nur waehrend render_script gesetzt, damit auch die Meldungen aus
        # der Wiederholschleife in der TUI landen und nicht nur auf stderr.
        self._progress: ProgressReporter | None = None

    def _backend(self, voice: VoiceProfile) -> TTSBackend:
        if voice.backend not in self.backends:
            self.backends[voice.backend] = backend_for_voice(self.config, voice)
        return self.backends[voice.backend]

    def _drop_backend(self, voice: VoiceProfile) -> None:
        backend = self.backends.pop(voice.backend, None)
        if backend is not None:
            close_backend(backend)

    def _voice_for_line(self, script: PodcastScript, line: ScriptLine) -> VoiceProfile:
        return self._voice_for_speaker_id(script, line.speaker_id)

    def _voice_for_speaker_id(self, script: PodcastScript, speaker_id: str) -> VoiceProfile:
        speaker = next((speaker for speaker in script.speakers if speaker.id == speaker_id), None)
        if speaker is None:
            raise ValueError(f"script references unknown speaker_id: {speaker_id}")
        voice = self.voices.get(speaker.voice_profile_id)
        if voice is None:
            raise ValueError(f"script references unknown voice_profile_id: {speaker.voice_profile_id}")
        return voice

    def _render_chunks_for_script(self, script: PodcastScript) -> list[ScriptRenderChunk]:
        if self._should_group_openai_single_speaker(script):
            return group_single_speaker_openai_lines(script.lines, self.config.tts.openai.max_input_chars)
        return [
            ScriptRenderChunk(line.speaker_id, " ".join(line.text.split()), (index,))
            for index, line in enumerate(script.lines, start=1)
        ]

    def _should_group_openai_single_speaker(self, script: PodcastScript) -> bool:
        if len(script.speakers) != 1 or not script.lines:
            return False
        voice = self._voice_for_speaker_id(script, script.speakers[0].id)
        return voice.backend == "openai"

    def render_script(
        self,
        script: PodcastScript,
        run_dir: Path,
        *,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> list[RenderedSegment]:
        segment_dir = run_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        chunks = self._render_chunks_for_script(script)
        total = len(chunks)
        prepared: list[PreparedRenderChunk] = []
        for index, chunk in enumerate(chunks, start=1):
            voice = self._voice_for_speaker_id(script, chunk.speaker_id)
            prepared.append(
                PreparedRenderChunk(
                    index=index,
                    chunk=chunk,
                    voice=voice,
                    wav_path=segment_dir / f"{index:04d}_{chunk.speaker_id}_{voice.id}.wav",
                )
            )

        self._progress = progress
        try:
            if self._should_render_openai_chunks_parallel(prepared):
                return self._render_prepared_chunks_parallel(prepared, total, progress, cancellation)
            return self._render_prepared_chunks_serial(prepared, total, progress, cancellation)
        finally:
            # Lokale Modelle geben ihren VRAM frei, bevor Assembly und Export laufen.
            self.close()
            self._progress = None

    def close(self) -> None:
        for backend in self.backends.values():
            close_backend(backend)
        self.backends.clear()

    def _should_render_openai_chunks_parallel(self, prepared: list[PreparedRenderChunk]) -> bool:
        return self.config.tts.openai.concurrency > 1 and len(prepared) > 1 and all(item.voice.backend == "openai" for item in prepared)

    def _render_prepared_chunks_serial(
        self,
        prepared: list[PreparedRenderChunk],
        total: int,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> list[RenderedSegment]:
        rendered: list[RenderedSegment] = []
        for item in prepared:
            if cancellation:
                cancellation.raise_if_cancelled()
            report_progress(
                progress,
                ProgressEvent(
                    "progress",
                    "tts",
                    f"Segment {item.index}/{total} rendern ({item.voice.display_name})",
                    item.index - 1,
                    total,
                ),
            )
            self._render_prepared_chunk(item)
            rendered.append(
                rendered_segment_for_chunk(item)
            )
            report_progress(progress, ProgressEvent("progress", "tts", f"Segment {item.index}/{total} fertig", item.index, total))
        return rendered

    def _render_prepared_chunks_parallel(
        self,
        prepared: list[PreparedRenderChunk],
        total: int,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> list[RenderedSegment]:
        max_workers = max(1, min(self.config.tts.openai.concurrency, total))
        rendered_by_index: dict[int, RenderedSegment] = {}
        completed = 0
        report_progress(progress, ProgressEvent("progress", "tts", f"{total} OpenAI-Segmente parallel rendern", 0, total))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for item in prepared:
                if cancellation:
                    cancellation.raise_if_cancelled()
                futures[executor.submit(self._render_prepared_chunk, item)] = item

            for future in as_completed(futures):
                if cancellation:
                    cancellation.raise_if_cancelled()
                item = futures[future]
                future.result()
                rendered_by_index[item.index] = rendered_segment_for_chunk(item)
                completed += 1
                report_progress(
                    progress,
                    ProgressEvent("progress", "tts", f"Segment {item.index}/{total} fertig", completed, total),
                )
        return [rendered_by_index[index] for index in sorted(rendered_by_index)]

    def _render_prepared_chunk(self, item: PreparedRenderChunk) -> None:
        if self._reuse_finished_segment(item):
            return
        self._render_chunk_with_oom_retry(item)

    def _fingerprint_path(self, item: PreparedRenderChunk) -> Path:
        return item.wav_path.with_name(f"{item.wav_path.stem}.fingerprint.json")

    def _segment_fingerprint(self, item: PreparedRenderChunk) -> dict[str, object]:
        """Alles, was den Klang dieses Segments bestimmt.

        Bewusst grosszuegig gefasst: ein Feld zu viel laesst unnoetig neu
        rendern, ein Feld zu wenig liefert altes Audio zu neuem Text. Nur der
        zweite Fehler ist teuer, weil er unbemerkt bleibt.
        """
        backend_config = getattr(self.config.tts, item.voice.backend, None)
        return {
            "text": item.chunk.text,
            "voice": item.voice.model_dump(mode="json"),
            "backend_config": (
                backend_config.model_dump(mode="json") if backend_config is not None else None
            ),
        }

    def _reuse_finished_segment(self, item: PreparedRenderChunk) -> bool:
        if not self.config.tts.reuse_segments:
            return False
        fingerprint_path = self._fingerprint_path(item)
        if not item.wav_path.exists() or not fingerprint_path.exists():
            return False
        try:
            stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if stored != self._segment_fingerprint(item):
            return False
        report_progress(
            self._progress,
            ProgressEvent("progress", "tts", f"Segment {item.index} unveraendert, uebernommen"),
        )
        return True

    def _render_chunk_with_oom_retry(self, item: PreparedRenderChunk) -> None:
        attempts = max(0, self.config.tts.gpu_oom_retries) + 1
        for attempt in range(1, attempts + 1):
            try:
                self._render_chunk_atomically(item)
                return
            except Exception as error:
                if not _is_out_of_memory(error):
                    raise
                if attempt == attempts:
                    raise RuntimeError(self._out_of_memory_advice(item, attempts)) from error
                wait = max(0, self.config.tts.gpu_oom_wait_sec)
                hint = gpu_memory_hint()
                freed = free_ollama_models() if self.config.tts.gpu_oom_free_ollama else []
                message = (
                    f"Segment {item.index}: GPU-Speicher voll (Versuch {attempt} von {attempts})"
                    + (f"; {hint}" if hint else "")
                    + f". Modell wird entladen, neuer Versuch in {wait} s."
                    + (
                        f" Ollama entladen: {', '.join(freed)}."
                        if freed
                        else " Wer jetzt Platz schafft, rettet den Lauf."
                    )
                )
                report_progress(
                    self._progress, ProgressEvent("log", "tts", message, level="warning")
                )
                print(f"[tts] {message}", file=sys.stderr)
                # Der gescheiterte Worker haelt seinen VRAM weiter fest. Ohne das
                # Entladen braeuchte der naechste Versuch genauso viel Speicher
                # und scheiterte genauso.
                self._drop_backend(item.voice)
                if wait:
                    time.sleep(wait)

    def _out_of_memory_advice(self, item: PreparedRenderChunk, attempts: int) -> str:
        """Die Meldung, die am Ende wirklich gelesen wird.

        Ein blosses "out of memory" laesst offen, was zu tun ist. Wer den
        Speicher haelt und wie es weitergeht, steht deshalb hier drin.
        """
        run_dir = item.wav_path.parent.parent
        lines = [
            f"Chatterbox hat nach {attempts} Versuchen keinen GPU-Speicher bekommen "
            f"(Segment {item.index}).",
        ]
        hint = gpu_memory_hint()
        if hint:
            lines.append(f"Belegung: {hint}.")
        lines.append(
            "Abhilfe: ein Ollama-Modell entladen (`ollama stop <modell>`), ein laufendes "
            "Spiel beenden, oder `tts.chatterbox.device: cpu` setzen (langsamer, aber unabhaengig)."
        )
        if self.config.tts.reuse_segments:
            lines.append(
                f"Fertige Segmente bleiben erhalten, fortsetzen mit: "
                f"codcast rerender {run_dir} --reuse-segments"
            )
        return " ".join(lines)

    def _render_chunk_atomically(self, item: PreparedRenderChunk) -> None:
        """Erst vollstaendig schreiben, dann umbenennen.

        Damit heisst "Datei existiert" auch wirklich "Segment ist fertig". Ohne
        das koennte ein abgebrochener Lauf eine halbe WAV hinterlassen, die beim
        naechsten Mal als fertig durchgeht. Die Endung bleibt `.wav`, weil
        torchaudio das Format aus ihr ableitet.
        """
        pending = item.wav_path.with_name(f"{item.wav_path.stem}.part.wav")
        fingerprint_path = self._fingerprint_path(item)
        fingerprint_path.unlink(missing_ok=True)
        self._backend(item.voice).render(item.chunk.text, item.voice, pending)
        fingerprint_path.write_text(
            json.dumps(self._segment_fingerprint(item), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(pending, item.wav_path)


def _is_out_of_memory(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "outofmemoryerror" in text


def _run_text(command: list[str], timeout: int = 15) -> str | None:
    """Fremdwerkzeug aufrufen und None liefern, wenn es fehlt oder klemmt."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _nvidia_smi(*args: str) -> str | None:
    return _run_text(["nvidia-smi", *args], timeout=5)


def free_ollama_models() -> list[str]:
    """Geladene Ollama-Modelle entladen und die Namen zurueckgeben.

    Nur auf ausdruecklichen Wunsch (`tts.gpu_oom_free_ollama`), denn hier
    greift der Renderer in einen fremden Dienst. Auf einem Rechner, auf dem
    Diktieren ein Modell in den VRAM legt und es dort minutenlang haelt, ist
    genau das aber der Unterschied zwischen fertigem Podcast und Abbruch:
    Nachladen kostet Sekunden, ein abgebrochener Lauf Minuten.
    """
    listing = _run_text(["ollama", "ps"])
    if listing is None:
        return []
    stopped: list[str] = []
    # Erste Zeile ist die Kopfzeile; ohne geladenes Modell bleibt nur sie uebrig.
    for line in listing.splitlines()[1:]:
        columns = line.split()
        if not columns:
            continue
        if _run_text(["ollama", "stop", columns[0]]) is not None:
            stopped.append(columns[0])
    return stopped


def gpu_memory_hint() -> str | None:
    """Wer den VRAM gerade belegt, in einer Zeile.

    Eine Wiederholung nach Speichermangel nuetzt nur, wenn jemand den Platz
    freigibt. Dafuer muss dastehen, welches Programm ihn haelt: ein
    Ollama-Modell laesst sich in Sekunden entladen, ein Browser nicht.
    Ohne nvidia-smi (CPU-Pfad, AMD) gibt es eben keinen Hinweis.
    """
    free = _nvidia_smi("--query-gpu=memory.free", "--format=csv,noheader,nounits")
    if free is None or not free.strip():
        return None
    holders: list[tuple[int, str]] = []
    apps = _nvidia_smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
    for line in (apps or "").splitlines():
        raw_pid, _, raw_used = line.partition(",")
        try:
            megabytes = int(raw_used.strip())
            name = Path(f"/proc/{int(raw_pid.strip())}/comm").read_text(encoding="utf-8").strip()
        except (ValueError, OSError):
            continue
        holders.append((megabytes, name))
    holders.sort(reverse=True)
    hint = f"frei sind {free.strip().splitlines()[0]} MiB"
    if not holders:
        return hint
    belegt = ", ".join(f"{name} {megabytes} MiB" for megabytes, name in holders[:3])
    return f"{hint}; groesste Belegung: {belegt}"


def rendered_segment_for_chunk(item: PreparedRenderChunk) -> RenderedSegment:
    return RenderedSegment(
        index=item.index,
        speaker_id=item.chunk.speaker_id,
        voice_profile_id=item.voice.id,
        text=item.chunk.text,
        wav_path=str(item.wav_path),
    )


def render_voice_sample(config: AppConfig, voice: VoiceProfile, text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend = backend_for_voice(config, voice)
    try:
        backend.render(text, voice, output_path)
    finally:
        # Sonst stapeln sich bei mehreren Samples in einem Prozess die Worker im VRAM.
        close_backend(backend)


def close_backend(backend: TTSBackend) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def group_single_speaker_openai_lines(lines: list[ScriptLine], max_chars: int) -> list[ScriptRenderChunk]:
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    chunks: list[ScriptRenderChunk] = []
    current_parts: list[str] = []
    current_indexes: list[int] = []
    current_speaker_id: str | None = None

    def flush() -> None:
        nonlocal current_parts, current_indexes, current_speaker_id
        if current_parts and current_speaker_id is not None:
            chunks.append(
                ScriptRenderChunk(
                    speaker_id=current_speaker_id,
                    text="\n\n".join(current_parts),
                    source_line_indexes=tuple(current_indexes),
                )
            )
        current_parts = []
        current_indexes = []
        current_speaker_id = None

    for index, line in enumerate(lines, start=1):
        for text_part in _split_text_for_openai_input(line.text, max_chars):
            speaker_changed = current_speaker_id is not None and current_speaker_id != line.speaker_id
            candidate_len = len(text_part) if not current_parts else len("\n\n".join([*current_parts, text_part]))
            if speaker_changed or (current_parts and candidate_len > max_chars):
                flush()
            current_speaker_id = line.speaker_id
            current_parts.append(text_part)
            current_indexes.append(index)
    flush()
    return chunks


def _split_text_for_openai_input(text: str, max_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return [normalized]

    parts: list[str] = []
    current = ""
    for word in normalized.split(" "):
        if len(word) > max_chars:
            raise ValueError(f"single word exceeds OpenAI TTS input limit: {word[:40]}")
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > max_chars:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def fish_payload_for_voice(text: str, voice: VoiceProfile, config: AppConfig) -> dict:
    fish = config.tts.fish
    references = []
    reference_id = voice.fish_reference_id
    if reference_id is None:
        if not voice.speaker_wav:
            raise ValueError(
                f"Fish voice {voice.id} needs speaker_wav or fish_reference_id. "
                "Import a high-quality reference WAV with `codcast voices import --backend fish ...`."
            )
        if not voice.speaker_wav.exists():
            raise FileNotFoundError(f"Fish speaker reference not found: {voice.speaker_wav}")
        if not voice.ref_text:
            raise ValueError(f"Fish voice {voice.id} needs ref_text matching the reference WAV")
        references.append({"audio": voice.speaker_wav.read_bytes(), "text": voice.ref_text})

    return {
        "text": text,
        "references": references,
        "reference_id": reference_id,
        "format": fish.output_format,
        "latency": fish.latency,
        "max_new_tokens": fish.max_new_tokens,
        "chunk_length": fish.chunk_length,
        "top_p": fish.top_p,
        "repetition_penalty": fish.repetition_penalty,
        "temperature": fish.temperature,
        "streaming": False,
        "use_memory_cache": fish.use_memory_cache,
        "seed": fish.seed,
        "normalize": True,
    }


def _api_key_from_env_or_file(env_name: str, env_file: Path) -> str | None:
    value = os.environ.get(env_name)
    if value:
        return value.strip()
    raw_key_candidate = None
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == env_name:
            return _strip_env_value(raw_value)
        if not separator and raw_key_candidate is None:
            raw_key_candidate = _strip_env_value(line)
    if raw_key_candidate:
        return raw_key_candidate
    return None


def _strip_env_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _load_kokoro_model(kmodel_cls, config_path: Path, model_path: Path):
    import torch

    model = kmodel_cls(repo_id="hexgrad/Kokoro-82M", config=str(config_path), model=str(model_path))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    for key, state_dict in state.items():
        module = getattr(model, key)
        converted = _convert_kokoro_state_dict(state_dict)
        result = module.load_state_dict(converted, strict=False)
        unexpected = list(result.unexpected_keys)
        critical_missing = [item for item in result.missing_keys if not _is_expected_missing_norm_key(item)]
        if unexpected or critical_missing:
            raise RuntimeError(
                f"Kokoro checkpoint did not load cleanly for {model_path}::{key}; "
                f"missing={critical_missing[:20]}, unexpected={unexpected[:20]}"
            )
    return model


def _convert_kokoro_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        key = key.replace(".parametrizations.weight.original0", ".weight_g")
        key = key.replace(".parametrizations.weight.original1", ".weight_v")
        converted[key] = value
    return converted


def _is_expected_missing_norm_key(key: str) -> bool:
    return ".norm.weight" in key or ".norm.bias" in key


@contextmanager
def _auto_accept_coqui_cpml():
    original_input = builtins.input

    def patched_input(prompt: str = "") -> str:
        if prompt.strip() == "| | >":
            return "y"
        return original_input(prompt)

    builtins.input = patched_input
    try:
        yield
    finally:
        builtins.input = original_input
