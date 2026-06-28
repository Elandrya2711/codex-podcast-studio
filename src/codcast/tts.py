from __future__ import annotations

import builtins
import os
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

    def _backend(self, voice: VoiceProfile) -> TTSBackend:
        if voice.backend not in self.backends:
            self.backends[voice.backend] = backend_for_voice(self.config, voice)
        return self.backends[voice.backend]

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

        if self._should_render_openai_chunks_parallel(prepared):
            return self._render_prepared_chunks_parallel(prepared, total, progress, cancellation)
        return self._render_prepared_chunks_serial(prepared, total, progress, cancellation)

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
        self._backend(item.voice).render(item.chunk.text, item.voice, item.wav_path)


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
    backend_for_voice(config, voice).render(text, voice, output_path)


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
