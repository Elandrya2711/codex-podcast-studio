"""Wiederverwendung fertiger Segmente und Verhalten bei GPU-Speichermangel.

Beides betrifft denselben Fall: ein langer Lauf auf einer geteilten Grafikkarte
bricht mitten im Rendern ab. Vorher kostete das alle bereits fertigen Segmente.
"""

from pathlib import Path

import pytest

from codcast import tts
from codcast.config import AppConfig, VoiceProfile
from codcast.models import PodcastScript
from codcast.tts import ScriptRenderer

OOM_MESSAGE = (
    "Chatterbox konnte Segment nicht rendern: OutOfMemoryError: CUDA out of memory. "
    "Tried to allocate 20.00 MiB."
)

# Vor dem Stummschalten festhalten: die Tests der Auskunft selbst brauchen das Echte.
REAL_GPU_HINT = tts.gpu_memory_hint


@pytest.fixture(autouse=True)
def no_real_gpu_queries(monkeypatch):
    """Kein Test befragt die echte Grafikkarte.

    Sonst haengt das Ergebnis daran, was auf dem Rechner gerade laeuft, und
    jeder Wiederholversuch startet einen nvidia-smi-Prozess.
    """
    monkeypatch.setattr(tts, "gpu_memory_hint", lambda: None)


class _CountingBackend(tts.TTSBackend):
    """Protokolliert jeden Renderaufruf und kann Fehler nach Plan werfen."""

    instances: list["_CountingBackend"] = []
    failures: list[Exception | None] = []

    def __init__(self, _config: AppConfig) -> None:
        self.rendered: list[str] = []
        self.closed = False
        _CountingBackend.instances.append(self)

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        self.rendered.append(text)
        if _CountingBackend.failures:
            error = _CountingBackend.failures.pop(0)
            if error is not None:
                raise error
        output_path.write_bytes(b"RIFF")

    def close(self) -> None:
        self.closed = True

    @classmethod
    def render_calls(cls) -> list[str]:
        return [text for backend in cls.instances for text in backend.rendered]


@pytest.fixture
def backend(monkeypatch):
    _CountingBackend.instances = []
    _CountingBackend.failures = []
    monkeypatch.setattr(tts, "backend_for_voice", lambda config, voice: _CountingBackend(config))
    return _CountingBackend


@pytest.fixture
def no_waiting(monkeypatch):
    """Wartezeiten mitschreiben statt sie abzusitzen."""
    waited: list[float] = []
    monkeypatch.setattr(tts.time, "sleep", waited.append)
    return waited


def _voice(**overrides) -> VoiceProfile:
    fields = {
        "id": "v",
        "display_name": "V",
        "backend": "chatterbox",
        "speaker_wav": Path("ref.wav"),
    }
    fields.update(overrides)
    return VoiceProfile(**fields)


def _script(*texts: str) -> PodcastScript:
    return PodcastScript.model_validate(
        {
            "title": "T",
            "topic": "T",
            "language": "de-DE",
            "target_min_minutes": 1,
            "target_max_minutes": 2,
            "speakers": [
                {"id": "s1", "display_name": "V", "role": "Host", "voice_profile_id": "v"}
            ],
            "lines": [{"speaker_id": "s1", "text": text} for text in texts],
            "estimated_words": len(texts),
            "production_notes": [],
        }
    )


def _render(config: AppConfig, script: PodcastScript, run_dir: Path, voice: VoiceProfile):
    return ScriptRenderer(config, {"v": voice}).render_script(script, run_dir)


def test_finished_segments_are_reused_and_the_model_stays_unloaded(backend, tmp_path: Path):
    script = _script("Erste Zeile.", "Zweite Zeile.")

    _render(AppConfig(), script, tmp_path, _voice())
    assert backend.render_calls() == ["Erste Zeile.", "Zweite Zeile."]

    segments = _render(AppConfig(), script, tmp_path, _voice())

    assert len(segments) == 2
    assert backend.render_calls() == ["Erste Zeile.", "Zweite Zeile."], "kein zweites Rendern"
    # Wenn alles uebernommen wird, darf das Modell gar nicht erst geladen werden.
    assert len(backend.instances) == 1


def test_only_the_changed_line_is_rendered_again(backend, tmp_path: Path):
    _render(AppConfig(), _script("Erste Zeile.", "Zweite Zeile."), tmp_path, _voice())

    _render(AppConfig(), _script("Erste Zeile.", "Zweite Zeile, neu."), tmp_path, _voice())

    assert backend.render_calls() == ["Erste Zeile.", "Zweite Zeile.", "Zweite Zeile, neu."]


def test_a_changed_voice_parameter_forces_a_new_take(backend, tmp_path: Path):
    script = _script("Erste Zeile.")
    _render(AppConfig(), script, tmp_path, _voice())

    # Dasselbe Wort, andere Stimme: das alte Audio waere schlicht falsch.
    _render(AppConfig(), script, tmp_path, _voice(chatterbox_temperature=0.45))

    assert backend.render_calls() == ["Erste Zeile.", "Erste Zeile."]


def test_a_changed_backend_setting_forces_a_new_take(backend, tmp_path: Path):
    script = _script("Erste Zeile.")
    _render(AppConfig(), script, tmp_path, _voice())

    config = AppConfig()
    config.tts.chatterbox.exaggeration = 0.7
    _render(config, script, tmp_path, _voice())

    assert backend.render_calls() == ["Erste Zeile.", "Erste Zeile."]


def test_reuse_can_be_switched_off(backend, tmp_path: Path):
    script = _script("Erste Zeile.")
    _render(AppConfig(), script, tmp_path, _voice())

    config = AppConfig()
    config.tts.reuse_segments = False
    _render(config, script, tmp_path, _voice())

    assert backend.render_calls() == ["Erste Zeile.", "Erste Zeile."]


def test_a_segment_without_fingerprint_is_rendered_again(backend, tmp_path: Path):
    script = _script("Erste Zeile.")
    _render(AppConfig(), script, tmp_path, _voice())

    # So sieht ein Lauf aus, der vor dieser Aenderung abgebrochen ist: Audio da,
    # aber kein Nachweis, wozu es gehoert.
    for fingerprint in (tmp_path / "segments").glob("*.fingerprint.json"):
        fingerprint.unlink()

    _render(AppConfig(), script, tmp_path, _voice())

    assert backend.render_calls() == ["Erste Zeile.", "Erste Zeile."]


def test_a_fingerprint_without_audio_is_rendered_again(backend, tmp_path: Path):
    script = _script("Erste Zeile.")
    _render(AppConfig(), script, tmp_path, _voice())

    for wav in (tmp_path / "segments").glob("*.wav"):
        wav.unlink()

    _render(AppConfig(), script, tmp_path, _voice())

    assert backend.render_calls() == ["Erste Zeile.", "Erste Zeile."]


def test_out_of_memory_is_retried_after_unloading_the_model(backend, no_waiting, tmp_path: Path):
    backend.failures = [RuntimeError(OOM_MESSAGE)]

    segments = _render(AppConfig(), _script("Erste Zeile."), tmp_path, _voice())

    assert len(segments) == 1
    assert backend.render_calls() == ["Erste Zeile.", "Erste Zeile."]
    # Der erste Worker haelt seinen VRAM fest, bis er geschlossen wird. Ohne das
    # haette der zweite Versuch genauso wenig Speicher wie der erste.
    assert len(backend.instances) == 2
    assert backend.instances[0].closed
    assert no_waiting == [AppConfig().tts.gpu_oom_wait_sec]


def test_out_of_memory_gives_up_after_the_configured_retries(backend, no_waiting, tmp_path: Path):
    backend.failures = [RuntimeError(OOM_MESSAGE)] * 5
    config = AppConfig()
    config.tts.gpu_oom_retries = 2

    with pytest.raises(RuntimeError, match="keinen GPU-Speicher bekommen") as failure:
        _render(config, _script("Erste Zeile."), tmp_path, _voice())

    assert len(backend.render_calls()) == 3, "ein Versuch plus zwei Wiederholungen"
    assert len(no_waiting) == 2
    assert "out of memory" in str(failure.value.__cause__).lower()


def test_other_failures_are_not_retried(backend, no_waiting, tmp_path: Path):
    backend.failures = [RuntimeError("worker died")]

    with pytest.raises(RuntimeError, match="worker died"):
        _render(AppConfig(), _script("Erste Zeile."), tmp_path, _voice())

    assert len(backend.render_calls()) == 1
    assert no_waiting == []


def test_a_failed_segment_leaves_no_usable_audio_behind(backend, no_waiting, tmp_path: Path):
    backend.failures = [RuntimeError("worker died")]

    with pytest.raises(RuntimeError):
        _render(AppConfig(), _script("Erste Zeile."), tmp_path, _voice())

    # Halbfertiges darf nicht unter dem endgueltigen Namen liegen, sonst gilt es
    # beim naechsten Lauf als fertig.
    assert list((tmp_path / "segments").glob("0001_*.wav")) == []
    assert list((tmp_path / "segments").glob("*.fingerprint.json")) == []


def test_audio_written_before_a_failure_is_not_mistaken_for_finished(
    monkeypatch, backend, no_waiting, tmp_path: Path
):
    """Der gefaehrliche Fall: die Datei ist schon da, dann scheitert der Rest.

    Etwa wenn die Tempoanpassung nach der Synthese abbricht. Genau hier trennt
    das Umbenennen fertig von halbfertig.
    """

    def write_then_fail(self, text, voice, output_path: Path) -> None:
        self.rendered.append(text)
        output_path.write_bytes(b"RIFF-halb")
        raise RuntimeError("ffmpeg died")

    healthy_render = _CountingBackend.render
    monkeypatch.setattr(_CountingBackend, "render", write_then_fail)
    script = _script("Erste Zeile.")

    with pytest.raises(RuntimeError, match="ffmpeg died"):
        _render(AppConfig(), script, tmp_path, _voice())

    segment_dir = tmp_path / "segments"
    assert list(segment_dir.glob("0001_s1_v.wav")) == [], "kein fertiger Name fuer halbes Audio"
    assert list(segment_dir.glob("*.fingerprint.json")) == []

    # Und der naechste Lauf rendert es wirklich neu, statt den Rest zu uebernehmen.
    monkeypatch.setattr(_CountingBackend, "render", healthy_render)
    _render(AppConfig(), script, tmp_path, _voice())
    assert (segment_dir / "0001_s1_v.wav").read_bytes() == b"RIFF"


def test_gpu_hint_names_the_biggest_holders(monkeypatch):
    def fake_run(command, **kwargs):
        if "--query-gpu=memory.free" in command:
            return type("R", (), {"returncode": 0, "stdout": "2988\n"})()
        return type("R", (), {"returncode": 0, "stdout": "143208, 2154\n547370, 4162\n"})()

    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    monkeypatch.setattr(
        tts.Path, "read_text", lambda self, encoding=None: {"143208": "python", "547370": "llama-server"}[self.parts[2]]
    )

    hint = REAL_GPU_HINT()

    assert hint is not None
    assert "2988 MiB" in hint
    # Der groesste Halter zuerst, denn den entlaedt man als erstes.
    assert hint.index("llama-server") < hint.index("python")


def test_gpu_hint_is_absent_without_nvidia_smi(monkeypatch):
    def missing(command, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(tts.subprocess, "run", missing)

    assert REAL_GPU_HINT() is None


def test_ollama_models_are_unloaded_by_name(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["ollama", "ps"]:
            stdout = (
                "NAME         ID              SIZE      PROCESSOR    UNTIL\n"
                "gemma3:4b    a2af6cc3eb7f    3.0 GB    100% GPU     28 minutes from now\n"
                "qwen3:8b     b1cf7dd4fa8e    5.2 GB    100% GPU     4 minutes from now\n"
            )
            return type("R", (), {"returncode": 0, "stdout": stdout})()
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(tts.subprocess, "run", fake_run)

    assert tts.free_ollama_models() == ["gemma3:4b", "qwen3:8b"]
    assert ["ollama", "stop", "gemma3:4b"] in calls
    assert ["ollama", "stop", "qwen3:8b"] in calls


def test_no_loaded_model_means_nothing_to_unload(monkeypatch):
    header_only = "NAME    ID    SIZE    PROCESSOR    UNTIL\n"
    monkeypatch.setattr(
        tts.subprocess,
        "run",
        lambda command, **kwargs: type("R", (), {"returncode": 0, "stdout": header_only})(),
    )

    assert tts.free_ollama_models() == []


def test_out_of_memory_unloads_ollama_when_allowed(backend, no_waiting, monkeypatch, tmp_path: Path):
    freed = []
    monkeypatch.setattr(tts, "free_ollama_models", lambda: freed.append("call") or ["gemma3:4b"])
    backend.failures = [RuntimeError(OOM_MESSAGE)]
    config = AppConfig()
    config.tts.gpu_oom_free_ollama = True

    _render(config, _script("Erste Zeile."), tmp_path, _voice())

    assert freed == ["call"], "genau einmal entladen, nicht pro Segment"


def test_ollama_is_left_alone_by_default(backend, no_waiting, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tts, "free_ollama_models", lambda: pytest.fail("fremder Dienst ungefragt angefasst"))
    backend.failures = [RuntimeError(OOM_MESSAGE)]

    assert AppConfig().tts.gpu_oom_free_ollama is False
    _render(AppConfig(), _script("Erste Zeile."), tmp_path, _voice())


def test_the_final_error_says_what_to_do(backend, no_waiting, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tts, "gpu_memory_hint", lambda: "frei sind 28 MiB; groesste Belegung: llama-server 4162 MiB")
    backend.failures = [RuntimeError(OOM_MESSAGE)] * 9
    config = AppConfig()
    config.tts.gpu_oom_retries = 1

    with pytest.raises(RuntimeError) as failure:
        _render(config, _script("Erste Zeile."), tmp_path, _voice())

    message = str(failure.value)
    assert "llama-server 4162 MiB" in message, "wer den Speicher haelt"
    assert "ollama stop" in message, "was zu tun ist"
    assert "device: cpu" in message, "der unabhaengige Ausweg"
    assert f"codcast rerender {tmp_path} --reuse-segments" in message, "wie es weitergeht"
    # Die eigentliche Ursache darf nicht verloren gehen.
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert "out of memory" in str(failure.value.__cause__).lower()


@pytest.mark.parametrize(
    ("roh", "erwartet"),
    [
        # Gemessen an Chatterbox: zwei Text-Tokens stuerzen ab, drei laufen.
        ("B.", "B. .."),
        ("Ja?", "Ja? .."),
        ("Hmm.", "Hmm. .."),
        # Ab der Schwelle bleibt der Text unangetastet.
        ("Stock.", "Stock."),
        ("Sag es.", "Sag es."),
        ("Ein ganz normaler Satz.", "Ein ganz normaler Satz."),
        # Leer bleibt leer, hier ist nichts zu retten.
        ("", ""),
        ("   ", "   "),
    ],
    ids=["zwei-zeichen", "drei-zeichen", "vier-zeichen", "genau-sechs", "sieben", "normal", "leer", "nur-leerraum"],
)
def test_very_short_lines_are_padded_with_unspoken_characters(roh, erwartet):
    assert tts._pad_short_text(roh, 6) == erwartet


def test_padding_threshold_is_configurable():
    assert tts._pad_short_text("Hmm.", 0) == "Hmm."
    assert tts._pad_short_text("Ein Satz.", 40) == "Ein Satz. .."


def test_any_segment_failure_says_how_to_continue(backend, no_waiting, tmp_path: Path):
    # Der echte Fall: Chatterbox stirbt an einer zu kurzen Zeile, nicht am Speicher.
    backend.failures = [RuntimeError("IndexError: max(): Expected reduction dim 1 to have non-zero size.")]

    with pytest.raises(RuntimeError) as failure:
        _render(AppConfig(), _script("Erste Zeile."), tmp_path, _voice())

    message = str(failure.value)
    assert "IndexError" in message, "die Ursache bleibt lesbar"
    assert f"codcast rerender {tmp_path} --reuse-segments" in message
    assert len(backend.render_calls()) == 1, "kein Wiederholen, das war kein Speicherfehler"


def test_the_resume_hint_is_omitted_when_nothing_is_kept(backend, no_waiting, tmp_path: Path):
    config = AppConfig()
    config.tts.reuse_segments = False
    backend.failures = [RuntimeError("worker died")]

    with pytest.raises(RuntimeError) as failure:
        _render(config, _script("Erste Zeile."), tmp_path, _voice())

    # Ohne Wiederverwendung waere der Rat falsch: der naechste Lauf faengt vorn an.
    assert "rerender" not in str(failure.value)
