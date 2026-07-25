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

    with pytest.raises(RuntimeError, match="out of memory"):
        _render(config, _script("Erste Zeile."), tmp_path, _voice())

    assert len(backend.render_calls()) == 3, "ein Versuch plus zwei Wiederholungen"
    assert len(no_waiting) == 2


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
