from pathlib import Path

from codcast import tts
from codcast.config import AppConfig, VoiceProfile
from codcast.models import PodcastScript
from codcast.tts import ScriptRenderer, render_voice_sample


class _RecordingBackend(tts.TTSBackend):
    """Zaehlt Renderaufrufe und ob der Worker geschlossen wurde."""

    instances: list["_RecordingBackend"] = []

    def __init__(self, _config: AppConfig) -> None:
        self.rendered: list[str] = []
        self.closed = False
        _RecordingBackend.instances.append(self)

    def render(self, text: str, voice: VoiceProfile, output_path: Path) -> None:
        self.rendered.append(text)
        output_path.write_bytes(b"RIFF")

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch) -> None:
    _RecordingBackend.instances = []
    monkeypatch.setattr(tts, "backend_for_voice", lambda config, voice: _RecordingBackend(config))


def _voice() -> VoiceProfile:
    return VoiceProfile(id="v", display_name="V", backend="chatterbox", speaker_wav=Path("ref.wav"))


def test_voice_sample_closes_the_backend(monkeypatch, tmp_path: Path):
    _install(monkeypatch)

    render_voice_sample(AppConfig(), _voice(), "Hallo", tmp_path / "a.wav")
    render_voice_sample(AppConfig(), _voice(), "Hallo", tmp_path / "b.wav")

    assert len(_RecordingBackend.instances) == 2
    assert all(backend.closed for backend in _RecordingBackend.instances)


def test_voice_sample_closes_the_backend_even_on_failure(monkeypatch, tmp_path: Path):
    _install(monkeypatch)

    def boom(self, text, voice, output_path):
        raise RuntimeError("worker died")

    monkeypatch.setattr(_RecordingBackend, "render", boom)
    try:
        render_voice_sample(AppConfig(), _voice(), "Hallo", tmp_path / "a.wav")
    except RuntimeError:
        pass

    assert _RecordingBackend.instances[0].closed


def test_script_renderer_releases_backends_after_rendering(monkeypatch, tmp_path: Path):
    _install(monkeypatch)
    voice = _voice()
    script = PodcastScript.model_validate(
        {
            "title": "T",
            "topic": "T",
            "language": "de-DE",
            "target_min_minutes": 1,
            "target_max_minutes": 2,
            "speakers": [{"id": "s1", "display_name": "V", "role": "Host", "voice_profile_id": "v"}],
            "lines": [{"speaker_id": "s1", "text": "Erste Zeile."}, {"speaker_id": "s1", "text": "Zweite Zeile."}],
            "estimated_words": 4,
            "production_notes": [],
        }
    )

    renderer = ScriptRenderer(AppConfig(), {"v": voice})
    segments = renderer.render_script(script, tmp_path)

    assert len(segments) == 2
    assert len(_RecordingBackend.instances) == 1, "ein Worker fuer alle Segmente"
    assert _RecordingBackend.instances[0].rendered == ["Erste Zeile.", "Zweite Zeile."]
    assert _RecordingBackend.instances[0].closed
    assert renderer.backends == {}
