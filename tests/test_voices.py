import pytest

from codcast.config import load_config
from codcast.voices import build_speaker_specs, select_voice_profiles


def test_select_default_openai_voices(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, None)
    assert [voice.id for voice in voices] == ["openai-cedar", "openai-marin"]
    speakers = build_speaker_specs(voices)
    assert speakers[0].id == "s1"
    assert speakers[0].voice_profile_id == "openai-cedar"
    assert speakers[1].id == "s2"
    assert speakers[1].voice_profile_id == "openai-marin"


def test_select_fast_kokoro_voices(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, "fast")
    assert [voice.id for voice in voices] == ["martin", "victoria"]


def test_best_uses_configured_backend(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, "best")
    assert [voice.id for voice in voices] == ["openai-cedar", "openai-marin"]


def test_openai_quality_selects_openai_voices(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, "openai")
    assert [voice.id for voice in voices] == ["openai-cedar", "openai-marin"]


def test_rejects_too_many_kokoro_speakers_without_reuse(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    with pytest.raises(ValueError, match="only 2 configured"):
        select_voice_profiles(config, 3, "fast")


def test_allows_voice_reuse_when_enabled(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    config.tts.allow_voice_reuse = True
    voices = select_voice_profiles(config, 3, "best")
    assert [voice.id for voice in voices] == ["openai-cedar", "openai-marin", "openai-cedar"]
