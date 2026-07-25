import pytest

from codcast.config import load_config
from codcast.voices import build_speaker_specs, select_voice_profiles


def test_select_default_chatterbox_voices(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, None)
    assert [voice.id for voice in voices] == ["chatterbox-host-m", "chatterbox-host-f"]
    speakers = build_speaker_specs(voices)
    assert speakers[0].id == "s1"
    assert speakers[0].voice_profile_id == "chatterbox-host-m"
    assert speakers[1].id == "s2"
    assert speakers[1].voice_profile_id == "chatterbox-host-f"


def test_select_fast_kokoro_voices(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, "fast")
    assert [voice.id for voice in voices] == ["martin", "victoria"]


def test_best_uses_configured_backend(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voices = select_voice_profiles(config, 2, "best")
    assert [voice.id for voice in voices] == ["chatterbox-host-m", "chatterbox-host-f"]

    config.tts.backend = "openai"
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
    assert [voice.id for voice in voices] == ["chatterbox-host-m", "chatterbox-host-f", "chatterbox-host-m"]


def test_voice_set_selects_the_named_cast(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    config.tts.voice_sets = {
        "standard": ["chatterbox-host-m", "chatterbox-host-f"],
        "gaeste": ["chatterbox-host-f", "chatterbox-host-m"],
    }

    voices = select_voice_profiles(config, 2, "chatterbox", voice_set="gaeste")
    assert [voice.id for voice in voices] == ["chatterbox-host-f", "chatterbox-host-m"]


def test_configured_voice_set_applies_without_an_argument(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    config.tts.voice_sets = {"gaeste": ["chatterbox-host-f"]}
    config.tts.voice_set = "gaeste"
    config.tts.allow_voice_reuse = True

    voices = select_voice_profiles(config, 2, "chatterbox")
    assert [voice.id for voice in voices] == ["chatterbox-host-f", "chatterbox-host-f"]


def test_voice_set_too_small_without_reuse_is_an_error(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    config.tts.voice_sets = {"solo": ["chatterbox-host-m"]}

    with pytest.raises(ValueError, match="only has 1"):
        select_voice_profiles(config, 2, "chatterbox", voice_set="solo")


def test_voice_set_with_wrong_backend_is_rejected(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    config.tts.voice_sets = {"gemischt": ["chatterbox-host-m", "openai-cedar"]}

    with pytest.raises(ValueError, match="another backend"):
        select_voice_profiles(config, 2, "chatterbox", voice_set="gemischt")


def test_unknown_voice_set_is_an_error(tmp_path):
    config = load_config(tmp_path / "missing.yml")

    with pytest.raises(ValueError, match="unknown voice set"):
        select_voice_profiles(config, 2, "chatterbox", voice_set="gibtsnicht")


def test_config_rejects_voice_set_with_unknown_member(tmp_path):
    from codcast.config import AppConfig

    with pytest.raises(ValueError, match="unknown voice profiles"):
        AppConfig.model_validate(
            {
                "tts": {
                    "voices": [{"id": "a", "display_name": "A", "backend": "chatterbox"}],
                    "voice_sets": {"set": ["a", "fehlt"]},
                }
            }
        )


def test_config_rejects_selected_set_that_is_not_defined():
    from codcast.config import AppConfig

    with pytest.raises(ValueError, match="not defined in voice_sets"):
        AppConfig.model_validate({"tts": {"voice_set": "gibtsnicht"}})
