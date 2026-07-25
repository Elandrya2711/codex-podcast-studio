import wave
from pathlib import Path

import pytest

import codcast.cli as cli
from codcast.config import load_config


def _reference(tmp_path: Path) -> Path:
    path = tmp_path / "ref.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 24000)
    return path


def test_chatterbox_import_needs_no_transcript(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "podcast.yml"
    config_path.write_text("language: de-DE\n", encoding="utf-8")

    exit_code = cli.main(
        [
            "voices",
            "import",
            "--config",
            str(config_path),
            "--backend",
            "chatterbox",
            "--name",
            "Host M",
            "--wav",
            str(_reference(tmp_path)),
        ]
    )

    assert exit_code == 0
    voice = next(v for v in load_config(config_path).tts.voices if v.id == "host-m")
    assert voice.backend == "chatterbox"
    assert voice.ref_text is None
    assert voice.speaker_wav == Path("voices/host-m/ref.wav")


def test_fish_import_still_requires_a_transcript(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "podcast.yml"
    config_path.write_text("language: de-DE\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="--transcript is required for the fish backend"):
        cli.main(
            [
                "voices",
                "import",
                "--config",
                str(config_path),
                "--backend",
                "fish",
                "--name",
                "fish-host",
                "--wav",
                str(_reference(tmp_path)),
            ]
        )


def test_import_defaults_to_the_local_backend(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "podcast.yml"
    config_path.write_text("language: de-DE\n", encoding="utf-8")

    cli.main(
        ["voices", "import", "--config", str(config_path), "--name", "standard", "--wav", str(_reference(tmp_path))]
    )

    voice = next(v for v in load_config(config_path).tts.voices if v.id == "standard")
    assert voice.backend == "chatterbox"
