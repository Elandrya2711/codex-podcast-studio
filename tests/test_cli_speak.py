import argparse
import io
from pathlib import Path

import pytest

import codcast.cli as cli
from codcast.config import load_config


def _args(**overrides):
    values = {
        "text": "Hallo aus Codex.",
        "text_file": None,
        "stdin": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_speak_text_accepts_stdin(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("  Eine Audioantwort.  \n"))

    assert cli._speak_text(_args(text=None, stdin=True)) == "Eine Audioantwort."


def test_speak_text_rejects_multiple_sources(tmp_path: Path):
    source = tmp_path / "reply.txt"
    source.write_text("Datei", encoding="utf-8")

    with pytest.raises(SystemExit, match="nicht mehrere"):
        cli._speak_text(_args(text_file=source))


def test_default_speak_voice_uses_first_active_voice_set_member(tmp_path: Path):
    config = load_config(tmp_path / "missing.yml")
    config.tts.voice_set = "standard"
    config.tts.voice_sets = {"standard": ["openai-cedar", "openai-marin"]}

    assert cli._default_speak_voice(config).id == "openai-cedar"


def test_split_speak_text_keeps_sentence_order_and_limits_chunks():
    text = (
        "Der erste Satz erklaert den Anfang. Der zweite Satz liefert mehr Kontext. "
        "Der dritte Satz schliesst den Absatz ab.\n\nEin neuer Absatz bleibt als eigener Take erhalten."
    )

    chunks = cli._split_speak_text(text, max_chars=80)

    assert chunks == [
        "Der erste Satz erklaert den Anfang. Der zweite Satz liefert mehr Kontext.",
        "Der dritte Satz schliesst den Absatz ab.",
        "Ein neuer Absatz bleibt als eigener Take erhalten.",
    ]
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_stabilize_speak_voice_uses_conservative_chatterbox_settings(tmp_path: Path):
    config = load_config(tmp_path / "missing.yml")
    voice = next(item for item in config.tts.voices if item.id == "chatterbox-host-m")

    stable = cli._stabilize_speak_voice(config, voice)

    assert stable is not voice
    assert stable.chatterbox_temperature == 0.45
    assert voice.chatterbox_temperature is None
    assert config.tts.chatterbox.max_retries == 4
    assert config.tts.chatterbox.min_duration_ratio == 0.55
    assert config.tts.chatterbox.max_duration_ratio == 1.45


def test_speak_writes_requested_wav(monkeypatch, tmp_path: Path, capsys):
    config = load_config(tmp_path / "missing.yml")
    monkeypatch.setattr(cli, "load_config", lambda _path: config)

    rendered = []

    def fake_render(_config, voice, text, output):
        rendered.append((voice.id, text, output))
        output.write_bytes(b"wav")

    monkeypatch.setattr(cli, "render_voice_sample", fake_render)
    output = tmp_path / "reply.wav"

    exit_code = cli.main(["speak", "Hallo Welt.", "--output", str(output)])

    assert exit_code == 0
    assert output.read_bytes() == b"wav"
    assert rendered == [("chatterbox-host-m", "Hallo Welt.", output)]
    assert capsys.readouterr().out.strip() == str(output)


def test_speak_transcodes_default_mp3(monkeypatch, tmp_path: Path):
    config = load_config(tmp_path / "missing.yml")
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "_default_speak_output", lambda _format: tmp_path / "reply.mp3")
    monkeypatch.setattr(cli, "render_voice_sample", lambda _config, _voice, _text, output: output.write_bytes(b"wav"))

    def fake_transcode(source, output, output_format):
        assert source.read_bytes() == b"wav"
        assert output_format == "mp3"
        output.write_bytes(b"mp3")

    monkeypatch.setattr(cli, "_transcode_speak_output", fake_transcode)

    assert cli.main(["speak", "Hallo Welt."]) == 0
    assert (tmp_path / "reply.mp3").read_bytes() == b"mp3"
