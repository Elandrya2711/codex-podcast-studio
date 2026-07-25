from pathlib import Path

from codcast.config import AppConfig, load_config


def test_load_default_config_from_missing_file(tmp_path: Path):
    config = load_config(tmp_path / "missing.yml")
    assert isinstance(config, AppConfig)
    assert config.language == "de-DE"
    assert config.output_root == Path("podcasts")
    assert config.research.depth == "standard"
    assert config.research.provider == "searxng"
    assert config.research.searxng_base_url == "http://127.0.0.1:8888"
    assert config.tts.backend == "chatterbox"
    assert config.tts.quality == "best"
    assert [voice.backend for voice in config.tts.voices[:2]] == ["chatterbox", "chatterbox"]


def test_load_config_merges_overrides(tmp_path: Path):
    path = tmp_path / "podcast.yml"
    path.write_text(
        "generation:\n  words_per_minute: 130\nresearch:\n  depth: deep\n  max_documents: 12\ntts:\n  allow_voice_reuse: true\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.generation.words_per_minute == 130
    assert config.research.depth == "deep"
    assert config.research.max_documents == 12
    assert config.tts.allow_voice_reuse is True
    assert config.tts.voices


def test_legacy_tavily_research_config_is_migrated(tmp_path: Path):
    path = tmp_path / "podcast.yml"
    path.write_text(
        "research:\n  provider: tavily\n  api_key_env: TAVILY_API_KEY\n  depth: dossier\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.research.provider == "searxng"
    assert config.research.depth == "dossier"
