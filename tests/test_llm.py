import argparse
from pathlib import Path

import pytest

from codcast import cli
from codcast.claude_runner import ClaudeRunner
from codcast.codex_runner import CodexRunner
from codcast.config import load_config
from codcast.llm import runner_for_config


def _config(tmp_path: Path):
    return load_config(tmp_path / "missing.yml")


def _fake_executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_claude_is_the_default_provider(tmp_path: Path):
    config = _config(tmp_path)
    assert config.llm.provider == "claude"
    assert config.llm.claude.model == "claude-opus-5"
    assert config.llm.claude.deep_effort == "xhigh"
    assert config.llm_deep_reasoning == "xhigh"
    assert config.llm_timeout_sec == config.llm.claude.timeout_sec


def test_legacy_config_without_llm_block_still_loads(tmp_path: Path):
    path = tmp_path / "podcast.yml"
    path.write_text(
        "codex:\n  model: gpt-legacy\n  timeout_sec: 999\n",
        encoding="utf-8",
    )
    config = load_config(path)

    assert config.codex.model == "gpt-legacy"
    # Existing files gain the new default provider without a migration step.
    assert config.llm.provider == "claude"


def test_runner_for_config_dispatches_on_provider(tmp_path: Path):
    config = _config(tmp_path)
    config.llm.claude.executable = str(_fake_executable(tmp_path, "claude"))
    assert isinstance(runner_for_config(config, tmp_path), ClaudeRunner)

    config.llm.provider = "codex"
    config.codex.executable = str(_fake_executable(tmp_path, "codex"))
    assert isinstance(runner_for_config(config, tmp_path), CodexRunner)


def test_codex_provider_reports_its_own_timeout_and_effort(tmp_path: Path):
    config = _config(tmp_path)
    config.llm.provider = "codex"
    config.codex.timeout_sec = 777

    assert config.llm_timeout_sec == 777
    assert config.llm_deep_reasoning == "xhigh"


def test_missing_executable_names_the_provider_and_the_escape_hatch(tmp_path: Path):
    config = _config(tmp_path)
    config.llm.claude.executable = str(tmp_path / "does-not-exist")

    with pytest.raises(RuntimeError) as excinfo:
        runner_for_config(config, tmp_path)

    message = str(excinfo.value)
    assert "claude" in message
    assert "--llm-provider codex" in message


def test_unsupported_provider_raises(tmp_path: Path):
    config = _config(tmp_path)
    config.llm.provider = "gemini"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="unsupported LLM provider"):
        runner_for_config(config, tmp_path)


def _args(**kwargs) -> argparse.Namespace:
    defaults = {
        "llm_provider": None,
        "model": None,
        "effort": None,
        "codex_model": None,
        "cached_search": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_apply_llm_args_routes_model_to_the_active_provider(tmp_path: Path):
    config = _config(tmp_path)
    cli.apply_llm_args(config, _args(model="claude-opus-4-8"))
    assert config.llm.claude.model == "claude-opus-4-8"
    assert config.codex.model is None

    other = _config(tmp_path)
    cli.apply_llm_args(other, _args(llm_provider="codex", model="gpt-x"))
    assert other.llm.provider == "codex"
    assert other.codex.model == "gpt-x"


def test_apply_llm_args_keeps_legacy_codex_model_flag(tmp_path: Path):
    config = _config(tmp_path)
    cli.apply_llm_args(config, _args(codex_model="gpt-legacy"))

    assert config.codex.model == "gpt-legacy"
    assert config.llm.claude.model == "claude-opus-5"


def test_apply_llm_args_cached_search_disables_both_providers(tmp_path: Path):
    config = _config(tmp_path)
    cli.apply_llm_args(config, _args(cached_search=True))

    assert config.llm.claude.live_search is False
    assert config.codex.live_search is False


def test_apply_llm_args_sets_effort_for_both_providers(tmp_path: Path):
    config = _config(tmp_path)
    cli.apply_llm_args(config, _args(effort="medium"))

    assert config.llm.claude.effort == "medium"
    assert config.codex.effort == "medium"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("fable", "claude-fable-5"),
        ("FABLE", "claude-fable-5"),
        ("opus", "claude-opus-5"),
        ("sonnet", "claude-sonnet-5"),
        # Anything unknown passes through so future or pinned models keep working.
        ("claude-opus-4-8", "claude-opus-4-8"),
    ],
)
def test_claude_model_aliases_resolve_to_pinned_ids(given: str, expected: str):
    assert cli.resolve_claude_model(given) == expected


def test_apply_llm_args_expands_claude_model_alias(tmp_path: Path):
    config = _config(tmp_path)
    cli.apply_llm_args(config, _args(model="fable"))

    assert config.llm.claude.model == "claude-fable-5"


def test_codex_provider_does_not_expand_claude_aliases(tmp_path: Path):
    config = _config(tmp_path)
    cli.apply_llm_args(config, _args(llm_provider="codex", model="fable"))

    assert config.codex.model == "fable"


def test_claude_model_choices_keeps_a_hand_configured_model_selectable():
    choices, default = cli.claude_model_choices("claude-opus-5")
    assert default == "opus"
    assert "fable" in choices

    choices, default = cli.claude_model_choices("claude-opus-4-8")
    assert default == "claude-opus-4-8"
    assert "claude-opus-4-8" in choices
    assert "fable" in choices
