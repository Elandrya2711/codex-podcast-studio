from pathlib import Path

from codcast.codex_runner import CodexRunner
from codcast.config import CodexConfig
from codcast.models import ResearchReport


def test_codex_command_shape(tmp_path: Path):
    runner = CodexRunner(CodexConfig(model="gpt-test"), tmp_path)
    cmd = runner.build_command("prompt", tmp_path / "schema.json", tmp_path / "out.json")
    assert cmd[:3] == ["codex", "--search", "--model"]
    assert cmd.index("--ask-for-approval") < cmd.index("exec")
    assert "exec" in cmd
    assert "--output-schema" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "-" == cmd[-1]
    assert "prompt" not in cmd


def test_codex_command_supports_config_overrides(tmp_path: Path):
    runner = CodexRunner(CodexConfig(model="gpt-test"), tmp_path)
    cmd = runner.build_command(
        "prompt",
        tmp_path / "schema.json",
        tmp_path / "out.json",
        config_overrides={"model_reasoning_effort": "xhigh"},
    )

    assert "-c" in cmd
    assert 'model_reasoning_effort="xhigh"' in cmd
    assert cmd.index("-c") < cmd.index("exec")


def test_codex_command_maps_reasoning_to_effort_override(tmp_path: Path):
    runner = CodexRunner(CodexConfig(model="gpt-test"), tmp_path)
    cmd = runner.build_command(
        "prompt",
        tmp_path / "schema.json",
        tmp_path / "out.json",
        reasoning="xhigh",
    )

    assert 'model_reasoning_effort="xhigh"' in cmd
    assert cmd.index("-c") < cmd.index("exec")


def test_codex_command_uses_configured_effort_and_prefers_per_call_reasoning(tmp_path: Path):
    runner = CodexRunner(CodexConfig(model="gpt-test", effort="low"), tmp_path)

    default_cmd = runner.build_command("prompt", tmp_path / "schema.json", tmp_path / "out.json")
    assert 'model_reasoning_effort="low"' in default_cmd

    override_cmd = runner.build_command(
        "prompt",
        tmp_path / "schema.json",
        tmp_path / "out.json",
        reasoning="max",
    )
    assert 'model_reasoning_effort="max"' in override_cmd
    assert 'model_reasoning_effort="low"' not in override_cmd


def test_codex_command_can_disable_live_search_per_call(tmp_path: Path):
    runner = CodexRunner(CodexConfig(model="gpt-test", live_search=True), tmp_path)
    cmd = runner.build_command("prompt", tmp_path / "schema.json", tmp_path / "out.json", live_search=False)

    assert "--search" not in cmd


def test_codex_runner_streams_logs_to_progress(tmp_path: Path):
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = '-o' ]; then shift; out=\"$1\"; fi\n"
        "  shift\n"
        "done\n"
        "cat > \"$out.prompt\"\n"
        "echo stdout-line\n"
        "echo stderr-line >&2\n"
        "printf '%s' '{\"topic\":\"T\",\"language\":\"de-DE\",\"summary\":\"S\",\"sources\":[],\"claims\":[]}' > \"$out\"\n",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    events = []
    runner = CodexRunner(CodexConfig(executable=str(fake_codex), live_search=False), tmp_path)

    report = runner.run_structured(
        prompt="prompt",
        schema_name="research",
        output_path=tmp_path / "research.json",
        model=ResearchReport,
        progress=events.append,
    )

    assert report.summary == "S"
    assert (tmp_path / "research.json.prompt").read_text(encoding="utf-8") == "prompt"
    assert any("stdout-line" in event.message for event in events)
    assert any("stderr-line" in event.message for event in events)
