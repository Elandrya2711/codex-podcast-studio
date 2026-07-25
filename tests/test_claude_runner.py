import json
from pathlib import Path

import pytest

from codcast.claude_runner import ClaudeRunner
from codcast.config import ClaudeConfig
from codcast.models import ResearchReport
from codcast.schemas import schema_for


RESEARCH_PAYLOAD = {
    "topic": "T",
    "language": "de-DE",
    "summary": "S",
    "sources": [],
    "claims": [],
}


def _schema_arg(cmd: list[str]) -> dict:
    return json.loads(cmd[cmd.index("--json-schema") + 1])


def test_claude_command_shape(tmp_path: Path):
    runner = ClaudeRunner(ClaudeConfig(), tmp_path)
    cmd = runner.build_command(schema=schema_for("research"))

    assert cmd[:4] == ["claude", "-p", "--model", "claude-opus-5"]
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in cmd
    assert "--no-session-persistence" in cmd
    # The prompt must travel on stdin, never in argv.
    assert "prompt" not in cmd


def test_claude_command_inlines_full_json_schema(tmp_path: Path):
    runner = ClaudeRunner(ClaudeConfig(), tmp_path)
    cmd = runner.build_command(schema=schema_for("research"))
    schema = _schema_arg(cmd)

    # The CLI only accepts inline JSON, and the project's schemas rely on $defs/$ref.
    assert "$defs" in schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def _effort_of(runner: ClaudeRunner, **kwargs) -> str:
    cmd = runner.build_command(schema=schema_for("research"), **kwargs)
    return cmd[cmd.index("--effort") + 1]


def test_claude_command_maps_reasoning_to_effort(tmp_path: Path):
    assert _effort_of(ClaudeRunner(ClaudeConfig(), tmp_path), reasoning="xhigh") == "xhigh"

    # Per-call reasoning wins over the configured default.
    configured = ClaudeRunner(ClaudeConfig(effort="low"), tmp_path)
    assert _effort_of(configured) == "low"
    assert _effort_of(configured, reasoning="max") == "max"


def test_claude_command_omits_effort_when_unset(tmp_path: Path):
    runner = ClaudeRunner(ClaudeConfig(), tmp_path)
    assert "--effort" not in runner.build_command(schema=schema_for("research"))


def test_claude_command_toggles_web_search(tmp_path: Path):
    runner = ClaudeRunner(ClaudeConfig(live_search=True), tmp_path)

    live = runner.build_command(schema=schema_for("research"))
    assert live[live.index("--tools") + 1] == "WebSearch"

    cached = runner.build_command(schema=schema_for("research"), live_search=False)
    assert cached[cached.index("--tools") + 1] == ""


def test_claude_command_isolates_user_environment(tmp_path: Path):
    runner = ClaudeRunner(ClaudeConfig(), tmp_path)
    cmd = runner.build_command(schema=schema_for("research"))

    # --tools alone does not filter MCP tools, so these must be present.
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--disable-slash-commands" in cmd
    assert "--system-prompt" in cmd

    relaxed = ClaudeRunner(ClaudeConfig(isolate=False), tmp_path).build_command(schema=schema_for("research"))
    assert "--strict-mcp-config" not in relaxed


def _write_fake_claude(tmp_path: Path, events: list[dict], *, exit_code: int = 0) -> Path:
    """A stand-in for `claude -p` that replays a fixed stream-json transcript."""
    stream_path = tmp_path / "fake-claude.ndjson"
    stream_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )
    fake = tmp_path / "fake-claude"
    fake.write_text(
        "#!/bin/sh\n"
        'cat > "$0.prompt"\n'
        "echo stderr-line >&2\n"
        f'cat "{stream_path}"\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def test_claude_runner_parses_structured_output_and_streams_logs(tmp_path: Path):
    fake = _write_fake_claude(
        tmp_path,
        [
            {"type": "system", "subtype": "init", "tools": ["StructuredOutput"], "mcp_servers": []},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "arbeite"}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch"}]}},
            {"type": "rate_limit_event"},
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "permission_denials": [],
                "structured_output": RESEARCH_PAYLOAD,
            },
        ],
    )
    events = []
    runner = ClaudeRunner(ClaudeConfig(executable=str(fake)), tmp_path)

    report = runner.run_structured(
        prompt="prompt",
        schema_name="research",
        output_path=tmp_path / "research.json",
        model=ResearchReport,
        progress=events.append,
    )

    assert report.summary == "S"
    assert json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))["summary"] == "S"
    assert (tmp_path / "research.schema.json").exists()
    assert (tmp_path / "research.stdout.log").read_text(encoding="utf-8").count("\n") == 5
    assert f"{fake}.prompt" and Path(f"{fake}.prompt").read_text(encoding="utf-8") == "prompt"

    messages = [event.message for event in events]
    assert any("arbeite" in message for message in messages)
    assert any("tool: WebSearch" in message for message in messages)
    assert any("0 MCP-Server" in message for message in messages)
    assert any("stderr-line" in message for message in messages)
    assert any(event.level == "warning" for event in events)


def test_claude_runner_falls_back_to_text_result(tmp_path: Path):
    fake = _write_fake_claude(
        tmp_path,
        [
            {
                "type": "result",
                "is_error": False,
                "subtype": "success",
                "result": f"```json\n{json.dumps(RESEARCH_PAYLOAD)}\n```",
            }
        ],
    )
    runner = ClaudeRunner(ClaudeConfig(executable=str(fake)), tmp_path)

    report = runner.run_structured(
        prompt="prompt",
        schema_name="research",
        output_path=tmp_path / "research.json",
        model=ResearchReport,
    )

    assert report.summary == "S"


def test_claude_runner_raises_on_error_result(tmp_path: Path):
    fake = _write_fake_claude(
        tmp_path,
        [
            {
                "type": "result",
                "is_error": True,
                "subtype": "error_during_execution",
                "permission_denials": [{"tool_name": "Bash"}],
            }
        ],
    )
    runner = ClaudeRunner(ClaudeConfig(executable=str(fake)), tmp_path)

    with pytest.raises(RuntimeError, match="error_during_execution"):
        runner.run_structured(
            prompt="prompt",
            schema_name="research",
            output_path=tmp_path / "research.json",
            model=ResearchReport,
        )


def test_claude_runner_raises_when_result_event_missing(tmp_path: Path):
    fake = _write_fake_claude(tmp_path, [{"type": "assistant", "message": {"content": []}}])
    runner = ClaudeRunner(ClaudeConfig(executable=str(fake)), tmp_path)

    with pytest.raises(RuntimeError, match="no result event"):
        runner.run_structured(
            prompt="prompt",
            schema_name="research",
            output_path=tmp_path / "research.json",
            model=ResearchReport,
        )


def test_claude_runner_raises_on_nonzero_exit(tmp_path: Path):
    fake = _write_fake_claude(tmp_path, [], exit_code=2)
    runner = ClaudeRunner(ClaudeConfig(executable=str(fake)), tmp_path)

    with pytest.raises(RuntimeError, match="exit code 2"):
        runner.run_structured(
            prompt="prompt",
            schema_name="research",
            output_path=tmp_path / "research.json",
            model=ResearchReport,
        )
