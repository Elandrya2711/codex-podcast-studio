from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import ClaudeConfig
from .progress import CancellationToken, ProgressEvent, ProgressReporter
from .schemas import schema_for
from .util import json_from_text, write_json

T = TypeVar("T", bound=BaseModel)

MAX_LOG_CHARS = 240


class ClaudeRunner:
    """Structured LLM calls via the `claude` CLI in non-interactive print mode.

    Mirrors :class:`~codcast.codex_runner.CodexRunner`: the prompt is delivered on
    stdin (never in argv), the JSON schema constrains the output, and the parsed
    result is written to ``output_path`` so resume logic can read it back.
    """

    def __init__(self, config: ClaudeConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def build_command(
        self,
        *,
        schema: dict[str, Any],
        live_search: bool | None = None,
        reasoning: str | None = None,
    ) -> list[str]:
        cmd = [self.config.executable, "-p", "--model", self.config.model]
        effort = reasoning or self.config.effort
        if effort:
            cmd.extend(["--effort", effort])
        cmd.extend(["--json-schema", json.dumps(schema)])
        # stream-json needs --verbose in print mode; it is what gives us live logs.
        cmd.extend(["--output-format", "stream-json", "--verbose"])
        use_live_search = self.config.live_search if live_search is None else live_search
        cmd.extend(["--tools", "WebSearch" if use_live_search else ""])
        if self.config.isolate:
            # Without these the user's global MCP servers, hooks and skills leak
            # into the run: --tools alone does not filter MCP tools.
            cmd.extend(
                [
                    "--strict-mcp-config",
                    "--setting-sources",
                    "",
                    "--disable-slash-commands",
                ]
            )
        cmd.append("--no-session-persistence")
        if self.config.system_prompt:
            cmd.extend(["--system-prompt", self.config.system_prompt])
        cmd.extend(self.config.extra_args)
        return cmd

    def run_structured(
        self,
        *,
        prompt: str,
        schema_name: str,
        output_path: Path,
        model: type[T],
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
        timeout_sec: int | None = None,
        config_overrides: dict[str, object] | None = None,
        live_search: bool | None = None,
        reasoning: str | None = None,
    ) -> T:
        del config_overrides  # codex-specific escape hatch, not applicable here
        schema = schema_for(schema_name)
        schema_path = output_path.with_suffix(".schema.json")
        stdout_path = output_path.with_suffix(".stdout.log")
        stderr_path = output_path.with_suffix(".stderr.log")
        write_json(schema_path, schema)

        cmd = self.build_command(schema=schema, live_search=live_search, reasoning=reasoning)
        if progress:
            progress(ProgressEvent("log", schema_name, f"Starte Claude-Schritt: {schema_name}"))
        process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        sink: dict[str, Any] = {}
        threads = [
            threading.Thread(
                target=self._copy_events,
                args=(process.stdout, stdout_path, schema_name, progress, sink),
                daemon=True,
            ),
            threading.Thread(
                target=self._copy_stream,
                args=(process.stderr, stderr_path, schema_name, "stderr", progress),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        stdin_thread = threading.Thread(
            target=self._write_stdin,
            args=(process.stdin, prompt),
            daemon=True,
        )
        stdin_thread.start()

        effective_timeout = timeout_sec or self.config.timeout_sec
        deadline = time.monotonic() + effective_timeout
        while process.poll() is None:
            if cancellation and cancellation.cancelled:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for thread in threads:
                    thread.join(timeout=1)
                stdin_thread.join(timeout=1)
                cancellation.raise_if_cancelled()
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                for thread in threads:
                    thread.join(timeout=1)
                stdin_thread.join(timeout=1)
                raise TimeoutError(f"claude -p timed out after {effective_timeout} seconds")
            time.sleep(0.1)

        returncode = process.wait()
        for thread in threads:
            thread.join(timeout=5)
        stdin_thread.join(timeout=5)

        logs = f"See {stderr_path} and {stdout_path}."
        if returncode != 0:
            raise RuntimeError(f"claude -p failed with exit code {returncode}. {logs}")

        result = sink.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"claude -p produced no result event. {logs}")
        denials = result.get("permission_denials") or []
        if result.get("is_error") or result.get("subtype") != "success":
            detail = result.get("subtype") or result.get("api_error_status") or "unknown error"
            suffix = f" Permission denials: {denials}." if denials else ""
            raise RuntimeError(f"claude -p reported an error ({detail}).{suffix} {logs}")

        data = result.get("structured_output")
        if data is None:
            text = result.get("result")
            if not isinstance(text, str):
                raise RuntimeError(f"claude -p returned no structured output. {logs}")
            data = json_from_text(text)
        parsed = model.model_validate(data)
        output_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        return parsed

    def _copy_events(
        self,
        stream,
        output_path: Path,
        phase: str,
        progress: ProgressReporter | None,
        sink: dict[str, Any],
    ) -> None:
        if stream is None:
            output_path.write_text("", encoding="utf-8")
            return
        with output_path.open("w", encoding="utf-8") as handle:
            for line in stream:
                handle.write(line)
                handle.flush()
                raw = line.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    if progress:
                        progress(ProgressEvent("log", phase, f"stdout: {raw[:MAX_LOG_CHARS]}"))
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "result":
                    sink["result"] = event
                    continue
                if progress:
                    for message, level in _describe_event(event):
                        progress(ProgressEvent("log", phase, message, level=level))

    def _copy_stream(
        self,
        stream,
        output_path: Path,
        phase: str,
        stream_name: str,
        progress: ProgressReporter | None,
    ) -> None:
        if stream is None:
            output_path.write_text("", encoding="utf-8")
            return
        with output_path.open("w", encoding="utf-8") as handle:
            for line in stream:
                handle.write(line)
                handle.flush()
                message = line.strip()
                if progress and message:
                    progress(
                        ProgressEvent("log", phase, f"{stream_name}: {message[:MAX_LOG_CHARS]}")
                    )

    def _write_stdin(self, stream, prompt: str) -> None:
        if stream is None:
            return
        try:
            stream.write(prompt)
            stream.close()
        except BrokenPipeError:
            return


def _describe_rate_limit(event: dict[str, Any]) -> list[tuple[str, str]]:
    """Nur melden, wenn das Kontingent wirklich klemmt.

    Die CLI schickt dieses Event bei jedem Aufruf mit, im Regelfall mit
    `status: "allowed"`. Das als Warnung zu fuehren, macht jede Warnung im Log
    wertlos. Der Rohtext steht ohnehin in `<stufe>.stdout.log`, falls jemand
    das Fenster nachrechnen will.
    """
    info = event.get("rate_limit_info")
    if not isinstance(info, dict):
        return []
    status = info.get("status")
    if not isinstance(status, str) or status == "allowed":
        # Kein lesbarer Status ist kein Engpass, sondern nur eine Aussage, die
        # wir nicht deuten koennen. Raten wuerde hier falschen Alarm erzeugen.
        return []
    parts = [f"Kontingent der Claude CLI: {status}"]
    window = info.get("rateLimitType")
    if window:
        parts.append(f"Fenster {window}")
    resets_at = info.get("resetsAt")
    if isinstance(resets_at, (int, float)) and not isinstance(resets_at, bool):
        parts.append(f"frei ab {datetime.fromtimestamp(resets_at):%H:%M}")
    return [(", ".join(parts), "warning")]


def _describe_event(event: dict[str, Any]) -> list[tuple[str, str]]:
    """Turn one stream-json event into zero or more human-readable log lines."""
    event_type = event.get("type")
    if event_type == "rate_limit_event":
        return _describe_rate_limit(event)
    if event_type == "system":
        if event.get("subtype") == "init":
            servers = event.get("mcp_servers") or []
            tools = event.get("tools") or []
            return [(f"init: {len(tools)} Tools, {len(servers)} MCP-Server", "info")]
        return []
    if event_type != "assistant":
        return []
    lines: list[tuple[str, str]] = []
    message = event.get("message")
    blocks = message.get("content") if isinstance(message, dict) else None
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_use":
            lines.append((f"tool: {block.get('name')}", "info"))
        elif block_type in {"text", "thinking"}:
            text = block.get("text") if block_type == "text" else block.get("thinking")
            if isinstance(text, str) and text.strip():
                prefix = "denkt" if block_type == "thinking" else "sagt"
                lines.append((f"{prefix}: {text.strip()[:MAX_LOG_CHARS]}", "info"))
    return lines
