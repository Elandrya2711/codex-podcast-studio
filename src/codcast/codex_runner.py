from __future__ import annotations

import subprocess
import threading
import time
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import CodexConfig
from .progress import CancellationToken, ProgressEvent, ProgressReporter
from .schemas import schema_for
from .util import json_from_text, write_json

T = TypeVar("T", bound=BaseModel)


class CodexRunner:
    def __init__(self, config: CodexConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def build_command(
        self,
        prompt: str,
        schema_path: Path,
        output_path: Path,
        *,
        config_overrides: dict[str, object] | None = None,
        live_search: bool | None = None,
    ) -> list[str]:
        cmd = [self.config.executable]
        use_live_search = self.config.live_search if live_search is None else live_search
        if use_live_search:
            cmd.append("--search")
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        cmd.extend(["--ask-for-approval", self.config.approval_policy])
        for key, value in (config_overrides or {}).items():
            cmd.extend(["-c", f"{key}={_toml_literal(value)}"])
        cmd.extend(self.config.extra_args)
        cmd.extend(
            [
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                self.config.sandbox,
                "--color",
                "never",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
        )
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
    ) -> T:
        schema_path = output_path.with_suffix(".schema.json")
        stdout_path = output_path.with_suffix(".stdout.log")
        stderr_path = output_path.with_suffix(".stderr.log")
        write_json(schema_path, schema_for(schema_name))

        cmd = self.build_command(
            prompt,
            schema_path,
            output_path,
            config_overrides=config_overrides,
            live_search=live_search,
        )
        if progress:
            progress(ProgressEvent("log", schema_name, f"Starte Codex-Schritt: {schema_name}"))
        process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threads = [
            threading.Thread(
                target=self._copy_stream,
                args=(process.stdout, stdout_path, schema_name, "stdout", progress),
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
                raise TimeoutError(f"codex exec timed out after {effective_timeout} seconds")
            time.sleep(0.1)

        returncode = process.wait()
        for thread in threads:
            thread.join(timeout=5)
        stdin_thread.join(timeout=5)
        if returncode != 0:
            raise RuntimeError(
                "codex exec failed with exit code "
                f"{returncode}. See {stderr_path} and {stdout_path}."
            )

        text = output_path.read_text(encoding="utf-8") if output_path.exists() else stdout_path.read_text(encoding="utf-8")
        data = json_from_text(text)
        parsed = model.model_validate(data)
        output_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        return parsed

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
                    progress(ProgressEvent("log", phase, f"{stream_name}: {message[:240]}"))

    def _write_stdin(self, stream, prompt: str) -> None:
        if stream is None:
            return
        try:
            stream.write(prompt)
            stream.close()
        except BrokenPipeError:
            return


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if value is None:
        return '""'
    return json.dumps(str(value))
