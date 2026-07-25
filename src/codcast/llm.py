from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .claude_runner import ClaudeRunner
from .codex_runner import CodexRunner
from .config import AppConfig
from .progress import CancellationToken, ProgressReporter

T = TypeVar("T", bound=BaseModel)


class LLMRunner(Protocol):
    """The contract every LLM provider in this project implements.

    Callers rely on both the return value and the artifacts written next to
    ``output_path`` (``.schema.json``, ``.stdout.log``, ``.stderr.log``), because
    the resume path reads those files back.
    """

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
    ) -> T: ...


_SETUP_HINTS = {
    "claude": (
        "Installiere die Claude CLI (https://claude.com/product/claude-code) und melde dich "
        "mit deinem Abo an ('claude' starten und einloggen). Ein API-Key ist nicht noetig. "
        "Alternativ auf den bisherigen Provider ausweichen: --llm-provider codex"
    ),
    "codex": (
        "Installiere die Codex CLI und melde dich mit 'codex login' an. "
        "Alternativ auf Claude ausweichen: --llm-provider claude"
    ),
}


def runner_for_config(config: AppConfig, project_root: Path) -> LLMRunner:
    provider = config.llm.provider
    if provider == "claude":
        claude_config = config.llm.claude
        _require_executable(claude_config.executable, "claude")
        return ClaudeRunner(claude_config, project_root)
    if provider == "codex":
        _require_executable(config.codex.executable, "codex")
        return CodexRunner(config.codex, project_root)
    raise ValueError(f"unsupported LLM provider: {provider}")


def _require_executable(executable: str, provider: str) -> None:
    if shutil.which(executable):
        return
    hint = _SETUP_HINTS.get(provider, "")
    raise RuntimeError(
        f"LLM-Provider '{provider}' ist ausgewaehlt, aber '{executable}' wurde nicht gefunden. {hint}"
    )
