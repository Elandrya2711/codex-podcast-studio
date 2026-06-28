from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .models import RunManifest
from .progress import CancellationToken, PodcastCancelled, ProgressEvent


PHASE_LABELS = [
    ("setup", "Setup"),
    ("evidence", "Belege"),
    ("research", "Recherche"),
    ("validation", "Validierung"),
    ("script", "Skript"),
    ("artifacts", "Dokumente"),
    ("tts", "TTS"),
    ("assembly", "Audio"),
    ("complete", "Fertig"),
]


@dataclass
class TuiState:
    topic: str
    started_at: float = field(default_factory=time.monotonic)
    statuses: dict[str, str] = field(default_factory=lambda: {phase: "wait" for phase, _ in PHASE_LABELS})
    messages: dict[str, str] = field(default_factory=dict)
    progress: dict[str, tuple[int, int]] = field(default_factory=dict)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    run_path: str | None = None
    manifest: RunManifest | None = None
    error: str | None = None
    cancelled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def apply(self, event: ProgressEvent) -> None:
        with self.lock:
            if event.phase not in self.statuses:
                self.statuses[event.phase] = "wait"
            if event.type == "start":
                self.statuses[event.phase] = "run"
            elif event.type == "done":
                self.statuses[event.phase] = "done"
            elif event.type == "failed":
                self.statuses[event.phase] = "fail"
                self.error = event.message
            elif event.type == "progress":
                self.statuses[event.phase] = "run"
            if event.message:
                self.messages[event.phase] = event.message
            if event.current is not None and event.total is not None:
                self.progress[event.phase] = (event.current, event.total)
            if event.path is not None:
                if event.phase in {"setup", "complete"}:
                    self.run_path = str(event.path)
            if event.type == "log" or event.level != "info":
                prefix = event.phase
                if event.level != "info":
                    prefix = f"{event.level}:{prefix}"
                self.logs.append(f"{prefix}: {event.message}")

    def set_manifest(self, manifest: RunManifest) -> None:
        with self.lock:
            self.manifest = manifest

    def set_error(self, error: BaseException) -> None:
        with self.lock:
            self.error = str(error)

    def set_cancelled(self) -> None:
        with self.lock:
            self.cancelled = True
            self.logs.append("system: Abbruch angefordert")


class PodcastTui:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def run_generator(self, generator: Any, **generate_kwargs: Any) -> RunManifest:
        token = CancellationToken()
        state = TuiState(topic=str(generate_kwargs.get("topic", "Podcast")))
        result: dict[str, Any] = {}

        def reporter(event: ProgressEvent) -> None:
            state.apply(event)

        def worker() -> None:
            try:
                manifest = generator.generate(progress=reporter, cancellation=token, **generate_kwargs)
            except PodcastCancelled as exc:
                state.set_cancelled()
                result["error"] = exc
            except Exception as exc:
                state.set_error(exc)
                state.apply(ProgressEvent("failed", "complete", str(exc), level="error"))
                result["error"] = exc
            else:
                state.set_manifest(manifest)
                result["manifest"] = manifest

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with Live(self._render(state), console=self.console, refresh_per_second=8, transient=False) as live:
            try:
                while thread.is_alive():
                    live.update(self._render(state))
                    time.sleep(0.12)
            except KeyboardInterrupt:
                token.cancel()
                state.set_cancelled()
                live.update(self._render(state))
                while thread.is_alive():
                    live.update(self._render(state))
                    time.sleep(0.12)
            live.update(self._render(state))

        error = result.get("error")
        if error is not None:
            raise error
        return result["manifest"]

    def _render(self, state: TuiState) -> Group:
        with state.lock:
            statuses = dict(state.statuses)
            messages = dict(state.messages)
            progress_values = dict(state.progress)
            logs = list(state.logs)
            run_path = state.run_path
            manifest = state.manifest
            error = state.error
            cancelled = state.cancelled
            elapsed = time.monotonic() - state.started_at
            topic = state.topic

        phase_table = Table.grid(expand=True)
        phase_table.add_column(ratio=1)
        phase_table.add_column(ratio=3)
        phase_table.add_column(ratio=6)
        for phase, label in PHASE_LABELS:
            status = statuses.get(phase, "wait")
            style = {"done": "green", "run": "cyan", "fail": "red", "wait": "dim"}.get(status, "white")
            marker = {"done": "OK", "run": "RUN", "fail": "ERR", "wait": "..."}.get(status, status)
            phase_table.add_row(f"[{style}]{marker}[/{style}]", label, messages.get(phase, ""))

        total_phases = len(PHASE_LABELS)
        completed_phases = sum(1 for phase, _ in PHASE_LABELS if statuses.get(phase) == "done")
        overall = Progress(
            TextColumn("[bold]Gesamt"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            expand=True,
        )
        overall.add_task("Gesamt", total=total_phases, completed=completed_phases)

        progress_group = [overall]
        for phase in ("tts", "assembly"):
            if phase in progress_values:
                current, total = progress_values[phase]
                bar = Progress(
                    TextColumn(f"[bold]{phase}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    expand=True,
                )
                bar.add_task(phase, total=max(total, 1), completed=min(current, total))
                progress_group.append(bar)

        log_table = Table.grid(expand=True)
        log_table.add_column()
        if logs:
            for line in logs[-12:]:
                log_table.add_row(line)
        else:
            log_table.add_row("[dim]Noch keine Logs.[/dim]")

        footer_lines = [f"Thema: {topic}", f"Laufzeit: {elapsed:0.1f}s"]
        if run_path:
            footer_lines.append(f"Run: {run_path}")
        if cancelled:
            footer_lines.append("Status: Abbruch angefordert")
        if error:
            footer_lines.append(f"Fehler: {error}")

        panels = [
            Panel("\n".join(footer_lines), title="Podcast", border_style="cyan"),
            Panel(Group(*progress_group), title="Fortschritt", border_style="blue"),
            Panel(phase_table, title="Schritte", border_style="green"),
            Panel(log_table, title="Logs", border_style="magenta"),
        ]
        if manifest is not None:
            artifact_table = Table.grid(expand=True)
            artifact_table.add_column(ratio=1)
            artifact_table.add_column(ratio=5)
            for name, path in manifest.artifacts.items():
                artifact_table.add_row(name, path)
            panels.append(Panel(artifact_table, title="Ergebnisse", border_style="green"))
        return Group(*panels)
