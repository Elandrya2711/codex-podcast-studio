from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


ProgressLevel = Literal["info", "warning", "error"]
ProgressEventType = Literal["start", "progress", "log", "done", "failed"]
ProgressReporter = Callable[["ProgressEvent"], None]


class PodcastCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise PodcastCancelled("Podcast generation cancelled")


@dataclass(frozen=True)
class ProgressEvent:
    type: ProgressEventType
    phase: str
    message: str
    current: int | None = None
    total: int | None = None
    path: Path | None = None
    level: ProgressLevel = "info"


def report_progress(progress: ProgressReporter | None, event: ProgressEvent) -> None:
    if progress is not None:
        progress(event)
