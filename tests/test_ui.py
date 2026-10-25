from __future__ import annotations

from rich.console import Console

from codcast.ui import PodcastTui, TuiState, one_line


def test_one_line_collapses_whitespace() -> None:
    assert one_line("Zeile eins\n\n  Zeile   zwei\t", 80) == "Zeile eins Zeile zwei"


def test_one_line_keeps_short_text_untouched() -> None:
    assert one_line("Kurzes Thema", 80) == "Kurzes Thema"


def test_one_line_truncates_with_ellipsis() -> None:
    result = one_line("a" * 200, 20)
    assert len(result) == 20
    assert result.endswith("\N{HORIZONTAL ELLIPSIS}")


def test_one_line_survives_absurd_width() -> None:
    assert one_line("egal", 0) == ""


def test_header_stays_single_line_for_a_briefing_topic() -> None:
    # Ein Thema aus --topic-file kann eine ganze Datei sein; der Kopfbereich
    # darf dadurch nicht den Fortschritt aus dem Bild schieben.
    topic = "\n".join(f"Absatz {index} mit reichlich Text." for index in range(200))
    tui = PodcastTui(console=Console(width=100))

    group = tui._render(TuiState(topic=topic))

    header = group.renderables[0]
    assert header.renderable.count("\n") == 1  # nur Thema und Laufzeit
    assert max(len(line) for line in header.renderable.splitlines()) <= 96
