from __future__ import annotations

import re

from .models import PodcastScript

WORD_RE = re.compile(r"[\wÄÖÜäöüß]+(?:[-'][\wÄÖÜäöüß]+)?", re.UNICODE)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def script_word_count(script: PodcastScript) -> int:
    return sum(count_words(line.text) for line in script.lines)


def estimate_minutes(word_count: int, words_per_minute: int) -> float:
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be positive")
    return word_count / words_per_minute


def target_word_range(min_minutes: float, max_minutes: float, words_per_minute: int) -> tuple[int, int]:
    if min_minutes <= 0 or max_minutes <= 0:
        raise ValueError("minutes must be positive")
    if min_minutes > max_minutes:
        raise ValueError("min_minutes cannot exceed max_minutes")
    return (round(min_minutes * words_per_minute), round(max_minutes * words_per_minute))


def duration_status(script: PodcastScript, words_per_minute: int) -> tuple[str, int, float]:
    words = script_word_count(script)
    minutes = estimate_minutes(words, words_per_minute)
    if minutes < script.target_min_minutes:
        return "too_short", words, minutes
    if minutes > script.target_max_minutes:
        return "too_long", words, minutes
    return "ok", words, minutes

