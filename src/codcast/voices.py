from __future__ import annotations

from itertools import cycle, islice

from .config import AppConfig, VoiceProfile
from .models import SpeakerSpec

ROLES = [
    "Host",
    "Analyst",
    "Skeptic",
    "Context",
    "Explainer",
    "Producer",
]


def backend_for_quality(config: AppConfig, quality: str | None = None) -> str:
    selected = quality or config.tts.quality
    if selected == "best":
        return config.tts.backend
    if selected == "chatterbox":
        return "chatterbox"
    if selected == "fast":
        return "kokoro"
    if selected == "openai":
        return "openai"
    if selected == "xtts":
        return "xtts"
    return config.tts.backend


def voice_set_members(config: AppConfig, name: str) -> list[VoiceProfile]:
    """Profile einer benannten Besetzung, in der konfigurierten Reihenfolge."""
    members = config.tts.voice_sets.get(name)
    if not members:
        raise ValueError(f"unknown voice set: {name}")
    by_id = {voice.id: voice for voice in config.tts.voices}
    missing = [member for member in members if member not in by_id]
    if missing:
        raise ValueError(f"voice set {name} references unknown voice profiles: {', '.join(missing)}")
    return [by_id[member] for member in members]


def select_voice_profiles(
    config: AppConfig,
    speaker_count: int,
    quality: str | None = None,
    voice_set: str | None = None,
) -> list[VoiceProfile]:
    if speaker_count < 1:
        raise ValueError("speaker_count must be at least 1")
    if speaker_count > config.generation.max_speakers:
        raise ValueError(f"speaker_count cannot exceed {config.generation.max_speakers}")

    backend = backend_for_quality(config, quality)
    selected_set = voice_set or config.tts.voice_set
    if selected_set:
        members = voice_set_members(config, selected_set)
        mismatched = [voice.id for voice in members if voice.backend != backend]
        if mismatched:
            raise ValueError(
                f"voice set {selected_set} contains voices for another backend than {backend}: "
                f"{', '.join(mismatched)}"
            )
        if len(members) >= speaker_count:
            return members[:speaker_count]
        if config.tts.allow_voice_reuse:
            return list(islice(cycle(members), speaker_count))
        raise ValueError(
            f"{speaker_count} voice(s) requested, but voice set {selected_set} only has {len(members)}. "
            "Add more voices to the set or set tts.allow_voice_reuse=true."
        )
    profiles = [voice for voice in config.tts.voices if voice.backend == backend]
    if len(profiles) >= speaker_count:
        return profiles[:speaker_count]
    if config.tts.allow_voice_reuse and profiles:
        return list(islice(cycle(profiles), speaker_count))

    if backend == "xtts":
        raise ValueError(
            f"{speaker_count} XTTS voice profile(s) requested, but only {len(profiles)} configured. "
            "Import private reference voices with `codcast voices import ...`."
        )
    if backend == "chatterbox":
        raise ValueError(
            f"{speaker_count} Chatterbox voice profile(s) requested, but only {len(profiles)} configured. "
            "Import reference voices with `codcast voices import --backend chatterbox ...`."
        )
    if backend == "fish":
        raise ValueError(
            f"{speaker_count} Fish voice profile(s) requested, but only {len(profiles)} configured. "
            "Import premium reference voices with `codcast voices import --backend fish ...`."
        )
    if backend == "openai":
        raise ValueError(
            f"{speaker_count} OpenAI voice profile(s) requested, but only {len(profiles)} configured. "
            "Add more OpenAI voice profiles to podcast.yml or set tts.allow_voice_reuse=true."
        )
    raise ValueError(
        f"{speaker_count} Kokoro voice profile(s) requested, but only {len(profiles)} configured. "
        "Add more high-quality voices to podcast.yml or set tts.allow_voice_reuse=true."
    )


def build_speaker_specs(profiles: list[VoiceProfile]) -> list[SpeakerSpec]:
    specs: list[SpeakerSpec] = []
    for index, profile in enumerate(profiles, start=1):
        role = ROLES[index - 1] if index <= len(ROLES) else f"Speaker {index}"
        specs.append(
            SpeakerSpec(
                id=f"s{index}",
                display_name=profile.display_name,
                role=role,
                voice_profile_id=profile.id,
            )
        )
    return specs


def voice_map(profiles: list[VoiceProfile]) -> dict[str, VoiceProfile]:
    return {profile.id: profile for profile in profiles}
