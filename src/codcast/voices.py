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
    if selected == "fast":
        return "kokoro"
    if selected == "openai":
        return "openai"
    if selected == "xtts":
        return "xtts"
    return config.tts.backend


def select_voice_profiles(config: AppConfig, speaker_count: int, quality: str | None = None) -> list[VoiceProfile]:
    if speaker_count < 1:
        raise ValueError("speaker_count must be at least 1")
    if speaker_count > config.generation.max_speakers:
        raise ValueError(f"speaker_count cannot exceed {config.generation.max_speakers}")

    backend = backend_for_quality(config, quality)
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
