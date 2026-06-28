from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def expand_path(value: str | Path | None, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    path = Path(text).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CodexConfig(ConfigModel):
    executable: str = "codex"
    model: str | None = None
    live_search: bool = True
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    approval_policy: Literal["never", "on-request", "untrusted"] = "never"
    timeout_sec: int = 1200
    extra_args: list[str] = Field(default_factory=list)


class GenerationConfig(ConfigModel):
    words_per_minute: int = 145
    rewrite_attempts: int = 2
    max_speakers: int = 6
    pause_between_lines_sec: float = 0.28


class ResearchConfig(ConfigModel):
    depth: Literal["standard", "deep", "dossier"] = "standard"
    provider: Literal["searxng"] = "searxng"
    searxng_base_url: str = "http://127.0.0.1:8888"
    searxng_categories: str = "general,news,science"
    searxng_language: str = "auto"
    max_minutes: int | None = None
    max_rounds: int | None = None
    max_documents: int | None = None
    per_query_results: int = 10
    queries_per_round: int = 4
    extraction_batch_size: int = 8
    codex_batch_chars: int = 24000
    crawl_high_authority_domains: bool = False
    timeout_sec: int = 60
    max_fetch_bytes: int = 2_000_000
    user_agent: str = "codex-podcast-studio/0.1 (+local research pipeline)"
    allow_private_networks: bool = False

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_research_config(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        migrated.pop("api_key_env", None)
        if migrated.get("provider") == "tavily":
            migrated["provider"] = "searxng"
        return migrated


class EvidenceConfig(ConfigModel):
    enabled: bool = True
    youtube_enabled: bool = True
    yt_dlp_executable: str = "yt-dlp"
    preferred_subtitle_languages: list[str] = Field(default_factory=lambda: ["de.*", "de", "en.*", "en"])
    max_youtube_urls: int = 8
    max_subtitle_bytes: int = 2_000_000
    max_transcript_chars: int = 45000
    max_total_transcript_chars: int = 120000
    timeout_sec: int = 240
    cookies_from_browser: str | None = None


class AudioConfig(ConfigModel):
    sample_rate: int = 44100
    channels: int = 1
    wav_loudnorm: bool = True
    mp3_bitrate: str = "192k"


class KokoroConfig(ConfigModel):
    lang_code: str = "de"
    device: str = "cuda"
    config_path: Path = Path("models/kokoro/config.json")

    @field_validator("config_path", mode="before")
    @classmethod
    def expand_config_path(cls, value: object) -> object:
        return Path(str(value)).expanduser()


class XttsConfig(ConfigModel):
    gpu: bool = True
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    accept_cpml_noncommercial: bool = False


class OpenAIConfig(ConfigModel):
    env_file: Path = Path(".env.tts.local")
    api_key_env: str = "OPENAI_TTS_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini-tts"
    voice: str = "cedar"
    response_format: Literal["wav"] = "wav"
    max_input_chars: int = 3800
    max_output_bytes: int = 50_000_000
    concurrency: int = 4
    instructions: str = (
        "Sprich natuerlich, ruhig und praezise auf Deutsch. "
        "Der Stil ist ein moderner, gut verstaendlicher Podcast."
    )
    timeout_sec: int = 600

    @field_validator("env_file", mode="before")
    @classmethod
    def expand_env_file(cls, value: object) -> object:
        return Path(str(value)).expanduser()


class FishConfig(ConfigModel):
    server_url: str = "http://127.0.0.1:8098/v1/tts"
    api_key: str | None = None
    api_key_env: str | None = "FISH_AUDIO_API_KEY"
    output_format: Literal["wav", "mp3", "opus"] = "wav"
    latency: Literal["normal", "balanced"] = "normal"
    max_new_tokens: int = 192
    max_output_bytes: int = 50_000_000
    chunk_length: int = 120
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    temperature: float = 0.8
    seed: int | None = None
    timeout_sec: int = 600
    use_memory_cache: Literal["on", "off"] = "on"

    @field_validator("use_memory_cache", mode="before")
    @classmethod
    def normalize_memory_cache(cls, value: object) -> object:
        if value is True:
            return "on"
        if value is False:
            return "off"
        return value


class VoiceProfile(ConfigModel):
    id: str
    display_name: str
    backend: Literal["fish", "kokoro", "openai", "xtts"]
    language: str = "de-DE"
    license: str = "unknown"
    speaker_wav: Path | None = None
    ref_text: str | None = None
    fish_reference_id: str | None = None
    openai_voice: str | None = None
    openai_instructions: str | None = None
    kokoro_model: Path | None = None
    kokoro_voice: Path | None = None
    speed: float = 1.0

    @field_validator("speaker_wav", "kokoro_model", "kokoro_voice", mode="before")
    @classmethod
    def expand_model_paths(cls, value: object) -> object:
        if value in (None, ""):
            return None
        return Path(str(value)).expanduser()


class TTSConfig(ConfigModel):
    quality: Literal["best", "fast", "openai", "xtts"] = "best"
    backend: Literal["fish", "kokoro", "openai", "xtts"] = "openai"
    language: str = "de"
    allow_voice_reuse: bool = False
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    xtts: XttsConfig = Field(default_factory=XttsConfig)
    fish: FishConfig = Field(default_factory=FishConfig)
    voices: list[VoiceProfile] = Field(default_factory=list)


class AppConfig(ConfigModel):
    language: str = "de-DE"
    output_root: Path = Path("podcasts")
    codex: CodexConfig = Field(default_factory=CodexConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)

    @field_validator("output_root", mode="before")
    @classmethod
    def expand_output_root(cls, value: object) -> object:
        return Path(str(value)).expanduser()


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_config_dict() -> dict[str, Any]:
    return {
        "language": "de-DE",
        "output_root": "podcasts",
        "codex": CodexConfig().model_dump(mode="json"),
        "generation": GenerationConfig().model_dump(mode="json"),
        "research": ResearchConfig().model_dump(mode="json"),
        "evidence": EvidenceConfig().model_dump(mode="json"),
        "audio": AudioConfig().model_dump(mode="json"),
        "tts": {
            "quality": "best",
            "backend": "openai",
            "language": "de",
            "allow_voice_reuse": False,
            "openai": OpenAIConfig().model_dump(mode="json"),
            "kokoro": KokoroConfig().model_dump(mode="json"),
            "xtts": XttsConfig().model_dump(mode="json"),
            "fish": FishConfig().model_dump(mode="json"),
            "voices": [
                {
                    "id": "openai-cedar",
                    "display_name": "Cedar",
                    "backend": "openai",
                    "language": "de-DE",
                    "license": "openai-api",
                    "openai_voice": "cedar",
                    "openai_instructions": (
                        "Sprich als ruhiger deutscher Podcast-Host: klar, konzentriert, "
                        "natuerlich und ohne uebertriebene Emotion."
                    ),
                    "speed": 1.0,
                },
                {
                    "id": "openai-marin",
                    "display_name": "Marin",
                    "backend": "openai",
                    "language": "de-DE",
                    "license": "openai-api",
                    "openai_voice": "marin",
                    "openai_instructions": (
                        "Sprich als analytische deutsche Podcast-Stimme: warm, praezise "
                        "und mit sauberer Betonung technischer Begriffe."
                    ),
                    "speed": 1.0,
                },
                {
                    "id": "martin",
                    "display_name": "Martin",
                    "backend": "kokoro",
                    "language": "de-DE",
                    "license": "apache-2.0",
                    "kokoro_model": "models/kokoro/martin/kikiri_german_martin_ep10.pth",
                    "kokoro_voice": "models/kokoro/martin/voices/martin.pt",
                    "speed": 1.0,
                },
                {
                    "id": "victoria",
                    "display_name": "Victoria",
                    "backend": "kokoro",
                    "language": "de-DE",
                    "license": "apache-2.0",
                    "kokoro_model": "models/kokoro/victoria/kikiri_german_victoria_ep10.pth",
                    "kokoro_voice": "models/kokoro/victoria/voices/victoria.pt",
                    "speed": 1.0,
                },
                {
                    "id": "fish-host-m",
                    "display_name": "Jonas",
                    "backend": "fish",
                    "language": "de-DE",
                    "license": "private",
                    "speaker_wav": "voices/fish/host-m.wav",
                    "ref_text": (
                        "Willkommen im Podcast Studio. Heute sprechen wir ruhig und praezise "
                        "ueber Twitch, YouTube, Anny the Duck und die Dynamik in Online-Communitys."
                    ),
                    "speed": 1.0,
                },
                {
                    "id": "fish-host-f",
                    "display_name": "Mara",
                    "backend": "fish",
                    "language": "de-DE",
                    "license": "private",
                    "speaker_wav": "voices/fish/host-f.wav",
                    "ref_text": (
                        "Ich ordne die Fakten ein, stelle kritische Nachfragen und achte darauf, "
                        "dass englische Begriffe wie Twitch, YouTube und Stream sauber klingen."
                    ),
                    "speed": 1.0,
                },
            ],
        },
    }


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path("podcast.yml")
    data = default_config_dict()
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        data = deep_merge(data, loaded)
    config = AppConfig.model_validate(data)
    base_dir = config_path.expanduser().resolve().parent
    if not config.tts.openai.env_file.is_absolute():
        config.tts.openai.env_file = base_dir / config.tts.openai.env_file
    return config


def write_config(config: AppConfig, path: Path) -> None:
    data = config.model_dump(mode="json")
    _make_openai_env_file_relative(data, path.expanduser().resolve().parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def config_to_yaml(config: AppConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def _make_openai_env_file_relative(data: dict[str, Any], base_dir: Path) -> None:
    try:
        env_file = Path(data["tts"]["openai"]["env_file"])
    except KeyError:
        return
    if not env_file.is_absolute():
        return
    try:
        data["tts"]["openai"]["env_file"] = str(env_file.relative_to(base_dir))
    except ValueError:
        data["tts"]["openai"]["env_file"] = str(env_file)
