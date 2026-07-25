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


EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]


class CodexConfig(ConfigModel):
    executable: str = "codex"
    model: str | None = None
    effort: EffortLevel | None = None
    live_search: bool = True
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only"
    approval_policy: Literal["never", "on-request", "untrusted"] = "never"
    timeout_sec: int = 1200
    extra_args: list[str] = Field(default_factory=list)


class ClaudeConfig(ConfigModel):
    executable: str = "claude"
    model: str = "claude-opus-5"
    effort: EffortLevel | None = None
    deep_effort: EffortLevel = "xhigh"
    live_search: bool = True
    timeout_sec: int = 1200
    isolate: bool = True
    system_prompt: str = (
        "Du bist ein praeziser Recherche- und Redaktionsassistent fuer deutschsprachige "
        "Podcast-Produktion. Arbeite ausschliesslich mit belegbaren Fakten und kennzeichne "
        "Unsicherheiten explizit. Gib das Ergebnis ausschliesslich ueber das "
        "StructuredOutput-Tool im vorgegebenen Schema zurueck: keine Vorrede, keine "
        "Nachbemerkung, keine Markdown-Codebloecke."
    )
    extra_args: list[str] = Field(default_factory=list)


class LLMConfig(ConfigModel):
    provider: Literal["claude", "codex"] = "claude"
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)


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


class ChatterboxConfig(ConfigModel):
    """Chatterbox Multilingual laeuft in einer eigenen venv (`codcast setup-chatterbox`),

    weil seine torch-Abhaengigkeiten nicht zu denen des Projekts passen muessen.
    Die Defaults sind die auf Deutsch gemessen beste Kombination: ruhiger Vortrag,
    Tempo und Betonung dicht an der Referenzstimme.
    """

    python_executable: Path = Path(".venv-chatterbox/bin/python")
    device: str = "cuda"
    language: str = "de"
    exaggeration: float = 0.35
    cfg_weight: float = 0.3
    temperature: float = 0.6
    repetition_penalty: float = 2.0
    # Zahlen und Kuerzel ausschreiben; ohne das wird aus "48-Volt" gemessen "88 Volt".
    normalize_text: bool = True
    # Schutz gegen Wiederholungs-Loops: das Modell spricht gelegentlich einen Satz
    # zweimal, was die Segmentdauer weit ueber das Erwartbare treibt. Solche
    # Segmente werden neu gerendert.
    max_retries: int = 2
    min_duration_ratio: float = 0.45
    max_duration_ratio: float = 1.7
    chars_per_second: float = 15.0
    startup_timeout_sec: int = 600
    timeout_sec: int = 300

    @field_validator("python_executable", mode="before")
    @classmethod
    def expand_python_executable(cls, value: object) -> object:
        return Path(str(value)).expanduser()


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
    backend: Literal["chatterbox", "fish", "kokoro", "openai", "xtts"]
    language: str = "de-DE"
    license: str = "unknown"
    speaker_wav: Path | None = None
    ref_text: str | None = None
    fish_reference_id: str | None = None
    openai_voice: str | None = None
    openai_instructions: str | None = None
    kokoro_model: Path | None = None
    kokoro_voice: Path | None = None
    # Pro Stimme nachschaerfen, ohne die anderen Stimmen anzufassen.
    # Nicht gesetzte Werte kommen aus tts.chatterbox.
    chatterbox_exaggeration: float | None = None
    chatterbox_cfg_weight: float | None = None
    chatterbox_temperature: float | None = None
    speed: float = 1.0

    @field_validator("speaker_wav", "kokoro_model", "kokoro_voice", mode="before")
    @classmethod
    def expand_model_paths(cls, value: object) -> object:
        if value in (None, ""):
            return None
        return Path(str(value)).expanduser()


class TTSConfig(ConfigModel):
    quality: Literal["best", "chatterbox", "fast", "openai", "xtts"] = "best"
    backend: Literal["chatterbox", "fish", "kokoro", "openai", "xtts"] = "chatterbox"
    language: str = "de"
    allow_voice_reuse: bool = False
    # Benannte Besetzungen: ohne das gewinnen immer die ersten passenden Profile,
    # sobald mehrere Stimmpaare fuer dasselbe Backend konfiguriert sind.
    voice_set: str | None = None
    voice_sets: dict[str, list[str]] = Field(default_factory=dict)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    chatterbox: ChatterboxConfig = Field(default_factory=ChatterboxConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    xtts: XttsConfig = Field(default_factory=XttsConfig)
    fish: FishConfig = Field(default_factory=FishConfig)
    voices: list[VoiceProfile] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_voice_sets(self) -> "TTSConfig":
        known = {voice.id for voice in self.voices}
        for name, members in self.voice_sets.items():
            if not members:
                raise ValueError(f"voice_set '{name}' is empty")
            unknown = [member for member in members if member not in known]
            if unknown:
                raise ValueError(f"voice_set '{name}' references unknown voice profiles: {', '.join(unknown)}")
        if self.voice_set is not None and self.voice_set not in self.voice_sets:
            raise ValueError(f"voice_set '{self.voice_set}' is not defined in voice_sets")
        return self


class DiarizeConfig(ConfigModel):
    """Sprechertrennung fuer `codcast voices extract`.

    Laeuft wie Chatterbox in einer eigenen venv (`codcast setup-diarize`), weil
    torch/speechbrain nicht in die Projekt-venv gehoeren. Die Defaults sind auf
    Klonmaterial getrimmt, nicht auf vollstaendige Diarisierung: lieber viel
    verwerfen als eine Silbe der zweiten Person mitnehmen.
    """

    python_executable: Path = Path(".venv-diarize/bin/python")
    device: str = "cuda"
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_dir: Path = Path("models/spkrec-ecapa-voxceleb")
    # Analysefenster fuer die Sprecher-Embeddings.
    window_sec: float = 1.5
    hop_sec: float = 0.75
    # An jedem Sprecherwechsel sitzt die Ueberlappung; beide Enden jedes Laufs fallen weg.
    erosion_sec: float = 0.75
    min_run_sec: float = 3.0
    min_chunk_sec: float = 4.0
    max_chunk_sec: float = 12.0
    # Qualitaetsgates pro Chunk.
    min_speech_ratio: float = 0.75
    min_margin: float = 0.10
    max_peak: float = 0.99
    min_rms_dbfs: float = -40.0
    max_rms_dbfs: float = -6.0
    # Auswahl ueber Zeit-Bins verteilen, damit das Material nicht aus einem Themenblock stammt.
    time_bins: int = 10
    analysis_sample_rate: int = 16000
    # Podcast-Quellen liegen praktisch immer bei 44.1 kHz; hochsamplen brächte nichts.
    master_sample_rate: int = 44100
    export_sample_rate: int = 24000
    # Ein Sprecherwechsel bricht einen Lauf immer; laengere Pausen zusaetzlich.
    max_gap_sec: float = 1.0
    loudnorm_i: float = -20.0
    loudnorm_tp: float = -1.5
    gap_between_chunks_sec: float = 0.15
    preview_sec: float = 30.0
    timeout_sec: int = 7200

    @field_validator("python_executable", "model_dir", mode="before")
    @classmethod
    def expand_python_executable(cls, value: object) -> object:
        return Path(str(value)).expanduser()


class AppConfig(ConfigModel):
    language: str = "de-DE"
    output_root: Path = Path("podcasts")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    codex: CodexConfig = Field(default_factory=CodexConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    diarize: DiarizeConfig = Field(default_factory=DiarizeConfig)

    @field_validator("output_root", mode="before")
    @classmethod
    def expand_output_root(cls, value: object) -> object:
        return Path(str(value)).expanduser()

    @property
    def llm_timeout_sec(self) -> int:
        """Per-call timeout of the active LLM provider."""
        if self.llm.provider == "claude":
            return self.llm.claude.timeout_sec
        return self.codex.timeout_sec

    @property
    def llm_deep_reasoning(self) -> str:
        """Reasoning effort for the expensive deep-research and revision stages."""
        if self.llm.provider == "claude":
            return self.llm.claude.deep_effort
        return "xhigh"


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
        "llm": LLMConfig().model_dump(mode="json"),
        "codex": CodexConfig().model_dump(mode="json"),
        "generation": GenerationConfig().model_dump(mode="json"),
        "research": ResearchConfig().model_dump(mode="json"),
        "evidence": EvidenceConfig().model_dump(mode="json"),
        "audio": AudioConfig().model_dump(mode="json"),
        "diarize": DiarizeConfig().model_dump(mode="json"),
        "tts": {
            "quality": "best",
            "backend": "chatterbox",
            "language": "de",
            "allow_voice_reuse": False,
            "openai": OpenAIConfig().model_dump(mode="json"),
            "chatterbox": ChatterboxConfig().model_dump(mode="json"),
            "kokoro": KokoroConfig().model_dump(mode="json"),
            "xtts": XttsConfig().model_dump(mode="json"),
            "fish": FishConfig().model_dump(mode="json"),
            "voices": [
                {
                    "id": "chatterbox-host-m",
                    "display_name": "Jonas",
                    "backend": "chatterbox",
                    "language": "de-DE",
                    "license": "private",
                    "speaker_wav": "voices/chatterbox/host-m.wav",
                    "speed": 1.0,
                },
                {
                    "id": "chatterbox-host-f",
                    "display_name": "Mara",
                    "backend": "chatterbox",
                    "language": "de-DE",
                    "license": "private",
                    "speaker_wav": "voices/chatterbox/host-f.wav",
                    "speed": 1.0,
                },
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
    # Der Worker-Interpreter muss auch gefunden werden, wenn codcast aus einem
    # anderen Arbeitsverzeichnis laeuft.
    if not config.tts.chatterbox.python_executable.is_absolute():
        config.tts.chatterbox.python_executable = base_dir / config.tts.chatterbox.python_executable
    if not config.diarize.python_executable.is_absolute():
        config.diarize.python_executable = base_dir / config.diarize.python_executable
    if not config.diarize.model_dir.is_absolute():
        config.diarize.model_dir = base_dir / config.diarize.model_dir
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
