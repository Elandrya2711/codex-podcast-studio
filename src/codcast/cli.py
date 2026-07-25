from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio import assemble_episode, command_available
from .config import AppConfig, VoiceProfile, default_config_dict, load_config, write_config
from .models import PodcastScript
from .pipeline import PodcastGenerator
from .progress import PodcastCancelled
from .tts import ScriptRenderer, render_voice_sample
from .ui import PodcastTui
from .util import write_json
from .voices import backend_for_quality, select_voice_profiles, voice_map


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALITY_CHOICES = ("best", "chatterbox", "fast", "openai", "xtts")
RESEARCH_DEPTH_CHOICES = ("standard", "deep", "dossier")
LLM_PROVIDER_CHOICES = ("claude", "codex")
EFFORT_CHOICES = ("low", "medium", "high", "xhigh", "max")
# Bewusst ungated: kein HuggingFace-Token noetig, anders als bei pyannote.
DIARIZE_PACKAGES = ("torch", "torchaudio", "speechbrain", "silero-vad", "scikit-learn", "soundfile", "numpy<2")
# Short aliases that resolve to pinned model IDs, so a run stays reproducible.
CLAUDE_MODEL_ALIASES = {
    "opus": "claude-opus-5",
    "fable": "claude-fable-5",
    "sonnet": "claude-sonnet-5",
}
EFFORT_DEFAULT_LABEL = "standard"

QUALITY_DESCRIPTIONS = {
    "best": "Premium-Pfad aus podcast.yml (tts.backend)",
    "chatterbox": "lokal auf der GPU, klont Referenzstimmen, kostenlos",
    "fast": "Kokoro, lokal und schnell",
    "openai": "OpenAI TTS, kostet Geld, benoetigt OPENAI_TTS_API_KEY",
    "xtts": "eigene XTTS-Referenzstimmen",
}
RESEARCH_DEPTH_DESCRIPTIONS = {
    "standard": "eine Recherche-Runde, schnellster Weg",
    "deep": "mehrere Runden ueber lokale SearXNG-Suche",
    "dossier": "gruendlichste Stufe mit Dossier und Revision",
}
LLM_PROVIDER_DESCRIPTIONS = {
    "claude": "Claude CLI, Abo-Login, Standard",
    "codex": "Codex CLI, Abo-Login",
}
CLAUDE_MODEL_DESCRIPTIONS = {
    "opus": "claude-opus-5, gute Balance (Standard)",
    "fable": "claude-fable-5, faehigstes Modell, hoeherer Verbrauch",
    "sonnet": "claude-sonnet-5, schnell und sparsam",
}
EFFORT_DESCRIPTIONS = {
    EFFORT_DEFAULT_LABEL: "Vorgabe der CLI verwenden",
    "low": "wenig Nachdenken, schnell und sparsam",
    "medium": "ausgeglichen",
    "high": "gruendlich",
    "xhigh": "sehr gruendlich, empfohlen fuer schwere Themen",
    "max": "maximal, hoechster Verbrauch",
}
WIZARD_DEFAULT_MIN_MINUTES = 10.0
WIZARD_DEFAULT_MAX_MINUTES = 15.0


def project_root() -> Path:
    return Path.cwd()


def add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("podcast.yml"), help="Path to podcast.yml")


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-provider",
        choices=list(LLM_PROVIDER_CHOICES),
        default=None,
        help="LLM provider for research and scripting (default from podcast.yml: claude)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model for the active LLM provider. For claude also accepts the aliases "
            + "/".join(CLAUDE_MODEL_ALIASES)
            + " (resolved to pinned IDs, e.g. fable -> claude-fable-5)"
        ),
    )
    parser.add_argument(
        "--effort",
        choices=list(EFFORT_CHOICES),
        default=None,
        help="Reasoning effort for the active LLM provider",
    )
    parser.add_argument("--codex-model", default=None, help="Deprecated alias for --model on the codex provider")
    parser.add_argument(
        "--cached-search",
        action="store_true",
        help="Disable live web search and rely on cached knowledge",
    )


def resolve_claude_model(value: str) -> str:
    """Expand a short alias like 'fable' to its pinned model ID; pass anything else through."""
    return CLAUDE_MODEL_ALIASES.get(value.strip().lower(), value)


def claude_model_choices(current: str) -> tuple[tuple[str, ...], str]:
    """Wizard choices for the Claude model plus the key matching the configured model."""
    by_id = {model_id: alias for alias, model_id in CLAUDE_MODEL_ALIASES.items()}
    choices = tuple(CLAUDE_MODEL_ALIASES)
    if current in by_id:
        return choices, by_id[current]
    # A model configured by hand stays selectable instead of being silently dropped.
    return choices + (current,), current


def apply_llm_args(config: AppConfig, args: argparse.Namespace) -> None:
    """Apply LLM-related CLI flags onto the loaded config."""
    provider = getattr(args, "llm_provider", None)
    if provider:
        config.llm.provider = provider
    codex_model = getattr(args, "codex_model", None)
    if codex_model:
        config.codex.model = codex_model
    model = getattr(args, "model", None)
    if model:
        if config.llm.provider == "claude":
            config.llm.claude.model = resolve_claude_model(model)
        else:
            config.codex.model = model
    effort = getattr(args, "effort", None)
    if effort:
        config.llm.claude.effort = effort
        config.codex.effort = effort
    if getattr(args, "cached_search", False):
        config.llm.claude.live_search = False
        config.codex.live_search = False


def _apply_voice_set(config: AppConfig, args: argparse.Namespace) -> None:
    name = getattr(args, "voice_set", None)
    if not name:
        return
    if name not in config.tts.voice_sets:
        available = ", ".join(config.tts.voice_sets) or "keine"
        raise SystemExit(f"Unbekanntes Stimmen-Set '{name}'. Verfuegbar: {available}")
    config.tts.voice_set = name


def print_manifest(manifest) -> None:
    print(f"Run: {manifest.run_id}")
    for name, path in manifest.artifacts.items():
        print(f"{name}: {path}")
    if manifest.warnings:
        print("Warnings:")
        for warning in manifest.warnings:
            print(f"- {warning}")


def cmd_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.topic_file and args.topic:
        raise SystemExit("Bitte entweder ein Thema als Argument oder --topic-file angeben, nicht beides.")
    if args.topic_file:
        try:
            topic = _read_text_file(args.topic_file, label="Thema-Datei")
        except OSError as exc:
            raise SystemExit(f"Thema-Datei konnte nicht gelesen werden: {exc}") from exc
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.topic:
        topic = args.topic
    else:
        raise SystemExit("Bitte ein Thema als Argument oder --topic-file angeben.")
    selected_quality = args.quality or config.tts.quality
    config.language = args.language
    config.tts.quality = selected_quality
    config.tts.backend = backend_for_quality(config, selected_quality)
    selected_research_depth = args.research_depth or config.research.depth
    config.research.depth = selected_research_depth
    _apply_voice_set(config, args)
    apply_llm_args(config, args)
    generator = PodcastGenerator(config, project_root())
    generate_kwargs = {
        "topic": topic,
        "min_minutes": args.min_minutes,
        "max_minutes": args.max_minutes,
        "speaker_count": args.speakers,
        "quality": selected_quality,
        "language": args.language,
        "research_depth": selected_research_depth,
        "render_audio": not args.no_render,
    }
    if args.ui:
        try:
            PodcastTui().run_generator(generator, **generate_kwargs)
        except PodcastCancelled:
            return 130
        return 0
    manifest = generator.generate(
        **generate_kwargs,
    )
    print_manifest(manifest)
    return 0


def _prompt_text(prompt: str, *, input_func: Callable[[str], str] = input) -> str:
    while True:
        value = input_func(prompt).strip()
        if value:
            return value
        print("Bitte einen Wert eingeben.")


def _read_text_file(path: Path, *, label: str) -> str:
    text = path.expanduser().read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} ist leer: {path}")
    return text


def _topic_from_value(value: str) -> str:
    topic = value.strip()
    if topic.startswith(("@/", "@~", "@.")):
        return _read_text_file(Path(topic[1:]), label="Thema-Datei")
    return topic


def _prompt_topic(*, input_func: Callable[[str], str] = input) -> str:
    while True:
        value = _prompt_text("Thema: ", input_func=input_func)
        try:
            return _topic_from_value(value)
        except OSError as exc:
            print(f"Thema-Datei konnte nicht gelesen werden: {exc}")
        except ValueError as exc:
            print(str(exc))


def _prompt_int(
    prompt: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
    input_func: Callable[[str], str] = input,
) -> int:
    while True:
        raw = input_func(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Bitte eine ganze Zahl eingeben.")
            continue
        if value < minimum or value > maximum:
            print(f"Bitte einen Wert zwischen {minimum} und {maximum} eingeben.")
            continue
        return value


def _prompt_float(
    prompt: str,
    *,
    default: float,
    minimum: float,
    input_func: Callable[[str], str] = input,
) -> float:
    while True:
        raw = input_func(f"{prompt} [{default:g}]: ").strip().replace(",", ".")
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Bitte eine Zahl eingeben.")
            continue
        if value < minimum:
            print(f"Bitte einen Wert ab {minimum:g} eingeben.")
            continue
        return value


def _prompt_yes_no(
    prompt: str,
    *,
    default: bool = True,
    input_func: Callable[[str], str] = input,
) -> bool:
    suffix = "J/n" if default else "j/N"
    while True:
        raw = input_func(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"j", "ja", "y", "yes"}:
            return True
        if raw in {"n", "nein", "no"}:
            return False
        print("Bitte ja oder nein eingeben.")


def _prompt_option(
    prompt: str,
    *,
    choices: tuple[str, ...],
    default: str,
    descriptions: dict[str, str] | None = None,
    input_func: Callable[[str], str] = input,
) -> str:
    """Show every choice as a numbered line and accept a number, a name or a prefix."""
    print("")
    print(f"{prompt}:")
    width = max(len(choice) for choice in choices)
    for index, choice in enumerate(choices, start=1):
        marker = ">" if choice == default else " "
        description = (descriptions or {}).get(choice, "")
        suffix = f"  {description}" if description else ""
        label = choice.ljust(width) if description else choice
        print(f" {marker} {index}) {label}{suffix}")
    while True:
        raw = input_func(f"Auswahl [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
            print(f"Bitte 1-{len(choices)} eingeben.")
            continue
        matches = [choice for choice in choices if choice.lower().startswith(raw)]
        if len(matches) == 1:
            return matches[0]
        print(f"Bitte eine Nummer 1-{len(choices)} oder einen der Namen eingeben.")


def cmd_podcast_wizard(*, input_func: Callable[[str], str] = input) -> int:
    config_path = PROJECT_ROOT / "podcast.yml"
    if not config_path.exists():
        raise SystemExit(f"Zentrale Podcast-Konfiguration nicht gefunden: {config_path}")

    config = load_config(config_path)
    default_quality = config.tts.quality
    default_research_depth = config.research.depth
    default_language = config.language
    output_root = config.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    print("Podcast Wizard")
    print(f"Config: {config_path}")
    print("Tipp: Lange Themen als @/pfad/zur/thema.txt eingeben.")
    print("")

    state = _WizardState(
        topic=_prompt_topic(input_func=input_func),
        speakers=2,
        min_minutes=WIZARD_DEFAULT_MIN_MINUTES,
        max_minutes=WIZARD_DEFAULT_MAX_MINUTES,
        quality=default_quality,
        research_depth=default_research_depth,
        provider=config.llm.provider,
        claude_model=config.llm.claude.model,
        codex_model=config.codex.model,
        effort=config.llm.claude.effort or EFFORT_DEFAULT_LABEL,
        live_search=config.llm.claude.live_search,
        render_audio=True,
        language=default_language,
    )

    if not _run_settings_menu(state, config, output_root, input_func=input_func):
        print("Abgebrochen.")
        return 1

    config.language = state.language
    config.tts.quality = state.quality
    config.tts.backend = backend_for_quality(config, state.quality)
    if state.voice_set:
        config.tts.voice_set = state.voice_set
    config.research.depth = state.research_depth
    config.llm.provider = state.provider
    config.llm.claude.model = state.claude_model
    config.codex.model = state.codex_model
    config.llm.claude.effort = None if state.effort == EFFORT_DEFAULT_LABEL else state.effort
    config.codex.effort = config.llm.claude.effort
    config.llm.claude.live_search = state.live_search
    config.codex.live_search = state.live_search

    generator = PodcastGenerator(config, PROJECT_ROOT)
    try:
        PodcastTui().run_generator(
            generator,
            topic=state.topic,
            min_minutes=state.min_minutes,
            max_minutes=state.max_minutes,
            speaker_count=state.speakers,
            quality=state.quality,
            language=state.language,
            research_depth=state.research_depth,
            render_audio=state.render_audio,
        )
    except PodcastCancelled:
        return 130
    return 0


@dataclass
class _WizardState:
    topic: str
    speakers: int
    min_minutes: float
    max_minutes: float
    quality: str
    research_depth: str
    provider: str
    claude_model: str
    codex_model: str | None
    effort: str
    live_search: bool
    render_audio: bool
    language: str
    voice_set: str | None = None

    @property
    def model_label(self) -> str:
        if self.provider == "claude":
            return self.claude_model
        return self.codex_model or "CLI-Default"


def _topic_preview(topic: str, limit: int = 60) -> str:
    single_line = " ".join(topic.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 1]}…"


def _quality_label(state: _WizardState, config: AppConfig) -> str:
    """Make clear which backend runs, so `best` cannot silently mean paid OpenAI."""
    backend = backend_for_quality(config, state.quality)
    suffix = " (kostet Geld)" if backend == "openai" else ""
    if backend == state.quality:
        return f"{state.quality}{suffix}"
    return f"{state.quality} -> {backend}{suffix}"


def _voice_set_label(state: _WizardState, config: AppConfig) -> str:
    name = state.voice_set or config.tts.voice_set
    if not name:
        return "Standard (erste passende Stimmen)"
    members = config.tts.voice_sets.get(name, [])
    names = ", ".join(
        voice.display_name for voice in config.tts.voices if voice.id in members
    )
    return f"{name} ({names})" if names else name


def _edit_voice_set(state: _WizardState, config: AppConfig, input_func: Callable[[str], str]) -> None:
    choices = tuple(config.tts.voice_sets)
    descriptions = {
        name: ", ".join(voice.display_name for voice in config.tts.voices if voice.id in members)
        for name, members in config.tts.voice_sets.items()
    }
    state.voice_set = _prompt_option(
        "Stimmen-Besetzung",
        choices=choices,
        default=state.voice_set or config.tts.voice_set or choices[0],
        descriptions=descriptions,
        input_func=input_func,
    )


def _run_settings_menu(
    state: _WizardState,
    config: AppConfig,
    output_root: Path,
    *,
    input_func: Callable[[str], str] = input,
) -> bool:
    """Show every option as an editable row. Empty input starts the run."""
    rows: list[tuple[str, Callable[[], str], Callable[[], None]]] = [
        ("Thema", lambda: _topic_preview(state.topic), lambda: _edit_topic(state, input_func)),
        ("Sprecher", lambda: str(state.speakers), lambda: _edit_speakers(state, config, input_func)),
        (
            "Laenge",
            lambda: f"{state.min_minutes:g}-{state.max_minutes:g} Minuten",
            lambda: _edit_length(state, input_func),
        ),
        ("Audio-Qualitaet", lambda: _quality_label(state, config), lambda: _edit_quality(state, input_func)),
        ("Recherche-Tiefe", lambda: state.research_depth, lambda: _edit_depth(state, input_func)),
        ("LLM-Provider", lambda: state.provider, lambda: _edit_provider(state, input_func)),
        ("Modell", lambda: state.model_label, lambda: _edit_model(state, input_func)),
        ("Reasoning-Effort", lambda: state.effort, lambda: _edit_effort(state, input_func)),
        (
            "Live-Websuche",
            lambda: "ein" if state.live_search else "aus",
            lambda: _edit_live_search(state, input_func),
        ),
        (
            "Audio rendern",
            lambda: "ja" if state.render_audio else "nein (nur Transkript)",
            lambda: _edit_render(state, input_func),
        ),
        ("Sprache", lambda: state.language, lambda: _edit_language(state, input_func)),
    ]
    if config.tts.voice_sets:
        rows.insert(
            4,
            ("Stimmen", lambda: _voice_set_label(state, config), lambda: _edit_voice_set(state, config, input_func)),
        )

    while True:
        print("")
        print("Einstellungen:")
        width = max(len(label) for label, _, _ in rows)
        for index, (label, render, _) in enumerate(rows, start=1):
            print(f"  {index:>2}) {label.ljust(width)}  {render()}")
        print(f"      {'Ausgabe'.ljust(width)}  {output_root}")
        print("")
        raw = input_func("Nummer aendern, leer = Podcast starten, q = abbrechen: ").strip().lower()
        if not raw:
            return True
        if raw in {"q", "quit", "abbrechen", "n", "nein"}:
            return False
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            rows[int(raw) - 1][2]()
            continue
        print(f"Bitte 1-{len(rows)} eingeben, leer zum Starten oder q zum Abbrechen.")


def _edit_topic(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.topic = _prompt_topic(input_func=input_func)


def _edit_speakers(state: _WizardState, config: AppConfig, input_func: Callable[[str], str]) -> None:
    state.speakers = _prompt_int(
        "Sprecher",
        default=state.speakers,
        minimum=1,
        maximum=config.generation.max_speakers,
        input_func=input_func,
    )


def _edit_length(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.min_minutes = _prompt_float(
        "Mindestlaenge in Minuten",
        default=state.min_minutes,
        minimum=0.1,
        input_func=input_func,
    )
    while True:
        max_minutes = _prompt_float(
            "Maximallaenge in Minuten",
            default=max(state.max_minutes, state.min_minutes),
            minimum=0.1,
            input_func=input_func,
        )
        if max_minutes >= state.min_minutes:
            state.max_minutes = max_minutes
            return
        print("Die Maximallaenge muss groesser oder gleich der Mindestlaenge sein.")


def _edit_quality(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.quality = _prompt_option(
        "Audio-Qualitaet",
        choices=QUALITY_CHOICES,
        default=state.quality,
        descriptions=QUALITY_DESCRIPTIONS,
        input_func=input_func,
    )


def _edit_depth(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.research_depth = _prompt_option(
        "Recherche-Tiefe",
        choices=RESEARCH_DEPTH_CHOICES,
        default=state.research_depth,
        descriptions=RESEARCH_DEPTH_DESCRIPTIONS,
        input_func=input_func,
    )


def _edit_provider(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.provider = _prompt_option(
        "LLM-Provider",
        choices=LLM_PROVIDER_CHOICES,
        default=state.provider,
        descriptions=LLM_PROVIDER_DESCRIPTIONS,
        input_func=input_func,
    )


def _edit_model(state: _WizardState, input_func: Callable[[str], str]) -> None:
    if state.provider == "claude":
        choices, default = claude_model_choices(state.claude_model)
        state.claude_model = resolve_claude_model(
            _prompt_option(
                "Claude-Modell",
                choices=choices,
                default=default,
                descriptions=CLAUDE_MODEL_DESCRIPTIONS,
                input_func=input_func,
            )
        )
        return
    current = state.codex_model or "CLI-Default"
    raw = input_func(f"Codex-Modell (leer = unveraendert, '-' = CLI-Default) [{current}]: ").strip()
    if raw == "-":
        state.codex_model = None
    elif raw:
        state.codex_model = raw


def _edit_effort(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.effort = _prompt_option(
        "Reasoning-Effort",
        choices=(EFFORT_DEFAULT_LABEL,) + EFFORT_CHOICES,
        default=state.effort,
        descriptions=EFFORT_DESCRIPTIONS,
        input_func=input_func,
    )


def _edit_live_search(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.live_search = _prompt_yes_no(
        "Live-Websuche des LLM nutzen?",
        default=state.live_search,
        input_func=input_func,
    )


def _edit_render(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.render_audio = _prompt_yes_no(
        "Audio rendern? (nein = nur Transkript und Quellen)",
        default=state.render_audio,
        input_func=input_func,
    )


def _edit_language(state: _WizardState, input_func: Callable[[str], str]) -> None:
    state.language = input_func(f"Sprache [{state.language}]: ").strip() or state.language


def cmd_init_config(args: argparse.Namespace) -> int:
    path = args.config
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists; use --force to overwrite")
    path.write_text(__import__("yaml").safe_dump(default_config_dict(), sort_keys=False), encoding="utf-8")
    print(f"Wrote {path}")
    return 0


def cmd_voices_list(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not config.tts.voices:
        print("No voices configured.")
        return 0
    for voice in config.tts.voices:
        if voice.backend == "kokoro":
            path = voice.kokoro_model
        elif voice.backend in {"fish", "xtts"}:
            path = voice.speaker_wav
        else:
            path = None
        exists = bool(path and path.exists())
        location = "cloud" if path is None else str(path)
        print(
            f"{voice.id}\t{voice.display_name}\t{voice.backend}\t{voice.language}"
            f"\tlicense={voice.license}\texists={exists}\t{location}"
        )
    return 0


def cmd_voices_test(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    voice = next((item for item in config.tts.voices if item.id == args.voice_id), None)
    if voice is None:
        raise SystemExit(f"Unknown voice profile: {args.voice_id}")
    output = args.output or Path("voice-test") / f"{voice.id}.wav"
    render_voice_sample(config, voice, args.text, output)
    print(f"Wrote {output}")
    return 0


def cmd_voices_import(args: argparse.Namespace) -> int:
    config_path = args.config
    config = load_config(config_path)
    voice_id = args.name.strip().lower().replace(" ", "-")
    if not voice_id:
        raise SystemExit("Voice name cannot be empty")
    if args.backend != "chatterbox" and not args.transcript:
        raise SystemExit(f"--transcript is required for the {args.backend} backend")
    existing = next((voice for voice in config.tts.voices if voice.id == voice_id), None)
    source_wav = args.wav.expanduser().resolve()
    if not source_wav.exists():
        raise SystemExit(f"Reference WAV not found: {source_wav}")
    voice_dir = Path("voices") / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    dest = voice_dir / "ref.wav"
    shutil.copy2(source_wav, dest)

    profile = VoiceProfile(
        id=voice_id,
        display_name=args.display_name or (existing.display_name if existing else voice_id),
        backend=args.backend,
        language=args.language,
        license=args.license,
        speaker_wav=dest,
        ref_text=args.transcript,
    )
    config.tts.voices = [voice for voice in config.tts.voices if voice.id != voice_id] + [profile]
    write_config(config, config_path)
    print(f"Imported {args.backend} voice {voice_id} into {dest}")
    return 0


def cmd_voices_extract(args: argparse.Namespace) -> int:
    from .diarize import extract_voices

    config = load_config(args.config)
    out_dir = args.out or Path("voices/source_candidates") / args.source.stem
    report = extract_voices(
        config,
        args.source,
        out_dir,
        speakers=args.speakers,
        minutes=args.minutes,
        device=args.device,
        skip_head=args.skip_head,
        skip_tail=args.skip_tail,
    )

    between = report["verification"]["between_speakers"]
    cross = max(
        between[i][j] for i in range(len(between)) for j in range(len(between)) if i != j
    )
    print()
    print(f"Bericht: {out_dir / 'report.json'}")
    for detail, within in zip(report["speakers_detail"], report["verification"]["within_speaker"]):
        print(
            f"  {detail['label']}: {detail['seconds']:.0f}s aus {detail['chunk_count']} Chunks, "
            f"Margin {detail['margin_mean']:.3f}, "
            f"Selbstaehnlichkeit {within['mean']:.3f} (min {within['min']:.3f})"
        )
        print(f"    {detail['export_wav']}")
    # Der Wert entscheidet, ob das Material taugt: hoch heisst, die beiden
    # Ergebnisdateien klingen nach derselben Person.
    print(f"  Aehnlichkeit zwischen den Sprechern: {cross:.3f} (klein ist gut)")
    if cross > 0.5:
        print("  WARNUNG: Die Sprecher wurden nicht sauber getrennt. Ergebnis vor Gebrauch pruefen.")
    print("  Previews anhoeren, dann z.B.:")
    first = report["speakers_detail"][0]
    print(f"    uv run codcast voices import --backend chatterbox --name meine-stimme --wav {first['export_wav']}")
    return 0


def resolve_run_dir(run: Path, config) -> Path:
    if run.is_absolute():
        return run
    output_root = config.output_root
    if not output_root.is_absolute():
        output_root = project_root() / output_root
    candidates = [run, output_root / run, Path("runs") / run]
    return next(
        (
            candidate
            for candidate in candidates
            if (candidate / "script.json").exists()
            or (candidate / "manifest.json").exists()
            or (candidate / "inputs.json").exists()
        ),
        output_root / run,
    )


def cmd_resume(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_dir = resolve_run_dir(args.run, config)
    generator = PodcastGenerator(config, project_root())
    resume_kwargs = {
        "run_dir": run_dir,
        "render_audio": not args.no_render,
    }
    if args.ui:
        class ResumeAdapter:
            def generate(self, **kwargs):
                return generator.resume(**resume_kwargs, progress=kwargs.get("progress"), cancellation=kwargs.get("cancellation"))

        try:
            PodcastTui().run_generator(ResumeAdapter(), topic=run_dir.name)
        except PodcastCancelled:
            return 130
        return 0
    manifest = generator.resume(**resume_kwargs)
    print_manifest(manifest)
    return 0


def cmd_rerender(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    selected_quality = args.quality or config.tts.quality
    config.tts.quality = selected_quality
    config.tts.backend = backend_for_quality(config, selected_quality)
    _apply_voice_set(config, args)

    run_dir = resolve_run_dir(args.run, config)
    script_path = run_dir / "script.json"
    if not script_path.exists():
        raise SystemExit(f"No script.json found at {script_path}")

    script = PodcastScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    profiles = select_voice_profiles(config, len(script.speakers), selected_quality)
    script.speakers = [
        speaker.model_copy(update={"display_name": profile.display_name, "voice_profile_id": profile.id})
        for speaker, profile in zip(script.speakers, profiles, strict=True)
    ]

    suffix = args.suffix.strip().strip("-")
    if not suffix:
        raise SystemExit("Suffix cannot be empty")
    output_stem = f"{run_dir.name}-{suffix}"

    def progress(event) -> None:
        if event.phase not in {"tts", "assembly"}:
            return
        if event.current is not None and event.total is not None:
            print(f"{event.phase}: {event.current}/{event.total} {event.message}")
        else:
            print(f"{event.phase}: {event.message}")

    rendered = ScriptRenderer(config, voice_map(profiles)).render_script(script, run_dir, progress=progress)
    segments_path = run_dir / f"segments-{suffix}.json"
    write_json(segments_path, [item.model_dump(mode="json") for item in rendered])
    final_wav, final_mp3, duration_sec = assemble_episode(
        rendered,
        run_dir,
        config.audio,
        config.generation.pause_between_lines_sec,
        output_stem,
        progress=progress,
    )
    print(f"segments: {segments_path}")
    print(f"wav: {final_wav}")
    print(f"mp3: {final_mp3}")
    print(f"duration_sec: {duration_sec:.2f}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = args.run
    if not run_dir.is_absolute():
        config = load_config(args.config)
        run_dir = resolve_run_dir(run_dir, config)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"No manifest found at {manifest}")
    print(manifest.read_text(encoding="utf-8"))
    return 0


def cmd_setup_xtts(_: argparse.Namespace) -> int:
    print("Install optional XTTS dependencies with:")
    print("  uv sync --extra xtts")
    print("Then import at least one private/non-commercial reference voice:")
    print('  codcast voices import --backend xtts --name host-a --wav /path/ref.wav --transcript "..."')
    return 0


def cmd_setup_fish(_: argparse.Namespace) -> int:
    print("Fish S2 Pro Premium setup fuer lokale GPU:")
    print("  1. Install Fish Speech in a separate environment:")
    print("     git clone https://github.com/fishaudio/fish-speech.git ../fish-speech")
    print("     cd ../fish-speech")
    print("     uv sync --python 3.12 --extra cu128")
    print("  2. Download S2 Pro weights:")
    print("     uv run hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro")
    print("  3. Quantize the text model for 16GB local GPU use:")
    print("     uv run python tools/llama/quantize.py --checkpoint-path checkpoints/s2-pro --mode int8 --timestamp s2-pro")
    print("  4. Start the local Fish server:")
    print("     uv run python tools/api_server.py --llama-checkpoint-path checkpoints/fs-1.2-int8-s2-pro --decoder-checkpoint-path checkpoints/s2-pro/codec.pth --listen 127.0.0.1:8098 --half --max-seq-len 8192")
    print("     Full BF16/FP16 S2 Pro is still a 24GB+ VRAM path; the int8 path is the tested local-GPU setup.")
    print("  5. Import two high-quality German reference voices:")
    print('     uv run codcast voices import --backend fish --name fish-host-m --wav /path/male.wav --transcript "Exact spoken reference text"')
    print('     uv run codcast voices import --backend fish --name fish-host-f --wav /path/female.wav --transcript "Exact spoken reference text"')
    print("The premium path intentionally has no Kokoro/Piper fallback.")
    return 0


def cmd_setup_claude(_: argparse.Namespace) -> int:
    print("Claude als LLM-Provider (Standard):")
    print("  1. Claude CLI installieren: https://claude.com/product/claude-code")
    print("  2. Einmal 'claude' starten und mit dem Abo einloggen.")
    print("     Es wird KEIN API-Key benoetigt, dieses Projekt speichert auch keinen.")
    print("  3. Pruefen: claude --version")
    print("  4. Nutzung (Claude ist der Default):")
    print("     uv run codcast generate \"Thema\" --min-minutes 10 --max-minutes 15 --speakers 2 --ui")
    print("  5. Modell/Effort anpassen: --model claude-opus-5 --effort xhigh")
    print("  6. Auf den alten Provider ausweichen: --llm-provider codex")
    return 0


def cmd_setup_chatterbox(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    python_executable = config.tts.chatterbox.python_executable
    if not python_executable.is_absolute():
        python_executable = project_root() / python_executable
    venv_dir = python_executable.parent.parent

    print("Chatterbox Multilingual (lokal, MIT-Lizenz, klont Referenzstimmen)")
    print(f"  venv: {venv_dir}")
    if not command_available("uv"):
        print("  uv wurde nicht gefunden. Installiere uv oder lege die venv manuell an:")
        print(f"    python3.12 -m venv {venv_dir}")
        print(f"    {venv_dir}/bin/pip install chatterbox-tts 'setuptools<81'")
        return 1

    steps = [
        (["uv", "venv", "--python", "3.12", str(venv_dir)], "venv anlegen"),
        # setuptools<81 wird gebraucht, weil resemble-perth (Wasserzeichen) pkg_resources importiert.
        (["uv", "pip", "install", "--python", str(python_executable), "chatterbox-tts", "setuptools<81"], "Pakete installieren"),
    ]
    for command, label in steps:
        print(f"  {label}: {' '.join(command)}")
        result = subprocess.run(command, cwd=project_root())
        if result.returncode != 0:
            print(f"  Abbruch: {label} ist fehlgeschlagen.")
            return result.returncode

    check = subprocess.run(
        [str(python_executable), "-c", "import perth, chatterbox.mtl_tts; assert perth.PerthImplicitWatermarker; print('ok')"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print("  Installationspruefung fehlgeschlagen:")
        print(check.stderr.strip()[-800:])
        return check.returncode

    print("  Pruefung: ok")
    print("  Modellgewichte (ca. 3 GB) laedt der erste Lauf automatisch nach ~/.cache/huggingface.")
    print("  Zwei Referenzstimmen importieren (10 Sekunden sauberes Deutsch reichen):")
    print("     uv run codcast voices import --backend chatterbox --name chatterbox-host-m --wav /pfad/maennlich.wav")
    print("     uv run codcast voices import --backend chatterbox --name chatterbox-host-f --wav /pfad/weiblich.wav")
    print("  Danach: uv run codcast voices test chatterbox-host-m")
    return 0


def cmd_setup_diarize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    python_executable = config.diarize.python_executable
    if not python_executable.is_absolute():
        python_executable = project_root() / python_executable
    venv_dir = python_executable.parent.parent

    print("Sprechertrennung fuer `voices extract` (lokal, ungated, kein HF-Token noetig)")
    print(f"  venv: {venv_dir}")
    if not command_available("uv"):
        print("  uv wurde nicht gefunden. Installiere uv oder lege die venv manuell an:")
        print(f"    python3.12 -m venv {venv_dir}")
        print(f"    {venv_dir}/bin/pip install {' '.join(DIARIZE_PACKAGES)}")
        return 1

    steps = [
        (["uv", "venv", "--python", "3.12", str(venv_dir)], "venv anlegen"),
        (["uv", "pip", "install", "--python", str(python_executable), *DIARIZE_PACKAGES], "Pakete installieren"),
    ]
    for command, label in steps:
        print(f"  {label}: {' '.join(command)}")
        result = subprocess.run(command, cwd=project_root())
        if result.returncode != 0:
            print(f"  Abbruch: {label} ist fehlgeschlagen.")
            return result.returncode

    check = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import torch, sklearn, soundfile; from silero_vad import load_silero_vad; "
            "from speechbrain.inference.speaker import EncoderClassifier; print('ok')",
        ],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print("  Installationspruefung fehlgeschlagen:")
        print(check.stderr.strip()[-800:])
        return check.returncode

    print("  Pruefung: ok")
    print("  ECAPA-Gewichte (ca. 80 MB) laedt der erste Lauf nach models/spkrec-ecapa-voxceleb.")
    print("  Fuenf Minuten pro Sprecher aus einer Zwei-Sprecher-Aufnahme ziehen:")
    print("     uv run codcast voices extract /pfad/folge.mp3 --speakers 2 --minutes 5")
    return 0


def cmd_setup_openai(_: argparse.Namespace) -> int:
    print("OpenAI TTS setup:")
    print("  1. Lege den Key in die lokale, gitignorierte Datei:")
    print("     printf 'OPENAI_TTS_API_KEY=sk-...\\n' > .env.tts.local")
    print("     Alternativ reicht auch nur der rohe Key in dieser Datei.")
    print("     chmod 600 .env.tts.local")
    print("  2. Nutze OpenAI fuer den Audiopfad:")
    print("     uv run codcast generate \"Thema\" --min-minutes 10 --max-minutes 15 --speakers 2 --quality openai --ui")
    print("  3. Der Key-Name ist absichtlich OPENAI_TTS_API_KEY, damit er nur fuer TTS genutzt wird.")
    return 0


def build_parser(prog: str = "codcast") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate researched podcasts with the Claude or Codex CLI and local TTS.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Research and generate a podcast episode")
    add_common_config_arg(generate)
    generate.add_argument("topic", nargs="?", help="Topic or question")
    generate.add_argument("--topic-file", type=Path, default=None, help="Read a long topic from a UTF-8 text file")
    generate.add_argument("--min-minutes", type=float, required=True)
    generate.add_argument("--max-minutes", type=float, required=True)
    generate.add_argument("--speakers", type=int, default=2)
    generate.add_argument("--quality", choices=list(QUALITY_CHOICES), default=None)
    generate.add_argument("--voice-set", default=None, help="Named voice set from podcast.yml (tts.voice_sets)")
    generate.add_argument("--research-depth", choices=list(RESEARCH_DEPTH_CHOICES), default=None)
    generate.add_argument("--language", default="de-DE")
    add_llm_args(generate)
    generate.add_argument("--no-render", action="store_true", help="Stop after transcript and sources")
    generate.add_argument("--ui", action="store_true", help="Show a live terminal UI while generating")
    generate.set_defaults(func=cmd_generate)

    resume = sub.add_parser("resume", help="Resume an existing podcast run from saved artifacts")
    add_common_config_arg(resume)
    resume.add_argument("run", type=Path)
    resume.add_argument("--no-render", action="store_true", help="Stop after transcript and sources")
    resume.add_argument("--ui", action="store_true", help="Show a live terminal UI while resuming")
    resume.set_defaults(func=cmd_resume)

    init_config = sub.add_parser("init-config", help="Write a default podcast.yml")
    add_common_config_arg(init_config)
    init_config.add_argument("--force", action="store_true")
    init_config.set_defaults(func=cmd_init_config)

    voices = sub.add_parser("voices", help="Manage local voice profiles")
    voice_sub = voices.add_subparsers(dest="voice_command", required=True)
    voices_list = voice_sub.add_parser("list")
    add_common_config_arg(voices_list)
    voices_list.set_defaults(func=cmd_voices_list)

    voices_test = voice_sub.add_parser("test")
    add_common_config_arg(voices_test)
    voices_test.add_argument("voice_id")
    voices_test.add_argument("--text", default="Willkommen im Codex Podcast Studio.")
    voices_test.add_argument("--output", type=Path, default=None)
    voices_test.set_defaults(func=cmd_voices_test)

    voices_import = voice_sub.add_parser("import")
    add_common_config_arg(voices_import)
    voices_import.add_argument("--backend", choices=["chatterbox", "fish", "xtts"], default="chatterbox")
    voices_import.add_argument("--name", required=True)
    voices_import.add_argument("--wav", type=Path, required=True)
    voices_import.add_argument(
        "--transcript",
        default=None,
        help="Exakt gesprochener Referenztext, erforderlich fuer fish und xtts; Chatterbox braucht keinen",
    )
    voices_import.add_argument("--display-name", default=None)
    voices_import.add_argument("--language", default="de-DE")
    voices_import.add_argument("--license", default="personal")
    voices_import.set_defaults(func=cmd_voices_import)

    voices_extract = voice_sub.add_parser(
        "extract",
        help="Sprecher aus einer Aufnahme trennen und pro Person sauberes Referenzmaterial schneiden",
    )
    add_common_config_arg(voices_extract)
    voices_extract.add_argument("source", type=Path, help="Quelldatei (mp3, wav, m4a, ...)")
    voices_extract.add_argument("--speakers", type=int, default=2)
    voices_extract.add_argument("--minutes", type=float, default=5.0, help="Zielmaterial pro Sprecher")
    voices_extract.add_argument("--out", type=Path, default=None)
    voices_extract.add_argument("--device", default=None, help="cuda oder cpu; Default aus podcast.yml")
    voices_extract.add_argument(
        "--skip-head", type=float, default=0.0, help="Sekunden am Anfang ignorieren (Jingle)"
    )
    voices_extract.add_argument("--skip-tail", type=float, default=0.0, help="Sekunden am Ende ignorieren")
    voices_extract.set_defaults(func=cmd_voices_extract)

    rerender = sub.add_parser("rerender", help="Render audio again from an existing script.json without research")
    add_common_config_arg(rerender)
    rerender.add_argument("run", type=Path)
    rerender.add_argument("--quality", choices=list(QUALITY_CHOICES), default=None)
    rerender.add_argument("--voice-set", default=None, help="Named voice set from podcast.yml (tts.voice_sets)")
    rerender.add_argument("--suffix", default="openai-tts")
    rerender.set_defaults(func=cmd_rerender)

    inspect = sub.add_parser("inspect", help="Print a run manifest")
    add_common_config_arg(inspect)
    inspect.add_argument("run", type=Path)
    inspect.set_defaults(func=cmd_inspect)

    setup = sub.add_parser("setup-xtts", help="Print XTTS setup commands")
    setup.set_defaults(func=cmd_setup_xtts)

    setup_chatterbox = sub.add_parser("setup-chatterbox", help="Install the local Chatterbox voice environment")
    add_common_config_arg(setup_chatterbox)
    setup_chatterbox.set_defaults(func=cmd_setup_chatterbox)

    setup_diarize = sub.add_parser("setup-diarize", help="Install the local speaker-separation environment")
    add_common_config_arg(setup_diarize)
    setup_diarize.set_defaults(func=cmd_setup_diarize)

    setup_fish = sub.add_parser("setup-fish", help="Print Fish S2 Pro premium setup commands")
    setup_fish.set_defaults(func=cmd_setup_fish)

    setup_openai = sub.add_parser("setup-openai", help="Print OpenAI TTS setup commands")
    setup_openai.set_defaults(func=cmd_setup_openai)

    setup_claude = sub.add_parser("setup-claude", help="Print Claude LLM provider setup steps")
    setup_claude.set_defaults(func=cmd_setup_claude)
    return parser


def main(argv: list[str] | None = None, prog: str = "codcast") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    return args.func(args)


def podcast_main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        return main(args, prog="podcast")
    return cmd_podcast_wizard()
