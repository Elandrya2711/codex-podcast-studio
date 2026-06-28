from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable

from .audio import assemble_episode
from .config import VoiceProfile, default_config_dict, load_config, write_config
from .models import PodcastScript
from .pipeline import PodcastGenerator
from .progress import PodcastCancelled
from .tts import ScriptRenderer, render_voice_sample
from .ui import PodcastTui
from .util import write_json
from .voices import backend_for_quality, select_voice_profiles, voice_map


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALITY_CHOICES = ("best", "fast", "openai", "xtts")
RESEARCH_DEPTH_CHOICES = ("standard", "deep", "dossier")
WIZARD_DEFAULT_MIN_MINUTES = 10.0
WIZARD_DEFAULT_MAX_MINUTES = 15.0


def project_root() -> Path:
    return Path.cwd()


def add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("podcast.yml"), help="Path to podcast.yml")


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
    if args.codex_model:
        config.codex.model = args.codex_model
    if args.cached_search:
        config.codex.live_search = False
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


def _prompt_choice(
    prompt: str,
    *,
    choices: tuple[str, ...],
    default: str,
    input_func: Callable[[str], str] = input,
) -> str:
    choice_text = "/".join(choices)
    while True:
        raw = input_func(f"{prompt} ({choice_text}) [{default}]: ").strip().lower()
        if not raw:
            return default
        matches = [choice for choice in choices if choice.startswith(raw)]
        if len(matches) == 1:
            return matches[0]
        print(f"Bitte einen dieser Werte eingeben: {choice_text}.")


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

    topic = _prompt_topic(input_func=input_func)
    speakers = _prompt_int(
        "Sprecher",
        default=2,
        minimum=1,
        maximum=config.generation.max_speakers,
        input_func=input_func,
    )
    min_minutes = _prompt_float(
        "Mindestlaenge in Minuten",
        default=WIZARD_DEFAULT_MIN_MINUTES,
        minimum=0.1,
        input_func=input_func,
    )
    while True:
        max_minutes = _prompt_float(
            "Maximallaenge in Minuten",
            default=WIZARD_DEFAULT_MAX_MINUTES,
            minimum=0.1,
            input_func=input_func,
        )
        if max_minutes >= min_minutes:
            break
        print("Die Maximallaenge muss groesser oder gleich der Mindestlaenge sein.")
    quality = _prompt_choice(
        "Qualitaet",
        choices=QUALITY_CHOICES,
        default=default_quality,
        input_func=input_func,
    )
    research_depth = _prompt_choice(
        "Recherche-Tiefe",
        choices=RESEARCH_DEPTH_CHOICES,
        default=default_research_depth,
        input_func=input_func,
    )
    language = input_func(f"Sprache [{default_language}]: ").strip() or default_language

    print("")
    print("Zusammenfassung:")
    print(f"Thema: {topic}")
    print(f"Sprecher: {speakers}")
    print(f"Laenge: {min_minutes:g}-{max_minutes:g} Minuten")
    print(f"Qualitaet: {quality}")
    print(f"Recherche-Tiefe: {research_depth}")
    print(f"Sprache: {language}")
    print(f"Ausgabe: {output_root}")
    if not _prompt_yes_no("Podcast jetzt starten?", default=True, input_func=input_func):
        print("Abgebrochen.")
        return 1

    config.language = language
    config.tts.quality = quality
    config.tts.backend = backend_for_quality(config, quality)
    config.research.depth = research_depth
    generator = PodcastGenerator(config, PROJECT_ROOT)
    try:
        PodcastTui().run_generator(
            generator,
            topic=topic,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            speaker_count=speakers,
            quality=quality,
            language=language,
            research_depth=research_depth,
            render_audio=True,
        )
    except PodcastCancelled:
        return 130
    return 0


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
    parser = argparse.ArgumentParser(prog=prog, description="Generate researched podcasts with Codex CLI and local TTS.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Research and generate a podcast episode")
    add_common_config_arg(generate)
    generate.add_argument("topic", nargs="?", help="Topic or question")
    generate.add_argument("--topic-file", type=Path, default=None, help="Read a long topic from a UTF-8 text file")
    generate.add_argument("--min-minutes", type=float, required=True)
    generate.add_argument("--max-minutes", type=float, required=True)
    generate.add_argument("--speakers", type=int, default=2)
    generate.add_argument("--quality", choices=list(QUALITY_CHOICES), default=None)
    generate.add_argument("--research-depth", choices=list(RESEARCH_DEPTH_CHOICES), default=None)
    generate.add_argument("--language", default="de-DE")
    generate.add_argument("--codex-model", default=None)
    generate.add_argument("--cached-search", action="store_true", help="Use Codex cached search instead of live --search")
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
    voices_import.add_argument("--backend", choices=["fish", "xtts"], default="fish")
    voices_import.add_argument("--name", required=True)
    voices_import.add_argument("--wav", type=Path, required=True)
    voices_import.add_argument("--transcript", required=True)
    voices_import.add_argument("--display-name", default=None)
    voices_import.add_argument("--language", default="de-DE")
    voices_import.add_argument("--license", default="personal")
    voices_import.set_defaults(func=cmd_voices_import)

    rerender = sub.add_parser("rerender", help="Render audio again from an existing script.json without research")
    add_common_config_arg(rerender)
    rerender.add_argument("run", type=Path)
    rerender.add_argument("--quality", choices=list(QUALITY_CHOICES), default=None)
    rerender.add_argument("--suffix", default="openai-tts")
    rerender.set_defaults(func=cmd_rerender)

    inspect = sub.add_parser("inspect", help="Print a run manifest")
    add_common_config_arg(inspect)
    inspect.add_argument("run", type=Path)
    inspect.set_defaults(func=cmd_inspect)

    setup = sub.add_parser("setup-xtts", help="Print XTTS setup commands")
    setup.set_defaults(func=cmd_setup_xtts)

    setup_fish = sub.add_parser("setup-fish", help="Print Fish S2 Pro premium setup commands")
    setup_fish.set_defaults(func=cmd_setup_fish)

    setup_openai = sub.add_parser("setup-openai", help="Print OpenAI TTS setup commands")
    setup_openai.set_defaults(func=cmd_setup_openai)
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
