import tomllib
from pathlib import Path

import pytest

import codcast.cli as cli
from codcast.models import PodcastScript, RenderedSegment, RunManifest, ScriptLine, SpeakerSpec


class FakeGenerator:
    calls = []

    def __init__(self, config, project_root):
        self.config = config
        self.project_root = project_root

    def generate(self, **kwargs):
        self.__class__.calls.append(
            {
                "method": "generate",
                "config": self.config,
                "project_root": self.project_root,
                "kwargs": kwargs,
            }
        )
        return RunManifest(
            run_id="test-run",
            topic=kwargs["topic"],
            language=kwargs["language"],
            min_minutes=kwargs["min_minutes"],
            max_minutes=kwargs["max_minutes"],
            speakers=kwargs["speaker_count"],
            quality=kwargs["quality"],
        )

    def resume(self, **kwargs):
        self.__class__.calls.append(
            {
                "method": "resume",
                "config": self.config,
                "project_root": self.project_root,
                "kwargs": kwargs,
            }
        )
        return RunManifest(
            run_id="test-run",
            topic="resumed",
            language="de-DE",
            min_minutes=3,
            max_minutes=5,
            speakers=2,
            quality="fast",
        )


class FakeTui:
    calls = []

    def run_generator(self, generator, **kwargs):
        self.__class__.calls.append({"generator": generator, "kwargs": kwargs})
        return RunManifest(
            run_id="test-run",
            topic=kwargs["topic"],
            language=kwargs["language"],
            min_minutes=kwargs["min_minutes"],
            max_minutes=kwargs["max_minutes"],
            speakers=kwargs["speaker_count"],
            quality=kwargs["quality"],
        )


def input_from(values):
    iterator = iter(values)

    def fake_input(_prompt):
        return next(iterator)

    return fake_input


def write_project_config(project_root):
    (project_root / "podcast.yml").write_text(
        "language: de-DE\n"
        "output_root: podcasts\n"
        "generation:\n"
        "  max_speakers: 4\n"
        "tts:\n"
        "  quality: fast\n"
        "  backend: kokoro\n",
        encoding="utf-8",
    )


def _wizard(monkeypatch, tmp_path, answers):
    """Run the wizard with a scripted answer list.

    Menu rows: 1 Thema, 2 Sprecher, 3 Laenge, 4 Audio-Qualitaet, 5 Recherche-Tiefe,
    6 LLM-Provider, 7 Modell, 8 Reasoning-Effort, 9 Live-Websuche, 10 Audio rendern,
    11 Sprache. An empty answer at the menu starts the run.
    """
    write_project_config(tmp_path)
    FakeGenerator.calls = []
    FakeTui.calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "PodcastTui", FakeTui)
    return cli.cmd_podcast_wizard(input_func=input_from(answers))


def test_podcast_wizard_starts_with_defaults_after_topic(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", ""])

    assert result == 0
    call = FakeTui.calls[0]
    assert call["generator"].project_root == tmp_path
    assert call["generator"].config.language == "de-DE"
    assert call["generator"].config.tts.quality == "fast"
    assert call["generator"].config.tts.backend == "kokoro"
    assert call["generator"].config.llm.provider == "claude"
    assert call["generator"].config.llm.claude.model == "claude-opus-5"
    assert call["kwargs"] == {
        "topic": "KI Thema",
        "min_minutes": 10.0,
        "max_minutes": 15.0,
        "speaker_count": 2,
        "quality": "fast",
        "language": "de-DE",
        "research_depth": "standard",
        "render_audio": True,
    }


def test_podcast_wizard_can_select_claude_fable_by_name(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "7", "fable", ""])

    assert result == 0
    assert FakeTui.calls[0]["generator"].config.llm.claude.model == "claude-fable-5"


def test_podcast_wizard_can_select_claude_fable_by_number(monkeypatch, tmp_path):
    # Row 7 is the model, option 2 in that submenu is fable.
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "7", "2", ""])

    assert result == 0
    assert FakeTui.calls[0]["generator"].config.llm.claude.model == "claude-fable-5"


def test_podcast_wizard_switches_provider_and_asks_for_codex_model(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "6", "codex", "7", "gpt-x", ""])

    assert result == 0
    config = FakeTui.calls[0]["generator"].config
    assert config.llm.provider == "codex"
    assert config.codex.model == "gpt-x"
    # The Claude model stays untouched so switching back keeps the choice.
    assert config.llm.claude.model == "claude-opus-5"


def test_podcast_wizard_covers_effort_live_search_and_render(monkeypatch, tmp_path):
    answers = [
        "KI Thema",
        "8", "xhigh",   # Reasoning-Effort
        "9", "n",       # Live-Websuche aus
        "10", "n",      # kein Audio-Rendering
        "",
    ]
    result = _wizard(monkeypatch, tmp_path, answers)

    assert result == 0
    config = FakeTui.calls[0]["generator"].config
    assert config.llm.claude.effort == "xhigh"
    assert config.codex.effort == "xhigh"
    assert config.llm.claude.live_search is False
    assert config.codex.live_search is False
    assert FakeTui.calls[0]["kwargs"]["render_audio"] is False


def test_podcast_wizard_effort_standard_maps_to_none(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "8", "standard", ""])

    assert result == 0
    assert FakeTui.calls[0]["generator"].config.llm.claude.effort is None


def test_podcast_wizard_can_change_quality_and_depth(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "4", "openai", "5", "dossier", ""])

    assert result == 0
    kwargs = FakeTui.calls[0]["kwargs"]
    assert kwargs["quality"] == "openai"
    assert kwargs["research_depth"] == "dossier"
    assert FakeTui.calls[0]["generator"].config.tts.backend == "openai"


def test_podcast_wizard_can_change_topic_from_the_menu(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["Erstes Thema", "1", "Zweites Thema", ""])

    assert result == 0
    assert FakeTui.calls[0]["kwargs"]["topic"] == "Zweites Thema"


def test_podcast_wizard_quit_cancels_without_running(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "q"])

    assert result == 1
    assert FakeTui.calls == []


def test_podcast_wizard_rejects_out_of_range_menu_entry(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "99", "nonsense", ""])

    assert result == 0
    assert FakeTui.calls[0]["kwargs"]["topic"] == "KI Thema"


def test_podcast_wizard_reprompts_invalid_numeric_values(monkeypatch, tmp_path):
    answers = [
        "Thema",
        "2", "0", "3",        # Sprecher: 0 unzulaessig, dann 3
        "3", "5", "4", "6",   # Laenge: max 4 < min 5, dann 6
        "",
    ]
    result = _wizard(monkeypatch, tmp_path, answers)

    assert result == 0
    kwargs = FakeTui.calls[0]["kwargs"]
    assert kwargs["speaker_count"] == 3
    assert kwargs["min_minutes"] == 5.0
    assert kwargs["max_minutes"] == 6.0


def test_podcast_wizard_reads_topic_from_file_reference(monkeypatch, tmp_path):
    write_project_config(tmp_path)
    topic_file = tmp_path / "topic.txt"
    topic_file.write_text("Zeile eins\nZeile zwei\n", encoding="utf-8")
    FakeGenerator.calls = []
    FakeTui.calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "PodcastTui", FakeTui)

    result = cli.cmd_podcast_wizard(input_func=input_from([f"@{topic_file}", ""]))

    assert result == 0
    assert FakeTui.calls[0]["kwargs"]["topic"] == "Zeile eins\nZeile zwei"


def test_generate_ui_uses_tui_runner(monkeypatch, tmp_path):
    FakeTui.calls = []
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "PodcastTui", FakeTui)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    args = cli.build_parser().parse_args(
        [
            "generate",
            "UI Thema",
            "--config",
            str(tmp_path / "missing.yml"),
            "--min-minutes",
            "3",
            "--max-minutes",
            "5",
            "--research-depth",
            "deep",
            "--ui",
        ]
    )

    result = cli.cmd_generate(args)

    assert result == 0
    assert FakeTui.calls[0]["generator"].project_root == tmp_path
    assert FakeTui.calls[0]["kwargs"]["topic"] == "UI Thema"
    assert FakeTui.calls[0]["kwargs"]["min_minutes"] == 3
    assert FakeTui.calls[0]["kwargs"]["max_minutes"] == 5
    assert FakeTui.calls[0]["kwargs"]["research_depth"] == "deep"
    assert FakeTui.calls[0]["generator"].config.research.depth == "deep"
    assert FakeTui.calls[0]["kwargs"]["render_audio"] is True


def test_generate_reads_topic_file(monkeypatch, tmp_path):
    FakeGenerator.calls = []
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    topic_file = tmp_path / "topic.txt"
    topic_file.write_text("Langes Thema\nmit Kontext\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--topic-file",
            str(topic_file),
            "--config",
            str(tmp_path / "missing.yml"),
            "--min-minutes",
            "3",
            "--max-minutes",
            "5",
            "--no-render",
        ]
    )

    result = cli.cmd_generate(args)

    assert result == 0
    assert FakeGenerator.calls[0]["kwargs"]["topic"] == "Langes Thema\nmit Kontext"


def test_resume_uses_inputs_only_run_dir(monkeypatch, tmp_path):
    FakeGenerator.calls = []
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    run_dir = tmp_path / "podcasts" / "failed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "inputs.json").write_text("{}", encoding="utf-8")
    (tmp_path / "podcast.yml").write_text("output_root: podcasts\n", encoding="utf-8")
    args = cli.build_parser().parse_args(
        [
            "resume",
            "failed-run",
            "--config",
            str(tmp_path / "podcast.yml"),
            "--no-render",
        ]
    )

    result = cli.cmd_resume(args)

    assert result == 0
    assert FakeGenerator.calls[0]["method"] == "resume"
    assert FakeGenerator.calls[0]["kwargs"]["run_dir"] == run_dir
    assert FakeGenerator.calls[0]["kwargs"]["render_audio"] is False


def test_rerender_uses_existing_script_without_generator(monkeypatch, tmp_path):
    run_dir = tmp_path / "podcasts" / "test-run"
    run_dir.mkdir(parents=True)
    (tmp_path / "podcast.yml").write_text("output_root: podcasts\n", encoding="utf-8")
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=[SpeakerSpec(id="s1", display_name="Old", role="Host", voice_profile_id="martin")],
        lines=[ScriptLine(speaker_id="s1", text="Hallo Welt.")],
    )
    (run_dir / "script.json").write_text(script.model_dump_json(), encoding="utf-8")
    calls = {}

    class FakeRenderer:
        def __init__(self, config, voices):
            calls["backend"] = config.tts.backend
            calls["voices"] = list(voices)

        def render_script(self, script, run_dir_arg, progress=None):
            calls["speaker_voice"] = script.speakers[0].voice_profile_id
            return [
                RenderedSegment(
                    index=1,
                    speaker_id="s1",
                    voice_profile_id=script.speakers[0].voice_profile_id,
                    text="Hallo Welt.",
                    wav_path=str(run_dir_arg / "segments" / "0001_s1_openai-cedar.wav"),
                )
            ]

    def fake_assemble_episode(rendered, run_dir_arg, audio_config, pause_between_lines_sec, output_stem, progress=None):
        calls["output_stem"] = output_stem
        return run_dir_arg / f"{output_stem}.wav", run_dir_arg / f"{output_stem}.mp3", 1.23

    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "ScriptRenderer", FakeRenderer)
    monkeypatch.setattr(cli, "assemble_episode", fake_assemble_episode)
    args = cli.build_parser().parse_args(
        [
            "rerender",
            "test-run",
            "--config",
            str(tmp_path / "podcast.yml"),
            "--quality",
            "openai",
            "--suffix",
            "openai-test",
        ]
    )

    result = cli.cmd_rerender(args)

    assert result == 0
    assert calls["backend"] == "openai"
    assert calls["voices"] == ["openai-cedar"]
    assert calls["speaker_voice"] == "openai-cedar"
    assert calls["output_stem"] == "test-run-openai-test"
    assert (run_dir / "segments-openai-test.json").exists()


def test_pyproject_exposes_podcast_entrypoint():
    data = tomllib.loads((cli.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["podcast"] == "codcast.cli:podcast_main"


def test_quality_label_names_the_effective_backend():
    from codcast.config import AppConfig

    state = cli._WizardState(
        topic="T",
        speakers=2,
        min_minutes=10,
        max_minutes=15,
        quality="best",
        research_depth="standard",
        provider="claude",
        claude_model="claude-opus-5",
        codex_model=None,
        effort=cli.EFFORT_DEFAULT_LABEL,
        live_search=True,
        render_audio=True,
        language="de-DE",
    )
    config = AppConfig()

    assert cli._quality_label(state, config) == "best -> chatterbox"

    config.tts.backend = "openai"
    assert cli._quality_label(state, config) == "best -> openai (kostet Geld)"

    state.quality = "openai"
    assert cli._quality_label(state, config) == "openai (kostet Geld)"

    state.quality = "chatterbox"
    assert cli._quality_label(state, config) == "chatterbox"


def test_podcast_wizard_can_switch_to_local_chatterbox(monkeypatch, tmp_path):
    result = _wizard(monkeypatch, tmp_path, ["KI Thema", "4", "chatterbox", ""])

    assert result == 0
    assert FakeTui.calls[0]["generator"].config.tts.quality == "chatterbox"
    assert FakeTui.calls[0]["generator"].config.tts.backend == "chatterbox"


def _rerender_run_dir(tmp_path) -> Path:
    run_dir = tmp_path / "podcasts" / "test-run"
    run_dir.mkdir(parents=True)
    (tmp_path / "podcast.yml").write_text("output_root: podcasts\n", encoding="utf-8")
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=[SpeakerSpec(id="s1", display_name="Old", role="Host", voice_profile_id="alt")],
        lines=[ScriptLine(speaker_id="s1", text="Hallo Welt.")],
    )
    (run_dir / "script.json").write_text(script.model_dump_json(), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize(
    ("extra_args", "expected_stem", "expected_reuse"),
    [
        ([], "test-run-chatterbox", False),
        (["--suffix", "zweiter-take"], "test-run-zweiter-take", False),
        (["--reuse-segments"], "test-run-chatterbox", True),
    ],
    ids=["standard-nennt-das-backend", "eigener-suffix", "fortsetzen-erlaubt"],
)
def test_rerender_names_output_after_the_backend_and_defaults_to_fresh_takes(
    monkeypatch, tmp_path, extra_args, expected_stem, expected_reuse
):
    run_dir = _rerender_run_dir(tmp_path)
    calls = {}

    class FakeRenderer:
        def __init__(self, config, voices):
            calls["reuse_segments"] = config.tts.reuse_segments

        def render_script(self, script, run_dir_arg, progress=None):
            return [
                RenderedSegment(
                    index=1,
                    speaker_id="s1",
                    voice_profile_id=script.speakers[0].voice_profile_id,
                    text="Hallo Welt.",
                    wav_path=str(run_dir_arg / "segments" / "0001.wav"),
                )
            ]

    def fake_assemble_episode(
        rendered, run_dir_arg, audio_config, pause_between_lines_sec, output_stem, progress=None
    ):
        calls["output_stem"] = output_stem
        return run_dir_arg / f"{output_stem}.wav", run_dir_arg / f"{output_stem}.mp3", 1.23

    monkeypatch.setattr(cli, "project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, "ScriptRenderer", FakeRenderer)
    monkeypatch.setattr(cli, "assemble_episode", fake_assemble_episode)
    args = cli.build_parser().parse_args(
        ["rerender", "test-run", "--config", str(tmp_path / "podcast.yml"), "--quality", "chatterbox"]
        + extra_args
    )

    assert cli.cmd_rerender(args) == 0
    assert calls["output_stem"] == expected_stem
    # Ein rerender soll neu sprechen, sonst ist der Befehl bei gleicher Eingabe wirkungslos.
    assert calls["reuse_segments"] is expected_reuse
    assert (run_dir / f"segments-{expected_stem.removeprefix('test-run-')}.json").exists()
