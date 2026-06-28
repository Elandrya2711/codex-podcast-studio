import tomllib

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


def test_podcast_wizard_uses_defaults_and_central_project(monkeypatch, tmp_path):
    write_project_config(tmp_path)
    FakeGenerator.calls = []
    FakeTui.calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "PodcastTui", FakeTui)

    result = cli.cmd_podcast_wizard(input_func=input_from(["KI Thema", "", "", "", "", "", "", ""]))

    assert result == 0
    call = FakeTui.calls[0]
    assert call["generator"].project_root == tmp_path
    assert call["generator"].config.language == "de-DE"
    assert call["generator"].config.tts.quality == "fast"
    assert call["generator"].config.tts.backend == "kokoro"
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


def test_podcast_wizard_reprompts_invalid_numeric_values(monkeypatch, tmp_path):
    write_project_config(tmp_path)
    FakeGenerator.calls = []
    FakeTui.calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "PodcastTui", FakeTui)

    result = cli.cmd_podcast_wizard(input_func=input_from(["Thema", "0", "3", "5", "4", "6", "", "", "", ""]))

    assert result == 0
    assert FakeTui.calls[0]["kwargs"]["speaker_count"] == 3
    assert FakeTui.calls[0]["kwargs"]["min_minutes"] == 5.0
    assert FakeTui.calls[0]["kwargs"]["max_minutes"] == 6.0
    assert FakeTui.calls[0]["kwargs"]["research_depth"] == "standard"


def test_podcast_wizard_reads_topic_from_file_reference(monkeypatch, tmp_path):
    write_project_config(tmp_path)
    topic_file = tmp_path / "topic.txt"
    topic_file.write_text("Zeile eins\nZeile zwei\n", encoding="utf-8")
    FakeGenerator.calls = []
    FakeTui.calls = []
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "PodcastGenerator", FakeGenerator)
    monkeypatch.setattr(cli, "PodcastTui", FakeTui)

    result = cli.cmd_podcast_wizard(input_func=input_from([f"@{topic_file}", "", "", "", "", "", "", ""]))

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
