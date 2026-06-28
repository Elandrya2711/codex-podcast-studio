import re

import codcast.pipeline as pipeline
from codcast.config import load_config
from codcast.deep_research import ResearchProviderError
from codcast.models import PodcastScript, ResearchReport, RunManifest, ScriptLine, ValidationReport
from codcast.pipeline import PodcastGenerator
from codcast.voices import build_speaker_specs, select_voice_profiles


def test_run_dir_uses_central_podcast_folder_and_dated_topic_name(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    generator = PodcastGenerator(config, tmp_path)

    run_id, run_dir = generator._run_dir("KI & Softwareentwicklung?")

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-ki-softwareentwicklung", run_id)
    assert run_dir == tmp_path / "podcasts" / run_id
    assert run_dir.exists()


def test_run_dir_keeps_duplicate_topic_runs_unique(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    generator = PodcastGenerator(config, tmp_path)

    first_id, first_dir = generator._run_dir("KI & Softwareentwicklung?")
    second_id, second_dir = generator._run_dir("KI & Softwareentwicklung?")

    assert second_id.startswith(first_id)
    assert second_id != first_id
    assert second_dir != first_dir
    assert second_dir.exists()


def test_normalize_script_assigns_speaker_contract(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    profiles = select_voice_profiles(config, 1, "best")
    speakers = build_speaker_specs(profiles)
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=speakers,
        lines=[ScriptLine(speaker_id="s1", text="Hallo und willkommen.")],
    )
    generator = PodcastGenerator(config, tmp_path)
    normalized = generator._normalize_script(script, speakers, 3, 4, "de-DE")
    assert normalized.target_min_minutes == 3
    assert normalized.target_max_minutes == 4
    assert normalized.estimated_words == 3


def test_normalize_script_splits_long_tts_lines(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    profiles = select_voice_profiles(config, 1, "best")
    speakers = build_speaker_specs(profiles)
    long_text = (
        "Das ist ein laengerer Abschnitt ueber Twitch, YouTube und Anny the Duck, "
        "der in einem echten Podcast zwar inhaltlich zusammenhaengt, fuer das lokale "
        "Fish TTS aber nicht als ein einzelner langer Monolog gerendert werden soll. "
        "Deshalb muss die Pipeline solche Zeilen in kleinere natuerliche Abschnitte "
        "zerlegen, ohne Sprecher oder Claim-Verweise zu verlieren."
    )
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=speakers,
        lines=[ScriptLine(speaker_id="s1", text=long_text, claim_ids=["C1"])],
    )

    normalized = PodcastGenerator(config, tmp_path)._normalize_script(script, speakers, 1, 2, "de-DE")

    assert len(normalized.lines) > 1
    assert all(len(line.text) <= 220 for line in normalized.lines)
    assert all(line.speaker_id == "s1" for line in normalized.lines)
    assert all(line.claim_ids == ["C1"] for line in normalized.lines)


def test_generate_falls_back_to_standard_research_when_deep_provider_is_unavailable(monkeypatch, tmp_path):
    class UnavailableDeepResearch:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            raise ResearchProviderError("offline", code="local_search_unavailable")

    class FallbackRunner:
        def __init__(self):
            self.calls = []

        def run_structured(self, *, schema_name, output_path, model, **kwargs):
            self.calls.append({"schema_name": schema_name, "output_path": output_path, "kwargs": kwargs})
            if model is ResearchReport:
                result = ResearchReport(topic="Thema", language="de-DE", summary="Standard fallback")
            elif model is ValidationReport:
                result = ValidationReport(topic="Thema", pass_status="pass")
            elif model is PodcastScript:
                result = PodcastScript(
                    title="Titel",
                    topic="Thema",
                    target_min_minutes=0.01,
                    target_max_minutes=1.0,
                    speakers=[],
                    lines=[ScriptLine(speaker_id="s1", text="Hallo Welt.")],
                )
            else:
                raise AssertionError(model)
            output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result

    monkeypatch.setattr(pipeline, "DeepResearchEngine", UnavailableDeepResearch)
    config = load_config(tmp_path / "missing.yml")
    config.research.depth = "dossier"
    config.evidence.enabled = False
    generator = PodcastGenerator(config, tmp_path)
    runner = FallbackRunner()
    generator.runner = runner

    manifest = generator.generate(
        topic="Thema",
        min_minutes=0.01,
        max_minutes=1.0,
        speaker_count=1,
        quality="openai",
        language="de-DE",
        research_depth="dossier",
        render_audio=False,
    )

    assert manifest.research_depth == "standard"
    assert "local_search_unavailable" in manifest.warnings
    assert "deep_research_fallback_standard" in manifest.warnings
    assert [call["schema_name"] for call in runner.calls] == ["research", "validation", "script"]
    assert "live_search" not in runner.calls[0]["kwargs"]


def test_resume_rejects_manifest_run_id_path_traversal(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = RunManifest(
        run_id="../../outside/pwn",
        topic="Thema",
        language="de-DE",
        min_minutes=1,
        max_minutes=2,
        speakers=1,
        quality="best",
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")

    try:
        PodcastGenerator(config, tmp_path).resume(run_dir=run_dir, render_audio=False)
    except ValueError as exc:
        assert "Unsafe run_id" in str(exc)
    else:
        raise AssertionError("expected unsafe run_id to fail")
