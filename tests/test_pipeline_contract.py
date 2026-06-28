import re

from codcast.config import load_config
from codcast.models import PodcastScript, ScriptLine
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
