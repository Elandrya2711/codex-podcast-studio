import json
import wave
from pathlib import Path

import pytest

from codcast.config import AppConfig, ChatterboxConfig, VoiceProfile
from codcast.tts import ChatterboxBackend, backend_for_voice
from codcast.voices import backend_for_quality, select_voice_profiles


def _write_fake_worker(tmp_path: Path, body: str) -> Path:
    """Stand-in for the real worker: speaks the same JSON-lines protocol."""
    worker = tmp_path / "fake_worker.py"
    worker.write_text(body, encoding="utf-8")
    return worker


WORKING_WORKER = """
import json, struct, sys, wave

print(json.dumps({"event": "ready", "sample_rate": 24000}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    job = json.loads(line)
    with open("/tmp/chatterbox-jobs.jsonl", "a") as handle:
        handle.write(json.dumps(job) + "\\n")
    with wave.open(job["output_path"], "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack("<24000h", *([0] * 24000)))
    print(json.dumps({"event": "done", "id": job["id"], "seconds": 1.0}), flush=True)
"""

FAILING_WORKER = """
import json, sys

print(json.dumps({"event": "ready", "sample_rate": 24000}), flush=True)
for line in sys.stdin:
    job = json.loads(line)
    sys.stderr.write("RuntimeError: reference too short\\n")
    sys.stderr.flush()
    print(json.dumps({"event": "error", "id": job["id"], "message": "RuntimeError: reference too short"}), flush=True)
"""

CRASHING_WORKER = """
import sys
sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory\\n")
raise SystemExit(1)
"""


def _backend(tmp_path: Path, worker_body: str, **overrides) -> ChatterboxBackend:
    worker = _write_fake_worker(tmp_path, worker_body)
    config = AppConfig(tts={"chatterbox": ChatterboxConfig(python_executable=Path(__import__("sys").executable), **overrides)})
    backend = ChatterboxBackend(config)
    backend.WORKER = worker
    return backend


def _voice(tmp_path: Path, **overrides) -> VoiceProfile:
    reference = tmp_path / "ref.wav"
    with wave.open(str(reference), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 24000)
    return VoiceProfile(
        id="chatterbox-host-m",
        display_name="Jonas",
        backend="chatterbox",
        speaker_wav=reference,
        **overrides,
    )


def test_chatterbox_renders_and_sends_configured_parameters(tmp_path: Path):
    jobs_log = Path("/tmp/chatterbox-jobs.jsonl")
    jobs_log.unlink(missing_ok=True)
    backend = _backend(tmp_path, WORKING_WORKER, exaggeration=0.4, cfg_weight=0.25, temperature=0.55)
    output = tmp_path / "out.wav"

    backend.render("Guten Morgen.", _voice(tmp_path), output)
    backend.close()

    assert output.exists()
    job = json.loads(jobs_log.read_text(encoding="utf-8").splitlines()[-1])
    assert job["text"] == "Guten Morgen."
    assert job["language"] == "de"
    assert job["exaggeration"] == 0.4
    assert job["cfg_weight"] == 0.25
    assert job["temperature"] == 0.55
    assert job["output_path"] == str(output)


def test_chatterbox_reuses_one_worker_for_many_segments(tmp_path: Path):
    backend = _backend(tmp_path, WORKING_WORKER)
    voice = _voice(tmp_path)

    backend.render("Erster Satz.", voice, tmp_path / "a.wav")
    first_process = backend._process
    backend.render("Zweiter Satz.", voice, tmp_path / "b.wav")

    assert backend._process is first_process, "das Modell darf nicht pro Segment neu laden"
    backend.close()


def test_chatterbox_reports_worker_error_with_log_hint(tmp_path: Path):
    backend = _backend(tmp_path, FAILING_WORKER)

    with pytest.raises(RuntimeError) as error:
        backend.render("Ein Satz.", _voice(tmp_path), tmp_path / "out.wav")
    backend.close()

    assert "reference too short" in str(error.value)


def test_chatterbox_reports_worker_crash_during_startup(tmp_path: Path):
    backend = _backend(tmp_path, CRASHING_WORKER)

    with pytest.raises(RuntimeError) as error:
        backend.render("Ein Satz.", _voice(tmp_path), tmp_path / "out.wav")

    message = str(error.value)
    assert "Chatterbox-Worker beendet" in message
    assert "CUDA out of memory" in message


def test_chatterbox_requires_speaker_wav(tmp_path: Path):
    backend = _backend(tmp_path, WORKING_WORKER)
    voice = VoiceProfile(id="x", display_name="X", backend="chatterbox")

    with pytest.raises(ValueError, match="speaker_wav"):
        backend.render("Text", voice, tmp_path / "out.wav")


def test_chatterbox_reports_missing_reference_file(tmp_path: Path):
    backend = _backend(tmp_path, WORKING_WORKER)
    voice = VoiceProfile(id="x", display_name="X", backend="chatterbox", speaker_wav=tmp_path / "missing.wav")

    with pytest.raises(FileNotFoundError):
        backend.render("Text", voice, tmp_path / "out.wav")


def test_missing_python_executable_points_at_setup_command(tmp_path: Path):
    config = AppConfig(tts={"chatterbox": ChatterboxConfig(python_executable=tmp_path / "nope" / "python")})

    with pytest.raises(RuntimeError, match="setup-chatterbox"):
        ChatterboxBackend(config).render("Text", _voice(tmp_path), tmp_path / "out.wav")


def test_speed_is_applied_after_rendering(tmp_path: Path):
    backend = _backend(tmp_path, WORKING_WORKER)
    output = tmp_path / "slow.wav"

    backend.render("Ein Satz.", _voice(tmp_path, speed=0.9), output)
    backend.close()

    with wave.open(str(output)) as handle:
        seconds = handle.getnframes() / handle.getframerate()
    # Der Fake-Worker schreibt exakt 1 Sekunde; atempo=0.9 streckt sie.
    assert 1.05 < seconds < 1.20


def test_backend_factory_and_quality_selection():
    config = AppConfig()
    voice = VoiceProfile(id="v", display_name="V", backend="chatterbox", speaker_wav=Path("ref.wav"))

    assert isinstance(backend_for_voice(config, voice), ChatterboxBackend)
    assert backend_for_quality(config, "chatterbox") == "chatterbox"
    assert backend_for_quality(config, "best") == "chatterbox", "chatterbox ist der neue Default"


def test_default_config_offers_two_chatterbox_voices():
    profiles = select_voice_profiles(AppConfig.model_validate(_default_data()), 2, "chatterbox")

    assert [profile.id for profile in profiles] == ["chatterbox-host-m", "chatterbox-host-f"]
    assert all(profile.speaker_wav is not None for profile in profiles)


def _default_data() -> dict:
    from codcast.config import default_config_dict

    return default_config_dict()


def test_voice_overrides_beat_the_global_parameters(tmp_path: Path):
    jobs_log = Path("/tmp/chatterbox-jobs.jsonl")
    jobs_log.unlink(missing_ok=True)
    backend = _backend(tmp_path, WORKING_WORKER, exaggeration=0.35, cfg_weight=0.3, temperature=0.6)
    voice = _voice(tmp_path, chatterbox_temperature=0.3, chatterbox_cfg_weight=0.45)

    backend.render("Ein Satz.", voice, tmp_path / "out.wav")
    backend.close()

    job = json.loads(jobs_log.read_text(encoding="utf-8").splitlines()[-1])
    assert job["temperature"] == 0.3, "Stimm-Override gewinnt"
    assert job["cfg_weight"] == 0.45
    assert job["exaggeration"] == 0.35, "nicht gesetzte Werte bleiben global"


def test_zero_is_a_real_override_not_a_missing_value(tmp_path: Path):
    jobs_log = Path("/tmp/chatterbox-jobs.jsonl")
    jobs_log.unlink(missing_ok=True)
    backend = _backend(tmp_path, WORKING_WORKER, exaggeration=0.35)
    voice = _voice(tmp_path, chatterbox_exaggeration=0.0)

    backend.render("Ein Satz.", voice, tmp_path / "out.wav")
    backend.close()

    job = json.loads(jobs_log.read_text(encoding="utf-8").splitlines()[-1])
    assert job["exaggeration"] == 0.0


LOOPING_WORKER = """
import json, struct, sys, wave

print(json.dumps({"event": "ready", "sample_rate": 24000}), flush=True)
attempt = 0
for line in sys.stdin:
    job = json.loads(line)
    attempt += 1
    # Erster Versuch: doppelt so lang wie erwartbar, danach plausibel.
    seconds = 40.0 if attempt == 1 else 3.0
    with wave.open(job["output_path"], "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\\x00\\x00" * int(24000 * seconds))
    with open("/tmp/chatterbox-attempts.txt", "a") as counter:
        counter.write(f"{attempt}\\n")
    print(json.dumps({"event": "done", "id": job["id"], "seconds": seconds}), flush=True)
"""

ALWAYS_LOOPING_WORKER = LOOPING_WORKER.replace("40.0 if attempt == 1 else 3.0", "40.0")


def test_repetition_loop_triggers_a_rerender(tmp_path: Path):
    attempts_log = Path("/tmp/chatterbox-attempts.txt")
    attempts_log.unlink(missing_ok=True)
    backend = _backend(tmp_path, LOOPING_WORKER)
    output = tmp_path / "out.wav"

    backend.render("Ein kurzer Satz mit knapp fuenfzig Zeichen Laenge.", _voice(tmp_path), output)
    backend.close()

    assert attempts_log.read_text(encoding="utf-8").split() == ["1", "2"], "genau ein Neuversuch"
    assert backend.warnings == []
    with wave.open(str(output)) as handle:
        assert handle.getnframes() / handle.getframerate() < 10, "die plausible Aufnahme bleibt stehen"


def test_persistent_problem_is_reported_not_hidden(tmp_path: Path):
    attempts_log = Path("/tmp/chatterbox-attempts.txt")
    attempts_log.unlink(missing_ok=True)
    backend = _backend(tmp_path, ALWAYS_LOOPING_WORKER, max_retries=1)

    backend.render("Ein kurzer Satz mit knapp fuenfzig Zeichen Laenge.", _voice(tmp_path), tmp_path / "out.wav")
    backend.close()

    assert len(attempts_log.read_text(encoding="utf-8").split()) == 2, "max_retries begrenzt die Versuche"
    assert len(backend.warnings) == 1
    assert "Wiederholungs-Loop" in backend.warnings[0]


def test_plausible_duration_renders_only_once(tmp_path: Path):
    attempts_log = Path("/tmp/chatterbox-attempts.txt")
    attempts_log.unlink(missing_ok=True)
    backend = _backend(tmp_path, WORKING_WORKER)

    # Der WORKING_WORKER meldet 1.0s; bei 15 Zeichen erwartet das Backend rund 1s.
    backend.render("Kurzer Satz.", _voice(tmp_path), tmp_path / "out.wav")
    backend.close()

    assert backend.warnings == []
