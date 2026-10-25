import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from codcast.config import AppConfig, PhononConfig, VoiceProfile, load_config
from codcast.tts import PhononBackend, backend_for_voice
from codcast.voices import backend_for_quality, select_voice_profiles

WORKER_PATH = Path(__file__).parents[1] / "src" / "codcast" / "workers" / "phonon_worker.py"

# Spricht dasselbe Protokoll wie der echte Worker und protokolliert Start und Jobs.
FAKE_WORKER = """
import json, struct, sys, wave

settings = json.loads(sys.argv[1])
with open(LOG, "a") as handle:
    handle.write(json.dumps({"start": settings}) + "\\n")
print(json.dumps({"event": "ready", "sample_rate": 24000}), flush=True)
for line in sys.stdin:
    job = json.loads(line)
    with open(LOG, "a") as handle:
        handle.write(json.dumps({"job": job}) + "\\n")
    if job["voice"] == "Kaputt":
        print(json.dumps({"event": "error", "id": job["id"], "message": "ValueError: unknown voice Kaputt"}), flush=True)
        continue
    with wave.open(job["output_path"], "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack("<2400h", *([0] * 2400)))
    print(json.dumps({"event": "done", "id": job["id"], "seconds": 0.1}), flush=True)
"""


def _backend(tmp_path: Path, **overrides) -> tuple[PhononBackend, Path]:
    log = tmp_path / "worker.jsonl"
    worker = tmp_path / "fake_phonon_worker.py"
    worker.write_text(f"LOG = {str(log)!r}\n" + FAKE_WORKER, encoding="utf-8")
    config = AppConfig(tts={"phonon": PhononConfig(python_executable=Path(sys.executable), **overrides)})
    backend = PhononBackend(config)
    backend.WORKER = worker
    return backend, log


def _voice(name: str = "Henrik") -> VoiceProfile:
    return VoiceProfile(id=f"phonon-{name.lower()}", display_name=name, backend="phonon", phonon_voice=name)


def _log(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_phonon_loads_the_model_once_for_all_segments(tmp_path: Path):
    backend, log = _backend(tmp_path, model="/modelle/phonon", threads=4, temperature=0.2)
    try:
        backend.render("Erster Satz.", _voice("Henrik"), tmp_path / "a.wav")
        backend.render("Zweiter Satz.", _voice("Nele"), tmp_path / "b.wav")
    finally:
        backend.close()

    entries = _log(log)
    starts = [entry["start"] for entry in entries if "start" in entry]
    jobs = [entry["job"] for entry in entries if "job" in entry]
    assert starts == [{"model": "/modelle/phonon", "lang": "de", "quant": "q8_0", "threads": 4}]
    assert [(job["voice"], job["text"]) for job in jobs] == [("Henrik", "Erster Satz."), ("Nele", "Zweiter Satz.")]
    assert jobs[0]["temperature"] == 0.2
    assert (tmp_path / "a.wav").exists() and (tmp_path / "b.wav").exists()


def test_phonon_leaves_number_normalization_to_ptts(tmp_path: Path):
    backend, log = _backend(tmp_path)
    try:
        backend.render("Das kostet 1.299,50 Euro.", _voice(), tmp_path / "a.wav")
    finally:
        backend.close()

    job = next(entry["job"] for entry in _log(log) if "job" in entry)
    assert job["text"] == "Das kostet 1.299,50 Euro."


def test_phonon_reports_worker_errors(tmp_path: Path):
    backend, _log_path = _backend(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="unknown voice Kaputt"):
            backend.render("Hallo.", _voice("Kaputt"), tmp_path / "a.wav")
    finally:
        backend.close()


def test_phonon_requires_a_builtin_voice(tmp_path: Path):
    backend, _log_path = _backend(tmp_path)
    voice = VoiceProfile(id="phonon-leer", display_name="Leer", backend="phonon")
    with pytest.raises(ValueError, match="phonon_voice"):
        backend.render("Hallo.", voice, tmp_path / "a.wav")


def test_phonon_points_to_setup_when_the_venv_is_missing(tmp_path: Path):
    config = AppConfig(tts={"phonon": PhononConfig(python_executable=tmp_path / "fehlt" / "python")})
    with pytest.raises(RuntimeError, match="setup-phonon"):
        PhononBackend(config).render("Hallo.", _voice(), tmp_path / "a.wav")


def test_real_worker_speaks_the_protocol(tmp_path: Path):
    """Der echte Worker gegen ein Ersatz-ptts, damit der Test ohne Modell laeuft."""
    fake_ptts = tmp_path / "fake" / "ptts"
    fake_ptts.mkdir(parents=True)
    (fake_ptts / "__init__.py").write_text(
        "THREADS = []\n"
        "def set_num_threads(n):\n"
        "    THREADS.append(n)\n"
        "class TTS:\n"
        "    sample_rate = 24000\n"
        "    def __init__(self, model, *, lang, quant):\n"
        "        assert (model, lang, quant) == ('/m', 'de', 'q8_0'), (model, lang, quant)\n"
        "    def save(self, path, text, *, voice=None, temperature=None):\n"
        "        if voice != 'Henrik':\n"
        "            raise ValueError(f'unknown voice {voice}')\n"
        "        open(path, 'wb').write(b'RIFF')\n"
        "        return 1.5\n",
        encoding="utf-8",
    )
    jobs = [
        {"id": 1, "text": "Hallo.", "voice": "Henrik", "output_path": str(tmp_path / "a.wav"), "temperature": None},
        {"id": 2, "text": "Hallo.", "voice": "Niemand", "output_path": str(tmp_path / "b.wav"), "temperature": None},
    ]
    result = subprocess.run(
        [sys.executable, str(WORKER_PATH), json.dumps({"model": "/m", "lang": "de", "quant": "q8_0", "threads": 2})],
        input="".join(json.dumps(job) + "\n" for job in jobs),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(fake_ptts.parent)},
        timeout=30,
        check=True,
    )

    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0] == {"event": "ready", "sample_rate": 24000}
    assert events[1] == {"event": "done", "id": 1, "seconds": 1.5}
    assert events[2]["event"] == "error" and "unknown voice Niemand" in events[2]["message"]
    assert (tmp_path / "a.wav").exists()


def test_phonon_quality_selects_the_builtin_default_pair(tmp_path: Path):
    config = load_config(tmp_path / "missing.yml")

    assert backend_for_quality(config, "phonon") == "phonon"
    profiles = select_voice_profiles(config, 2, "phonon")
    assert [profile.phonon_voice for profile in profiles] == ["Henrik", "Nele"]
    assert all(profile.license == "cc-by-nc-4.0" for profile in profiles)
    assert isinstance(backend_for_voice(config, profiles[0]), PhononBackend)


def test_phonon_paths_resolve_against_the_config_file(tmp_path: Path):
    config_path = tmp_path / "podcast.yml"
    config_path.write_text(yaml.safe_dump({"tts": {"phonon": {"model": "models/phonon"}}}), encoding="utf-8")

    config = load_config(config_path)

    assert config.tts.phonon.python_executable == tmp_path / ".venv-phonon/bin/python"
    assert config.tts.phonon.model == str(tmp_path / "models/phonon")


def test_phonon_hub_model_stays_a_hub_reference(tmp_path: Path):
    config = load_config(tmp_path / "missing.yml")

    assert config.tts.phonon.model.startswith("hf://gradium/phonon-40m-multilingual@")
