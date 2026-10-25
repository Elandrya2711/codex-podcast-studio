"""Persistenter Phonon-Worker, gestartet von PhononBackend.

Laeuft in einer eigenen venv (`codcast setup-phonon`) und importiert deshalb
nichts aus `codcast`. ptts bringt seine Laufzeit als Rust-Erweiterung mit,
ohne PyTorch und ohne GPU.

Aufruf: phonon_worker.py '{"model": "...", "lang": "de", "quant": "q8_0", "threads": null}'

Protokoll, eine JSON-Zeile pro Nachricht (wie workers/chatterbox_worker.py):
  stdout  {"event": "ready", "sample_rate": 24000}
  stdin   {"id": 1, "text": "...", "voice": "Henrik", "output_path": "/pfad/out.wav", "temperature": null}
  stdout  {"event": "done", "id": 1, "seconds": 3.4}
  stdout  {"event": "error", "id": 1, "message": "..."}

Diagnostik geht ausschliesslich nach stderr, damit stdout das Protokoll bleibt.
"""

from __future__ import annotations

import json
import sys
import traceback


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    import ptts

    settings = json.loads(sys.argv[1])
    if settings.get("threads"):
        ptts.set_num_threads(int(settings["threads"]))
    model = ptts.TTS(settings["model"], lang=settings["lang"], quant=settings["quant"])
    emit({"event": "ready", "sample_rate": int(model.sample_rate)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"event": "error", "id": None, "message": f"invalid job json: {exc}"})
            continue
        job_id = job.get("id")
        try:
            seconds = model.save(
                job["output_path"],
                job["text"],
                voice=job["voice"],
                temperature=job.get("temperature"),
            )
            emit({"event": "done", "id": job_id, "seconds": float(seconds)})
        except Exception as exc:  # eine kaputte Zeile darf den Worker nicht beenden
            traceback.print_exc(file=sys.stderr)
            emit({"event": "error", "id": job_id, "message": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
