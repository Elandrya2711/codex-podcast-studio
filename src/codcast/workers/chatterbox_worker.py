"""Persistenter Chatterbox-Worker, gestartet von ChatterboxBackend.

Laeuft absichtlich in einer eigenen venv (`codcast setup-chatterbox`), damit die
torch-Abhaengigkeiten von Chatterbox nicht mit denen des Projekts kollidieren.
Deshalb importiert diese Datei nichts aus `codcast`.

Protokoll, eine JSON-Zeile pro Nachricht:
  stdout  {"event": "ready", "sample_rate": 24000}
  stdin   {"id": 1, "text": "...", "reference": "/pfad/ref.wav", "output_path": "/pfad/out.wav",
           "language": "de", "exaggeration": 0.35, "cfg_weight": 0.3, "temperature": 0.6,
           "repetition_penalty": 2.0}
  stdout  {"event": "done", "id": 1, "seconds": 3.4}
  stdout  {"event": "error", "id": 1, "message": "..."}

Diagnostik geht ausschliesslich nach stderr, damit stdout das Protokoll bleibt.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# Muss vor dem torch-Import stehen. Auf einem Arbeitsrechner teilen sich Spiel,
# Ollama und dieser Worker die GPU; ohne das kippt die Synthese bei knappem VRAM
# an Fragmentierung, obwohl in Summe genug frei waere.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    emit({"event": "ready", "sample_rate": int(model.sr)})

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
            wav = model.generate(
                job["text"],
                language_id=job.get("language", "de"),
                audio_prompt_path=job["reference"],
                exaggeration=job.get("exaggeration", 0.35),
                cfg_weight=job.get("cfg_weight", 0.3),
                temperature=job.get("temperature", 0.6),
                repetition_penalty=job.get("repetition_penalty", 2.0),
            )
            torchaudio.save(job["output_path"], wav.cpu(), int(model.sr))
            emit({"event": "done", "id": job_id, "seconds": wav.shape[-1] / float(model.sr)})
        except Exception as exc:  # eine kaputte Zeile darf den Worker nicht beenden
            traceback.print_exc(file=sys.stderr)
            emit({"event": "error", "id": job_id, "message": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
