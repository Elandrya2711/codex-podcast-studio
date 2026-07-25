"""Persistenter Diarisierungs-Worker, gestartet von codcast.diarize.

Laeuft absichtlich in einer eigenen venv (`codcast setup-diarize`), damit torch
und speechbrain nicht in der Projekt-venv landen. Deshalb importiert diese Datei
nichts aus `codcast`.

Persistent statt einmalig, weil derselbe Lauf das Modell dreimal braucht:
erst fuer die Fensteranalyse, dann fuer die Bewertung der Kandidaten-Chunks und
zuletzt fuer die Verifikation der fertigen Dateien. Ein Modellstart reicht dafuer.

Protokoll, eine JSON-Zeile pro Nachricht:
  stdout  {"event": "ready", "device": "cuda"}
  stdin   {"cmd": "analyze", "audio": "/pfad/16k.wav", "speakers": 2,
           "window_sec": 1.5, "hop_sec": 0.75, ...}
  stdout  {"event": "result", "cmd": "analyze", "duration": 4186.5,
           "speech": [[start, end], ...], "windows": [[start, end, label], ...],
           "centroid_similarity": [[1.0, 0.31], [0.31, 1.0]]}
  stdin   {"cmd": "score", "audio": "/pfad/16k.wav",
           "chunks": [[start, end, speaker], ...]}
  stdout  {"event": "result", "cmd": "score", "scores": [[own, other, margin], ...]}
  stdin   {"cmd": "verify", "files": ["a.wav", "b.wav"], "window_sec": 10.0}
  stdout  {"event": "result", "cmd": "verify", "between": [[...]], "within": [{...}]}
  stdout  {"event": "error", "cmd": "...", "message": "..."}

Diagnostik geht ausschliesslich nach stderr, damit stdout das Protokoll bleibt.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# Muss vor dem torch-Import stehen. Auf einem Arbeitsrechner teilen sich Spiel,
# Ollama und dieser Worker die GPU; ohne das kippt die Analyse bei knappem VRAM
# an Fragmentierung, obwohl in Summe genug frei waere.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BATCH_SIZE = 64
# Kuerzere Ausschnitte liefern instabile Sprecher-Embeddings.
MIN_EMBED_SEC = 1.0


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


class Diarizer:
    def __init__(self, device: str, embedding_model: str, savedir: str) -> None:
        import torch
        from silero_vad import load_silero_vad
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        self.device = self._resolve_device(device)
        log(f"device={self.device}")
        self.vad = load_silero_vad()
        self.encoder = EncoderClassifier.from_hparams(
            source=embedding_model,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        # Zentroide aus dem letzten `analyze`, gebraucht von `score`.
        self.centroids = None

    def _resolve_device(self, requested: str) -> str:
        if requested != "cuda":
            return requested
        if not self.torch.cuda.is_available():
            log("cuda nicht verfuegbar, fallback auf cpu")
            return "cpu"
        try:
            free, _total = self.torch.cuda.mem_get_info()
        except Exception:  # ältere Treiber melden das nicht
            return "cuda"
        if free < 700 * 1024 * 1024:
            log(f"nur {free / 1024 / 1024:.0f} MiB VRAM frei, fallback auf cpu")
            return "cpu"
        return "cuda"

    # --- Audio ---------------------------------------------------------

    def _read_mono(self, path: str, target_rate: int = 16000):
        import soundfile as sf
        import torchaudio

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        wav = self.torch.from_numpy(data.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if rate != target_rate:
            wav = torchaudio.functional.resample(wav, rate, target_rate)
        return wav[0]

    # --- Embeddings ----------------------------------------------------

    def _embed_spans(self, wav, rate: int, spans: list[tuple[float, float]]):
        """L2-normalisierte ECAPA-Embeddings fuer (start, end)-Paare in Sekunden."""
        import torch

        out = []
        total = len(spans)
        for offset in range(0, total, BATCH_SIZE):
            batch_spans = spans[offset : offset + BATCH_SIZE]
            pieces = []
            for start, end in batch_spans:
                lo = max(0, int(start * rate))
                hi = min(wav.shape[0], int(end * rate))
                pieces.append(wav[lo:hi])
            longest = max(piece.shape[0] for piece in pieces)
            padded = torch.zeros(len(pieces), longest)
            lengths = torch.zeros(len(pieces))
            for index, piece in enumerate(pieces):
                padded[index, : piece.shape[0]] = piece
                lengths[index] = piece.shape[0] / longest
            with torch.no_grad():
                embeddings = self.encoder.encode_batch(
                    padded.to(self.device), lengths.to(self.device)
                ).squeeze(1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            out.append(embeddings.cpu())
            if offset % (BATCH_SIZE * 10) == 0:
                log(f"embed {offset}/{total}")
        return torch.cat(out, dim=0) if out else torch.zeros(0, 192)

    # --- Kommandos -----------------------------------------------------

    def analyze(self, job: dict) -> dict:
        import torch
        from silero_vad import get_speech_timestamps
        from sklearn.cluster import AgglomerativeClustering

        rate = int(job.get("sample_rate", 16000))
        wav = self._read_mono(job["audio"], rate)
        duration = wav.shape[0] / rate

        stamps = get_speech_timestamps(
            wav,
            self.vad,
            sampling_rate=rate,
            return_seconds=True,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=0,
        )
        speech = [(float(item["start"]), float(item["end"])) for item in stamps]
        log(f"vad: {len(speech)} regionen, {sum(e - s for s, e in speech):.1f}s sprache")

        window_sec = float(job.get("window_sec", 1.5))
        hop_sec = float(job.get("hop_sec", 0.75))
        spans: list[tuple[float, float]] = []
        for start, end in speech:
            if end - start < MIN_EMBED_SEC:
                continue
            if end - start <= window_sec:
                spans.append((start, end))
                continue
            cursor = start
            while cursor + window_sec <= end:
                spans.append((cursor, cursor + window_sec))
                cursor += hop_sec
            # Rest am Ende der Region nicht verlieren.
            if end - cursor >= MIN_EMBED_SEC:
                spans.append((end - window_sec, end))
        if not spans:
            raise RuntimeError("keine Sprachfenster gefunden")
        log(f"{len(spans)} fenster")

        embeddings = self._embed_spans(wav, rate, spans)
        speakers = int(job.get("speakers", 2))
        labels = (
            AgglomerativeClustering(n_clusters=speakers, metric="cosine", linkage="average")
            .fit_predict(embeddings.numpy())
            .tolist()
        )

        centroids = []
        for speaker in range(speakers):
            picked = [i for i, label in enumerate(labels) if label == speaker]
            centroid = embeddings[picked].mean(dim=0)
            centroids.append(torch.nn.functional.normalize(centroid, dim=-1))
        self.centroids = torch.stack(centroids)
        similarity = (self.centroids @ self.centroids.T).tolist()

        return {
            "duration": duration,
            "speech": [[s, e] for s, e in speech],
            "windows": [[span[0], span[1], int(label)] for span, label in zip(spans, labels)],
            "centroid_similarity": similarity,
            "counts": [labels.count(speaker) for speaker in range(speakers)],
        }

    def score(self, job: dict) -> dict:
        if self.centroids is None:
            raise RuntimeError("score vor analyze aufgerufen")
        rate = int(job.get("sample_rate", 16000))
        wav = self._read_mono(job["audio"], rate)
        chunks = job["chunks"]
        spans = [(float(item[0]), float(item[1])) for item in chunks]
        embeddings = self._embed_spans(wav, rate, spans)
        similarity = embeddings @ self.centroids.T

        scores = []
        for row, chunk in zip(similarity, chunks):
            speaker = int(chunk[2])
            own = float(row[speaker])
            others = [float(value) for index, value in enumerate(row) if index != speaker]
            other = max(others) if others else 0.0
            scores.append([own, other, own - other])
        return {"scores": scores}

    def verify(self, job: dict) -> dict:
        """Misst an den fertigen Dateien, ob die Trennung wirklich sauber ist."""
        import torch

        rate = 16000
        window_sec = float(job.get("window_sec", 10.0))
        centroids = []
        within = []
        for path in job["files"]:
            wav = self._read_mono(path, rate)
            duration = wav.shape[0] / rate
            spans = []
            cursor = 0.0
            while cursor + window_sec <= duration:
                spans.append((cursor, cursor + window_sec))
                cursor += window_sec
            if not spans:
                spans = [(0.0, duration)]
            embeddings = self._embed_spans(wav, rate, spans)
            centroid = torch.nn.functional.normalize(embeddings.mean(dim=0), dim=-1)
            similarity = embeddings @ centroid
            centroids.append(centroid)
            within.append(
                {
                    "windows": len(spans),
                    "duration": duration,
                    "mean": float(similarity.mean()),
                    "std": float(similarity.std()) if len(spans) > 1 else 0.0,
                    "min": float(similarity.min()),
                }
            )
        matrix = torch.stack(centroids)
        return {"between": (matrix @ matrix.T).tolist(), "within": within}


def main() -> int:
    job_line = sys.stdin.readline()
    setup = json.loads(job_line)
    diarizer = Diarizer(
        device=setup.get("device", "cuda"),
        embedding_model=setup.get("embedding_model", "speechbrain/spkrec-ecapa-voxceleb"),
        savedir=setup["savedir"],
    )
    emit({"event": "ready", "device": diarizer.device})

    handlers = {"analyze": diarizer.analyze, "score": diarizer.score, "verify": diarizer.verify}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"event": "error", "cmd": None, "message": f"invalid job json: {exc}"})
            continue
        command = job.get("cmd")
        handler = handlers.get(command)
        if handler is None:
            emit({"event": "error", "cmd": command, "message": f"unbekanntes Kommando: {command}"})
            continue
        try:
            payload = handler(job)
            emit({"event": "result", "cmd": command, **payload})
        except Exception as exc:  # eine kaputte Anfrage darf den Worker nicht beenden
            traceback.print_exc(file=sys.stderr)
            emit({"event": "error", "cmd": command, "message": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
