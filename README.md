# Codex Podcast Studio

Codex Podcast Studio erzeugt aus einem Thema oder einer Fragestellung einen recherchierten Podcast: Codex CLI recherchiert, validiert Claims und schreibt ein Dialogskript; die Standard-Audioausgabe fuer `best` laeuft ueber OpenAI TTS, der schnelle lokale Fallback ueber Kokoro.

## Lizenz

Codex Podcast Studio ist source-available fuer nicht-kommerzielle Nutzung unter
der Codex Podcast Studio Non-Commercial Upstream License 1.0.

- Kommerzielle Nutzung erfordert vorherige schriftliche Erlaubnis von `Elandrya2711`.
- Weiterentwicklungen, Forks und deployte Aenderungen muessen als Pull Request
  oder Patch an das offizielle Upstream-Repository zurueckgespiegelt werden:
  `https://github.com/Elandrya2711/codex-podcast-studio`.
- Drittanbieter-Abhaengigkeiten, Modelle, Stimmen, Datensaetze und generierte
  Podcast-Ausgaben unterliegen jeweils ihren eigenen Bedingungen.

Siehe [LICENSE](LICENSE), [NOTICE](NOTICE) und [CONTRIBUTING.md](CONTRIBUTING.md).

## Setup

```bash
git clone https://github.com/Elandrya2711/codex-podcast-studio.git
cd codex-podcast-studio
uv sync --extra dev --python /usr/bin/python3.11
cp podcast.yml.example podcast.yml
uv run codcast voices list
```

Die Default-Konfiguration nutzt zwei OpenAI-TTS-Stimmen fuer `best`:

- `Cedar`: ruhiger Podcast-Host
- `Marin`: analytische Podcast-Stimme

Der OpenAI-Key wird nicht aus `OPENAI_API_KEY` gelesen, sondern absichtlich aus `OPENAI_TTS_API_KEY` in `.env.tts.local`, damit er nur fuer TTS genutzt wird:

```bash
printf 'OPENAI_TTS_API_KEY=sk-...\n' > .env.tts.local
chmod 600 .env.tts.local
```

`.env.tts.local` ist gitignoriert. Das Tool liest daraus nur den konfigurierten TTS-Key-Namen.

Fish bleibt als optionaler Premium-/Experimentierpfad mit kuratierten Referenzstimmen:

- `Jonas`: maennlicher Fish-Host, erwartet `voices/fish/host-m.wav`
- `Mara`: weiblicher Fish-Host, erwartet `voices/fish/host-f.wav`

Diese Referenzdateien muessen bewusst gesetzt werden. Der Premium-Pfad hat keinen Kokoro- oder Piper-Fallback.

## Fish S2 Pro Setup

```bash
uv run codcast setup-fish
```

Fish S2 Pro sollte in einer separaten Umgebung laufen. Der lokale Server wird standardmaessig unter `http://127.0.0.1:8098/v1/tts` erwartet.

Der getestete lokale GPU-Pfad nutzt Fish S2 Pro mit int8-quantisiertem Textmodell. Das laeuft auf der RTX 4080 SUPER mit 16 GB VRAM lokal auf CUDA. Full BF16/FP16 S2 Pro bleibt ein 24-GB+-VRAM-Pfad.

Wenn du Fish Audio API statt lokalem Server nutzen willst, setze in `podcast.yml` `tts.fish.server_url` auf den API-Endpunkt und lege den Key als Environment-Variable ab:

```bash
export FISH_AUDIO_API_KEY="..."
```

Fish Speech installieren und S2 Pro laden:

```bash
git clone https://github.com/fishaudio/fish-speech.git ../fish-speech
cd ../fish-speech
uv sync --python 3.12 --extra cu128
uv run hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

Einmalig quantisieren:

```bash
uv run python tools/llama/quantize.py \
  --checkpoint-path checkpoints/s2-pro \
  --mode int8 \
  --timestamp s2-pro
```

Lokalen Server starten:

```bash
uv run python tools/api_server.py \
  --llama-checkpoint-path checkpoints/fs-1.2-int8-s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 127.0.0.1:8098 \
  --half \
  --max-seq-len 8192
```

Fuer 16-GB-GPUs kann ein int8-Checkpoint noetig sein. Full BF16/FP16 S2 Pro bleibt ein 24-GB+-VRAM-Pfad.

Referenzstimmen importieren:

```bash
uv run codcast voices import \
  --backend fish \
  --name fish-host-m \
  --wav /pfad/maennliche-referenz.wav \
  --transcript "Der exakt gesprochene Referenztext."

uv run codcast voices import \
  --backend fish \
  --name fish-host-f \
  --wav /pfad/weibliche-referenz.wav \
  --transcript "Der exakt gesprochene Referenztext."
```

## Schnelltest Ohne Codex

```bash
uv run codcast voices test fish-host-m --text "Anny the Duck reagiert auf Twitch, YouTube und Discord."
uv run codcast voices test fish-host-f --text "Der Clip heisst How to fix input lag, aber der Kontext ist wichtig."
```

Das schreibt Test-WAVs unter `voice-test/`, sofern der Fish-Server laeuft.

## Podcast Generieren

Ein sofort nutzbarer Zwei-Sprecher-Podcast:

```bash
uv run codcast generate "Wie veraendert KI die Softwareentwicklung?" \
  --min-minutes 5 \
  --max-minutes 8
```

Ohne weitere Flags erzeugt `generate` ein Gespraech aus 2 OpenAI-Stimmen: Cedar und Marin.

## Vorhandenes Skript neu rendern

Ohne neue Recherche oder Skript-Erzeugung kann ein vorhandener Run erneut vertont werden:

```bash
uv run codcast rerender podcasts/<run-ordner> --quality openai --suffix openai-tts
```

Das liest nur `script.json` und schreibt neue Dateien mit Suffix, z. B. `segments-openai-tts.json` und `<run-id>-openai-tts.mp3`.

Bei OpenAI werden Einsprecher-Skripte automatisch zu groesseren Requests unter `tts.openai.max_input_chars` gebuendelt. OpenAI-Requests laufen bis `tts.openai.concurrency` parallel. Mehrsprecher-Skripte bleiben zeilenweise, damit kein teurer Zusatzkontext pro Sprecherwechsel noetig ist.

## Systemweiter Podcast-Wizard

Installiere den Shortcut einmalig userweit:

```bash
uv tool install --force --editable .
uv tool update-shell
```

Danach kannst du aus jedem Ordner starten:

```bash
podcast
```

Der Wizard fragt Thema, Sprecher, Mindestlaenge, Maximallaenge, Qualitaet, Recherche-Tiefe und Sprache ab. Er nutzt die zentrale Konfiguration `podcast.yml` aus diesem Checkout und schreibt die Ergebnisse in den dort konfigurierten Podcast-Ordner.

Nach der Bestaetigung oeffnet `podcast` eine Terminal-UI mit Live-Fortschritt, Schrittstatus, TTS-Segmentzaehler, kurzen Logs und Ergebnisuebersicht. Den gleichen Fortschrittsmodus gibt es auch fuer direkte CLI-Aufrufe:

```bash
uv run codcast generate "Thema" \
  --min-minutes 10 \
  --max-minutes 15 \
  --ui
```

Ausgaben liegen zentral unter `podcasts/<YYYY-MM-DD-thema>/`:

- `research.json`
- `validation.json`
- `script.json`
- `<YYYY-MM-DD-thema>-transcript.md`
- `<YYYY-MM-DD-thema>-sources.md`
- `segments.json`
- `<YYYY-MM-DD-thema>.wav`
- `<YYYY-MM-DD-thema>.mp3`

## Recherche-Tiefe

Standardmaessig nutzt die Pipeline den bisherigen Codex-Recherchepfad:

```bash
uv run codcast generate "Thema" \
  --min-minutes 10 \
  --max-minutes 15 \
  --research-depth standard
```

Fuer echte Tiefenrecherche gibt es `deep` und `dossier`. Diese Modi nutzen keine paid Search-API. Die Web-Suche laeuft ueber eine lokale SearXNG-Instanz, Seiteninhalte werden lokal per Trafilatura/Fallback-Extraktion verarbeitet. YouTube-Treffer werden lokal mit `yt-dlp` als Transkripte geholt und als Dossier-Quellen verarbeitet, sofern Untertitel verfuegbar sind. Codex bleibt das einzige paid Tool in diesem Pfad.

```bash
# Einmalig lokale Open-Source-Suche starten:
podman compose -f compose.research.yml up -d
# alternativ mit Docker:
docker compose -f compose.research.yml up -d

uv run codcast generate "Thema" \
  --min-minutes 20 \
  --max-minutes 30 \
  --research-depth dossier \
  --ui
```

Die zentrale SearXNG-Adresse steht in `podcast.yml` unter `research.searxng_base_url`, standardmaessig `http://127.0.0.1:8888`. Die mitgelieferte `research/searxng/settings.yml` aktiviert JSON-Ausgabe fuer die lokale API.

Die zentralen Regler stehen in `podcast.yml` unter `research`. Ohne Overrides laeuft `deep` bis ca. 20 Minuten mit bis zu 100 Dokumenten, `dossier` bis ca. 60 Minuten mit bis zu 300 Dokumenten. Wichtige Zusatzartefakte:

- `research_plan.json`
- `deep_research/frontier.jsonl`
- `deep_research/documents/*.json`
- `deep_research/evidence.jsonl`
- `deep_research/topics.json`
- `deep_research/research_dossier.json`
- `deep_research/research_dossier.md`
- `deep_research/quality_report.json`

## XTTS / Voice Cloning

```bash
uv sync --extra dev --extra xtts --python /usr/bin/python3.11
```

XTTS bleibt nur fuer Voice-Cloning mit eigenen Referenzstimmen vorgesehen. Der Standardpfad nutzt keine synthetischen Bootstrap-Referenzen und keine Fallback-Stimmen.

## Schneller Kokoro-Modus

Kokoro/Kikiri bleibt nur als explizit schneller Modus erhalten:

```bash
uv run codcast generate "Thema" \
  --min-minutes 5 \
  --max-minutes 8 \
  --quality fast
```

## Mehrere Sprecher

Mehr als zwei Sprecher brauchen weitere hochwertige Voice-Profile in `podcast.yml`. Fuer eigene Stimmen kann XTTS genutzt werden:

```bash
uv sync --extra xtts
uv run codcast voices import \
  --backend xtts \
  --name host-a \
  --wav /pfad/zur/referenz.wav \
  --transcript "Der genaue Text der Referenzaufnahme."
```

Danach:

```bash
uv run codcast generate "Thema" \
  --min-minutes 20 \
  --max-minutes 35 \
  --speakers 2
```

Hinweis: `--quality best` nutzt den in `tts.backend` konfigurierten Premium-Pfad, aktuell OpenAI. `--quality openai` erzwingt OpenAI, `--quality fast` nutzt Kokoro/Kikiri. `--quality xtts` bleibt fuer eigene XTTS-Referenzstimmen verfuegbar.

## Codex-Verhalten

Die Standard-Recherche ruft bei aktivierter Live-Suche `codex --search exec --ephemeral --output-schema ...` auf. Codex schreibt strukturierte JSON-Artefakte; `stdout` und `stderr` jedes Codex-Schritts werden im Run-Ordner gespeichert. Mit `--cached-search` wird Live-Websuche deaktiviert und Codex nutzt den konfigurierten Standard.

Bei `--research-depth deep` und `--research-depth dossier` setzt die Pipeline fuer Planungs-, Extraktions- und Dossier-Schritte `model_reasoning_effort="xhigh"` und hoehere Timeouts. Diese Codex-Schritte laufen ohne Codex-Live-Suche; die Web-Datenbeschaffung laeuft ueber den lokalen Research-Provider `searxng`. Paid Drittanbieter-Suchdienste sind dafuer nicht erforderlich.

## Nuetzliche Kommandos

```bash
uv run codcast init-config --force
uv run codcast setup-fish
uv run codcast voices list
uv run codcast voices test fish-host-m
uv run codcast voices test fish-host-f
uv run codcast inspect <run-id>
uv run pytest
```
