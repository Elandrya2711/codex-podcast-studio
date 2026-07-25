# Codex Podcast Studio

Codex Podcast Studio erzeugt aus einem Thema oder einer Fragestellung einen recherchierten Podcast: eine Agent-CLI recherchiert, validiert Claims und schreibt ein Dialogskript; die Standard-Audioausgabe fuer `best` laeuft lokal auf der GPU ueber Chatterbox Multilingual, OpenAI TTS bleibt als bezahlte Alternative, Kokoro als schneller Fallback.

Als LLM-Provider stehen **Claude (Standard, `claude-opus-5`)** und **Codex** zur Verfuegung, umschaltbar mit `--llm-provider`. Beide nutzen die jeweilige CLI mit dem vorhandenen Abo-Login, das Projekt speichert fuer den LLM-Pfad keinen API-Key.

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

Die Default-Konfiguration nutzt fuer `best` zwei lokale Chatterbox-Stimmen, die eine eigene Referenzaufnahme klonen:

- `Jonas`: maennlicher Host, erwartet `voices/chatterbox/host-m.wav`
- `Mara`: weibliche Stimme, erwartet `voices/chatterbox/host-f.wav`

Siehe [Chatterbox Setup](#chatterbox-setup-lokal-und-kostenlos). OpenAI TTS bleibt vollstaendig nutzbar (`--quality openai`) und kostet pro Podcast Geld.

Der OpenAI-Key wird nicht aus `OPENAI_API_KEY` gelesen, sondern absichtlich aus `OPENAI_TTS_API_KEY` in `.env.tts.local`, damit er nur fuer TTS genutzt wird:

```bash
printf 'OPENAI_TTS_API_KEY=sk-...\n' > .env.tts.local
chmod 600 .env.tts.local
```

`.env.tts.local` ist gitignoriert. Das Tool liest daraus nur den konfigurierten TTS-Key-Namen.

## Chatterbox Setup (lokal und kostenlos)

Chatterbox Multilingual (MIT-Lizenz) laeuft auf der GPU, klont eine Referenzstimme aus wenigen Sekunden Audio und kostet nichts pro Podcast. Auf der RTX 4080 SUPER braucht es rund 4 GB VRAM und rendert etwa doppelt so schnell wie Echtzeit.

```bash
uv run codcast setup-chatterbox
```

Das legt `.venv-chatterbox` an und installiert `chatterbox-tts` dort. Eine eigene venv ist Absicht: Chatterbox pinnt andere torch-Versionen als das Projekt. Die Modellgewichte (rund 3 GB) laedt der erste Lauf nach `~/.cache/huggingface`.

`setuptools<81` wird mitinstalliert, weil das Wasserzeichen-Paket `resemble-perth` noch `pkg_resources` importiert. Ohne das faellt der Watermarker still auf `None` und Chatterbox startet nicht.

Referenzaufnahmen liegen diesem Repository bewusst nicht bei. Die Beispielkonfiguration verweist auf `voices/chatterbox/host-m.wav`, diese Datei bringst du selbst mit. Eine Referenz ist entweder eine fremde Aufnahme oder die Ausgabe eines fremden TTS-Dienstes, und in beiden Faellen laesst sich eine Stimme nicht einfach weiterverteilen. Mitgeliefert ist deshalb der Weg dorthin, nicht das Material: `voices extract` schneidet aus einer eigenen Aufnahme Klonmaterial, `voices import` uebernimmt es.

Zwei Referenzstimmen importieren, je 8 bis 15 Sekunden sauber gesprochenes Deutsch ohne Hintergrundgeraeusche:

```bash
uv run codcast voices import --backend chatterbox --name chatterbox-host-m --wav /pfad/maennlich.wav
uv run codcast voices import --backend chatterbox --name chatterbox-host-f --wav /pfad/weiblich.wav
uv run codcast voices test chatterbox-host-m
```

Die Qualitaet des Ergebnisses haengt direkt an der Referenzaufnahme: eine studioreine Referenz klingt hoerbar besser als eine Hobbyaufnahme. Ein `ref_text` ist fuer Chatterbox nicht noetig.

### Die Referenzaufnahme ist der groesste Hebel

Sieben Referenzen derselben Sprecherin, gleicher Text, gleiche Parameter, je acht Takes: der Anteil fehlerhafter Takes lag zwischen 0 und 62 Prozent. Es lohnt sich also, mehrere Kandidaten durchzuprobieren, statt an den Parametern zu drehen. Was sich als Referenz bewaehrt hat:

- 10 bis 12 Sekunden, ruhig gesprochener Fliesstext mit vollstaendigen Saetzen
- keine Aufzaehlung einzelner Buchstaben (`P, R, N, D`), keine Ziffern, keine Abkuerzungen
- echte Umlaute im gesprochenen Text: eine Referenz, in der jemand "fuer" statt "für" liest, vererbt genau das
- laenger ist nicht besser: zwei aneinandergehaengte Aufnahmen (21 bis 23 Sekunden) waren in der Messung deutlich schlechter als eine einzelne gute

### Mehrere Besetzungen nebeneinander

Sobald mehr als ein Stimmpaar fuer dasselbe Backend konfiguriert ist, gewinnen sonst immer die ersten passenden Profile. Benannte Besetzungen machen die Wahl explizit:

```yaml
tts:
  voice_set: standard
  voice_sets:
    standard: [chatterbox-host-m, chatterbox-host-f]
    gaeste: [gast-a, gast-b]
```

Im Wizard erscheint dann eine Zeile `Stimmen` mit den verfuegbaren Besetzungen und den Anzeigenamen dahinter. Fuer Skripte gibt es `--voice-set gaeste` bei `generate` und `rerender`. Die Reihenfolge im Set bestimmt, welche Stimme Sprecher eins wird. Ein Set mit Profilen aus verschiedenen Backends wird abgelehnt, ebenso ein Set, das auf unbekannte Profile zeigt.

### Warum Zahlen normalisiert werden

`tts.chatterbox.normalize_text` (Standard `true`) schreibt Zahlen und Kuerzel vor der Synthese aus. Ohne das wurde gemessen aus `48-Volt` ein gesprochenes "88 Volt" und aus `MY2025` ein "Mai 1050". Das ist inhaltlich falsch, nicht nur unschoen. Betroffen sind Ziffern, Dezimalkommas, Jahre und kurze Grossbuchstaben-Kuerzel; als Wort gesprochene Kuerzel wie `ABS` oder `ESP` bleiben unangetastet (`src/codcast/text_normalization.py`).

### Tempo und Betonung

Die Defaults sind auf Deutsch gemessen: `exaggeration: 0.35`, `cfg_weight: 0.3`, `temperature: 0.6`. Hoehere Werte klingen aufgeregter und sprechen schneller. Das Sprechtempo laesst sich pro Stimme mit `speed` feinjustieren (`speed: 0.95` streckt die Ausgabe per ffmpeg um 5 Prozent), was die Laengenplanung ueber `generation.words_per_minute` treffsicherer macht.

Wenn nur eine von mehreren Stimmen unruhig klingt, laesst sie sich einzeln nachschaerfen, ohne die anderen anzufassen. Nicht gesetzte Werte kommen weiter aus `tts.chatterbox`:

```yaml
  - id: chatterbox-host-f
    display_name: Mara
    backend: chatterbox
    speaker_wav: voices/chatterbox/host-f.wav
    chatterbox_temperature: 0.3   # weniger Streuung, weniger Ausrutscher
    chatterbox_cfg_weight: 0.4    # naeher an der Referenz
    speed: 0.95
```

Eine niedrigere `temperature` reduziert gelegentliche Aussprachefehler, weil das Modell weniger wuerfelt. Aber nicht beliebig weit: bei `0.3` trat im Test ein Wiederholungs-Loop auf, das Modell sprach einen kompletten Satz zweimal. Das ist der groebere Fehler. `0.45` hat sich als brauchbarer Kompromiss erwiesen.

### Schutz gegen Wiederholungs-Loops

Chatterbox spricht gelegentlich einen Satz zweimal oder bricht mitten drin ab. Beides erkennt das Backend an der Segmentdauer im Vergleich zur Textlaenge und rendert das Segment neu (`max_retries`, Standard 2). Bleibt es auffaellig, erscheint eine Warnung auf stderr, statt still ein kaputtes Segment auszuliefern. Die Schwellen sind ueber `min_duration_ratio`, `max_duration_ratio` und `chars_per_second` einstellbar.

### VRAM neben anderen Programmen

Der Worker setzt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, damit die Synthese nicht an Fragmentierung scheitert, wenn parallel ein Spiel oder ein Ollama-Modell auf der GPU liegt. Bleiben weniger als etwa 5 GB frei, reicht es trotzdem nicht: dann entweder ein Modell entladen (`ollama stop <modell>`) oder `tts.chatterbox.device: cpu` setzen. CPU funktioniert, ist aber deutlich langsamer.

### Ein Abbruch kostet keine fertigen Segmente

Eine 20-Minuten-Folge sind rund 180 Segmente und knapp zehn Minuten GPU-Zeit. Auf einer geteilten Karte kann das mitten im Lauf am Speicher scheitern, etwa weil ein Modell nachgeladen wird. Zwei Vorkehrungen dagegen:

- **Speicherfehler werden wiederholt** (`tts.gpu_oom_retries`, Standard 4). Vor jedem neuen Versuch wird das Modell entladen, denn der gescheiterte Worker haelt seinen VRAM sonst weiter fest, und danach `tts.gpu_oom_wait_sec` Sekunden gewartet (Standard 30). Jede Warnung nennt, wer den Speicher gerade belegt, mit dem groessten Halter zuerst. Vier mal dreissig Sekunden sind vor allem Reaktionszeit: blind zu warten hilft wenig, weil ein geladenes Ollama-Modell je nach Haltezeit eine halbe Stunde liegen bleibt. Wer die Warnung liest und ein Modell entlaedt, rettet den Lauf.
- **Optional entlaedt der Renderer Ollama selbst** (`tts.gpu_oom_free_ollama`, Standard aus). Aus gutem Grund nicht der Standard, denn damit greift codcast in einen fremden Dienst. Wer Ollama nebenbei nutzt, etwa in einer Diktierkette, will es meist an: das Modell laedt in Sekunden nach, ein abgebrochener Podcast kostet Minuten.
- **Gibt es auf, sagt es was zu tun ist.** Die Abschlussmeldung nennt die Belegung, die drei Auswege (Modell entladen, Spiel beenden, `device: cpu`) und den fertigen Befehl zum Fortsetzen.
- **Fertige Segmente werden uebernommen** (`tts.reuse_segments`, Standard an). Neben jedem Segment liegt ein Fingerabdruck aus Text, Stimmprofil und Backend-Einstellungen. Nur wenn der exakt passt, wird das vorhandene Audio verwendet. Geschrieben wird zuerst unter einem Zwischennamen und erst am Ende umbenannt, damit "Datei existiert" auch wirklich "Segment ist fertig" heisst.

`rerender` schaltet die Wiederverwendung bewusst ab, denn dort ist ein neuer Take die Absicht. Wer einen abgebrochenen `rerender` fortsetzen will, gibt `--reuse-segments` mit.

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

Das liest nur `script.json` und schreibt neue Dateien mit Suffix, z. B. `segments-openai-tts.json` und `<run-id>-openai-tts.mp3`. Ohne `--suffix` benennt sich die Ausgabe nach dem aktiven Backend, also `<run-id>-chatterbox.mp3` bei einem lokalen Lauf.

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

Der Wizard fragt nur das Thema ab und zeigt danach **alle** Optionen als Liste. Eine Nummer aendert den jeweiligen Punkt, leere Eingabe startet den Podcast, `q` bricht ab. Kommandozeilen-Flags braucht man dafuer nicht.

```
Einstellungen:
   1) Thema             Warum Kaffee muede machen kann
   2) Sprecher          2
   3) Laenge            10-15 Minuten
   4) Audio-Qualitaet   best
   5) Recherche-Tiefe   standard
   6) LLM-Provider      claude
   7) Modell            claude-opus-5
   8) Reasoning-Effort  standard
   9) Live-Websuche     ein
  10) Audio rendern     ja
  11) Sprache           de-DE
      Ausgabe           /pfad/zu/podcasts

Nummer aendern, leer = Podcast starten, q = abbrechen:
```

Jeder Unterpunkt listet die Auswahl mit Erklaerung, `>` markiert den aktuellen Wert. Auswahl per Nummer oder Name:

```
Claude-Modell:
 > 1) opus    claude-opus-5, gute Balance (Standard)
   2) fable   claude-fable-5, faehigstes Modell, hoeherer Verbrauch
   3) sonnet  claude-sonnet-5, schnell und sparsam
Auswahl [opus]:
```

Der Wizard nutzt die zentrale Konfiguration `podcast.yml` aus diesem Checkout als Startwerte und schreibt die Ergebnisse in den dort konfigurierten Podcast-Ordner. Dauerhafte Vorgaben aendert man in `podcast.yml`, dann stehen sie beim naechsten Start direkt im Menue.

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

Fuer echte Tiefenrecherche gibt es `deep` und `dossier`. Diese Modi nutzen keine paid Search-API. Die Web-Suche laeuft ueber eine lokale SearXNG-Instanz, Seiteninhalte werden lokal per Trafilatura/Fallback-Extraktion verarbeitet. YouTube-Treffer werden lokal mit `yt-dlp` als Transkripte geholt und als Dossier-Quellen verarbeitet, sofern Untertitel verfuegbar sind. Der LLM-Provider (Claude oder Codex) bleibt das einzige kostenpflichtige Tool in diesem Pfad, abgedeckt durch das jeweilige Abo.

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

## Referenzstimmen Aus Einer Aufnahme Schneiden

`voices import` erwartet ein fertiges WAV mit genau einer Stimme. Echtes Quellmaterial ist aber meist ein Gespraech. `voices extract` trennt die Sprecher und schneidet pro Person sauberes Klonmaterial:

```bash
uv run codcast setup-diarize          # einmalig, legt .venv-diarize an
uv run codcast voices extract folge.mp3 --speakers 2 --minutes 5
```

Alles laeuft lokal. Bewusst ohne pyannote, dessen Modelle auf HuggingFace gated sind: hier arbeiten Silero VAD (MIT) und SpeechBrain ECAPA (Apache-2.0), beide frei ladbar, kein Token noetig.

Ergebnis pro Sprecher in `voices/source_candidates/<name>/speaker_A/`: die einzelnen Chunks, eine zusammengesetzte und lautheitsnormalisierte Datei in Quellqualitaet, dieselbe in 24 kHz fuer den direkten Chatterbox-Import, sowie eine 30-Sekunden-Preview zum Zuordnen. `report.json` enthaelt alle Zeitstempel und Messwerte.

### Warum Das Sauber Wird

Das Clustering allein reicht nicht: an jedem Sprecherwechsel sitzt Ueberlappung, und die landet sonst im Klonmaterial. Zwei Filter danach machen den Unterschied.

- **Randerosion**: von jedem zusammenhaengenden Lauf eines Sprechers fallen `erosion_sec` an beiden Enden weg, also genau die Uebergangszonen.
- **Margin-Score**: pro Chunk die Aehnlichkeit zum eigenen minus die zum fremden Sprecherzentroid. Ueberlappende Rede, Musikbett und Stoergeraeusche druecken ihn automatisch, ohne dass es dafuer eigene Detektoren braucht.

Ausgewaehlt wird dann nicht stur nach Score, sondern per Round-Robin ueber zehn Zeit-Bins. Das kostet etwas Margin und bringt Prosodie-Varianz, die beim Klonen mehr wiegt als die eine gleichmaessigste Passage.

Findet `--minutes` nicht genug Material, kommt zurueck was da ist, mit Angabe der tatsaechlichen Dauer. Die Schwellen werden nicht still aufgeweicht.

### Ergebnis Pruefen

Der Lauf misst am Ende die fertigen Dateien nach und gibt zwei Zahlen aus:

- **Aehnlichkeit zwischen den Sprechern**: klein ist gut. Ueber 0.5 kommt eine Warnung, dann wurden die Stimmen nicht getrennt.
- **Selbstaehnlichkeit**: gross ist gut, die Datei klingt durchgehend nach derselben Person.

Bleibt im Ergebnis noch die zweite Stimme hoerbar, zuerst `erosion_sec` und `min_margin` in `podcast.yml` anheben.

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

Hinweis: `--quality best` nutzt den in `tts.backend` konfigurierten Pfad, standardmaessig das lokale Chatterbox. `--quality chatterbox` erzwingt den lokalen Pfad, `--quality openai` erzwingt OpenAI (kostet Geld), `--quality fast` nutzt Kokoro/Kikiri. `--quality xtts` bleibt fuer eigene XTTS-Referenzstimmen verfuegbar. Im Wizard zeigt die Zeile `Audio-Qualitaet` immer das effektive Backend, damit `best` nicht unbemerkt Geld kostet.

### Gemessener Vergleich der Audiopfade

Acht deutsche Saetze aus einer echten Episode, jeweils mit derselben Referenzstimme. `WER` ist die Wortfehlerrate nach Transkription mit faster-whisper large-v3 (misst Aussprachefehler), `Sprecher-Aehnlichkeit` die CAMPPlus-Cosine-Similarity gegen die Referenzaufnahme, `MOS` der SQUIM-Schaetzwert gegen das OpenAI-Original.

| Pfad | WER | Sprecher-Aehnlichkeit | MOS | Kosten |
|---|---|---|---|---|
| OpenAI TTS (`gpt-4o-mini-tts`) | 0,072 | 0,879 | 4,59 | ca. 5 Euro pro Episode |
| Chatterbox, Defaults dieses Projekts | **0,016** | **0,887** | 4,38 | 0 |
| Chatterbox, lizenzfreie Referenz | 0,047 | 0,304 | 4,27 | 0 |
| Kokoro/Kikiri (`fast`) | 0,077 | 0,429 | 4,39 | 0 |

Die Sprecher-Aehnlichkeit ist nur zwischen Pfaden mit derselben Zielstimme vergleichbar: 0,879 ist der Referenzwert derselben Stimme in anderen Saetzen, die lizenzfreie Variante klingt absichtlich nach einer anderen Person. Kokoros WER liegt scheinbar gleichauf, produziert aber echte Aussprachefehler wie "Corsa-Brit" fuer "Corsa Hybrid". CosyVoice 3 (0,5B) wurde ebenfalls getestet und lieferte in diesem Setup keine brauchbare deutsche Sprache (WER 0,99, halb Stille), sowohl `inference_zero_shot` als auch `inference_cross_lingual`.

## LLM-Provider

Der Provider wird in `podcast.yml` unter `llm.provider` gesetzt (Standard `claude`) und pro Run mit `--llm-provider {claude,codex}` ueberschrieben. Weitere Flags: `--model` (Modell des aktiven Providers), `--effort {low,medium,high,xhigh,max}`, `--cached-search` (Live-Websuche aus). `--codex-model` bleibt als Alias fuer den Codex-Provider erhalten.

Beide Provider liefern dieselben strukturierten JSON-Artefakte pro Schritt, und `stdout`/`stderr` jedes Schritts landen im Run-Ordner. Damit funktionieren `resume` und `inspect` unabhaengig vom Provider.

```bash
uv run codcast setup-claude
uv run codcast generate "Thema" --min-minutes 10 --max-minutes 15 --speakers 2 --ui
uv run codcast generate "Thema" --min-minutes 10 --max-minutes 15 --model fable
uv run codcast generate "Thema" --min-minutes 10 --max-minutes 15 --llm-provider codex
```

### Modellwahl

`--model` akzeptiert Kurz-Aliase, die auf feste Modell-IDs aufgeloest werden, damit ein Run reproduzierbar bleibt:

| Alias | Modell-ID | Einsatz |
| --- | --- | --- |
| `opus` | `claude-opus-5` | Standard, gute Balance aus Qualitaet und Verbrauch |
| `fable` | `claude-fable-5` | Anthropics faehigstes Modell, fuer besonders anspruchsvolle Themen und Tiefenrecherche. Verbraucht deutlich mehr Abo-Kontingent pro Run. |
| `sonnet` | `claude-sonnet-5` | schneller und sparsamer, fuer einfachere Themen |

Jede andere Angabe wird unveraendert an die CLI weitergegeben, z. B. `--model claude-opus-4-8`. Dauerhaft umstellen laesst sich das Modell in `podcast.yml` unter `llm.claude.model`.

Im Wizard (`podcast`) ist das Punkt 7 im Menue: dort genuegt `2` oder `fable`. Die Flags in diesem Abschnitt sind nur fuer Skripte und den `generate`-Unterbefehl noetig.

### Claude-Verhalten (Standard)

Ein Schritt ruft `claude -p` auf, der Prompt kommt ueber `stdin` und nie via argv:

```
claude -p --model claude-opus-5 [--effort xhigh] \
  --json-schema '<Schema inline>' \
  --output-format stream-json --verbose \
  --tools WebSearch            # ohne Live-Suche: --tools "" \
  --strict-mcp-config --setting-sources "" --disable-slash-commands \
  --no-session-persistence \
  --system-prompt '<Extraktor-Prompt>'
```

`--json-schema` erzwingt das Ausgabeschema, das Ergebnis wird aus dem `result`-Event (`structured_output`) gelesen. `--output-format stream-json` liefert die Live-Logs fuer die TUI.

Die Isolationsflags sind nicht optional: `--tools` filtert **keine** MCP-Tools, ohne `--strict-mcp-config` und `--setting-sources ""` landen global konfigurierte MCP-Server, Hooks und Skills im Recherche-Run. Der eigene `--system-prompt` ersetzt zusaetzlich den Coding-Agent-Standardprompt und senkt den Overhead pro Schritt deutlich. Zum Abschalten (nur fuer Debugging) `llm.claude.isolate: false`.

Voraussetzung ist eine installierte Claude CLI mit Abo-Login (`uv run codcast setup-claude` zeigt die Schritte). Fehlt sie, bricht der Run mit einem Hinweis auf `--llm-provider codex` ab.

### Codex-Verhalten

Die Standard-Recherche ruft bei aktivierter Live-Suche `codex --search exec --ephemeral --output-schema ...` auf. Mit `--cached-search` wird Live-Websuche deaktiviert und Codex nutzt den konfigurierten Standard.

### Reasoning bei Tiefenrecherche

Bei `--research-depth deep` und `--research-depth dossier` setzt die Pipeline fuer Planungs-, Extraktions- und Dossier-Schritte die hoechste Reasoning-Stufe und hoehere Timeouts: bei Claude `--effort xhigh` (konfigurierbar ueber `llm.claude.deep_effort`), bei Codex `model_reasoning_effort="xhigh"`. Diese Schritte laufen ohne Live-Suche des LLM; die Web-Datenbeschaffung laeuft ueber den lokalen Research-Provider `searxng`. Paid Drittanbieter-Suchdienste sind dafuer nicht erforderlich.

## Nuetzliche Kommandos

```bash
uv run codcast init-config --force
uv run codcast setup-claude
uv run codcast setup-chatterbox
uv run codcast setup-diarize
uv run codcast setup-fish
uv run codcast voices list
uv run codcast voices extract folge.mp3 --speakers 2 --minutes 5
uv run codcast voices test chatterbox-host-m
uv run codcast voices test chatterbox-host-f
uv run codcast inspect <run-id>
uv run pytest
```
