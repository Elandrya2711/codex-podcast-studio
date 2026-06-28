from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .audio import assemble_episode
from .codex_runner import CodexRunner
from .config import AppConfig, config_to_yaml
from .deep_research import DeepResearchEngine
from .duration import duration_status, script_word_count
from .local_evidence import LocalEvidenceCollector
from .models import LocalEvidenceReport, PodcastScript, ResearchReport, RunManifest, ScriptLine, ValidationReport
from .progress import CancellationToken, ProgressEvent, ProgressReporter, report_progress
from .prompts import (
    build_evidence_enriched_research_prompt,
    build_research_prompt,
    build_research_from_dossier_prompt,
    build_research_revision_prompt,
    build_rewrite_prompt,
    build_script_prompt,
    build_validation_prompt,
)
from .tts import ScriptRenderer
from .util import slugify, write_json
from .voices import backend_for_quality, build_speaker_specs, select_voice_profiles, voice_map


TTS_LINE_CHAR_LIMIT = 220
SAFE_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,180}")


def _safe_output_stem(run_id: str, *, source: str = "run_id") -> str:
    value = run_id.strip()
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or path.name != value
        or value in {".", ".."}
        or not SAFE_RUN_ID_RE.fullmatch(value)
    ):
        raise ValueError(f"Unsafe {source}: {run_id!r}")
    return value


class PodcastGenerator:
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.runner = CodexRunner(config.codex, project_root)

    def _run_dir(self, topic: str) -> tuple[str, Path]:
        now = datetime.now()
        base_run_id = f"{now.strftime('%Y-%m-%d')}-{slugify(topic)}"
        root = self.config.output_root
        if not root.is_absolute():
            root = self.project_root / root
        for attempt in range(100):
            if attempt == 0:
                run_id = base_run_id
            elif attempt == 1:
                run_id = f"{base_run_id}-{now.strftime('%H%M%S')}"
            else:
                run_id = f"{base_run_id}-{now.strftime('%H%M%S')}-{attempt}"
            run_dir = root / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_id, run_dir
            except FileExistsError:
                continue
        raise FileExistsError(f"Could not create a unique podcast run directory below {root}")

    def generate(
        self,
        *,
        topic: str,
        min_minutes: float,
        max_minutes: float,
        speaker_count: int,
        quality: str,
        language: str,
        research_depth: str | None = None,
        render_audio: bool = True,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunManifest:
        selected_research_depth = research_depth or self.config.research.depth
        self.config.research.depth = selected_research_depth
        if cancellation:
            cancellation.raise_if_cancelled()
        report_progress(progress, ProgressEvent("start", "setup", "Stimmen und Run-Ordner vorbereiten"))
        profiles = select_voice_profiles(self.config, speaker_count, quality)
        speakers = build_speaker_specs(profiles)
        run_id, run_dir = self._run_dir(topic)
        report_progress(progress, ProgressEvent("done", "setup", f"Run angelegt: {run_id}", path=run_dir))

        manifest = RunManifest(
            run_id=run_id,
            topic=topic,
            language=language,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            speakers=speaker_count,
            quality=quality,
            research_depth=selected_research_depth,
        )
        write_json(run_dir / "inputs.json", manifest.model_dump(mode="json"))
        (run_dir / "effective_config.yml").write_text(config_to_yaml(self.config), encoding="utf-8")

        evidence_collector = LocalEvidenceCollector(self.config.evidence, self.project_root)
        report_progress(progress, ProgressEvent("start", "evidence", "Lokale Belege suchen"))
        local_evidence = evidence_collector.collect(
            topic=topic,
            run_dir=run_dir,
            progress=progress,
            cancellation=cancellation,
        )
        initial_evidence_items = len(local_evidence.items)
        report_progress(progress, ProgressEvent("done", "evidence", f"{len(local_evidence.items)} lokale Belege gefunden"))
        if cancellation:
            cancellation.raise_if_cancelled()
        deep_artifacts: dict[str, Path] = {}
        local_research_only = selected_research_depth != "standard"
        if selected_research_depth == "standard":
            report_progress(progress, ProgressEvent("start", "research", "Recherche mit Codex starten"))
            research = self.runner.run_structured(
                prompt=build_research_prompt(topic, language, local_evidence),
                schema_name="research",
                output_path=run_dir / "research_initial.json",
                model=ResearchReport,
                progress=progress,
                cancellation=cancellation,
            )
            report_progress(progress, ProgressEvent("done", "research", "Erste Recherche abgeschlossen"))
        else:
            report_progress(progress, ProgressEvent("start", "research", f"Tiefenrecherche starten: {selected_research_depth}"))
            deep_result = DeepResearchEngine(self.config, self.runner, self.project_root).run(
                topic=topic,
                language=language,
                run_dir=run_dir,
                local_evidence=local_evidence,
                progress=progress,
                cancellation=cancellation,
            )
            for warning in deep_result.warnings:
                self._append_warning(manifest, warning)
            deep_artifacts = deep_result.artifacts
            report_progress(progress, ProgressEvent("start", "research", "Finale Recherche aus Dossier erstellen"))
            research = self.runner.run_structured(
                prompt=build_research_from_dossier_prompt(
                    topic,
                    language,
                    deep_result.dossier,
                    deep_result.documents,
                    local_evidence,
                ),
                schema_name="research",
                output_path=run_dir / "research_initial.json",
                model=ResearchReport,
                progress=progress,
                cancellation=cancellation,
                timeout_sec=max(self.config.codex.timeout_sec, 2400),
                config_overrides={"model_reasoning_effort": "xhigh"},
                live_search=False,
            )
            report_progress(progress, ProgressEvent("done", "research", "Finale Recherche aus Dossier erstellt"))
        return self._continue_after_research(
            run_id=run_id,
            run_dir=run_dir,
            manifest=manifest,
            topic=topic,
            language=language,
            min_minutes=min_minutes,
            max_minutes=max_minutes,
            speakers=speakers,
            profiles=profiles,
            local_evidence=local_evidence,
            initial_evidence_items=initial_evidence_items,
            research=research,
            selected_research_depth=selected_research_depth,
            local_research_only=local_research_only,
            deep_artifacts=deep_artifacts,
            render_audio=render_audio,
            progress=progress,
            cancellation=cancellation,
        )

    def resume(
        self,
        *,
        run_dir: Path,
        render_audio: bool = True,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunManifest:
        run_dir = run_dir.resolve()
        manifest_path = run_dir / "manifest.json"
        inputs_path = run_dir / "inputs.json"
        if manifest_path.exists():
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        elif inputs_path.exists():
            manifest = RunManifest.model_validate_json(inputs_path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"No inputs.json or manifest.json found at {run_dir}")

        run_id = _safe_output_stem(manifest.run_id or run_dir.name)
        topic = manifest.topic
        language = manifest.language
        selected_research_depth = manifest.research_depth
        self.config.language = language
        self.config.research.depth = selected_research_depth
        self.config.tts.quality = manifest.quality
        self.config.tts.backend = backend_for_quality(self.config, manifest.quality)
        profiles = select_voice_profiles(self.config, manifest.speakers, manifest.quality)
        speakers = build_speaker_specs(profiles)

        local_evidence_path = run_dir / "local_evidence.json"
        if local_evidence_path.exists():
            local_evidence = LocalEvidenceReport.model_validate_json(local_evidence_path.read_text(encoding="utf-8"))
        else:
            evidence_collector = LocalEvidenceCollector(self.config.evidence, self.project_root)
            report_progress(progress, ProgressEvent("start", "evidence", "Lokale Belege suchen"))
            local_evidence = evidence_collector.collect(
                topic=topic,
                run_dir=run_dir,
                progress=progress,
                cancellation=cancellation,
            )
            report_progress(progress, ProgressEvent("done", "evidence", f"{len(local_evidence.items)} lokale Belege gefunden"))
        initial_evidence_items = len(local_evidence.items)
        local_research_only = selected_research_depth != "standard"
        deep_artifacts = self._existing_deep_artifacts(run_dir)

        research_path = run_dir / "research.json"
        research_initial_path = run_dir / "research_initial.json"
        if research_path.exists():
            research = ResearchReport.model_validate_json(research_path.read_text(encoding="utf-8"))
            report_progress(progress, ProgressEvent("done", "research", "Gespeicherte finale Recherche geladen"))
        elif research_initial_path.exists():
            research = ResearchReport.model_validate_json(research_initial_path.read_text(encoding="utf-8"))
            report_progress(progress, ProgressEvent("done", "research", "Gespeicherte initiale Recherche geladen"))
        elif selected_research_depth != "standard":
            report_progress(progress, ProgressEvent("start", "research", "Tiefenrecherche aus Artefakten fortsetzen"))
            deep_result = DeepResearchEngine(self.config, self.runner, self.project_root).resume_from_artifacts(
                topic=topic,
                language=language,
                run_dir=run_dir,
                progress=progress,
                cancellation=cancellation,
            )
            for warning in deep_result.warnings:
                self._append_warning(manifest, warning)
            deep_artifacts = deep_result.artifacts
            report_progress(progress, ProgressEvent("start", "research", "Finale Recherche aus Dossier erstellen"))
            research = self.runner.run_structured(
                prompt=build_research_from_dossier_prompt(
                    topic,
                    language,
                    deep_result.dossier,
                    deep_result.documents,
                    local_evidence,
                ),
                schema_name="research",
                output_path=research_initial_path,
                model=ResearchReport,
                progress=progress,
                cancellation=cancellation,
                timeout_sec=max(self.config.codex.timeout_sec, 2400),
                config_overrides={"model_reasoning_effort": "xhigh"},
                live_search=False,
            )
            report_progress(progress, ProgressEvent("done", "research", "Finale Recherche aus Dossier erstellt"))
        else:
            raise FileNotFoundError(f"No research.json or research_initial.json found at {run_dir}")

        return self._continue_after_research(
            run_id=run_id,
            run_dir=run_dir,
            manifest=manifest,
            topic=topic,
            language=language,
            min_minutes=manifest.min_minutes,
            max_minutes=manifest.max_minutes,
            speakers=speakers,
            profiles=profiles,
            local_evidence=local_evidence,
            initial_evidence_items=initial_evidence_items,
            research=research,
            selected_research_depth=selected_research_depth,
            local_research_only=local_research_only,
            deep_artifacts=deep_artifacts,
            render_audio=render_audio,
            progress=progress,
            cancellation=cancellation,
        )

    def _existing_deep_artifacts(self, run_dir: Path) -> dict[str, Path]:
        research_root = run_dir / "deep_research"
        paths = {
            "research_plan": run_dir / "research_plan.json",
            "deep_research_frontier": research_root / "frontier.jsonl",
            "deep_research_evidence": research_root / "evidence.jsonl",
            "deep_research_topics": research_root / "topics.json",
            "research_dossier": research_root / "research_dossier.json",
            "research_dossier_notes": research_root / "research_dossier.md",
            "deep_research_quality": research_root / "quality_report.json",
        }
        return {name: path for name, path in paths.items() if path.exists()}

    def _continue_after_research(
        self,
        *,
        run_id: str,
        run_dir: Path,
        manifest: RunManifest,
        topic: str,
        language: str,
        min_minutes: float,
        max_minutes: float,
        speakers,
        profiles,
        local_evidence: LocalEvidenceReport,
        initial_evidence_items: int,
        research: ResearchReport,
        selected_research_depth: str,
        local_research_only: bool,
        deep_artifacts: dict[str, Path],
        render_audio: bool,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunManifest:
        research_path = run_dir / "research.json"
        if research_path.exists():
            research = ResearchReport.model_validate_json(research_path.read_text(encoding="utf-8"))
            report_progress(progress, ProgressEvent("done", "research", "Gespeicherte finale Recherche geladen"))
        else:
            evidence_collector = LocalEvidenceCollector(self.config.evidence, self.project_root)
            report_progress(progress, ProgressEvent("start", "evidence", "Belege aus Recherchequellen nachladen"))
            local_evidence = evidence_collector.collect(
                topic=topic,
                research=research,
                run_dir=run_dir,
                existing=local_evidence,
                progress=progress,
                cancellation=cancellation,
            )
            report_progress(progress, ProgressEvent("done", "evidence", f"{len(local_evidence.items)} lokale Belege verfuegbar"))
            if local_evidence.has_items() and len(local_evidence.items) > initial_evidence_items:
                report_progress(progress, ProgressEvent("start", "research", "Recherche mit lokalen Belegen verfeinern"))
                research = self.runner.run_structured(
                    prompt=build_evidence_enriched_research_prompt(topic, language, research, local_evidence),
                    schema_name="research",
                    output_path=research_path,
                    model=ResearchReport,
                    progress=progress,
                    cancellation=cancellation,
                    live_search=False if local_research_only else None,
                )
                report_progress(progress, ProgressEvent("done", "research", "Angereicherte Recherche abgeschlossen"))
            else:
                research_path.write_text(research.model_dump_json(indent=2), encoding="utf-8")

        if cancellation:
            cancellation.raise_if_cancelled()
        validation_path = run_dir / "validation.json"
        if validation_path.exists():
            validation = ValidationReport.model_validate_json(validation_path.read_text(encoding="utf-8"))
            report_progress(progress, ProgressEvent("done", "validation", "Gespeicherte Validierung geladen"))
        else:
            report_progress(progress, ProgressEvent("start", "validation", "Claims validieren"))
            validation = self.runner.run_structured(
                prompt=build_validation_prompt(research),
                schema_name="validation",
                output_path=validation_path,
                model=ValidationReport,
                progress=progress,
                cancellation=cancellation,
                live_search=False if local_research_only else None,
            )
            report_progress(progress, ProgressEvent("done", "validation", "Validierung abgeschlossen"))
        validation_revision_path = run_dir / "validation_revision_1.json"
        if (
            selected_research_depth == "dossier"
            and validation.pass_status == "needs_revision"
            and not validation_revision_path.exists()
        ):
            report_progress(progress, ProgressEvent("start", "research", "Dossier-Recherche gezielt nachvalidieren"))
            research = self.runner.run_structured(
                prompt=build_research_revision_prompt(research, validation),
                schema_name="research",
                output_path=run_dir / "research_revision_1.json",
                model=ResearchReport,
                progress=progress,
                cancellation=cancellation,
                timeout_sec=max(self.config.codex.timeout_sec, 2400),
                config_overrides={"model_reasoning_effort": "xhigh"},
                live_search=False,
            )
            research_path.write_text(research.model_dump_json(indent=2), encoding="utf-8")
            validation = self.runner.run_structured(
                prompt=build_validation_prompt(research),
                schema_name="validation",
                output_path=validation_revision_path,
                model=ValidationReport,
                progress=progress,
                cancellation=cancellation,
                timeout_sec=max(self.config.codex.timeout_sec, 2400),
                config_overrides={"model_reasoning_effort": "xhigh"},
                live_search=False,
            )
            validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
            if validation.pass_status == "needs_revision":
                self._append_warning(manifest, "central_claims_weak")
            report_progress(progress, ProgressEvent("done", "validation", "Dossier-Nachvalidierung abgeschlossen"))
        elif selected_research_depth != "standard" and validation.pass_status == "needs_revision":
            self._append_warning(manifest, "central_claims_weak")

        script_path = run_dir / "script.json"
        if script_path.exists():
            script = PodcastScript.model_validate_json(script_path.read_text(encoding="utf-8"))
            script = self._normalize_script(script, speakers, min_minutes, max_minutes, language)
            script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
            report_progress(progress, ProgressEvent("done", "script", f"Gespeichertes Skript mit {len(script.lines)} TTS-Zeilen geladen"))
        else:
            report_progress(progress, ProgressEvent("start", "script", "Podcast-Skript schreiben"))
            script = self.runner.run_structured(
                prompt=build_script_prompt(
                    topic,
                    research,
                    validation,
                    speakers,
                    min_minutes,
                    max_minutes,
                    self.config.generation.words_per_minute,
                    language,
                ),
                schema_name="script",
                output_path=script_path,
                model=PodcastScript,
                progress=progress,
                cancellation=cancellation,
                live_search=False if local_research_only else None,
            )
            script = self._normalize_script(script, speakers, min_minutes, max_minutes, language)
            report_progress(progress, ProgressEvent("done", "script", f"Skript mit {len(script.lines)} TTS-Zeilen erstellt"))
            script = self._fit_script_duration(
                script,
                run_dir,
                progress=progress,
                cancellation=cancellation,
                live_search=False if local_research_only else None,
            )
        report_progress(progress, ProgressEvent("start", "artifacts", "Transkript und Quellen schreiben"))
        script_artifacts = self._write_script_artifacts(script, research, validation, run_dir, run_id)
        report_progress(progress, ProgressEvent("done", "artifacts", "Transkript und Quellen geschrieben"))

        manifest.artifacts.update(
            {
                "research": str(research_path),
                "research_initial": str(run_dir / "research_initial.json"),
                "local_evidence": str(run_dir / "local_evidence.json"),
                "local_evidence_notes": str(run_dir / "local_evidence.md"),
                "validation": str(validation_path),
                "script": str(script_path),
                "transcript": str(script_artifacts["transcript"]),
                "sources": str(script_artifacts["sources"]),
            }
        )
        manifest.artifacts.update({name: str(path) for name, path in deep_artifacts.items()})

        if render_audio:
            report_progress(progress, ProgressEvent("start", "tts", "Audiosegmente rendern", current=0, total=len(script.lines)))
            rendered = ScriptRenderer(self.config, voice_map(profiles)).render_script(
                script,
                run_dir,
                progress=progress,
                cancellation=cancellation,
            )
            write_json(run_dir / "segments.json", [item.model_dump(mode="json") for item in rendered])
            report_progress(progress, ProgressEvent("done", "tts", f"{len(rendered)} Audiosegmente gerendert"))
            report_progress(progress, ProgressEvent("start", "assembly", "Episode zusammensetzen", current=0, total=len(rendered)))
            final_wav, final_mp3, duration_sec = assemble_episode(
                rendered,
                run_dir,
                self.config.audio,
                self.config.generation.pause_between_lines_sec,
                run_id,
                progress=progress,
                cancellation=cancellation,
            )
            report_progress(progress, ProgressEvent("done", "assembly", "Finale Audiodateien exportiert", path=final_mp3))
            manifest.artifacts.update({"final_wav": str(final_wav), "final_mp3": str(final_mp3)})
            actual_minutes = duration_sec / 60.0
            if actual_minutes < min_minutes or actual_minutes > max_minutes:
                self._append_warning(
                    manifest,
                    f"Rendered audio duration is {actual_minutes:.2f} minutes, outside target "
                    f"{min_minutes:.2f}-{max_minutes:.2f}.",
                )

        write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
        report_progress(progress, ProgressEvent("done", "complete", "Podcast fertig", path=run_dir))
        return manifest

    def _append_warning(self, manifest: RunManifest, warning: str) -> None:
        if warning not in manifest.warnings:
            manifest.warnings.append(warning)

    def _normalize_script(
        self,
        script: PodcastScript,
        speakers,
        min_minutes: float,
        max_minutes: float,
        language: str,
    ) -> PodcastScript:
        script.speakers = speakers
        script.target_min_minutes = min_minutes
        script.target_max_minutes = max_minutes
        script.language = language
        valid_speaker_ids = {speaker.id for speaker in speakers}
        bad_lines = [line.speaker_id for line in script.lines if line.speaker_id not in valid_speaker_ids]
        if bad_lines:
            raise ValueError(f"script contains unknown speaker ids: {sorted(set(bad_lines))}")
        script.lines = split_script_lines_for_tts(script.lines)
        script.estimated_words = script_word_count(script)
        return script

    def _fit_script_duration(
        self,
        script: PodcastScript,
        run_dir: Path,
        *,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
        live_search: bool | None = None,
    ) -> PodcastScript:
        current = script
        for attempt in range(self.config.generation.rewrite_attempts + 1):
            if cancellation:
                cancellation.raise_if_cancelled()
            status, words, minutes = duration_status(current, self.config.generation.words_per_minute)
            current.estimated_words = words
            report_progress(
                progress,
                ProgressEvent(
                    "progress",
                    "script",
                    f"Skriptlaenge: {minutes:.1f} Minuten ({status})",
                    current=attempt,
                    total=self.config.generation.rewrite_attempts,
                ),
            )
            if status == "ok" or attempt >= self.config.generation.rewrite_attempts:
                (run_dir / "script.json").write_text(current.model_dump_json(indent=2), encoding="utf-8")
                return current
            report_progress(progress, ProgressEvent("start", "script", f"Skript rewrite {attempt + 1} starten"))
            current = self.runner.run_structured(
                prompt=build_rewrite_prompt(current, status, words, minutes, self.config.generation.words_per_minute),
                schema_name="script",
                output_path=run_dir / f"script_rewrite_{attempt + 1}.json",
                model=PodcastScript,
                progress=progress,
                cancellation=cancellation,
                live_search=live_search,
            )
            current = self._normalize_script(
                current,
                script.speakers,
                script.target_min_minutes,
                script.target_max_minutes,
                script.language,
            )
        return current

    def _write_script_artifacts(
        self,
        script: PodcastScript,
        research: ResearchReport,
        validation: ValidationReport,
        run_dir: Path,
        output_stem: str,
    ) -> dict[str, Path]:
        transcript_path = run_dir / f"{output_stem}-transcript.md"
        sources_path = run_dir / f"{output_stem}-sources.md"

        transcript_lines = [f"# {script.title}", "", f"Thema: {script.topic}", ""]
        speakers = {speaker.id: speaker for speaker in script.speakers}
        for line in script.lines:
            speaker = speakers[line.speaker_id]
            transcript_lines.append(f"**{speaker.display_name}:** {line.text}")
            transcript_lines.append("")
        transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")

        validation_by_claim = {finding.claim_id: finding for finding in validation.findings}
        source_by_id = {source.id: source for source in research.sources}
        source_lines = [f"# Quellen zu {script.title}", ""]
        for source in research.sources:
            source_lines.append(f"- **{source.id}**: [{source.title}]({source.url})")
            if source.publisher:
                source_lines.append(f"  - Publisher: {source.publisher}")
            if source.published_at:
                source_lines.append(f"  - Published: {source.published_at}")
            source_lines.append(f"  - Relevanz: {source.relevance}")
        source_lines.append("")
        source_lines.append("## Claims")
        for claim in research.claims:
            finding = validation_by_claim.get(claim.id)
            status = finding.status.value if finding else "not_validated"
            source_titles = [source_by_id[sid].title for sid in claim.source_ids if sid in source_by_id]
            source_lines.append(f"- **{claim.id}** ({status}): {claim.text}")
            if source_titles:
                source_lines.append(f"  - Quellen: {', '.join(source_titles)}")
        sources_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
        return {"transcript": transcript_path, "sources": sources_path}


def split_script_lines_for_tts(lines: list[ScriptLine], limit: int = TTS_LINE_CHAR_LIMIT) -> list[ScriptLine]:
    split_lines: list[ScriptLine] = []
    for line in lines:
        parts = split_text_for_tts(line.text, limit)
        for index, part in enumerate(parts):
            split_lines.append(
                ScriptLine(
                    speaker_id=line.speaker_id,
                    text=part,
                    claim_ids=list(line.claim_ids),
                    stage_direction=line.stage_direction if index == 0 else None,
                )
            )
    return split_lines


def split_text_for_tts(text: str, limit: int = TTS_LINE_CHAR_LIMIT) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return [normalized]

    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_split_oversized_piece(sentence, limit))

    packed: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
            current = f"{current} {piece}"
        else:
            packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return packed


def _split_oversized_piece(text: str, limit: int) -> list[str]:
    result: list[str] = []
    remaining = text.strip()
    minimum_break = max(1, int(limit * 0.45))
    while len(remaining) > limit:
        candidates: list[int] = []
        for separator in (". ", "! ", "? ", "; ", ": ", ", ", " "):
            index = remaining.rfind(separator, 0, limit + 1)
            if index >= minimum_break:
                candidates.append(index + (1 if separator != " " else 0))
        split_at = max(candidates) if candidates else limit
        part = remaining[:split_at].strip()
        if not part:
            part = remaining[:limit].strip()
        result.append(part)
        remaining = remaining[len(part) :].strip()
    if remaining:
        result.append(remaining)
    return result
