from __future__ import annotations

import html
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import EvidenceConfig
from .models import LocalEvidenceFailure, LocalEvidenceItem, LocalEvidenceReport, ResearchReport
from .progress import CancellationToken, ProgressEvent, ProgressReporter, report_progress
from .util import write_json

URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}


def normalize_youtube_url(raw_url: str) -> str | None:
    url = raw_url.strip().rstrip(".,;:)]}")
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in YOUTUBE_HOSTS:
        return None

    video_id: str | None = None
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0] or None
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    else:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"}:
            video_id = parts[1]

    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"


def extract_youtube_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in URL_RE.finditer(text):
        normalized = normalize_youtube_url(match.group(0))
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)
    return urls


def vtt_to_plain_text(content: str) -> str:
    lines: list[str] = []
    previous = ""
    skipping_note = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            skipping_note = False
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "STYLE", "REGION")):
            continue
        if line.startswith("NOTE"):
            skipping_note = True
            continue
        if skipping_note:
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue

        line = re.sub(r"<[^>]+>", "", line)
        line = html.unescape(line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line == previous:
            continue
        previous = line
        lines.append(line)
    return "\n".join(lines)


class LocalEvidenceCollector:
    def __init__(self, config: EvidenceConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def collect(
        self,
        *,
        topic: str,
        run_dir: Path,
        research: ResearchReport | None = None,
        existing: LocalEvidenceReport | None = None,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> LocalEvidenceReport:
        report = existing or LocalEvidenceReport()
        if not self.config.enabled or not self.config.youtube_enabled:
            report_progress(progress, ProgressEvent("log", "evidence", "Lokale Belege sind deaktiviert"))
            self._write_report(report, run_dir)
            return report

        urls = self._candidate_urls(topic, research)
        report_progress(progress, ProgressEvent("log", "evidence", f"{len(urls)} YouTube-Kandidaten gefunden"))
        attempted = report.attempted_urls()
        total_chars = sum(item.transcript_chars for item in report.items)
        next_index = len(report.items) + 1

        for url in urls:
            if cancellation:
                cancellation.raise_if_cancelled()
            if url in attempted:
                continue
            if len(attempted) >= self.config.max_youtube_urls:
                break
            attempted.add(url)
            report_progress(
                progress,
                ProgressEvent(
                    "progress",
                    "evidence",
                    f"YouTube-Transcript abrufen: {url}",
                    current=min(len(attempted), self.config.max_youtube_urls),
                    total=min(len(urls), self.config.max_youtube_urls),
                ),
            )
            try:
                item = self._fetch_youtube_transcript(url, run_dir, next_index, total_chars)
            except Exception as exc:
                report.failures.append(LocalEvidenceFailure(url=url, reason=str(exc)))
                report_progress(progress, ProgressEvent("log", "evidence", f"Transcript fehlgeschlagen: {exc}", level="warning"))
                continue
            report.items.append(item)
            report_progress(progress, ProgressEvent("log", "evidence", f"Transcript gespeichert: {item.id}"))
            total_chars += item.transcript_chars
            next_index += 1
            if total_chars >= self.config.max_total_transcript_chars:
                break

        self._write_report(report, run_dir)
        return report

    def _candidate_urls(self, topic: str, research: ResearchReport | None) -> list[str]:
        texts = [topic]
        if research is not None:
            texts.extend(str(source.url) for source in research.sources)
        seen: set[str] = set()
        urls: list[str] = []
        for text in texts:
            for url in extract_youtube_urls(text):
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        return urls

    def _fetch_youtube_transcript(
        self,
        url: str,
        run_dir: Path,
        index: int,
        existing_chars: int,
    ) -> LocalEvidenceItem:
        evidence_root = run_dir / "local_evidence" / f"youtube_{index:02d}"
        evidence_root.mkdir(parents=True, exist_ok=True)
        output_template = evidence_root / "%(id)s.%(ext)s"
        cmd = [
            self.config.yt_dlp_executable,
            "--skip-download",
            "--write-info-json",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            ",".join(self.config.preferred_subtitle_languages),
            "--sub-format",
            "vtt",
            "--no-playlist",
            "-o",
            str(output_template),
        ]
        if self.config.cookies_from_browser:
            cmd.extend(["--cookies-from-browser", self.config.cookies_from_browser])
        cmd.append(url)

        stdout_path = evidence_root / "yt-dlp.stdout.log"
        stderr_path = evidence_root / "yt-dlp.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                text=True,
                stdout=stdout,
                stderr=stderr,
                timeout=self.config.timeout_sec,
            )
        if result.returncode != 0:
            detail = self._stderr_summary(stderr_path)
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"yt-dlp failed with exit code {result.returncode}{suffix}")

        transcript_file = self._select_transcript_file(evidence_root)
        if transcript_file is None:
            raise RuntimeError("yt-dlp did not download subtitles for the preferred languages")
        subtitle_size = transcript_file.stat().st_size
        if subtitle_size > self.config.max_subtitle_bytes:
            raise RuntimeError(
                f"subtitle file is too large: {subtitle_size} bytes "
                f"(limit {self.config.max_subtitle_bytes})"
            )

        transcript = vtt_to_plain_text(transcript_file.read_text(encoding="utf-8", errors="replace"))
        if not transcript.strip():
            raise RuntimeError(f"downloaded subtitle file was empty after parsing: {transcript_file.name}")

        remaining_chars = max(0, self.config.max_total_transcript_chars - existing_chars)
        max_chars = min(self.config.max_transcript_chars, remaining_chars or self.config.max_transcript_chars)
        excerpt = transcript[:max_chars].rstrip()
        is_truncated = len(transcript) > len(excerpt)
        text_path = evidence_root / f"{transcript_file.stem}.txt"
        text_path.write_text(transcript, encoding="utf-8")
        info = self._read_info_json(evidence_root)

        return LocalEvidenceItem(
            id=f"E{index}",
            url=url,
            title=info.get("title"),
            publisher=info.get("uploader") or info.get("channel"),
            published_at=self._format_upload_date(info.get("upload_date")),
            language=self._language_from_subtitle_name(transcript_file),
            transcript_path=str(text_path.relative_to(run_dir)),
            transcript_chars=len(transcript),
            transcript_excerpt=excerpt,
            is_truncated=is_truncated,
        )

    def _select_transcript_file(self, root: Path) -> Path | None:
        files = sorted(root.glob("*.vtt"))
        if not files:
            return None

        def score(path: Path) -> tuple[int, str]:
            language = self._language_from_subtitle_name(path) or ""
            for index, preferred in enumerate(self.config.preferred_subtitle_languages):
                prefix = preferred.rstrip(".*").lower()
                if language.lower() == prefix or language.lower().startswith(f"{prefix}-"):
                    return (index, path.name)
            return (len(self.config.preferred_subtitle_languages), path.name)

        return sorted(files, key=score)[0]

    def _language_from_subtitle_name(self, path: Path) -> str | None:
        parts = path.name.split(".")
        if len(parts) < 3:
            return None
        return parts[-2]

    def _read_info_json(self, root: Path) -> dict:
        info_files = sorted(root.glob("*.info.json"))
        if not info_files:
            return {}
        try:
            return json.loads(info_files[0].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _stderr_summary(self, stderr_path: Path) -> str:
        stderr = _read_tail_text(stderr_path, max_bytes=16_384)
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(("ERROR:", "WARNING:")):
                return line[:500]
        return " ".join(lines[-3:])[:500]

    def _format_upload_date(self, value: object) -> str | None:
        if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
            return None
        return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"

    def _write_report(self, report: LocalEvidenceReport, run_dir: Path) -> None:
        write_json(run_dir / "local_evidence.json", report.model_dump(mode="json"))
        lines = ["# Lokale Belege", ""]
        if not report.items and not report.failures:
            lines.append("Keine lokalen Belege gefunden oder konfiguriert.")
        for item in report.items:
            label = item.title or item.url
            lines.append(f"- **{item.id}**: [{label}]({item.url})")
            if item.publisher:
                lines.append(f"  - Kanal/Publisher: {item.publisher}")
            if item.published_at:
                lines.append(f"  - Veroeffentlicht: {item.published_at}")
            if item.language:
                lines.append(f"  - Transcript-Sprache: {item.language}")
            lines.append(f"  - Transcript: `{item.transcript_path}`")
            if item.is_truncated:
                lines.append("  - Hinweis: Auszug fuer den Prompt gekuerzt, Rohtranskript ist vollstaendig gespeichert.")
        if report.failures:
            lines.extend(["", "## Fehlgeschlagene Abrufe"])
            for failure in report.failures:
                lines.append(f"- {failure.url}: {failure.reason}")
        (run_dir / "local_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_tail_text(path: Path, *, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""
