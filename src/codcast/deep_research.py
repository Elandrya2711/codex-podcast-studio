from __future__ import annotations

import json
import ipaddress
import re
import socket
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol, TypeVar
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests

from .codex_runner import CodexRunner
from .config import AppConfig
from .local_evidence import LocalEvidenceCollector, normalize_youtube_url
from .models import (
    DeepResearchDocument,
    DeepResearchPlan,
    DiscoveredTopic,
    EvidenceBatch,
    LocalEvidenceReport,
    ResearchDossier,
    ResearchQuery,
)
from .progress import CancellationToken, ProgressEvent, ProgressReporter, report_progress
from .prompts import build_deep_research_plan_prompt, build_evidence_batch_prompt, build_research_dossier_prompt
from .util import write_json

T = TypeVar("T")


@dataclass(frozen=True)
class ResearchLimits:
    max_seconds: float
    max_rounds: int
    max_documents: int


@dataclass
class WebResult:
    url: str
    title: str | None = None
    content: str | None = None
    raw_content: str | None = None
    score: float | None = None
    published_at: str | None = None
    publisher: str | None = None


@dataclass
class DeepResearchResult:
    documents: list[DeepResearchDocument]
    evidence_batches: list[EvidenceBatch]
    dossier: ResearchDossier
    artifacts: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ResearchProviderError(RuntimeError):
    def __init__(self, message: str, *, code: str = "provider_error") -> None:
        super().__init__(message)
        self.code = code


class ResearchProvider(Protocol):
    def search(self, query: str, *, max_results: int, depth: str) -> list[WebResult]: ...

    def extract(self, urls: list[str], *, query: str, depth: str) -> list[WebResult]: ...

    def crawl(self, url: str, *, instructions: str, limit: int, depth: str) -> list[WebResult]: ...


class SearxngResearchProvider:
    def __init__(self, config: AppConfig) -> None:
        self.base_url = config.research.searxng_base_url.rstrip("/")
        self.timeout_sec = config.research.timeout_sec
        self.categories = config.research.searxng_categories
        self.language = config.research.searxng_language
        self.max_fetch_bytes = config.research.max_fetch_bytes
        self.allow_private_networks = config.research.allow_private_networks
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.research.user_agent})

    def search(self, query: str, *, max_results: int, depth: str) -> list[WebResult]:
        wanted = max(1, max_results)
        results: list[WebResult] = []
        seen_urls: set[str] = set()
        max_pages = 3 if depth == "dossier" and wanted > 20 else 1
        for page in range(1, max_pages + 1):
            payload = self._search_json(query, page)
            for item in payload.get("results", []):
                url = _normalize_result_url(str(item.get("url") or ""))
                if not _is_http_url(url) or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    WebResult(
                        url=url,
                        title=_clean_inline_text(str(item.get("title") or "")) or None,
                        content=_clean_inline_text(str(item.get("content") or "")) or None,
                        score=_float_or_none(item.get("score")),
                        published_at=item.get("publishedDate")
                        or item.get("published_date")
                        or item.get("published_at"),
                        publisher=str(item.get("engine") or "") or _domain(url),
                    )
                )
                if len(results) >= wanted:
                    return results
            if len(payload.get("results", [])) == 0:
                break
        return results

    def extract(self, urls: list[str], *, query: str, depth: str) -> list[WebResult]:
        extracted: list[WebResult] = []
        for url in urls:
            page = self._fetch_and_extract(url, depth=depth)
            if page is not None:
                extracted.append(page)
        return extracted

    def crawl(self, url: str, *, instructions: str, limit: int, depth: str) -> list[WebResult]:
        if not _is_http_url(url):
            return []
        max_depth = 2 if depth == "dossier" else 1
        root_domain = _domain(url)
        queue: deque[tuple[str, int]] = deque([(url, 0)])
        seen = {url}
        documents: list[WebResult] = []
        while queue and len(documents) < max(1, limit):
            current_url, current_depth = queue.popleft()
            fetched = self._fetch_html(current_url)
            if fetched is None:
                continue
            final_url, html = fetched
            title, text, published_at = _extract_readable_text(html, final_url, depth=depth)
            page = WebResult(
                url=final_url,
                title=title,
                content=_trim_text(text, 1200) or None,
                raw_content=text,
                published_at=published_at,
                publisher=_domain(final_url),
            )
            documents.append(page)
            if current_depth >= max_depth:
                continue
            for link in _extract_links(html, final_url):
                if len(seen) > limit * 4:
                    break
                if link in seen or _domain(link) != root_domain or not _looks_like_content_url(link):
                    continue
                seen.add(link)
                queue.append((link, current_depth + 1))
        return documents

    def _search_json(self, query: str, page: int) -> dict:
        params = {
            "q": query,
            "format": "json",
            "categories": self.categories,
            "language": self.language,
            "pageno": str(page),
        }
        try:
            response = self.session.get(f"{self.base_url}/search", params=params, timeout=self.timeout_sec)
        except requests.RequestException as exc:
            raise ResearchProviderError(
                f"Lokaler SearXNG-Dienst ist nicht erreichbar unter {self.base_url}. "
                "Starte SearXNG lokal und aktiviere das JSON-Format.",
                code="local_search_unavailable",
            ) from exc
        if response.status_code == 403:
            raise ResearchProviderError(
                "SearXNG lehnt JSON-Ergebnisse ab. Aktiviere in settings.yml unter search.formats den Wert json.",
                code="searxng_json_disabled",
            )
        if response.status_code == 429:
            raise ResearchProviderError("SearXNG oder eine Upstream-Suche rate-limited die Anfrage.", code="provider_rate_limited")
        if response.status_code >= 400:
            raise ResearchProviderError(f"SearXNG-Fehler {response.status_code}: {response.text[:500]}", code="local_search_error")
        try:
            return response.json()
        except ValueError as exc:
            raise ResearchProviderError(
                "SearXNG hat kein gueltiges JSON geliefert. Pruefe search.formats und die lokale Instanz.",
                code="searxng_invalid_json",
            ) from exc

    def _fetch_and_extract(self, url: str, *, depth: str) -> WebResult | None:
        fetched = self._fetch_html(url)
        if fetched is None:
            return None
        final_url, html = fetched
        title, text, published_at = _extract_readable_text(html, final_url, depth=depth)
        return WebResult(
            url=final_url,
            title=title,
            content=_trim_text(text, 1200) or None,
            raw_content=text,
            published_at=published_at,
            publisher=_domain(final_url),
        )

    def _fetch_html(self, url: str) -> tuple[str, str] | None:
        url = _normalize_result_url(url)
        if not _is_fetchable_url(url, allow_private_networks=self.allow_private_networks):
            return None
        try:
            response = self.session.get(url, timeout=self.timeout_sec, allow_redirects=True, stream=True)
        except requests.RequestException:
            return None
        if response.status_code >= 400:
            return None
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
            return None
        body = bytearray()
        try:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) >= self.max_fetch_bytes:
                    break
        finally:
            response.close()
        raw_body = bytes(body)
        encoding = response.encoding
        if not encoding or encoding.lower().replace("_", "-") in {"iso-8859-1", "latin-1"}:
            try:
                html = raw_body.decode("utf-8")
            except UnicodeDecodeError:
                html = raw_body.decode(encoding or "utf-8", errors="replace")
        else:
            html = raw_body.decode(encoding, errors="replace")
        return response.url, html


class DeepResearchEngine:
    def __init__(
        self,
        config: AppConfig,
        runner: CodexRunner,
        project_root: Path,
        provider: ResearchProvider | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.project_root = project_root
        self.provider = provider

    def run(
        self,
        *,
        topic: str,
        language: str,
        run_dir: Path,
        local_evidence: LocalEvidenceReport | None = None,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> DeepResearchResult:
        if self.config.research.depth == "standard":
            raise ValueError("DeepResearchEngine cannot run with research.depth=standard")

        research_root = run_dir / "deep_research"
        document_root = research_root / "documents"
        document_root.mkdir(parents=True, exist_ok=True)
        limits = self._limits()
        deadline = time.monotonic() + limits.max_seconds
        warnings: list[str] = []

        provider = self.provider or SearxngResearchProvider(self.config)
        plan = self._plan(topic, language, run_dir, progress=progress, cancellation=cancellation)
        frontier = self._initial_frontier(plan, topic)
        seen_queries = {_normalize_query(query.query) for query in frontier}
        seen_urls: set[str] = set()
        documents: list[DeepResearchDocument] = []
        evidence_batches: list[EvidenceBatch] = []
        known_topics: list[str] = []

        self._write_frontier(research_root, [{"event": "plan", "queries": [q.model_dump(mode="json") for q in frontier]}])

        local_evidence_report = local_evidence or LocalEvidenceReport()
        local_documents = self._collect_local_evidence_documents(
            local_evidence_report,
            run_dir,
            documents,
            seen_urls,
            document_root,
            limits,
        )
        if local_documents:
            self._write_frontier(
                research_root,
                [{"event": "local_evidence", "new_documents": len(local_documents)}],
            )
            batches = self._extract_evidence(
                topic,
                language,
                local_documents,
                known_topics,
                research_root,
                0,
                progress,
                cancellation,
            )
            evidence_batches.extend(batches)
            for batch in batches:
                self._append_jsonl(research_root / "evidence.jsonl", batch.model_dump(mode="json"))
                for discovered in batch.discovered_topics:
                    self._merge_topic(discovered, known_topics)
                for query in self._follow_up_queries(batch):
                    normalized = _normalize_query(query.query)
                    if normalized and normalized not in seen_queries:
                        frontier.append(query)
                        seen_queries.add(normalized)

        if self.config.research.crawl_high_authority_domains and plan.crawl_urls:
            crawled_documents = self._collect_crawl_documents(
                provider,
                plan,
                documents,
                seen_urls,
                document_root,
                limits,
                progress,
                cancellation,
            )
            if crawled_documents:
                batches = self._extract_evidence(
                    topic,
                    language,
                    crawled_documents,
                    known_topics,
                    research_root,
                    0,
                    progress,
                    cancellation,
                )
                evidence_batches.extend(batches)
                for batch in batches:
                    self._append_jsonl(research_root / "evidence.jsonl", batch.model_dump(mode="json"))
                    for discovered in batch.discovered_topics:
                        self._merge_topic(discovered, known_topics)
                    for query in self._follow_up_queries(batch):
                        normalized = _normalize_query(query.query)
                        if normalized and normalized not in seen_queries:
                            frontier.append(query)
                            seen_queries.add(normalized)

        for round_index in range(1, limits.max_rounds + 1):
            if cancellation:
                cancellation.raise_if_cancelled()
            if time.monotonic() >= deadline:
                warnings.append("research_depth_budget_exhausted")
                break
            if len(documents) >= limits.max_documents:
                break
            if not frontier:
                break

            queries = self._pop_queries(frontier, self.config.research.queries_per_round)
            report_progress(progress, ProgressEvent("start", "research", f"Tiefenrecherche Runde {round_index}: {len(queries)} Queries"))
            try:
                round_documents = self._search_round(
                    provider,
                    queries,
                    run_dir,
                    local_evidence_report,
                    documents,
                    seen_urls,
                    document_root,
                    limits,
                    progress,
                    cancellation,
                )
            except ResearchProviderError as exc:
                warnings.append(exc.code)
                if exc.code == "provider_rate_limited":
                    warnings.append("provider_rate_limited")
                    break
                raise RuntimeError(str(exc)) from exc
            self._write_frontier(
                research_root,
                [{"event": "round", "round": round_index, "queries": [q.model_dump(mode="json") for q in queries], "new_documents": len(round_documents)}],
            )
            if round_documents:
                batches = self._extract_evidence(
                    topic,
                    language,
                    round_documents,
                    known_topics,
                    research_root,
                    round_index,
                    progress,
                    cancellation,
                )
                evidence_batches.extend(batches)
                for batch in batches:
                    self._append_jsonl(research_root / "evidence.jsonl", batch.model_dump(mode="json"))
                    for discovered in batch.discovered_topics:
                        self._merge_topic(discovered, known_topics)
                    for query in self._follow_up_queries(batch):
                        normalized = _normalize_query(query.query)
                        if normalized and normalized not in seen_queries:
                            frontier.append(query)
                            seen_queries.add(normalized)
            report_progress(progress, ProgressEvent("done", "research", f"Tiefenrecherche Runde {round_index} abgeschlossen"))

        if not documents:
            raise RuntimeError("Tiefenrecherche hat keine Webdokumente gefunden.")

        if time.monotonic() >= deadline:
            warnings.append("research_depth_budget_exhausted")
        if self._has_low_source_diversity(documents):
            warnings.append("low_source_diversity")

        write_json(research_root / "topics.json", known_topics)
        dossier = self._build_dossier(topic, language, documents, evidence_batches, run_dir, research_root, progress, cancellation)
        return self._finish_result(
            documents,
            evidence_batches,
            dossier,
            run_dir,
            research_root,
            limits,
            warnings,
        )

    def resume_from_artifacts(
        self,
        *,
        topic: str,
        language: str,
        run_dir: Path,
        progress: ProgressReporter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> DeepResearchResult:
        research_root = run_dir / "deep_research"
        documents = self._load_documents(research_root)
        if not documents:
            raise RuntimeError(f"Keine gespeicherten Deep-Research-Dokumente gefunden: {research_root}")
        evidence_batches = self._load_evidence_batches(research_root)
        if not evidence_batches:
            raise RuntimeError(f"Keine gespeicherten Deep-Research-Belege gefunden: {research_root}")
        if cancellation:
            cancellation.raise_if_cancelled()

        known_topics = self._load_topics(research_root)
        if not known_topics:
            for batch in evidence_batches:
                for discovered in batch.discovered_topics:
                    self._merge_topic(discovered, known_topics)
            write_json(research_root / "topics.json", known_topics)

        dossier_path = research_root / "research_dossier.json"
        if dossier_path.exists():
            dossier = ResearchDossier.model_validate_json(dossier_path.read_text(encoding="utf-8"))
            write_json(run_dir / "research_dossier.json", dossier.model_dump(mode="json"))
            report_progress(progress, ProgressEvent("done", "research", f"Gespeichertes Dossier mit {len(dossier.claims)} Claims geladen"))
        else:
            dossier = self._build_dossier(topic, language, documents, evidence_batches, run_dir, research_root, progress, cancellation)

        warnings: list[str] = []
        if self._has_low_source_diversity(documents):
            warnings.append("low_source_diversity")
        return self._finish_result(
            documents,
            evidence_batches,
            dossier,
            run_dir,
            research_root,
            self._limits(),
            warnings,
        )

    def _plan(
        self,
        topic: str,
        language: str,
        run_dir: Path,
        *,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> DeepResearchPlan:
        report_progress(progress, ProgressEvent("start", "research", "Tiefenrecherche planen"))
        plan = self.runner.run_structured(
            prompt=build_deep_research_plan_prompt(topic, language, self.config.research.depth),
            schema_name="research_plan",
            output_path=run_dir / "research_plan.json",
            model=DeepResearchPlan,
            progress=progress,
            cancellation=cancellation,
            timeout_sec=self._codex_timeout(),
            config_overrides=self._codex_overrides(),
            live_search=False,
        )
        report_progress(progress, ProgressEvent("done", "research", f"Rechercheplan mit {len(plan.seed_queries)} Queries erstellt"))
        return plan

    def _collect_crawl_documents(
        self,
        provider: ResearchProvider,
        plan: DeepResearchPlan,
        documents: list[DeepResearchDocument],
        seen_urls: set[str],
        document_root: Path,
        limits: ResearchLimits,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> list[DeepResearchDocument]:
        crawled_documents: list[DeepResearchDocument] = []
        crawl_limit = 20 if self.config.research.depth == "dossier" else 10
        for crawl_url in plan.crawl_urls[:3]:
            if cancellation:
                cancellation.raise_if_cancelled()
            remaining = limits.max_documents - len(documents)
            if remaining <= 0:
                return crawled_documents
            report_progress(progress, ProgressEvent("log", "research", f"Crawl: {crawl_url}"))
            results = provider.crawl(
                crawl_url,
                instructions="Finde relevante Quellen, Daten, FAQ-, Dokumentations- und Hintergrundseiten zum Recherchethema.",
                limit=min(crawl_limit, remaining),
                depth=self.config.research.depth,
            )
            for result in results:
                document = self._add_document(result, f"crawl:{crawl_url}", documents, seen_urls, document_root, limits)
                if document:
                    crawled_documents.append(document)
        return crawled_documents

    def _collect_local_evidence_documents(
        self,
        local_evidence: LocalEvidenceReport | None,
        run_dir: Path,
        documents: list[DeepResearchDocument],
        seen_urls: set[str],
        document_root: Path,
        limits: ResearchLimits,
    ) -> list[DeepResearchDocument]:
        if local_evidence is None or not local_evidence.items:
            return []
        local_documents: list[DeepResearchDocument] = []
        for item in local_evidence.items:
            transcript_path = run_dir / item.transcript_path
            if transcript_path.exists():
                transcript = transcript_path.read_text(encoding="utf-8", errors="replace")
            else:
                transcript = item.transcript_excerpt
            label = item.title or item.url
            result = WebResult(
                url=item.url,
                title=f"YouTube-Transcript: {label}",
                content=item.transcript_excerpt,
                raw_content=transcript,
                publisher=item.publisher or "YouTube",
                published_at=item.published_at,
                score=1.0,
            )
            document = self._add_document(result, f"youtube_transcript:{item.id}", documents, seen_urls, document_root, limits)
            if document:
                local_documents.append(document)
        return local_documents

    def _search_round(
        self,
        provider: ResearchProvider,
        queries: list[ResearchQuery],
        run_dir: Path,
        local_evidence: LocalEvidenceReport,
        documents: list[DeepResearchDocument],
        seen_urls: set[str],
        document_root: Path,
        limits: ResearchLimits,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> list[DeepResearchDocument]:
        round_documents: list[DeepResearchDocument] = []
        for query in queries:
            if cancellation:
                cancellation.raise_if_cancelled()
            remaining = limits.max_documents - len(documents)
            if remaining <= 0:
                break
            try:
                results = provider.search(
                    query.query,
                    max_results=min(self.config.research.per_query_results, remaining),
                    depth=self.config.research.depth,
                )
            except ResearchProviderError as exc:
                report_progress(progress, ProgressEvent("log", "research", str(exc), level="warning"))
                if exc.code in {"provider_rate_limited", "local_search_unavailable", "searxng_json_disabled", "searxng_invalid_json"}:
                    raise
                continue
            youtube_urls = [normalized for result in results if (normalized := normalize_youtube_url(result.url))]
            if youtube_urls:
                round_documents.extend(
                    self._collect_youtube_result_documents(
                        youtube_urls,
                        local_evidence,
                        run_dir,
                        documents,
                        seen_urls,
                        document_root,
                        limits,
                        progress,
                        cancellation,
                    )
                )
            urls = [
                result.url
                for result in results
                if result.url and result.url not in seen_urls and not normalize_youtube_url(result.url)
            ]
            extracted = self._extract_urls(provider, urls, query.query, progress)
            by_url = {result.url: result for result in extracted}
            for result in results:
                enriched = by_url.get(result.url)
                if enriched:
                    result.raw_content = enriched.raw_content or result.raw_content
                    result.content = enriched.content or result.content
                    result.title = enriched.title or result.title
                document = self._add_document(result, query.query, documents, seen_urls, document_root, limits)
                if document:
                    round_documents.append(document)
        return round_documents

    def _collect_youtube_result_documents(
        self,
        urls: list[str],
        local_evidence: LocalEvidenceReport,
        run_dir: Path,
        documents: list[DeepResearchDocument],
        seen_urls: set[str],
        document_root: Path,
        limits: ResearchLimits,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> list[DeepResearchDocument]:
        if not self.config.evidence.enabled or not self.config.evidence.youtube_enabled:
            return []
        before = len(local_evidence.items)
        collector = LocalEvidenceCollector(self.config.evidence, self.project_root)
        collector.collect(
            topic="\n".join(urls),
            run_dir=run_dir,
            existing=local_evidence,
            progress=progress,
            cancellation=cancellation,
        )
        if len(local_evidence.items) <= before:
            return []
        return self._collect_local_evidence_documents(local_evidence, run_dir, documents, seen_urls, document_root, limits)

    def _extract_urls(
        self,
        provider: ResearchProvider,
        urls: list[str],
        query: str,
        progress: ProgressReporter | None,
    ) -> list[WebResult]:
        extracted: list[WebResult] = []
        for batch in _chunks(urls, self.config.research.extraction_batch_size):
            try:
                extracted.extend(provider.extract(batch, query=query, depth=self.config.research.depth))
            except ResearchProviderError as exc:
                report_progress(progress, ProgressEvent("log", "research", str(exc), level="warning"))
        return extracted

    def _extract_evidence(
        self,
        topic: str,
        language: str,
        documents: list[DeepResearchDocument],
        known_topics: list[str],
        research_root: Path,
        round_index: int,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> list[EvidenceBatch]:
        batches: list[EvidenceBatch] = []
        for batch_index, document_batch in enumerate(self._document_prompt_batches(documents), start=1):
            report_progress(progress, ProgressEvent("start", "research", f"Belege extrahieren {round_index}.{batch_index}"))
            batch = self.runner.run_structured(
                prompt=build_evidence_batch_prompt(topic, language, document_batch, known_topics),
                schema_name="evidence_batch",
                output_path=research_root / f"evidence_round_{round_index}_{batch_index}.json",
                model=EvidenceBatch,
                progress=progress,
                cancellation=cancellation,
                timeout_sec=self._codex_timeout(),
                config_overrides=self._codex_overrides(),
                live_search=False,
            )
            batches.append(batch)
        return batches

    def _build_dossier(
        self,
        topic: str,
        language: str,
        documents: list[DeepResearchDocument],
        evidence_batches: list[EvidenceBatch],
        run_dir: Path,
        research_root: Path,
        progress: ProgressReporter | None,
        cancellation: CancellationToken | None,
    ) -> ResearchDossier:
        report_progress(progress, ProgressEvent("start", "research", "Research-Dossier synthetisieren"))
        dossier = self.runner.run_structured(
            prompt=build_research_dossier_prompt(
                topic,
                language,
                self._metadata_only_documents(documents),
                self._bounded_evidence(evidence_batches),
            ),
            schema_name="research_dossier",
            output_path=research_root / "research_dossier.json",
            model=ResearchDossier,
            progress=progress,
            cancellation=cancellation,
            timeout_sec=self._codex_timeout(),
            config_overrides=self._codex_overrides(),
            live_search=False,
        )
        write_json(run_dir / "research_dossier.json", dossier.model_dump(mode="json"))
        report_progress(progress, ProgressEvent("done", "research", f"Dossier mit {len(dossier.claims)} Claims erstellt"))
        return dossier

    def _finish_result(
        self,
        documents: list[DeepResearchDocument],
        evidence_batches: list[EvidenceBatch],
        dossier: ResearchDossier,
        run_dir: Path,
        research_root: Path,
        limits: ResearchLimits,
        warnings: list[str],
    ) -> DeepResearchResult:
        quality_report = self._quality_report(documents, evidence_batches, dossier, limits)
        write_json(research_root / "quality_report.json", quality_report)
        dossier_md = research_root / "research_dossier.md"
        dossier_md.write_text(self._dossier_markdown(dossier, documents), encoding="utf-8")
        all_warnings = [*warnings, *quality_report["warnings"]]
        return DeepResearchResult(
            documents=documents,
            evidence_batches=evidence_batches,
            dossier=dossier,
            artifacts={
                "research_plan": run_dir / "research_plan.json",
                "deep_research_frontier": research_root / "frontier.jsonl",
                "deep_research_evidence": research_root / "evidence.jsonl",
                "deep_research_topics": research_root / "topics.json",
                "research_dossier": research_root / "research_dossier.json",
                "research_dossier_notes": dossier_md,
                "deep_research_quality": research_root / "quality_report.json",
            },
            warnings=sorted(set(all_warnings)),
        )

    def _add_document(
        self,
        result: WebResult,
        query: str,
        documents: list[DeepResearchDocument],
        seen_urls: set[str],
        document_root: Path,
        limits: ResearchLimits,
    ) -> DeepResearchDocument | None:
        if not result.url or result.url in seen_urls or len(documents) >= limits.max_documents:
            return None
        seen_urls.add(result.url)
        doc_id = f"D{len(documents) + 1:04d}"
        content = (result.raw_content or result.content or "").strip()
        document = DeepResearchDocument(
            id=doc_id,
            url=result.url,
            title=result.title,
            query=query,
            content=_trim_text(content, 20000),
            snippet=_trim_text(result.content or "", 1200) or None,
            publisher=result.publisher or _domain(result.url),
            published_at=result.published_at,
            score=result.score,
        )
        documents.append(document)
        write_json(document_root / f"{doc_id}.json", document.model_dump(mode="json"))
        self._append_jsonl(document_root.parent / "documents.jsonl", document.model_dump(mode="json"))
        return document

    def _initial_frontier(self, plan: DeepResearchPlan, topic: str) -> deque[ResearchQuery]:
        queries = sorted(plan.seed_queries, key=lambda item: item.priority, reverse=True)
        if not queries:
            queries = [ResearchQuery(query=topic, perspective="Ausgangsfrage", rationale="Fallback aus dem Thema", priority=5)]
        return deque(queries)

    def _pop_queries(self, frontier: deque[ResearchQuery], limit: int) -> list[ResearchQuery]:
        candidates = list(frontier)
        frontier.clear()
        candidates.sort(key=lambda item: item.priority, reverse=True)
        selected: list[ResearchQuery] = []
        used_buckets: set[str] = set()
        for candidate in candidates:
            bucket = _query_bucket(candidate)
            if bucket in used_buckets:
                continue
            selected.append(candidate)
            used_buckets.add(bucket)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            selected_ids = {id(query) for query in selected}
            for candidate in candidates:
                if id(candidate) in selected_ids:
                    continue
                selected.append(candidate)
                if len(selected) >= limit:
                    break
        selected_ids = {id(query) for query in selected}
        for remaining in candidates:
            if id(remaining) not in selected_ids:
                frontier.append(remaining)
        return selected

    def _follow_up_queries(self, batch: EvidenceBatch) -> list[ResearchQuery]:
        queries = list(batch.follow_up_queries)
        for topic in batch.discovered_topics:
            if topic.relevance == "low":
                continue
            for query in topic.follow_up_queries:
                queries.append(
                    ResearchQuery(
                        query=query,
                        perspective=topic.title,
                        rationale="Aus entdecktem Unterthema abgeleitet",
                        priority=4 if topic.relevance == "high" else 3,
                    )
                )
        return queries

    def _document_prompt_batches(self, documents: list[DeepResearchDocument]) -> list[list[DeepResearchDocument]]:
        batches: list[list[DeepResearchDocument]] = []
        current: list[DeepResearchDocument] = []
        current_chars = 0
        for document in documents:
            size = len(document.content) + len(document.snippet or "")
            if current and current_chars + size > self.config.research.codex_batch_chars:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(document)
            current_chars += size
        if current:
            batches.append(current)
        return batches

    def _metadata_only_documents(self, documents: list[DeepResearchDocument]) -> list[DeepResearchDocument]:
        return [document.model_copy(update={"content": "", "snippet": document.snippet or ""}) for document in documents]

    def _bounded_evidence(self, evidence_batches: list[EvidenceBatch]) -> list[EvidenceBatch]:
        bounded: list[EvidenceBatch] = []
        total = 0
        for batch in evidence_batches:
            remaining = 120 - total
            if remaining <= 0:
                break
            bounded.append(batch.model_copy(update={"evidence": batch.evidence[:remaining]}))
            total += len(bounded[-1].evidence)
        return bounded

    def _merge_topic(self, topic: DiscoveredTopic, known_topics: list[str]) -> None:
        normalized = {item.lower() for item in known_topics}
        if topic.title.lower() not in normalized:
            known_topics.append(topic.title)

    def _limits(self) -> ResearchLimits:
        if self.config.research.depth == "dossier":
            defaults = (60, 6, 300)
        else:
            defaults = (20, 3, 100)
        max_minutes = self.config.research.max_minutes or defaults[0]
        max_rounds = self.config.research.max_rounds or defaults[1]
        max_documents = self.config.research.max_documents or defaults[2]
        return ResearchLimits(max_seconds=max_minutes * 60.0, max_rounds=max_rounds, max_documents=max_documents)

    def _codex_timeout(self) -> int:
        minimum = 3600 if self.config.research.depth == "dossier" else 2400
        budget_seconds = int(self._limits().max_seconds)
        return max(60, min(max(self.config.codex.timeout_sec, minimum), budget_seconds))

    def _codex_overrides(self) -> dict[str, object]:
        return {"model_reasoning_effort": "xhigh"}

    def _has_low_source_diversity(self, documents: list[DeepResearchDocument]) -> bool:
        if len(documents) < 10:
            return False
        domains = Counter(_domain(document.url) for document in documents)
        return len(domains) < 3 or domains.most_common(1)[0][1] / len(documents) > 0.7

    def _load_documents(self, research_root: Path) -> list[DeepResearchDocument]:
        document_root = research_root / "documents"
        paths = sorted(document_root.glob("D*.json"))
        if paths:
            return [
                DeepResearchDocument.model_validate_json(path.read_text(encoding="utf-8"))
                for path in paths
            ]
        jsonl_path = research_root / "documents.jsonl"
        if not jsonl_path.exists():
            return []
        return [
            DeepResearchDocument.model_validate_json(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _load_evidence_batches(self, research_root: Path) -> list[EvidenceBatch]:
        jsonl_path = research_root / "evidence.jsonl"
        if jsonl_path.exists():
            return [
                EvidenceBatch.model_validate_json(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        paths = sorted(
            (
                path
                for path in research_root.glob("evidence_round_*_*.json")
                if re.fullmatch(r"evidence_round_\d+_\d+\.json", path.name)
            ),
            key=_evidence_path_key,
        )
        return [
            EvidenceBatch.model_validate_json(path.read_text(encoding="utf-8"))
            for path in paths
        ]

    def _load_topics(self, research_root: Path) -> list[str]:
        topics_path = research_root / "topics.json"
        if not topics_path.exists():
            return []
        data = json.loads(topics_path.read_text(encoding="utf-8"))
        return [str(item) for item in data if str(item).strip()]

    def _quality_report(
        self,
        documents: list[DeepResearchDocument],
        evidence_batches: list[EvidenceBatch],
        dossier: ResearchDossier,
        limits: ResearchLimits,
    ) -> dict[str, object]:
        domains = Counter(_domain(document.url) for document in documents)
        evidence_count = sum(len(batch.evidence) for batch in evidence_batches)
        high_confidence_evidence = sum(
            1
            for batch in evidence_batches
            for evidence in batch.evidence
            if evidence.confidence == "high"
        )
        discovered_topic_count = sum(len(batch.discovered_topics) for batch in evidence_batches)
        sourced_claim_count = sum(1 for claim in dossier.claims if claim.source_document_ids)
        coverage_text = "\n".join(
            [dossier.summary]
            + [document.title or "" for document in documents]
            + [document.url for document in documents]
            + [topic.title + " " + topic.summary for topic in dossier.key_topics]
            + [claim.text for claim in dossier.claims]
            + [evidence.text for batch in evidence_batches for evidence in batch.evidence]
        ).lower()
        required_terms = _coverage_terms(dossier.topic)
        covered_terms = [term for term in required_terms if term in coverage_text]
        missing_terms = [term for term in required_terms if term not in coverage_text]
        expected_documents = min(limits.max_documents, 20 if self.config.research.depth == "dossier" else 8)
        expected_domains = max(1, min(5 if self.config.research.depth == "dossier" else 3, expected_documents // 4 or 1))
        expected_evidence = max(1, min(25 if self.config.research.depth == "dossier" else 8, expected_documents))
        expected_claims = max(1, min(8 if self.config.research.depth == "dossier" else 4, expected_documents // 2 or 1))
        warnings: list[str] = []
        if len(documents) < expected_documents:
            warnings.append("dossier_document_count_low")
        if len(domains) < expected_domains:
            warnings.append("dossier_source_diversity_low")
        if evidence_count < expected_evidence:
            warnings.append("dossier_evidence_count_low")
        if sourced_claim_count < expected_claims:
            warnings.append("dossier_claims_low")
        if self.config.research.depth == "dossier" and high_confidence_evidence == 0:
            warnings.append("dossier_high_confidence_evidence_missing")
        if missing_terms:
            warnings.append("dossier_topic_coverage_low")
        return {
            "status": "pass" if not warnings else "warn",
            "warnings": warnings,
            "document_count": len(documents),
            "source_domain_count": len(domains),
            "top_domains": domains.most_common(10),
            "evidence_count": evidence_count,
            "high_confidence_evidence_count": high_confidence_evidence,
            "discovered_topic_count": discovered_topic_count,
            "dossier_claim_count": len(dossier.claims),
            "sourced_dossier_claim_count": sourced_claim_count,
            "required_coverage_terms": required_terms,
            "covered_coverage_terms": covered_terms,
            "missing_coverage_terms": missing_terms,
            "thresholds": {
                "min_documents": expected_documents,
                "min_source_domains": expected_domains,
                "min_evidence": expected_evidence,
                "min_sourced_claims": expected_claims,
            },
        }

    def _dossier_markdown(self, dossier: ResearchDossier, documents: list[DeepResearchDocument]) -> str:
        docs_by_id = {document.id: document for document in documents}
        lines = [f"# Research-Dossier: {dossier.topic}", "", dossier.summary, "", "## Themen"]
        for topic in dossier.key_topics:
            lines.append(f"- **{topic.title}**: {topic.summary}")
        lines.extend(["", "## Claims"])
        for claim in dossier.claims:
            sources = [docs_by_id[doc_id].url for doc_id in claim.source_document_ids if doc_id in docs_by_id]
            suffix = f" Quellen: {', '.join(sources)}" if sources else ""
            lines.append(f"- ({claim.confidence}) {claim.text}{suffix}")
            if claim.conflict_notes:
                lines.append(f"  - Konflikt: {claim.conflict_notes}")
        if dossier.open_questions:
            lines.extend(["", "## Offene Fragen"])
            lines.extend(f"- {question}" for question in dossier.open_questions)
        if dossier.recommended_angles:
            lines.extend(["", "## Podcast-Winkel"])
            lines.extend(f"- {angle}" for angle in dossier.recommended_angles)
        return "\n".join(lines) + "\n"

    def _write_frontier(self, root: Path, events: list[dict]) -> None:
        for event in events:
            self._append_jsonl(root / "frontier.jsonl", event)

    def _append_jsonl(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def _evidence_path_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"evidence_round_(\d+)_(\d+)\.json$", path.name)
    if not match:
        return (9999, 9999, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def _chunks(items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), max(1, size))]


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _query_bucket(query: ResearchQuery) -> str:
    text = f"{query.query} {query.perspective or ''}".lower()
    for marker in ("searxng", "trafilatura", "crawl4ai", "youtube", "transcript", "scrapy", "crawlee", "nutch"):
        if marker in text:
            return marker
    words = [word for word in re.findall(r"[a-z0-9]+", text) if len(word) > 3]
    return words[0] if words else _normalize_query(query.query)


def _coverage_terms(topic: str) -> list[str]:
    text = topic.lower()
    known_terms = [
        "searxng",
        "trafilatura",
        "crawl4ai",
        "youtube",
        "transcript",
        "transkript",
        "scrapy",
        "crawlee",
        "nutch",
    ]
    terms: list[str] = []
    for term in known_terms:
        if term in text and term not in terms:
            terms.append(term)
    return terms


def _trim_text(text: str, limit: int) -> str:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip()


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower() or url


def _extract_readable_text(html: str, url: str, *, depth: str) -> tuple[str | None, str, str | None]:
    title: str | None = None
    published_at: str | None = None
    text: str | None = None
    try:
        import trafilatura

        text = trafilatura.extract(
            html,
            url=url,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            favor_recall=depth == "dossier",
        )
        metadata = trafilatura.extract_metadata(html, default_url=url)
        if metadata is not None:
            title = getattr(metadata, "title", None) or None
            published_at = getattr(metadata, "date", None) or None
    except Exception:
        text = None
    if not text:
        parser = _ReadableTextParser()
        try:
            parser.feed(html)
        except Exception:
            pass
        title = title or parser.title
        text = parser.text()
    return title, _trim_text(text or "", 20000), published_at


class _ReadableTextParser(HTMLParser):
    _block_tags = {"article", "section", "main", "p", "div", "br", "li", "h1", "h2", "h3", "h4", "blockquote", "tr"}
    _skip_tags = {"script", "style", "noscript", "svg", "canvas", "form", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.title: str | None = None
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if self._skip_depth == 0 and tag in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            if self.title is None:
                self.title = _clean_inline_text(" ".join(self.title_parts)) or None
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._skip_depth == 0 and tag in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._skip_depth > 0 or self._in_title:
            return
        cleaned = _clean_inline_text(data)
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return _trim_text(" ".join(self.parts), 20000)


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        link = _normalize_result_url(urljoin(self.base_url, href))
        if _is_http_url(link):
            self.links.append(link)


def _extract_links(html: str, base_url: str) -> list[str]:
    parser = _LinkParser(base_url)
    try:
        parser.feed(html)
    except Exception:
        return []
    seen: set[str] = set()
    links: list[str] = []
    for link in parser.links:
        parsed = urlparse(link)
        normalized = parsed._replace(fragment="").geturl()
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def _normalize_result_url(url: str) -> str:
    url = unquote(url.strip())
    parsed = urlparse(url)
    if parsed.path == "/url":
        query = parse_qs(parsed.query)
        if query.get("q"):
            return _normalize_result_url(query["q"][0])
    return parsed._replace(fragment="").geturl()


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_fetchable_url(url: str, *, allow_private_networks: bool) -> bool:
    if not _is_http_url(url):
        return False
    if allow_private_networks:
        return True
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip_text = address[4][0]
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _looks_like_content_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if re.search(r"\.(?:jpg|jpeg|png|gif|webp|svg|css|js|zip|tar|gz|mp3|mp4|avi|mov|woff2?|ttf|ico)$", path):
        return False
    blocked_parts = {"/login", "/signup", "/sign-in", "/privacy", "/terms", "/contact", "/cart", "/account"}
    return not any(part in path for part in blocked_parts)


def _clean_inline_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", unescape(text))
    return " ".join(without_tags.split())


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
