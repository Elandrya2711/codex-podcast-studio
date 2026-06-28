from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimStatus(str, Enum):
    supported = "supported"
    weak = "weak"
    conflicting = "conflicting"
    unverified = "unverified"


class Source(StrictModel):
    id: str = Field(pattern=r"^S[0-9]+$")
    title: str
    url: HttpUrl | str
    publisher: str | None = None
    published_at: str | None = None
    accessed_at: str | None = None
    relevance: str = Field(min_length=1)


class ResearchClaim(StrictModel):
    id: str = Field(pattern=r"^C[0-9]+$")
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: str | None = None


class ResearchReport(StrictModel):
    topic: str
    language: str = "de-DE"
    generated_at: str = Field(default_factory=utc_now_iso)
    summary: str
    sources: list[Source] = Field(default_factory=list)
    claims: list[ResearchClaim] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ResearchQuery(StrictModel):
    query: str = Field(min_length=1)
    perspective: str | None = None
    rationale: str | None = None
    priority: int = Field(default=3, ge=1, le=5)


class DeepResearchPlan(StrictModel):
    topic: str
    language: str = "de-DE"
    perspectives: list[str] = Field(default_factory=list)
    seed_queries: list[ResearchQuery] = Field(default_factory=list)
    crawl_urls: list[str] = Field(default_factory=list)
    source_priorities: list[str] = Field(default_factory=list)
    stop_criteria: list[str] = Field(default_factory=list)


class DeepResearchDocument(StrictModel):
    id: str = Field(pattern=r"^D[0-9]+$")
    url: str
    title: str | None = None
    query: str
    content: str = ""
    snippet: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    score: float | None = None
    retrieved_at: str = Field(default_factory=utc_now_iso)


class ExtractedEvidence(StrictModel):
    text: str = Field(min_length=1)
    source_document_id: str = Field(pattern=r"^D[0-9]+$")
    source_url: str
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: str | None = None


class DiscoveredTopic(StrictModel):
    title: str = Field(min_length=1)
    summary: str
    relevance: Literal["high", "medium", "low"] = "medium"
    source_document_ids: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class EvidenceBatch(StrictModel):
    topic: str
    document_ids: list[str] = Field(default_factory=list)
    evidence: list[ExtractedEvidence] = Field(default_factory=list)
    discovered_topics: list[DiscoveredTopic] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    follow_up_queries: list[ResearchQuery] = Field(default_factory=list)


class DossierTopic(StrictModel):
    title: str
    summary: str
    source_document_ids: list[str] = Field(default_factory=list)


class DossierClaim(StrictModel):
    text: str
    source_document_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    conflict_notes: str | None = None


class ResearchDossier(StrictModel):
    topic: str
    language: str = "de-DE"
    generated_at: str = Field(default_factory=utc_now_iso)
    summary: str
    key_topics: list[DossierTopic] = Field(default_factory=list)
    claims: list[DossierClaim] = Field(default_factory=list)
    source_assessment: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommended_angles: list[str] = Field(default_factory=list)


class LocalEvidenceItem(StrictModel):
    id: str = Field(pattern=r"^E[0-9]+$")
    kind: Literal["youtube_transcript"] = "youtube_transcript"
    url: str
    title: str | None = None
    publisher: str | None = None
    published_at: str | None = None
    accessed_at: str = Field(default_factory=utc_now_iso)
    language: str | None = None
    transcript_path: str
    transcript_chars: int
    transcript_excerpt: str
    is_truncated: bool = False


class LocalEvidenceFailure(StrictModel):
    url: str
    reason: str
    attempted_at: str = Field(default_factory=utc_now_iso)


class LocalEvidenceReport(StrictModel):
    generated_at: str = Field(default_factory=utc_now_iso)
    items: list[LocalEvidenceItem] = Field(default_factory=list)
    failures: list[LocalEvidenceFailure] = Field(default_factory=list)

    def has_items(self) -> bool:
        return bool(self.items)

    def attempted_urls(self) -> set[str]:
        return {item.url for item in self.items} | {failure.url for failure in self.failures}


class ValidationFinding(StrictModel):
    claim_id: str
    status: ClaimStatus
    source_ids: list[str] = Field(default_factory=list)
    notes: str


class ValidationReport(StrictModel):
    topic: str
    generated_at: str = Field(default_factory=utc_now_iso)
    pass_status: Literal["pass", "needs_revision"] = "needs_revision"
    findings: list[ValidationFinding] = Field(default_factory=list)
    weak_claim_ids: list[str] = Field(default_factory=list)


class SpeakerSpec(StrictModel):
    id: str = Field(pattern=r"^s[0-9]+$")
    display_name: str
    role: str
    voice_profile_id: str


class ScriptLine(StrictModel):
    speaker_id: str
    text: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    stage_direction: str | None = None

    @field_validator("text")
    @classmethod
    def no_empty_lines(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("script line text cannot be empty")
        return normalized


class PodcastScript(StrictModel):
    title: str
    topic: str
    language: str = "de-DE"
    target_min_minutes: float
    target_max_minutes: float
    speakers: list[SpeakerSpec]
    lines: list[ScriptLine]
    estimated_words: int | None = None
    production_notes: list[str] = Field(default_factory=list)


class RenderedSegment(StrictModel):
    index: int
    speaker_id: str
    voice_profile_id: str
    text: str
    wav_path: str
    duration_sec: float | None = None


class RunManifest(StrictModel):
    run_id: str
    topic: str
    created_at: str = Field(default_factory=utc_now_iso)
    language: str
    min_minutes: float
    max_minutes: float
    speakers: int
    quality: str
    research_depth: str = "standard"
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
