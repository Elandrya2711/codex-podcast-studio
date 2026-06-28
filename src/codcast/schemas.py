from __future__ import annotations

from typing import Any

from .models import DeepResearchPlan, EvidenceBatch, PodcastScript, ResearchDossier, ResearchReport, ValidationReport


def schema_for(name: str) -> dict[str, Any]:
    if name == "research":
        return codex_strict_schema(ResearchReport.model_json_schema())
    if name == "research_plan":
        return codex_strict_schema(DeepResearchPlan.model_json_schema())
    if name == "evidence_batch":
        return codex_strict_schema(EvidenceBatch.model_json_schema())
    if name == "research_dossier":
        return codex_strict_schema(ResearchDossier.model_json_schema())
    if name == "validation":
        return codex_strict_schema(ValidationReport.model_json_schema())
    if name == "script":
        return codex_strict_schema(PodcastScript.model_json_schema())
    raise KeyError(f"unknown schema: {name}")


def codex_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic JSON Schema to Codex/OpenAI strict structured-output schema."""
    normalized = dict(schema)
    _normalize_node(normalized)
    return normalized


def _normalize_node(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        node.pop("format", None)
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
            node["additionalProperties"] = False
        for value in node.values():
            _normalize_node(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_node(item)
