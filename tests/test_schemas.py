from codcast.schemas import schema_for


def _walk_objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for value in node.values():
            yield from _walk_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_objects(item)


def _walk_nodes(node):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_nodes(item)


def test_codex_schemas_require_all_properties_and_remove_defaults():
    for name in ["research", "research_plan", "evidence_batch", "research_dossier", "validation", "script"]:
        schema = schema_for(name)
        for obj in _walk_objects(schema):
            properties = obj.get("properties", {})
            assert obj.get("additionalProperties") is False
            assert obj.get("required") == list(properties.keys())
        assert all(not (isinstance(node, dict) and "default" in node) for node in _walk_nodes(schema))
        assert all(not (isinstance(node, dict) and "format" in node) for node in _walk_nodes(schema))
