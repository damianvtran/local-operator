"""Model identity and positive Bedrock capability evidence, not vendor guesses.

Sources: AWS Bedrock model cards (2026-11-19), under
https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-:
amazon-nova-2-lite.md, amazon-nova-pro.md, qwen-qwen3-coder-next.md,
meta-llama-3-3-70b-instruct.md, deepseek-deepseek-r1.html.
Unknowns deliberately remain unknown: listing TEXT/streaming metadata does not
prove tool, image, document, caching or reasoning-configuration support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class BedrockCapabilities:
    protocol: str = "converse"
    streaming: bool | None = None
    tools: bool | None = None
    required_tools: bool | None = None
    streaming_tools: bool | None = None
    images: bool | None = None
    tool_result_images: bool | None = None
    tool_result_status: bool = False
    document_formats: frozenset[str] = frozenset()
    tool_result_document_formats: frozenset[str] = frozenset()
    cache_locations: frozenset[str] = frozenset()
    cache_ttls: frozenset[str] = frozenset()
    cache_max_points: int = 0
    cache_min_tokens: int | None = None
    cache_max_tokens: int | None = None
    reasoning_config: str | None = None
    context_window: int | None = None
    max_output: int | None = None


def foundation_identity(model_id: str) -> str | None:
    """Parse resources exactly; an application profile is intentionally opaque.

    Region on an ARN identifies its resource, never the signing/source Region.
    Keep the original ID on the wire; this projection is solely for policy.
    """
    if model_id.startswith("arn:"):
        parts = model_id.split(":", 5)
        if len(parts) != 6 or parts[2] != "bedrock":
            return None
        kind, sep, model_id = parts[5].partition("/")
        if not sep or kind not in {"foundation-model", "inference-profile"}:
            return None
    return re.sub(r"^(?:us|eu|apac|global|us-gov|jp|au|ca)\.", "", model_id, count=1)


# Native documents: Nova v1/v2 user guides, modalities-document.html and
# nova2-userguide/using-multimodal-models.html (2026-11-19). Tool-result
# document support is not inferred from ordinary document support.
_NOVA = BedrockCapabilities(
    document_formats=frozenset({"pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"}),
    streaming=True,
    tools=True,
    required_tools=True,
    streaming_tools=True,
    images=True,
    tool_result_images=True,
    tool_result_status=True,
    cache_locations=frozenset({"system", "messages"}),
    cache_ttls=frozenset({"5m"}),
    cache_max_points=4,
    cache_min_tokens=1000,
    cache_max_tokens=20_000,
)
CAPABILITIES = {
    "amazon.nova-2-lite-v1:0": replace(
        _NOVA, reasoning_config="nova2", context_window=1_000_000, max_output=64_000
    ),
    "amazon.nova-pro-v1:0": replace(_NOVA, context_window=300_000, max_output=10_000),
    "qwen.qwen3-coder-next": BedrockCapabilities(
        streaming=True, tools=True, images=False, context_window=256_000, max_output=16_000
    ),
    "meta.llama3-3-70b-instruct-v1:0": BedrockCapabilities(
        streaming=True,
        tools=True,
        streaming_tools=False,
        images=False,
        context_window=128_000,
        max_output=4_000,
    ),
    "deepseek.r1-v1:0": BedrockCapabilities(
        streaming=True, tools=False, images=False, context_window=128_000, max_output=8_000
    ),
}


def capabilities_for(model_id: str) -> BedrockCapabilities:
    identity = foundation_identity(model_id)
    if identity and identity.startswith("anthropic.claude-"):
        return BedrockCapabilities(protocol="anthropic")
    return CAPABILITIES.get(identity or "", BedrockCapabilities())


def capabilities_for_spec(spec) -> BedrockCapabilities:
    """Join persisted, connection-scoped listing facts without runtime lookups.

    A profile's multiple backing models intersect positive capabilities. Unknown
    backing identities cannot inherit a positive from another region/model.
    """
    metadata = getattr(spec, "bedrock_metadata", None) or {}
    identities = metadata.get("foundation_ids") or []
    caps = capabilities_for(spec.model_id)
    if identities:
        rows = [capabilities_for(identity) for identity in identities]
        updates = {}
        for field in fields(BedrockCapabilities):
            values = [getattr(row, field.name) for row in rows]
            if all(value == values[0] for value in values):
                updates[field.name] = values[0]
            elif isinstance(values[0], frozenset):
                updates[field.name] = frozenset.intersection(*values)
            elif any(value is False for value in values):
                updates[field.name] = False
            else:
                updates[field.name] = getattr(BedrockCapabilities(), field.name)
        caps = BedrockCapabilities(**updates)
    if isinstance(metadata.get("streaming"), bool):
        caps = replace(caps, streaming=metadata["streaming"])
    if isinstance(metadata.get("images"), bool):
        caps = replace(caps, images=metadata["images"])
    return caps


def client_for_bedrock_spec(spec, **kwargs):
    """Pure dispatch: no credential resolution, network or implicit ARN lookup."""
    if capabilities_for_spec(spec).protocol == "anthropic":
        from .client import BedrockAnthropicClient

        return BedrockAnthropicClient(**kwargs)
    from .converse import BedrockConverseClient

    # Claude-only TTL controls must never become generic Converse knobs.
    kwargs.pop("cache_ttl_1h_min_context_tokens", None)
    kwargs.pop("subagent_cache_ttl_1h_min_context_tokens", None)
    return BedrockConverseClient(**kwargs)
