"""Region/account scoped all-vendor Bedrock TEXT discovery.

Foundation metadata is fetched without ON_DEMAND filtering so profile-only
backing models can be joined. Bare invocation still requires ON_DEMAND. Listing
is not a subscription/access probe; denial retains static fallback and never
switches from bearer to ambient credentials. A partial inventory is useful but
cannot authoritatively delete the failed leg's static rows.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from local_operator.providers.bedrock import auth as bauth
from local_operator.providers.bedrock import connection as bconn
from local_operator.providers.bedrock.models import (
    capabilities_for_spec,
    foundation_identity,
)

logger = logging.getLogger(__name__)
PROFILES_PATH = "/inference-profiles"
FOUNDATION_MODELS_PATH = "/foundation-models"


def catalogue_scope(api_key: str | None) -> str | None:
    if not api_key:
        return None
    try:
        conn = bconn.parse(api_key)
        return f"{bauth.region_for(conn)}|{conn.scope_key()}"
    except Exception:  # noqa: BLE001 - unavailable credentials keep static fallback
        return None


def listing_cache_key(api_key: str | None) -> str:
    scope = catalogue_scope(api_key)
    return (
        f"bedrock.{hashlib.sha256(scope.encode()).hexdigest()[:12]}.listing"
        if scope
        else "bedrock.listing"
    )


def _signed_get(ctx, resolved, url, params):
    import httpx

    request_url = str(httpx.URL(url, params=params))
    headers = bauth.sign_headers(resolved, "GET", request_url, b"", {"Accept": "application/json"})
    response = ctx.client.get(request_url, headers=headers, timeout=ctx.timeout)
    if not 200 <= int(getattr(response, "status_code", 0)) < 300:
        return None
    return response.json()


def _text(entry):
    # TEXT output alone is insufficient for an embedding endpoint: embeddings
    # and image/video/speech generators have no conversation input contract.
    identity = str(entry.get("modelId", ""))
    return (
        "TEXT" in (entry.get("outputModalities") or [])
        and "TEXT" in (entry.get("inputModalities") or [])
        and not any(marker in identity.lower() for marker in ("embed", "rerank"))
        and (entry.get("modelLifecycle") or {}).get("status", "ACTIVE") == "ACTIVE"
    )


def parse_foundation_models(body: Any) -> list[tuple[str, str]]:
    if not isinstance(body, dict):
        return []
    return [
        (entry["modelId"], entry.get("modelName") or entry["modelId"])
        for entry in body.get("modelSummaries", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("modelId"), str)
        and _text(entry)
        and "ON_DEMAND" in (entry.get("inferenceTypesSupported") or [])
    ]


def parse_profiles(body: Any) -> list[tuple[str, str]]:
    """Shape-only public parser; eligibility joins backing metadata in fetch."""
    if not isinstance(body, dict):
        return []
    rows = []
    for entry in body.get("inferenceProfileSummaries") or []:
        if not isinstance(entry, dict) or entry.get("status") != "ACTIVE":
            continue
        model_id = (
            entry.get("inferenceProfileArn")
            if entry.get("type") == "APPLICATION"
            else entry.get("inferenceProfileId")
        )
        if isinstance(model_id, str):
            rows.append((model_id, entry.get("inferenceProfileName") or model_id))
    return rows


def fetch(ctx: Any) -> list[Any] | None:
    from types import SimpleNamespace

    from local_operator.model.discovery import DiscoveredModel

    if not ctx.api_key:
        return None
    try:
        resolved = bauth.resolve_sync(ctx.api_key)
    except bauth.BedrockCredentialError:
        return None
    base = bconn.control_base(resolved.region)
    complete = True
    profiles = []
    token = None
    seen_tokens = set()
    while True:
        # No type filter: includes both SYSTEM_DEFINED and APPLICATION. AWS
        # SDK typeEquals is HTTP `type`, NOT a `typeEquals` query parameter.
        params = {"maxResults": "1000"}
        if token:
            params["nextToken"] = token
        try:
            page = _signed_get(ctx, resolved, base + PROFILES_PATH, params)
        except Exception:  # noqa: BLE001 - isolate failures to this inventory leg
            page = None
        if not isinstance(page, dict) or not isinstance(
            page.get("inferenceProfileSummaries"), list
        ):
            complete = False
            break
        profiles.extend(page["inferenceProfileSummaries"])
        token = page.get("nextToken")
        if not token:
            break
        if not isinstance(token, str) or token in seen_tokens:
            complete = False
            break
        seen_tokens.add(token)
    try:
        foundation = _signed_get(
            ctx, resolved, base + FOUNDATION_MODELS_PATH, {"byOutputModality": "TEXT"}
        )
    except Exception:  # noqa: BLE001 - preserve successful profile leg
        foundation = None
    if not isinstance(foundation, dict) or not isinstance(foundation.get("modelSummaries"), list):
        complete = False
        foundation = {"modelSummaries": []}
    metadata = {
        entry["modelId"]: entry
        for entry in foundation["modelSummaries"]
        if isinstance(entry, dict) and isinstance(entry.get("modelId"), str)
    }
    rows = []
    seen = set()

    def append(model_id, name, identities, backing):
        if model_id in seen:
            return
        seen.add(model_id)
        listing = {"foundation_ids": identities, "inventory_complete": complete}
        stream = [entry.get("responseStreamingSupported") for entry in backing]
        if stream and all(isinstance(value, bool) for value in stream):
            listing["streaming"] = all(stream)
        caps = capabilities_for_spec(SimpleNamespace(model_id=model_id, bedrock_metadata=listing))
        inputs = [entry.get("inputModalities") for entry in backing]
        images = (
            all("IMAGE" in value for value in inputs)
            if inputs and all(isinstance(value, list) for value in inputs)
            else caps.images
        )
        if isinstance(images, bool):
            listing["images"] = images
        rows.append(
            DiscoveredModel(
                id=model_id,
                name=name,
                context_window=caps.context_window or 0,
                max_tokens=caps.max_output or 0,
                supports_images=images,
                supports_tools=caps.tools,
                supports_prompt_cache=bool(caps.cache_locations),
                reasoning=(
                    True if caps.reasoning_config or identities == ["deepseek.r1-v1:0"] else None
                ),
                reasoning_efforts=(
                    ("none", "low", "medium", "high") if caps.reasoning_config == "nova2" else None
                ),
                bedrock_metadata=listing,
            )
        )

    for entry in profiles:
        parsed = parse_profiles({"inferenceProfileSummaries": [entry]})
        if not parsed:
            continue
        identities = sorted(
            {
                identity
                for model in entry.get("models", [])
                if isinstance(model, dict)
                and (identity := foundation_identity(str(model.get("modelArn", ""))))
            }
        )
        backing = [metadata[identity] for identity in identities if identity in metadata]
        # When the other leg is denied, preserve known conversation seeds and
        # unknown profiles (an attempt, not an invented capability). Positive
        # metadata stating non-TEXT or inactive always excludes the profile.
        if backing and (
            len(backing) != len(identities) or not all(_text(item) for item in backing)
        ):
            continue
        append(*parsed[0], identities, backing)
    for model_id, name in parse_foundation_models(foundation):
        append(model_id, name, [model_id], [metadata[model_id]])
    return rows if rows or complete else None
