"""Bedrock model discovery: Claude inference profiles + on-demand models.

Called from ``model/discovery.py``'s wire table (``_WIRE_TRANSPORTS["bedrock"]``)
inside the listing worker, so it is synchronous and signs each request with
the same resolver the runtime uses (design §6.4):

1. ``GET /inference-profiles?typeEquals=SYSTEM_DEFINED&maxResults=1000`` —
   keep ``status == ACTIVE`` ids containing ``.anthropic.claude``. These are
   the cross-Region ids (``us.``, ``global.`` ...) THIS source Region can call;
   which prefixes exist depends on the Region, so they are listed, never
   inferred from a prefix.
2. ``GET /foundation-models?byProvider=anthropic&byInferenceType=ON_DEMAND`` —
   keep ACTIVE, streaming-capable models: the bare in-Region ids. Profile-only
   models are excluded by the ON_DEMAND filter because they cannot be invoked
   bare.

Rows carry only id + name; ``merge_models`` restores windows and prices from
the static Bedrock rows / the family resolver. ANY failure returns ``None``,
which keeps the static list (the ``fetch_models`` contract) — a role that may
invoke but not list (``bedrock:InvokeModel*`` only) still gets a usable picker.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from local_operator.providers.bedrock import auth as bauth
from local_operator.providers.bedrock import connection as bconn

logger = logging.getLogger(__name__)

PROFILES_PATH = "/inference-profiles"
FOUNDATION_MODELS_PATH = "/foundation-models"


def catalogue_scope(api_key: str | None) -> str | None:
    """``{region}|{connection scope}`` for cache isolation, or ``None``.

    Never contains a secret: a sealed ref and a raw key never enter it (a raw
    key contributes a short fingerprint only, see ``Connection.scope_key``).
    Listings therefore never cross Regions or accounts.
    """
    if not api_key:
        return None
    try:
        conn = bconn.parse(api_key)
    except bconn.DescriptorError:
        return None
    try:
        region = bauth.region_for(conn)
    except Exception:  # noqa: BLE001 - a broken profile has no listing either
        return None
    return f"{region}|{conn.scope_key()}"


def listing_cache_key(api_key: str | None) -> str:
    """Document name for this connection's listing (hashed scope, nothing raw)."""
    scope = catalogue_scope(api_key)
    if not scope:
        return "bedrock.listing"
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:12]
    return f"bedrock.{digest}.listing"


def _signed_get(ctx: Any, resolved: bauth.ResolvedAuth, url: str, params: dict[str, str]) -> Any:
    import httpx

    # Sign the exact URL httpx will send: build it first so the canonical
    # query string matches byte for byte.
    request_url = str(httpx.URL(url, params=params))
    headers = bauth.sign_headers(resolved, "GET", request_url, b"", {"Accept": "application/json"})
    response = ctx.client.get(request_url, headers=headers, timeout=ctx.timeout)
    status = int(getattr(response, "status_code", 0))
    if not 200 <= status < 300:
        logger.debug("bedrock listing returned HTTP %s for %s", status, url)
        return None
    return response.json()


def parse_profiles(body: Any) -> list[tuple[str, str]]:
    """``(id, name)`` for ACTIVE Claude inference profiles."""
    rows: list[tuple[str, str]] = []
    if not isinstance(body, dict):
        return rows
    for entry in body.get("inferenceProfileSummaries") or []:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("inferenceProfileId")
        if not isinstance(model_id, str) or ".anthropic.claude" not in model_id:
            continue
        if entry.get("status") != "ACTIVE":
            continue
        name = entry.get("inferenceProfileName")
        rows.append((model_id, name if isinstance(name, str) and name else model_id))
    return rows


def parse_foundation_models(body: Any) -> list[tuple[str, str]]:
    """``(id, name)`` for ACTIVE, streaming, on-demand Anthropic Claude models."""
    rows: list[tuple[str, str]] = []
    if not isinstance(body, dict):
        return rows
    for entry in body.get("modelSummaries") or []:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("modelId")
        if not isinstance(model_id, str) or not model_id.startswith("anthropic.claude"):
            continue
        lifecycle = entry.get("modelLifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("status") not in (None, "ACTIVE"):
            continue
        if entry.get("responseStreamingSupported") is False:
            continue
        name = entry.get("modelName")
        rows.append((model_id, name if isinstance(name, str) and name else model_id))
    return rows


def fetch(ctx: Any) -> list[Any] | None:
    """The ``_Transport`` for wire ``bedrock``; see the module docstring."""
    from local_operator.model.discovery import DiscoveredModel

    if not ctx.api_key:
        return None
    try:
        resolved = bauth.resolve_sync(ctx.api_key)
    except bauth.BedrockCredentialError as exc:
        logger.debug("bedrock listing: no usable credentials (%s)", exc)
        return None
    base = bconn.control_base(resolved.region)
    try:
        profiles = _signed_get(
            ctx,
            resolved,
            f"{base}{PROFILES_PATH}",
            {"typeEquals": "SYSTEM_DEFINED", "maxResults": "1000"},
        )
        foundation = _signed_get(
            ctx,
            resolved,
            f"{base}{FOUNDATION_MODELS_PATH}",
            {"byProvider": "anthropic", "byInferenceType": "ON_DEMAND"},
        )
    except Exception as exc:  # noqa: BLE001 - every failure keeps the static list
        logger.debug("bedrock listing failed: %s", exc)
        return None
    if profiles is None and foundation is None:
        return None
    seen: set[str] = set()
    rows: list[Any] = []
    for model_id, name in parse_profiles(profiles) + parse_foundation_models(foundation):
        if model_id in seen:
            continue
        seen.add(model_id)
        rows.append(DiscoveredModel(id=model_id, name=name))
    return rows
