"""Provider-native continuations, scoped to the wire protocol that made them.

Opaque reasoning is durable protocol state, not assistant prose. Preserve the
provider's ordering and signatures, but never replay stale native content after
an edit or send it to another endpoint/model. The harness still owns the visible
message and tool calls; the fingerprint below prevents native replay bypassing
compaction, argument repair, or imported-history changes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from local_operator.harness.types import Message, ModelSpec


def credential_scope(api_key: str | None, oauth_access: Any = None) -> str:
    """Opaque identity, never the credential, for native-state provenance.

    OAuth tokens refresh frequently, so use the stable account when present.
    API keys (and OAuth grants without account metadata) use a one-way digest;
    replaying under a changed credential is less safe than rebuilding context.
    """
    account = getattr(oauth_access, "account_id", None)
    if account:
        identity = f"oauth:{account}:{getattr(oauth_access, 'org_id', None)}"
    else:
        identity = getattr(oauth_access, "access_token", None) or api_key or "anonymous"
    return hashlib.sha256(identity.encode()).hexdigest()


def visible_fingerprint(text: str, calls: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps([text, calls], sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def native_payload(
    model: ModelSpec,
    endpoint: str,
    protocol: str,
    items: list[dict[str, Any]],
    text: str,
    calls: list[dict[str, Any]],
    scope: str | None = None,
) -> dict[str, Any]:
    return {
        "native_replay": {
            "protocol": protocol,
            "provider": model.provider,
            "model": model.model_id,
            "endpoint": endpoint,
            "credential_scope": scope,
            "visible": visible_fingerprint(text, calls),
            "items": items,
        }
    }


def replay_items(
    message: Message, model: ModelSpec, endpoint: str, protocol: str, scope: str | None = None
) -> list[dict[str, Any]] | None:
    if message.role != "assistant" or message.stop_reason in ("error", "aborted", "length"):
        return None
    payload = (message.provider_payload or {}).get("native_replay")
    if not isinstance(payload, dict):
        return None
    if any(
        payload.get(key) != value
        for key, value in (
            ("protocol", protocol),
            ("provider", model.provider),
            ("model", model.model_id),
            ("endpoint", endpoint),
            ("credential_scope", scope),
        )
    ):
        return None
    calls = [
        {"id": call.id, "name": call.name, "args": call.arguments} for call in message.tool_calls
    ]
    if payload.get("visible") != visible_fingerprint(message.text, calls):
        return None
    items = payload.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return None
    return items
