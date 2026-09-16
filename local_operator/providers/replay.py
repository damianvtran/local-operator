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

#: What a model that REQUIRES a reasoning echo is sent for an assistant turn the
#: harness has no reasoning for. See ``ModelSpec.requires_reasoning_echo`` for
#: what is measured about the requirement (and for what is deliberately not
#: claimed about it).
#:
#: THIS TEXT IS MODEL-VISIBLE AND BILLED. The API concatenates the echo into the
#: context it reads, so the model sees the sentence and the provider charges the
#: input: measured on the wire at +330 input tokens for one real request (66
#: placeholders x 5, exactly what :func:`reasoning_echo_placeholder_tokens`
#: estimates). Deliberately a visible sentence rather than an empty string or a
#: space, because the corpus on what the validator reads is thin and
#: contradictory -- on one shape an all-blank body answered 200 where the same
#: body with the keys ABSENT answered 400 -- so a blank relies on leniency no
#: replica has promised, and it is also a value an intermediary could normalise
#: away. Text cannot be normalised into absence.
#:
#: It is deliberately not a harness bookkeeping word either: a phrase like
#: "payload" or "details" would collide with the names the harness uses for its
#: own provider state (``provider_payload["details"]`` and friends, which are
#: never shipped), and the model reads this line as part of its own conversation.
REASONING_ECHO_PLACEHOLDER = "[thinking not recorded]"


def reasoning_echo_placeholder_tokens() -> int:
    """Token estimate for one echo placeholder, on the counting ruler used here.

    Held next to the constant so the body builder (which spends it) and
    ``providers.context`` (which counts it) cannot drift into disagreeing about
    what a placeholder costs.
    """
    return max(1, len(REASONING_ECHO_PLACEHOLDER) // 4)


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
