"""Owner-fetched native catalogue METADATA; never a quota-bypassing bearer grant.

Catalogue selection may use inference-exhausted logins. Keeping authentication
and HTTP on the approved owner is therefore a security boundary, not a transport
optimization. This module intentionally has no model selection or cache writes.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from local_operator.network.types import MeshRefusal
from local_operator.network.wire import CREDENTIAL_CATALOGUE_METADATA_V1 as CAPABILITY

# Leave room for the broker/control envelope within legacy 16KiB readers.
# Oversized catalogues fail as a whole; silently truncating would lie to B.
MAX_BYTES = 12288
MAX_MODELS = 256
MAX_TTL = 300
ERROR_CODES = frozenset(
    {
        "catalogue_unsupported",
        "catalogue_invalid",
        "catalogue_unavailable",
        "not_a_holder",
        "scope_unknown",
        "owner_offline",
        "local_only",
        "identity_mismatch",
        "epoch_stale",
        "not_owner",
    }
)
_MODEL_FIELDS = {
    "id",
    "provider",
    "context_window",
    "default_context_window",
    "max_context_window",
    "max_tokens",
    "supports_images",
}
_FIELDS = {
    "kind",
    "network_id",
    "owner_device",
    "credential_key",
    "provider",
    "models",
    "account_scope",
    "incarnation",
    "material_generation",
    "login_chain_revision",
    "ttl",
    "validator",
}


def negotiated(link: Any) -> bool:
    return CAPABILITY in getattr(link, "local_capabilities", ()) and CAPABILITY in getattr(
        link, "capabilities", ()
    )


def validate_response(value: Any, *, network: str, owner: str, key: str) -> dict[str, Any]:
    """Slice B consumes only this validated document, never grant/API-key fallback."""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("invalid catalogue metadata fields")
    if len(json.dumps(value).encode()) > MAX_BYTES:
        raise ValueError("catalogue metadata exceeds byte bound")
    if (
        value["kind"],
        value["network_id"],
        value["owner_device"],
        value["credential_key"],
        value["provider"],
    ) != ("catalogue_metadata", network, owner, key, "openai"):
        raise ValueError("catalogue metadata namespace mismatch")
    for field in ("account_scope", "incarnation", "validator"):
        if not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            raise ValueError("invalid catalogue scope or revision")
    for field in ("material_generation", "login_chain_revision", "ttl"):
        item = value[field]
        if type(item) is not int or not 1 <= item <= (MAX_TTL if field == "ttl" else 2**53 - 1):
            raise ValueError("invalid catalogue lifetime")
    models = value["models"]
    if not isinstance(models, list) or len(models) > MAX_MODELS:
        raise ValueError("invalid catalogue model count")
    ids = set()
    for row in models:
        if not isinstance(row, dict) or set(row) != _MODEL_FIELDS or row["provider"] != "openai":
            raise ValueError("invalid catalogue model fields")
        model = row["id"]
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}", model)
            or model.startswith(("sk-", "eyJ", "Bearer"))
            or model in ids
        ):
            raise ValueError("invalid catalogue model identifier")
        ids.add(model)
        for name in ("context_window", "max_tokens"):
            if type(row[name]) is not int or not 0 <= row[name] <= 10**9:
                raise ValueError("invalid catalogue model limit")
        for name in ("default_context_window", "max_context_window"):
            if row[name] is not None and (
                type(row[name]) is not int or not 1 <= row[name] <= 10**9
            ):
                raise ValueError("invalid catalogue context bound")
        if row["supports_images"] is not None and type(row["supports_images"]) is not bool:
            raise ValueError("invalid catalogue modality")
    return value


def _approved_rows(
    broker: Any, link: Any, frame: dict[str, Any]
) -> tuple[str, dict[int, tuple[str, dict[str, str]]]]:
    """Re-read authority and native metadata; no bearer/IdP access at this gate."""
    from local_operator.network.credentials import consent, reconcile
    from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState
    from local_operator.providers.auth_store import AuthStore

    broker._server.authorizer.check(link.context, frame)
    by, refusal = broker._caller(link, frame, "openai", "openai")
    if refusal is not None:
        raise MeshRefusal("identity_mismatch", "catalogue requester identity refused")
    entry = broker._entry("openai")
    if (
        entry is None
        or entry.owner_device != broker.self_device
        or entry.provider != "openai"
        or not entry.is_holder(by)
    ):
        raise MeshRefusal("not_a_holder", "catalogue metadata is not shared with this device")
    if SyncState.load(broker.network_id, broker.root).mark_for("openai") == MARK_LOCAL_ONLY:
        raise MeshRefusal("local_only", "catalogue metadata is local-only")
    consent.check_borrow(broker.network_id, broker.self_device, by, "openai", root=broker.root)
    derived = reconcile.is_derived(entry.holder(by))
    allowed = None
    if derived:
        allowed, _, code = reconcile.serve_gate(
            broker.network_id, broker.self_device, by, "openai", broker.root
        )
        if not allowed:
            raise MeshRefusal(code or "scope_unknown", "catalogue account approval is unavailable")
    native = broker._auth_store_instance()
    inventory = AuthStore.native_oauth_inventory(native._db_path)
    if inventory is None:
        raise MeshRefusal("scope_unknown", "native catalogue metadata unavailable")
    rows = {}
    for provider, row_id, marker, account, email, org, origin, kind in inventory:
        if (
            provider != "openai"
            or kind != "oauth"
            or origin is not None
            or not isinstance(marker, str)
            or not marker
            or not account
        ):
            continue
        if allowed is not None and row_id not in allowed:
            continue
        context = {
            name: str(v)
            for name, v in (("account_id", account), ("email", email), ("org_id", org))
            if v
        }
        try:
            with consent.serving_guard(
                broker.network_id, broker.self_device, by, "openai", context, broker.root
            ):
                if derived:
                    reconcile.derived_mint_check(
                        broker.network_id,
                        broker.self_device,
                        by,
                        "openai",
                        context,
                        broker.root,
                        credential_id=int(row_id),
                        kind="oauth",
                    )
                rows[int(row_id)] = (marker, context)
        except MeshRefusal:
            continue
    if not rows:
        raise MeshRefusal("scope_unknown", "no approved native catalogue account")
    return by, rows


def owner_catalogue(broker: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
    """Closed request shape; owner authority before refresh and after HTTP."""
    from local_operator.network.credentials.types import BrokerError

    try:
        allowed = {
            "op",
            "req",
            "kind",
            "network_id",
            "epoch",
            "from_device",
            "key",
            "provider",
            "ttl",
            "validator",
        }
        if (
            set(frame) - allowed
            or frame.get("key") != "openai"
            or frame.get("provider") != "openai"
        ):
            raise MeshRefusal(
                "catalogue_unsupported", "only native OpenAI catalogue metadata is supported"
            )
        if not negotiated(link):
            raise MeshRefusal("catalogue_unsupported", "peer lacks catalogue metadata capability")
        ttl = frame.get("ttl", 60)
        if type(ttl) is not int or not 1 <= ttl <= MAX_TTL:
            raise MeshRefusal("catalogue_invalid", "invalid catalogue TTL")
        validator = frame.get("validator", "")
        if not isinstance(validator, str) or (
            validator and not re.fullmatch(r"[0-9a-f]{64}", validator)
        ):
            raise MeshRefusal("catalogue_invalid", "invalid catalogue validator")
        by, rows = _approved_rows(broker, link, frame)
        return broker._loop.submit(_fetch(broker, link, frame, rows, ttl), timeout=20)
    except MeshRefusal as exc:
        # Native/consent refusal prose can name an account. This protocol returns
        # only a bounded known code, never auth-adjacent exception text.
        code = exc.code if exc.code in ERROR_CODES else "catalogue_unavailable"
        return BrokerError(
            code=code, message="owner catalogue metadata refused", key="openai"
        ).to_detail()
    except Exception:
        # Provider/transport exceptions may embed auth material: never echo them.
        return BrokerError(
            code="catalogue_unavailable",
            message="owner catalogue metadata unavailable",
            key="openai",
        ).to_detail()


async def _fetch(
    broker: Any, link: Any, frame: dict[str, Any], rows: dict[int, Any], ttl: int
) -> dict[str, Any]:
    import asyncio

    from local_operator.network.credentials import consent, lifecycle

    native = broker._auth_store_instance()
    access = await native.get_catalogue_access("openai", eligible_credential_ids=tuple(rows))
    if (
        access is None
        or access.kind != "oauth"
        or not access.account_id
        or access.credential_id not in rows
    ):
        raise MeshRefusal("catalogue_unavailable", "approved native OAuth catalogue unavailable")
    _, current = _approved_rows(broker, link, frame)
    if current.get(access.credential_id) != rows[access.credential_id]:
        raise MeshRefusal("scope_unknown", "catalogue account changed during refresh")
    models = await asyncio.to_thread(_fetch_models, access)
    by, current = _approved_rows(broker, link, frame)
    if current.get(access.credential_id) != rows[access.credential_id]:
        raise MeshRefusal("scope_unknown", "catalogue account changed during fetch")
    marker, context = current[access.credential_id]
    if context.get("account_id") != access.account_id:
        raise MeshRefusal("scope_unknown", "catalogue account identity changed")
    # Final authority check and response materialization serialize with revoke,
    # but the native refresh and provider HTTP above hold NO consent lock.
    with consent.serving_guard(
        broker.network_id, broker.self_device, by, "openai", context, broker.root
    ):
        # Mirror the existing mint boundary instead of recursively re-entering
        # admission: its pre-resolution reads may acquire locks already held by
        # serving_guard. Recheck membership/epoch and placement before emitting.
        from local_operator.network.credentials import reconcile
        from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

        broker._server.authorizer.check(link.context, frame)
        entry = broker._entry("openai")
        if (
            entry is None
            or entry.owner_device != broker.self_device
            or entry.provider != "openai"
            or not entry.is_holder(by)
            or SyncState.load(broker.network_id, broker.root).mark_for("openai") == MARK_LOCAL_ONLY
        ):
            raise MeshRefusal("not_a_holder", "catalogue sharing changed")
        if reconcile.is_derived(entry.holder(by)):
            reconcile.derived_mint_check(
                broker.network_id,
                broker.self_device,
                by,
                "openai",
                context,
                broker.root,
                credential_id=access.credential_id,
                kind="oauth",
            )
        contexts = lifecycle._native_contexts(broker.root, "openai") or {}
        candidates = [
            (scope, facts) for scope, (facts, identity) in contexts.items() if identity == context
        ]
        if len(candidates) != 1:
            raise MeshRefusal("scope_unknown", "catalogue lifecycle identity unavailable")
        scope, facts = candidates[0]
        if facts.native_login_ref != marker:
            raise MeshRefusal("scope_unknown", "catalogue login incarnation changed")
        counters = lifecycle.LifecycleStore(broker.network_id, broker.root).record_native(
            "openai", scope, facts
        )
        if counters.state != lifecycle.STATE_AVAILABLE:
            raise MeshRefusal("scope_unknown", "catalogue lifecycle is not available")

        def digest(text: str) -> str:
            return hashlib.sha256(text.encode()).hexdigest()

        response = {
            "kind": "catalogue_metadata",
            "network_id": broker.network_id,
            "owner_device": broker.self_device,
            "credential_key": "openai",
            "provider": "openai",
            "models": models,
            "account_scope": digest(f"{broker.network_id}:{broker.self_device}:{scope}"),
            "incarnation": digest(f"{broker.network_id}:{broker.self_device}:{marker}"),
            "material_generation": counters.material_generation,
            "login_chain_revision": counters.login_chain_revision,
            "ttl": ttl,
            "validator": digest(json.dumps(models, sort_keys=True)),
        }
        return validate_response(
            response, network=broker.network_id, owner=broker.self_device, key="openai"
        )


def _fetch_models(access: Any) -> list[dict[str, Any]]:
    """Reuse native Codex parsing and fixed owner endpoint; bound HTTP before JSON."""
    import httpx

    from local_operator.model import discovery

    class BoundedHTTP(httpx.Client):
        def get(self, url: Any, **kwargs: Any) -> httpx.Response:
            with self.stream("GET", url, **kwargs) as response:
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > 1048576:
                        raise ValueError("upstream catalogue exceeds byte bound")
                # Only status/content are needed by the shared native parser;
                # do not retain authenticated request/header objects downstream.
                return httpx.Response(response.status_code, content=bytes(data))

    with BoundedHTTP() as bounded_client:
        ctx = discovery._FetchContext(
            provider_id="openai",
            base_url="",
            api_key=access.access_token,
            is_oauth=True,
            account_id=access.account_id,
            timeout=10,
            client=bounded_client,
        )
        models = discovery._fetch_openai_oauth(ctx)
    if models is None or len(models) > MAX_MODELS:
        raise ValueError("native catalogue is invalid")
    # Identifier syntax alone cannot reject a provider echoing an opaque bearer
    # or account header in a model slug. Those values are known only HERE; reject
    # the entire listing instead of allowing an authenticated echo onto the wire.
    for row in models:
        if any(secret and secret in row.id for secret in (access.access_token, access.account_id)):
            raise ValueError("native catalogue echoed authentication identity")
    return [
        {
            "id": row.id,
            "provider": "openai",
            "context_window": row.context_window,
            "default_context_window": row.default_context_window,
            "max_context_window": row.max_context_window,
            "max_tokens": row.max_tokens,
            "supports_images": row.supports_images,
        }
        for row in models
    ]
