"""The SHARED offline authority fence (S1/H5-4 remediation).

WHY THIS MODULE EXISTS. "Which facts authorize a bridge action" was answered in
three places (host construction, one op fence, one recovery check) with
different coverage — and the security preflight showed the union was reached
by none of them: a stale runtime could schedule an authenticated status fetch,
an already-admitted runtime could act after a local disable/consent loss, and
the decide doors ran with their own partial checks. This module is the ONE
answer, offline and synchronous, invoked BEFORE any schedule/fetch/op/local
mutation/decide/retry — at boot, at an op edge, at consumption, and at every
retry/recovery door.

WHAT IT CHECKS (all persisted, no network):
- feature ON (``team_bridge.enabled`` read from disk NOW; a toml/yml torn read
  is NOT "still enabled" — fail-closed, matching read_config_values' rule);
- a UNIQUE committed owner (``store.active_owner``: zero OR ambiguous -> no);
- a live, NON-EMPTY original generation (fenced generation must still be the
  committed one — no silent rebind, no backfill);
- the consent receipt is still present;
- the runtime's bound binding still matches EXACTLY: canonical base, account,
  tenant, endpoint, credential row, enrollment generation;
- tenant consistency (``runtime.tenant_id`` == the committed owner's tenant);
- the runtime's session is still the NATIVE designated session.
ANY miss returns a typed refusal (``feature_off`` / ``owner_changed`` /
``consent_revoked``) and the caller performs NO fetch, NO mutation, NO POST.

WHAT IT IS NOT. Not the admission gate: a fresh MUTATION still additionally
requires admission (and still re-checks server auth/kill per op). But a
trusted-HUMAN resolution of an ALREADY-frozen answer (m8 identical replay) and
a local recorded-receipt retrieval do NOT depend on admission — they depend on
THIS fence plus the frozen-record validation, because the server's recorded
result is authoritative even after a deny/kill landed (§6.2 :970–996).

FAIL-CLOSED ERRORS ONLY. Store damage/ambiguity -> ``owner_changed``. Only
store/config-typed errors are caught; programmer errors propagate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

__all__ = [
    "authority_fence",
    "scope_from_runtime",
    "scope_matches_runtime",
]


def _refusal(code: str, error: str) -> dict[str, str]:
    return {"code": code, "error": error}


def _owner_changed() -> dict[str, str]:
    return _refusal(
        "owner_changed",
        "the Radient account binding for this session changed — reopen the "
        "session to bind the current enrollment (no silent rebind)",
    )


def authority_fence(runtime: Any) -> dict[str, str] | None:
    """``None`` when this runtime is still authorized; else a typed refusal."""

    from local_operator.config import read_config_values

    from .settings import TeamBridgeSettings
    from .store import Store, StoreRefused

    try:
        config_dir = Path(runtime.config_dir)
        values = read_config_values(config_dir)
        if values is None:
            # A torn read is not "still enabled": fail closed, zero traffic.
            return _refusal(
                "feature_off", "agent messages are turned off (the configuration can't be read)"
            )
        settings = TeamBridgeSettings.from_values(values)
        if not settings.enabled:
            return _refusal("feature_off", "agent messages are turned off")
        store = Store(config_dir)
        owner = store.active_owner()
        if owner is None:
            return _owner_changed()
        generation = int(getattr(owner, "enrollment_generation", 0) or 0)
        if generation <= 0 or not store.current_generation_is_live(owner):
            return _owner_changed()
        if store.owner_consent_receipt(owner) is None:
            return _refusal(
                "consent_revoked",
                "consent for agent messages was withdrawn — nothing is sent until it is "
                "given again",
            )
        bound = runtime.owner
        if (
            str(getattr(bound, "canonical_base_url", ""))
            != str(getattr(owner, "canonical_base_url", ""))
            or str(getattr(bound, "account_id", "")) != str(owner.account_id)
            or str(getattr(bound, "tenant_id", "")) != str(owner.tenant_id)
            or str(getattr(bound, "endpoint_id", "")) != str(getattr(owner, "endpoint_id", ""))
            or int(getattr(bound, "credential_id", -1)) != int(owner.credential_id)
            or int(getattr(bound, "enrollment_generation", -1)) != generation
        ):
            return _owner_changed()
        if str(getattr(runtime, "tenant_id", "")) != str(owner.tenant_id):
            return _owner_changed()
        session_id = str(getattr(runtime, "session_id", "") or "")
        if not session_id or store.designated_session_for(owner) != session_id:
            return _owner_changed()
        return None
    except (StoreRefused, sqlite3.Error, OSError):
        return _owner_changed()


def scope_from_owner(owner: Any, session_id: str) -> dict[str, Any]:
    """The ORIGINAL-scope block for a bound OWNER + designated session.

    Shared by the receive-side :func:`scope_from_runtime` AND by the native
    PRODUCER (H5 F3/S-8): the producer stamps THIS block — built from its
    authenticated pinned owner snapshot and the native designation — before
    handoff, and the receiver consumes the STORED block verbatim. Re-deriving
    it from the receiver's current fields at arrival is exactly the backfill
    the contract forbids.
    """

    return {
        "v": 1,
        "kind": "bridge",
        "base_url": str(getattr(owner, "canonical_base_url", "") or ""),
        "account_id": str(getattr(owner, "account_id", "") or ""),
        "tenant_id": str(getattr(owner, "tenant_id", "") or ""),
        "endpoint_id": str(getattr(owner, "endpoint_id", "") or ""),
        "credential_id": int(getattr(owner, "credential_id", -1)),
        "generation": str(getattr(owner, "enrollment_generation", "") or ""),
        "session_id": str(session_id or ""),
    }


def scope_from_runtime(runtime: Any) -> dict[str, Any]:
    """The ORIGINAL-scope block persisted with a bridge ask at ISSUANCE.

    Bounded and JSON-safe on purpose: the ask log stores plain dicts, and the
    recovery loader reads this block — never the current runtime — to decide
    what the human actually answered. Kept as a thin delegate to
    :func:`scope_from_owner` so the receive-side door and the producer stamp
    can never drift on the shape (H5 F3).
    """

    return scope_from_owner(runtime.owner, str(getattr(runtime, "session_id", "") or ""))


def scope_matches_runtime(scope: Any, runtime: Any) -> bool:
    """Whether a persisted ORIGINAL scope names EXACTLY this runtime's binding.

    Every field must match, and the scope's generation must be NON-EMPTY: a
    legacy/empty scope is non-authorizing (never backfilled from current).
    """

    if not isinstance(scope, dict):
        return False
    bound = runtime.owner
    generation = str(scope.get("generation") or "")
    if not generation or generation != str(getattr(bound, "enrollment_generation", "") or ""):
        return False
    return (
        str(scope.get("kind") or "") == "bridge"
        and str(scope.get("base_url") or "") == str(getattr(bound, "canonical_base_url", "") or "")
        and str(scope.get("account_id") or "") == str(getattr(bound, "account_id", "") or "")
        and str(scope.get("tenant_id") or "") == str(getattr(bound, "tenant_id", "") or "")
        and str(scope.get("endpoint_id") or "") == str(getattr(bound, "endpoint_id", "") or "")
        and int(scope.get("credential_id", -1)) == int(getattr(bound, "credential_id", -2))
        and str(scope.get("session_id") or "") == str(getattr(runtime, "session_id", "") or "")
    )
