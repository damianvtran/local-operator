"""Keyless effective-access evidence shared by resolution and host publication.

Permission is not a bearer. Only a successful native broker resolve in THIS
process proves borrowed access, bounded by both upstream/grant expiry and the
current placement. Receipts contain no credential or refresh material and are
not persisted: after restart the next request must verify the owner again.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.network.credentials.types import (
    DEVICE_BOUND_PROVIDERS,
    BrokerError,
    Grant,
)

_lock = threading.Lock()
_receipts: dict[tuple[str, str], EffectiveAccess] = {}


@dataclass(frozen=True)
class EffectiveAccess:
    state: str
    eligible: bool = False
    owner: str = ""
    owner_label: str = ""
    observed_at: float | None = None
    last_grant_at: float | None = None
    last_grant_expires_at: float | None = None

    def context(self) -> dict[str, Any]:
        """Explicit keyless wire allowlist; historical bounds are NOT a bearer.

        Keep the last successful resolve's time/expiry when the next request is
        refused, so a UI can explain what changed without suggesting that an
        old cached bearer is usable or serializing any native credential data.
        """
        return {
            "owner_device": self.owner,
            "owner_label": self.owner_label,
            "observed_at": self.observed_at,
            "last_grant_at": self.last_grant_at,
            "last_grant_expires_at": self.last_grant_expires_at,
        }

    @property
    def usable(self) -> bool:
        return self.state in {"local", "effective_borrow"}


def _root(root: Path | None) -> str:
    from local_operator.paths import config_dir

    return str((root if root is not None else config_dir()).resolve())


def note_resolution(root: Path | None, provider: str, result: Grant | BrokerError) -> None:
    """Remember a resolve's outcome, never its bearer or refresh secret."""
    now = time.time()
    key = (_root(root), provider)
    with _lock:
        previous = _receipts.get(key)
    if isinstance(result, Grant):
        expiry = result.grant_expires_at_ms
        if result.token_expires_at_ms:
            expiry = min(expiry, result.token_expires_at_ms)
        receipt = EffectiveAccess(
            "effective_borrow",
            owner=result.credential_ref.owner_device,
            owner_label=result.credential_ref.owner_device_name,
            observed_at=now,
            last_grant_at=now,
            last_grant_expires_at=expiry / 1000,
        )
    else:
        state = (
            "owner_unavailable"
            if result.code in {"owner_offline", "owner_unreachable", "refresh_failed"}
            else (
                "withdrawn"
                if result.code
                in {
                    "not_a_holder",
                    "grant_invalid",
                    "no_local_credential",
                    "scope_mismatch",
                    "scope_unknown",
                    "consent_withdrawn",
                }
                else "device_bound" if result.code == "non_portable" else "pending"
            )
        )
        same_owner = previous is not None and previous.owner == result.owner_device
        receipt = EffectiveAccess(
            state,
            owner=result.owner_device,
            owner_label=result.owner_device_name
            or (previous.owner_label if same_owner and previous else ""),
            observed_at=now,
            last_grant_at=previous.last_grant_at if same_owner and previous else None,
            last_grant_expires_at=(
                previous.last_grant_expires_at if same_owner and previous else None
            ),
        )
    with _lock:
        _receipts[key] = receipt


#: What a lifecycle ANNOUNCEMENT may record. Deliberately closed and with no usable
#: member: an announcement says that metadata changed, never that a login works.
_OBSERVED_STATES = {
    "withdrawn": "withdrawn",
    "non_portable": "device_bound",
    "unsupported": "unsupported",
    # available/absent: the owner's state moved, so any earlier borrowed evidence is
    # stale and the next request must re-verify with the owner.
    "available": "pending",
    "absent": "pending",
}


def note_lifecycle_observation(
    root: Path | None, provider: str, state: str, *, owner: str = ""
) -> EffectiveAccess | None:
    """Record owner-announced lifecycle metadata as evidence, NEVER as usable access.

    Only :func:`note_resolution` given a ``Grant`` may produce a usable state
    (``effective_borrow``): a lifecycle frame carries no bearer and its ack proves
    only that metadata was applied. So the result here is always one of
    ``pending``/``withdrawn``/``device_bound``/``unsupported``, an unrecognised state
    is ignored (returns ``None``), and the last successful resolve's timestamps are
    kept only to explain what changed, exactly as a refused resolve keeps them.
    """
    mapped = _OBSERVED_STATES.get(state)
    if mapped is None:
        return None
    key = (_root(root), provider)
    with _lock:
        previous = _receipts.get(key)
        same_owner = previous is not None and previous.owner in ("", owner)
        receipt = EffectiveAccess(
            mapped,
            owner=owner or (previous.owner if previous else ""),
            owner_label=previous.owner_label if previous and same_owner else "",
            observed_at=time.time(),
            last_grant_at=previous.last_grant_at if previous and same_owner else None,
            last_grant_expires_at=(
                previous.last_grant_expires_at if previous and same_owner else None
            ),
        )
        _receipts[key] = receipt
    return receipt


def effective_access(
    provider: str, *, root: Path | None = None, local: bool = False, mesh: Any = None
) -> EffectiveAccess:
    """Read current local/borrow eligibility and evidence without dialing an owner.

    Runtime uses ``eligible`` to attempt native resolution; publication uses
    ``usable``. A holder row alone can NEVER promote publication to usable.
    Withdrawal is checked ahead of a receipt, including document deletion.
    """
    if local:
        return EffectiveAccess("local")
    from local_operator.network.credentials.client import MeshCredentialClient

    with _lock:
        receipt = _receipts.get((_root(root), provider))
    # Both readers and runtime use the broker's recipient-bound predicate. A
    # status-only client reads placement; it never requests a grant or dials.
    source = mesh if mesh is not None else MeshCredentialClient.for_this_device(root)
    from dataclasses import replace

    if provider in DEVICE_BOUND_PROVIDERS:
        if receipt is not None:
            return replace(receipt, state="device_bound", eligible=False)
        return EffectiveAccess(
            "device_bound",
            owner=source.owner_of(provider) if source is not None else "",
            owner_label=source.owner_label(provider) if source is not None else "",
        )
    if source is None or not source.should_borrow(provider):
        return (
            replace(receipt, state="withdrawn", eligible=False)
            if receipt
            else EffectiveAccess("absent")
        )
    owner = source.owner_of(provider)
    if receipt is None or receipt.owner != owner:
        return EffectiveAccess(
            "eligible_borrow_unverified", True, owner, source.owner_label(provider)
        )
    state = receipt.state
    if (
        state == "effective_borrow"
        and receipt.last_grant_expires_at is not None
        and receipt.last_grant_expires_at <= time.time()
    ):
        # Permission is still present: expiry needs owner resolution, not a new
        # authorization. Withdrawal must remain reserved for lost permission.
        state = "expired"
    return replace(receipt, state=state, eligible=True)
