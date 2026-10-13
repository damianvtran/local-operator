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
_receipts: dict[tuple[str, str], tuple[str, float, str]] = {}


@dataclass(frozen=True)
class EffectiveAccess:
    state: str
    eligible: bool = False
    owner: str = ""

    @property
    def usable(self) -> bool:
        return self.state in {"local", "effective_borrow"}


def _root(root: Path | None) -> str:
    from local_operator.paths import config_dir

    return str((root if root is not None else config_dir()).resolve())


def note_resolution(root: Path | None, provider: str, result: Grant | BrokerError) -> None:
    """Remember a resolve's outcome, never its bearer or refresh secret."""
    if isinstance(result, Grant):
        expiry = result.grant_expires_at_ms
        if result.token_expires_at_ms:
            expiry = min(expiry, result.token_expires_at_ms)
        receipt = (result.credential_ref.owner_device, expiry / 1000, "effective_borrow")
    else:
        state = (
            "owner_unavailable"
            if result.code in {"owner_offline", "owner_unreachable", "refresh_failed"}
            else (
                "withdrawn"
                if result.code in {"not_a_holder", "grant_invalid", "no_local_credential"}
                else "device_bound" if result.code == "non_portable" else "pending"
            )
        )
        receipt = (result.owner_device, 0.0, state)
    with _lock:
        _receipts[(_root(root), provider)] = receipt


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
    if provider in DEVICE_BOUND_PROVIDERS:
        return EffectiveAccess("device_bound")
    from local_operator.network.credentials.client import MeshCredentialClient

    with _lock:
        receipt = _receipts.get((_root(root), provider))
    # Both readers and runtime use the broker's recipient-bound predicate. A
    # status-only client reads placement; it never requests a grant or dials.
    source = mesh if mesh is not None else MeshCredentialClient.for_this_device(root)
    if source is None or not source.should_borrow(provider):
        return EffectiveAccess("withdrawn" if receipt else "absent")
    owner = source.owner_of(provider)
    if receipt is None or receipt[0] != owner:
        return EffectiveAccess("eligible_borrow_unverified", True, owner)
    _owner, expiry, state = receipt
    if state == "effective_borrow" and expiry <= time.time():
        state = "withdrawn"
    return EffectiveAccess(state, True, owner)
