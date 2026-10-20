"""Enrollment/designation authority (H4d).

The wake authority is the NATIVE enrollment record — the captured-owner
consent receipt committed atomically with the designated CoS session id
(``Store.commit_enrollment(..., cos_session_id=...)``), generation-fenced so
``invalidate`` cannot leave a stale snapshot naming a wake destination. A
generic ``{enabled, session}`` preference has NO path into that record
(H2's seam law), so this module exposes exactly two writes:

* :func:`commit_consent_and_designation` — called ONLY after the real
  captured-owner consent flow succeeded, with the receipt that flow produced.
* :func:`commit_default_enrollment` — called ONLY after the opt-out default's
  authenticated settings PUT answered 200 with an ACTIVE endpoint. The server's
  recorded consent is the source; served/cached text alone never reaches it.
  This is what makes the default-on state able to receive (owner + receipt),
  while a missing ``team_bridge.session`` stays badge-only per RFC §13.6.

Neither invents a consent: the first needs the captured-owner receipt, the
second needs the server's successful write. Neither writes a designation
without a user choice.

The settings keys ``team_bridge.session``/``team_bridge.enabled`` stay SURFACE
write-through: an onboarding/settings surface calls this function as part of
the consent commit; a bare preference edit away from the consent flow does
not, and therefore cannot wake anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
)


def commit_consent_and_designation(
    owner: Any,
    receipt: Any,
    *,
    cos_session_id: str | None,
    config_dir: Path,
) -> None:
    """Commit the consent receipt + CoS designation in ONE native transaction.

    Raises ``StoreRefused`` (``invalid_receipt`` / ``stale_enrollment``) when
    the caller offers no real receipt or a generation at/below the live one —
    the same fence the seam already enforces; this wrapper exists so H4 has
    ONE named entry point to review — it never softens the seam.
    """

    # An empty designation is a legitimate commit: consent without a session
    # (badge-only arrivals, §5.6) — NOT a missing receipt.
    Store(config_dir).commit_enrollment(owner, receipt, cos_session_id=cos_session_id or None)


def commit_default_enrollment(
    *,
    config_dir: Path,
    credential_id: int,
    account_id: str,
    tenant_id: str,
    endpoint_id: str,
    consent_version: str,
    canonical_base_url: str,
    cos_session_id: str | None,
) -> bool:
    """Commit the NATIVE receipt for the opt-out default's consent write.

    WHY THIS IS A REAL CONSENT STEP, NOT A DERIVED ONE. The caller runs it only
    after the server PROVES an ACTIVE endpoint with recorded consent: either
    this machine's authenticated settings PUT returned one, or a fresh status
    answer reports the endpoint another surface enabled. The served consent TEXT
    is never the source and never reaches this function. The v1 server exposes
    no opaque receipt id, so the endpoint resource id (the record the consent is
    attached to) is the non-secret ``consent_ref``; the version is the server's.

    WHY THE REFUSALS. A machine already holding a committed owner returns
    ``False`` instead of replacing it: the store's one-owner model gives no safe
    way to retire another namespace implicitly, and a different owner needs an
    explicit re-enrollment. ``Store.commit_enrollment`` keeps its own monotonic
    fence, so a concurrent commit refuses locally rather than rolling authority
    backwards. The caller treats any exception as "no local authority" and keeps
    the status answer.

    DESIGNATION. ``cos_session_id`` is the user's existing ``team_bridge.session``
    choice, or None. None is badge-only per RFC §13.6 — the designation is the
    wake consent and is never guessed from "the current session".
    """

    store = Store(config_dir)
    # NOT ``active_owner() is not None``: None means zero OR ambiguous, and an
    # ambiguous store must never receive a third row from an automatic path.
    if store.committed_owner_count() != 0:
        return False
    owner = OwnerSnapshot(
        credential_id=credential_id,
        account_id=account_id,
        tenant_id=tenant_id,
        endpoint_id=endpoint_id,
        canonical_base_url=canonical_base_url,
        enrollment_generation=1,
    ).validated()
    store.commit_enrollment(
        owner,
        OwnerReceipt(consent_version=consent_version, consent_ref=endpoint_id),
        cos_session_id=cos_session_id or None,
    )
    return True


def designation(owner: Any, *, config_dir: Path) -> str | None:
    """Read the committed designation under the live-owner/generation fence."""

    return Store(config_dir).designated_session_for(owner)


__all__ = [
    "commit_consent_and_designation",
    "commit_default_enrollment",
    "designation",
    "StoreRefused",
]
