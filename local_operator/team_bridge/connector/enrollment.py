"""Enrollment/designation authority (H4d).

The wake authority is the NATIVE enrollment record — the captured-owner
consent receipt committed atomically with the designated CoS session id
(``Store.commit_enrollment(..., cos_session_id=...)``), generation-fenced so
``invalidate`` cannot leave a stale snapshot naming a wake destination. A
generic ``{enabled, session}`` preference has NO path into that record
(H2's seam law), so this module exposes exactly two writes:

* :func:`commit_consent_and_designation` — called ONLY after the real
  captured-owner consent flow succeeded, with the receipt that flow produced.
* :func:`commit_settings_enrollment` — called ONLY by the client after a
  USER's explicit ``PUT /messaging/settings`` answered 200 with an ACTIVE
  endpoint and the consent echo that machine sent. That user action is the
  source; served/cached text alone never reaches it, and a status read never
  calls it (consent hold: #2712's auto-enrollment is disabled) — except the
  hub-created ``default_on`` endpoint of the gated ``bridge_consent_v2``
  enrolment (``TeamBridgeClient.enroll_default_on``), which commits through the
  same function only after the hub's answer proves it created that endpoint.

Neither invents a consent: the first needs the captured-owner receipt, the
second needs the server's successful write to the user's own enable. Neither
writes a designation without a user choice.

The settings keys ``team_bridge.session``/``team_bridge.enabled`` stay SURFACE
write-through: the onboarding consent surface calls
:func:`commit_consent_and_designation` as part of its consent commit; a bare
preference edit away from the consent flow does not, and therefore cannot wake
anything.
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
    _connector_after_commit(config_dir)


def commit_settings_enrollment(
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
    """Commit the NATIVE receipt for a user's explicit settings enable.

    WHY THIS IS A REAL CONSENT STEP, NOT A DERIVED ONE. The caller runs it only
    after the USER's own authenticated settings PUT returned an ACTIVE endpoint
    with the consent echo this machine sent. The served consent TEXT is never
    the source and never reaches this function. The v1 server exposes no opaque
    receipt id, so the endpoint resource id (the record the consent is attached
    to) is the non-secret ``consent_ref``; the version is the server's.

    WHY NOT ON A STATUS READ. A fresh answer that says the endpoint is active is
    not proof that THIS user chose it: another surface or an older build may
    have enabled it. The consent hold (#2712) therefore forbids the status
    discipline from reaching this function; only the explicit write may.

    WHICH CONSENT AN EXPLICIT WRITE COMMITS UNDER (D1, review R1-2). Either the
    echo the surface just sent, or, for a resume that sends none, the hub's own
    CURRENT record for this member on the answer to that same guarded write
    (``TeamBridgeClient._commit_explicit_enable``). That second source is a
    deliberate change from "a bare resume is never an opt-in": without it a
    reinstalled or wiped machine had no product path to bind itself to an
    endpoint whose consent still binds. A status read still never reaches here.

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
    owner = OwnerSnapshot(
        credential_id=credential_id,
        account_id=account_id,
        tenant_id=tenant_id,
        endpoint_id=endpoint_id,
        canonical_base_url=canonical_base_url,
        enrollment_generation=1,
    ).validated()
    try:
        # The one-owner rule is enforced INSIDE the store's immediate
        # transaction (review R2-2), never by a count taken out here: the two
        # steps were atomic only while both ran on the event loop, and two
        # concurrent enables from worker threads produced a two-owner store
        # (3/40) that reads as "no owner" for ever.
        store.commit_enrollment(
            owner,
            OwnerReceipt(consent_version=consent_version, consent_ref=endpoint_id),
            cos_session_id=cos_session_id or None,
            require_no_owner=True,
        )
    except StoreRefused as refusal:
        if refusal.code == "owner_present":
            # NOT an error: the answer to "may this machine own another
            # endpoint" was no. Ambiguity is included — a store with two rows
            # must never receive a third.
            return False
        raise
    _connector_after_commit(config_dir)
    return True


def _connector_after_commit(config_dir: Path) -> None:
    """Install/start the connector after a successful LOCAL commit (D2a).

    A successful commit is the earliest honest event: before it there is no
    local authority to receive for. Called AFTER the store transaction, so a
    service failure can never undo or mask the commit; ``sync_connector_service``
    itself never raises. Both native entry points share this hook, including
    the consent-surface commit that bypasses the HTTP client.
    """

    from local_operator.team_bridge.connector.install import sync_connector_service

    sync_connector_service(config_dir)


def designation(owner: Any, *, config_dir: Path) -> str | None:
    """Read the committed designation under the live-owner/generation fence."""

    return Store(config_dir).designated_session_for(owner)


__all__ = [
    "commit_consent_and_designation",
    "commit_settings_enrollment",
    "designation",
    "StoreRefused",
]
