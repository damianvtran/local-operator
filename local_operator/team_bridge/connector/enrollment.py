"""Enrollment/designation authority (H4d).

The wake authority is the NATIVE enrollment record — the captured-owner
consent receipt committed atomically with the designated CoS session id
(``Store.commit_enrollment(..., cos_session_id=...)``), generation-fenced so
``invalidate`` cannot leave a stale snapshot naming a wake destination. A
generic ``{enabled, session}`` preference has NO path into that record
(H2's seam law), so this module exposes exactly one write:

* :func:`commit_consent_and_designation` — called ONLY after the real
  captured-owner consent flow succeeded, with the receipt that flow produced.
  It never invents a receipt, never re-derives consent from served text, and
  never writes a designation without one.

The settings keys ``team_bridge.session``/``team_bridge.enabled`` stay SURFACE
write-through: an onboarding/settings surface calls this function as part of
the consent commit; a bare preference edit away from the consent flow does
not, and therefore cannot wake anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.team_bridge.store import Store, StoreRefused


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


def designation(owner: Any, *, config_dir: Path) -> str | None:
    """Read the committed designation under the live-owner/generation fence."""

    return Store(config_dir).designated_session_for(owner)


__all__ = ["commit_consent_and_designation", "designation", "StoreRefused"]
