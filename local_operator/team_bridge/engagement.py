"""May a Team Bridge card ENGAGE a designated chief-of-staff session nobody has typed in?

WHY THIS MODULE EXISTS (live E2E F8, local-operator-ui #1314). The peer gate in
``session/runtime/server.py`` refuses every ``peer_message`` to a session that has
never run a user turn, so a row can never become the opening row of a
conversation its owner did not start. That rule is right for ``lop send``. It also
meant a member's designated chief of staff — designated in settings but never
opened — could never receive a Team Bridge card: every delivery failed
``conversation_failed`` and retried forever.

The bridge may cross that gate for ONE session only, and only when the member
was TOLD it could: ``bridge_consent_v2`` discloses that an incoming message may
engage a never-used chief-of-staff session; ``bridge_consent_v1`` does not. So
the exception is gated on the consent version the member ACCEPTED, read from the
native enrollment receipt in this machine's own store
(:meth:`Store.owner_consent_receipt_readonly`) — never from the served status
payload, which is the CURRENT offer and not an acceptance.

CONSTRAINTS, each load-bearing:
- The accepted set is EXPLICIT (:data:`ENGAGING_CONSENT_VERSIONS`). "v2 or later"
  is never inferred from string or number ordering: a future version that drops
  the disclosure must not inherit the permission. Adding a version is a
  deliberate edit here, next to the disclosure it relies on.
- Unknown, missing, unreadable, fenced or v1 → NO engagement. The caller holds
  the message with :data:`HOLD_OPEN_CHIEF_OF_STAFF` (visible to the owner,
  retried on the bounded backoff, never dropped) until the owner opens the
  session once, after which the ordinary engaged path delivers it.
- What crosses the gate is unchanged: the same ``team_bridge_message`` card,
  the same untrusted framing, delivered on the mailbox path. It is not a user
  turn and it carries no instruction of its own.
- Stdlib + the bridge store only: the runtime server imports this inside its
  refusal branch, and must not pay for the session graph to answer it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Consent versions whose ACCEPTED text discloses that an incoming message may
#: engage a never-used chief-of-staff session. Explicit by design — see the
#: module docstring. ``bridge_consent_v1`` is deliberately absent.
ENGAGING_CONSENT_VERSIONS: frozenset[str] = frozenset({"bridge_consent_v2"})

#: The hold code a card waits under while the accepted consent does not allow
#: engagement. Machine token; the owner-facing sentence is rendered by the UI
#: from this code (read_context passes it through HOLD_CODES).
HOLD_OPEN_CHIEF_OF_STAFF = "open_chief_of_staff"


def consent_allows_engagement(consent_version: object) -> bool:
    """Whether an ACCEPTED consent version permits engaging an unused session."""

    return isinstance(consent_version, str) and consent_version in ENGAGING_CONSENT_VERSIONS


def designated_session_may_be_engaged(session_id: str, *, config_dir: Path | str) -> bool:
    """Whether ``session_id`` may receive a Team Bridge card before its first user turn.

    True only when ALL hold, read fresh from local state (no network):
    - ``team_bridge.enabled`` is on in the config document on disk;
    - exactly one committed enrollment exists and its generation is live;
    - ``session_id`` IS that enrollment's designated chief-of-staff session;
    - the enrollment's ACCEPTED consent version is in
      :data:`ENGAGING_CONSENT_VERSIONS`.

    Every failure — absent store, damaged row, unreadable config — answers
    False, the direction that keeps the ordinary refusal in force.
    """

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import TeamBridgeSettings
    from local_operator.team_bridge.store import Store, StoreRefused

    if not session_id:
        return False
    try:
        root = Path(config_dir)
        values = read_config_values(root)
        if values is None or not TeamBridgeSettings.from_values(values).enabled:
            return False
        store = Store(root)
        owner = store.active_owner_readonly()
        if owner is None:
            return False
        receipt = store.owner_consent_receipt_readonly(owner)
        if receipt is None or not consent_allows_engagement(receipt.get("consent_version")):
            return False
        return store.designated_session_for_readonly(owner) == session_id
    except (StoreRefused, sqlite3.Error, OSError, ValueError):
        return False


__all__ = [
    "ENGAGING_CONSENT_VERSIONS",
    "HOLD_OPEN_CHIEF_OF_STAFF",
    "consent_allows_engagement",
    "designated_session_may_be_engaged",
]
