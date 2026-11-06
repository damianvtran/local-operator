"""Re-point a Team Bridge enrollment whose pinned login row is gone or dead.

WHY THIS EXISTS (the 2026-11-06 incident). The enrollment pins the EXACT
credential row id it committed (``enrollments.credential_id``; H2's owner pin).
Signing out of Radient and back in as the SAME identity deletes that row and
inserts a new one, so ``TeamBridgeClient._pinned_row`` finds nothing and every
bridge call answers "sign in to Radient" to someone who is signed in — and the
receiver logs ``bridge_sign_in_required`` forever, because nothing on the login
path touched the bridge store. The bridge has no token store of its own; only
the id was frozen, so the fix is moving that one id.

WHY RE-POINTING IS SAFE. ``owner_key`` (``store._owner_key``) is a LOCAL cache
key over hub/account/tenant/endpoint with no credential id in it, nothing
server-side keys on ``enrollment_generation``, and no dependent table keys on
``credential_id`` — so the re-point is ONE enrollment row, the same owner. The
identity proof is the tunnel's, shared
(:func:`~local_operator.providers.account_successor.same_account_successor`):
the pin is gone or dead, and exactly one live Radient OAuth row carries the
enrollment's COMMITTED ``account_id`` on its committed hub. A different
account, two candidates, or a live pin answers "nothing to do" and touches
nothing — consent and the designated session are carried, never re-granted.

TWO TRIGGERS, ONE FUNCTION. A verified Radient login (``auth_store``'s
post-write hook, off the event loop) and the receiver's identity-refusal arm
(``connector.loop.run_forever``) — the second is what heals an install that
was ALREADY broken when it updated and never signs in again. It runs only on a
``bridge_sign_in_required`` refusal, so a healthy receiver never reaches it,
and once healed the pin is live and the proof declines by construction.
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import closing
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The ``credential_heals.trigger`` vocabulary.
TRIGGER_LOGIN = "login"
TRIGGER_RECEIVER_START = "receiver_start"


def _open_auth_store(config_dir: Path) -> Any:
    """The bridge's own credential root — the spelling ``TeamBridgeClient`` reads.

    A PLAIN store at ``<config_dir>/auth.db`` (never the mesh-aware build): the
    successor must be a row THIS device holds, never a bearer lent by a peer.
    """

    from local_operator.providers.auth_store import AuthStore

    return AuthStore(db_path=config_dir / "auth.db", config_dir=config_dir)


def _successor(owner: Any, auth_store: Any, fresh_id: int | None) -> int | None:
    from local_operator.providers.account_successor import same_account_successor

    return same_account_successor(
        owner.credential_id,
        store=auth_store,
        hub=owner.canonical_base_url,
        # The COMMITTED account is the authority the re-point must preserve.
        owner_account=owner.account_id,
        fresh_id=fresh_id,
    )


def heal_credential_pin(
    *,
    config_dir: Path,
    trigger: str,
    auth_store: Any = None,
    fresh_id: int | None = None,
) -> str:
    """Re-point the committed enrollment at its provable same-account successor.

    Returns one log sentence when it re-pointed, ``""`` otherwise. NEVER raises:
    both callers are someone else's success path (a login that already
    committed, a receiver loop that must keep idling). A decline writes
    nothing; a re-point is one ``commit_enrollment`` transaction that re-checks
    the read under its write lock (``expected_credential_id``) and records the
    ``credential_heals`` receipt. ``fresh_id`` (login path) requires the one
    successor to be the row that login just wrote.
    """

    try:
        from local_operator.team_bridge.store import OwnerReceipt, Store, state_db_path

        if not state_db_path(config_dir).exists():
            return ""  # never create a bridge store on someone's login
        store = Store(config_dir)
        owner = store.active_owner()
        if owner is None:
            return ""
        if auth_store is not None:
            successor = _successor(owner, auth_store, fresh_id)
        else:
            with closing(_open_auth_store(config_dir)) as opened:
                successor = _successor(owner, opened, fresh_id)
        if successor is None:
            return ""
        receipt = store.owner_consent_receipt(owner)
        if receipt is None:
            return ""  # no committed consent to carry: re-pointing would invent one
        healed = dataclasses.replace(
            owner,
            credential_id=successor,
            enrollment_generation=owner.enrollment_generation + 1,
        )
        store.commit_enrollment(
            healed,
            OwnerReceipt(**receipt),
            cos_session_id=store.designated_session_for(owner),
            expected_credential_id=owner.credential_id,
            heal_trigger=trigger,
        )
    except Exception as exc:  # noqa: BLE001 — a heal must never fail its caller
        logger.warning("team bridge credential heal skipped: code=%s", type(exc).__name__)
        return ""
    return (
        f"team bridge enrollment re-pointed from credential {owner.credential_id} "
        f"to {successor} (same account; trigger={trigger}, "
        f"generation {healed.enrollment_generation})"
    )


def pin_gone_with_login_present(auth_store: Any, credential_id: int) -> bool:
    """Is the pinned row GONE while a usable Radient login sits on this device?

    The refusal surfaces' discriminator: "sign in to Radient" is false in that
    state, so they say what is actually wrong. Never raises (``False``).
    """

    try:
        if auth_store.get_credential(credential_id) is not None:
            return False
        return any(
            row.credential_type == "oauth" and not auth_store.grant_is_dead(row.id)
            for row in auth_store.list_credentials("radient")
        )
    except Exception:  # noqa: BLE001 — a wording choice must not fail a refusal
        return False
