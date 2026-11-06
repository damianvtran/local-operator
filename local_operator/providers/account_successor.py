"""Prove that a live Radient login is the SAME account as a pin that no longer serves.

WHY THIS IS A SHARED MODULE. Two owners pin an exact credential row id: the
tunnel (``tunnel/config.json``) and the Team Bridge enrollment
(``team_bridge/state.sqlite3``). A sign-in to the same Radient account can land
on a NEW row (a logout deletes the old one; a legacy-keyed row is not the row a
fresh sign-in writes), and both owners then refuse with "sign in" to someone
who is signed in. The re-point each needs is the same local proof, so it lives
here once — the tunnel keeps a thin wrapper (``tunnels.api``) that supplies its
ambient store and hub; the bridge supplies its own config root's store and its
enrollment's committed hub. Two copies of an identity proof drift, and a drift
here is a message sent as the wrong account.

Store reads only, no network, never a write: the callers run it on a
credential-write hook and on a receiver's refusal path, both cheap by contract.
"""

from __future__ import annotations

from typing import Any


def _hub(endpoint: object) -> str | None:
    """A row's stamped hub, trailing slash folded; ``None`` when it carries none."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        return None
    return endpoint.strip().rstrip("/")


def _account(data: object) -> str | None:
    account = data.get("account_id") if isinstance(data, dict) else None
    return account if isinstance(account, str) and account.strip() else None


def same_account_successor(
    pin: object,
    *,
    store: Any,
    hub: str,
    owner_account: object = None,
    fresh_id: int | None = None,
) -> int | None:
    """The live row that should own ``pin``'s owner instead of a dead ``pin``, if PROVABLE.

    WHY THIS EXISTS (the 2026-11-06 incident). A sign-in writes the row its
    identity key names, and a row written before sign-ins stamped
    ``api_endpoint`` carries a LEGACY key — so a fresh sign-in to the very same
    Radient account lands on a NEW row while ``tunnel/config.json`` keeps
    pinning the old one, whose grant the IdP has refused. Every surface then
    reports ``login_required`` and offers a sign-in, which writes the new row
    again and changes nothing: the loop issue #1711 describes for a pin that is
    GONE, here for a pin that is PRESENT BUT DEAD.

    WHY RE-POINTING IS SAFE HERE WITHOUT THE CLOUD READ ``report.login_verdict``
    says a general re-point needs: the tunnel is owned by a Radient ACCOUNT, and
    every condition below is local proof that the successor is that same
    account on that same hub — so the re-point changes which row carries the
    bearer, never who owns the tunnel. Anything short of proof answers ``None``
    and the surfaces keep their honest verdict.

    The proof, all of it required:

    * the account: the pinned row's own ``account_id`` when it has one, else
      the cached record's ``owner_account_id`` (``owner_account``, written at
      the last create/connect/configure) — for a gone pin AND for a
      pre-identity pin row that never carried an account (agent review round
      1, m-4): the cached owner is the tunnel's owner itself, which is the
      fact the re-point has to preserve. When both exist they must agree;
      when neither does there is nothing to compare and the answer is ``None``.
    * the pin is NOT USABLE: gone, or present with a dead-grant tombstone or a
      disabled cause. A live pin is the operator's choice (two logins to one
      account are a legitimate setup) and is never second-guessed.
    * the pin's hub, when it carries one, is the tunnel's hub — a missing
      endpoint is the legacy shape, which ``_radient_account_identity``
      deliberately does not invent, and the owner itself only ever spoke to
      ``hub``.
    * exactly ONE enabled, non-dead Radient oauth row other than the pin carries
      that account on ``hub``. Two would be a choice this function
      must not make — and that holds on the login path too (agent review round
      1, m-2): ``fresh_id`` is a MEMBERSHIP check (the one candidate must be
      the row the login just wrote), never a filter that would let a fresh
      sign-in win over another live same-account row the status and configure
      paths would call ambiguous. One rule, so the heal never re-points a
      store those surfaces would refuse to choose in.

    Store reads only — no network — because the credential-write hook calls it
    and that hook is cheap by construction (see ``install.rearm_if_parked``).
    ``store`` is an open ``AuthStore`` the caller owns (and closes); ``hub``
    is the owner's hub in a row's ``api_endpoint`` spelling (folded like one).
    """
    if not isinstance(pin, int) or isinstance(pin, bool):
        return None
    wanted_hub = _hub(hub)
    if wanted_hub is None:
        return None
    owner = owner_account if isinstance(owner_account, str) and owner_account.strip() else None
    row = store.get_credential(pin)
    if row is not None and row.provider == "radient" and row.credential_type == "oauth":
        account = _account(row.data) or owner
        if account is None or (owner is not None and owner != account):
            return None
        if row.disabled_cause is None and not store.grant_is_dead(pin):
            return None
        if _hub(row.data.get("api_endpoint")) not in (None, wanted_hub):
            return None
    else:
        account = owner
        if account is None:
            return None
    candidates = [
        other.id
        for other in store.list_credentials("radient")
        if other.id != pin
        and other.credential_type == "oauth"
        and _account(other.data) == account
        and _hub(other.data.get("api_endpoint")) == wanted_hub
        and not store.grant_is_dead(other.id)
    ]
    if len(candidates) != 1 or (fresh_id is not None and candidates[0] != fresh_id):
        return None
    return candidates[0]
