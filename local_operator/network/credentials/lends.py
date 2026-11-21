"""The owner's ids-only record of what it LENT, so a ``member rm`` can say what is left.

WHY THIS EXISTS (G9). A member removal ends what it can at the source: the owner
DELETEs lent GitHub App installation tokens (#3947 M2) and ends renewal lends
(#3955 Q1). Everything else it lent — model-provider OAuth bearers, MCP bearers,
GitHub PAT/``gh`` tokens — has no issuer-side revoke the owner can call. A
member that is REACHABLE at removal drops those itself (the epoch frame tells it
it was removed, and its client denies the network and drops every borrowed
grant). A member that is NOT reachable — the lost or stolen laptop, the very
reason one is removed — never learns: the owner refuses its later handshakes
silently by design, so it reads the owner as asleep and keeps serving those
bearers from its offline hold, up to ``offline_hold_s`` (24 h by default). The
removal receipt said nothing about any of this; it reported only copies.

This module is the minimum the owner needs to say it honestly: one row per
(holder, credential key) with the LATEST lease id and the latest moment that
lease may still be served on the holder (``max(grant end, hold end)``), plus
whether the issuer can revoke it. That is all — NO token, NO account label, NO
session id. A row is never a secret, so a stolen copy of this file reveals only
"device X borrowed key K until T", which the owner's audit log already records.

CONSTRAINTS.
- Device-local and never copied: it describes THIS owner's lends; a peer must
  never receive it (declared in ``test_sync_copy_set``'s path table).
- Never raises into the grant or removal paths: the grant was already served, the
  removal already committed. A row that cannot be written makes the receipt
  understate, so a write failure is logged at WARNING rather than swallowed.
- Single writer: only the owner relay mints grants (``owner.grant``), and the CLI's
  local ``member rm`` spelling runs only when NO relay is running. A process lock
  serialises the relay's own threads; the network store's writer replaces the
  file atomically (0600) so a reader never sees a torn document.
- Rows past their end are pruned on every write: a lease that can no longer be
  served is not something a receipt needs to name.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ``<config>/network/credentials/<network_id>/lends.json`` — beside placement.json.
LENDS_FILENAME = "lends.json"

_LOCK = threading.Lock()


def lends_path(network_id: str, root: Path | None = None) -> Path:
    """The ledger's path. NEVER CREATES anything (placement's discipline)."""
    from local_operator.network.credentials import placement as placement_mod

    return placement_mod.credentials_root(root) / network_id / LENDS_FILENAME


def _load(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.warning("lend ledger %s is unreadable; starting it again", path, exc_info=True)
        return []
    rows = document.get("lends") if isinstance(document, dict) else None
    return [dict(row) for row in rows or [] if isinstance(row, dict)]


def _save(path: Path, rows: list[dict[str, Any]]) -> None:
    # The network store's own staged 0600 write-then-rename (the writer the
    # sibling MCP evidence uses), not a second atomic writer.
    from local_operator.network import store

    path.parent.mkdir(parents=True, exist_ok=True)
    store._write_private_json(path, {"version": 1, "lends": rows})  # noqa: SLF001


def _live(rows: list[dict[str, Any]], now_ms: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("until_ms") or 0) > now_ms]


def note_lend(
    root: Path | None,
    network_id: str,
    *,
    holder: str,
    key: str,
    grant_id: str,
    until_ms: int,
    revocable: bool,
    now_ms: int | None = None,
) -> None:
    """Record that ``holder`` may serve ``key`` until ``until_ms``. Never raises.

    ``until_ms`` is the latest moment the lease may be served on the holder:
    ``max(grant_expires_at_ms, hold_until_ms)`` — the hold, when the owner stated
    one, outlives the grant window, and it is the hold a lost member keeps using.
    ``revocable`` marks a lend the removal path revokes at the issuer (a GitHub
    App installation token); those are counted as revoked, never as left usable.
    """
    if not (network_id and holder and key and until_ms):
        return
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        path = lends_path(network_id, root)
        with _LOCK:
            rows = _live(_load(path), now)
            kept = [r for r in rows if (r.get("holder"), r.get("key")) != (holder, key)]
            previous = next(
                (r for r in rows if (r.get("holder"), r.get("key")) == (holder, key)), None
            )
            until = max(int(until_ms), int((previous or {}).get("until_ms") or 0))
            # A key keeps ONE row per holder: the receipt names the credential and
            # the LAST moment any of its leases can be served, not every session's.
            # Revocable only while EVERY live lease of it is: one PAT lease beside an
            # App lease leaves something no DELETE reaches.
            still_revocable = bool(revocable) and bool(
                previous is None or previous.get("revocable")
            )
            kept.append(
                {
                    "holder": holder,
                    "key": key,
                    "grant_id": grant_id,
                    "until_ms": until,
                    "revocable": still_revocable,
                }
            )
            _save(path, kept)
    except Exception:  # noqa: BLE001 — the grant was already served
        logger.warning("could not record a lend in the owner's ledger", exc_info=True)


def close_holder(
    root: Path | None, network_id: str, holder: str, *, now_ms: int | None = None
) -> list[dict[str, Any]]:
    """Remove ``holder``'s rows and return those still servable at ``now_ms``. Never raises.

    The caller is a member removal: the rows leave the ledger because nothing will
    ever be lent to that device again, and the returned rows are what the receipt
    must name. Other holders' rows are untouched.
    """
    if not (network_id and holder):
        return []
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        path = lends_path(network_id, root)
        with _LOCK:
            rows = _live(_load(path), now)
            mine = [r for r in rows if r.get("holder") == holder]
            if mine or path.exists():
                _save(path, [r for r in rows if r.get("holder") != holder])
    except Exception:  # noqa: BLE001 — the removal already committed
        logger.warning("could not close a removed member's lends", exc_info=True)
        return []
    return sorted(mine, key=lambda r: (str(r.get("key")), int(r.get("until_ms") or 0)))


def removal_lends(
    root: Path | None,
    network_id: str,
    device_id: str,
    *,
    told: bool,
    revokes_at_issuer: bool = True,
    audit: Any = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """The lend half of a ``member rm`` receipt. Never raises.

    - ``lends_revoked``: lends ended at the source — App installation tokens (the
      relay's removal revoke DELETEs them at GitHub right after this) plus renewal
      lends ended here (``renewal.end_lends``; the revoke thread's own call then
      finds none, so nothing is ended or audited twice).
    - ``lends_open``: when the member was NOT told over a live link, each lend no
      issuer can revoke, with the moment it stops being servable on that device —
      ids and times only. Its remedy is the copies' remedy: rotate at the source.
    - ``lends_told``: when the member WAS told, how many such lends its client was
      told to drop. "Told", not "confirmed": the frame was accepted by a live link,
      and the client's drop is the member's own act.

    ``revokes_at_issuer=False`` is the CLI's relay-less spelling: no broker runs
    there, so nothing DELETEs an App token and it is reported as left usable like
    any other lend rather than counted as revoked.
    """
    from local_operator.network.credentials import renewal as renewal_mod

    rows = close_holder(root, network_id, device_id, now_ms=now_ms)
    try:
        renewal_ended = renewal_mod.end_lends(
            root, network_id, devices=[device_id], cause="member_removed", audit=audit
        )
    except Exception:  # noqa: BLE001 — the removal already committed
        logger.warning("member removal: ending renewal lends failed", exc_info=True)
        renewal_ended = 0
    revocable = [r for r in rows if revokes_at_issuer and r.get("revocable")]
    unrevocable = [r for r in rows if not (revokes_at_issuer and r.get("revocable"))]
    return {
        "lends_revoked": len(revocable) + int(renewal_ended or 0),
        "lends_open": (
            []
            if told
            else [{"key": str(r["key"]), "until_ms": int(r["until_ms"])} for r in unrevocable]
        ),
        "lends_told": len(unrevocable) if told else 0,
    }
