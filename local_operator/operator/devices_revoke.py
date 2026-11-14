"""Remove a paired phone: ONE implementation for the CLI, the desktop and the TUI.

WHY THIS MODULE (pairing design v2 §2.4, D9/U7/U8). Before it, a revoke was two stores
nobody linked: ``lop operator devices --revoke`` wrote ``revoked.json`` and the
root-owned anchor, while the phone's push row (``mobile/push_devices``) kept receiving
approval notifications under a machine-minted id with no tie to the operator device.
And the only way to run it was a terminal. This is that CLI body, extracted, with the
push half added and the phases in the order that makes a half-finished removal SAFE:

1. ``devices.record_revocation`` -- unprivileged, immediate: the relay refuses the
   phone's certificate from the next request (``operator_device_revoked``);
2. ``push_devices.revoke_linked`` -- every push row PROVEN to belong to this phone
   (design v2's push link) is tombstoned, so notifications stop now too;
3. the anchor's revocation entry, staged and installed through the OS admin prompt
   (``run_elevated`` with purpose ``remove``) -- the runtime's own list, and the one
   that travels to computers that join later.

Phases 1-2 never need a person; phase 3 always does (it is a root-owned write). A
cancelled or refused prompt therefore leaves the phone ``removing``: it can no longer
approve here or get notifications, and "Finish removing" re-runs this function, which
skips what is already done (idempotent).

``removing`` IS DERIVED, NOT STORED (U8): the id is in ``revoked_ids`` while the
installed anchor does not yet revoke it, or a linked push row is still live. Nothing
new is written to say so, so the state survives a refresh and a restart and cannot
disagree with the facts it summarises.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from local_operator.operator import devices

#: The desktop row states (design v2 §2: ``GET /v1/desktop/devices``). Closed, because
#: the Paired-phones list renders copy per value.
STATE_PAIRED = "paired"
STATE_UNCONFIRMED = "unconfirmed"
STATE_REMOVING = "removing"
STATE_REVOKED = "revoked"

#: ``revoke``'s outcomes.
OUTCOME_REVOKED = "revoked"
OUTCOME_REMOVING = "removing"


def _anchor_revokes(device_id: str) -> bool | None:
    """Whether the INSTALLED anchor revokes ``device_id``; ``None`` with no usable anchor."""
    from local_operator.operator.trust import device_is_revoked, load_anchor

    loaded = load_anchor()
    if not loaded.usable or loaded.anchor is None:
        return None
    return device_is_revoked(loaded.anchor, device_id)


def device_state(root: Path, device_id: str) -> str:
    """One phone's row state, from the facts alone (never a stored flag)."""
    from local_operator.mobile import push_devices

    if device_id in devices.revoked_ids(root):
        anchored = _anchor_revokes(device_id)
        # No usable anchor means there is no privileged list to finish: the local
        # record is then the whole revocation (the runtime cannot honour the phone
        # without an anchor either).
        if anchored is False or push_devices.live_linked(root, device_id):
            return STATE_REMOVING
        return STATE_REVOKED
    stored = devices.read_device(root, device_id)
    if stored is not None and not devices.is_admitted(stored.certificate):
        return STATE_UNCONFIRMED
    return STATE_PAIRED


def list_phones(root: Path) -> list[dict[str, Any]]:
    """The Paired-phones rows: certificates, revoked ids and push rows, joined by id.

    ``{device_id, name, platform, paired_at, last_seen, push, state}``. A revoked
    phone whose certificate is gone still lists (it may be ``removing``); a push row
    with no operator link is not a paired phone and is not listed here (it stays in
    the push list, removable there).
    """
    from local_operator.mobile import push_devices

    try:
        push_rows = push_devices.list_devices(root)["devices"]
    except Exception:  # noqa: BLE001 - an unreadable push store must not hide phones
        push_rows = []
    linked: dict[str, list[dict[str, Any]]] = {}
    for row in push_rows:
        operator_id = row.get("operator_device_id")
        if isinstance(operator_id, str) and operator_id:
            linked.setdefault(operator_id, []).append(row)
    rows: dict[str, dict[str, Any]] = {}
    for stored in devices.list_devices(root):
        rows[stored.device_id] = {
            "device_id": stored.device_id,
            "name": stored.name,
            "platform": _platform(linked.get(stored.device_id)),
            "paired_at": stored.issued_at,
        }
    for device_id in devices.revoked_ids(root):
        rows.setdefault(
            device_id,
            {
                "device_id": device_id,
                "name": "",
                "platform": _platform(linked.get(device_id)),
                "paired_at": None,
            },
        )
    out: list[dict[str, Any]] = []
    for device_id, row in rows.items():
        push = linked.get(device_id) or []
        last_seen = max((int(p.get("last_seen_at") or 0) for p in push), default=0) or None
        out.append(
            {
                **row,
                "last_seen": last_seen,
                "push": any(p.get("state") != "revoked" for p in push),
                "state": device_state(root, device_id),
            }
        )
    return out


def _platform(push: list[dict[str, Any]] | None) -> str:
    for row in push or []:
        if row.get("platform"):
            return str(row["platform"])
    return ""


def revoke(
    root: Path, device_id: str, *, elevate: bool = True, runner: Any = None
) -> dict[str, Any]:
    """Remove ``device_id`` in the three phases above; idempotent; never raises on a prompt.

    Returns ``{device_id, state, push_revoked, anchor: done|cancelled|refused|absent,
    detail}``. ``elevate=False`` runs phases 1-2 only (a caller that cannot show a
    prompt); ``runner`` is the gesture seam the E2E uses (``handlers.admit_device``'s
    own), so CI proves the order without a real sudo.
    """
    from local_operator.mobile import push_devices
    from local_operator.operator.handlers import (
        ADMIT_OK,
        _admission_install,
        _installed_admission_anchor,
    )
    from local_operator.operator.pair_handlers import _stage_anchor_revocation

    if not device_id or len(device_id) > 64 or not device_id.isalnum():
        raise ValueError("bad device id")
    # Phase 1: the relay stops honouring the phone now.
    if device_id not in devices.revoked_ids(root):
        devices.record_revocation(root, device_id)
    # Phase 2: notifications stop now (before any prompt, D9).
    push_revoked = push_devices.revoke_linked(root, device_id)
    anchor = "absent"
    detail = ""
    anchored = _anchor_revokes(device_id)
    if anchored is True:
        anchor = "done"
    elif anchored is False and elevate:
        current = _installed_admission_anchor()
        if current is None or not _stage_anchor_revocation(root, device_id, revoked=True):
            anchor = "refused"
        else:
            from local_operator.operator.elevate import PURPOSE_REMOVE

            # The install prints its own CLI sentences; a server caller has no terminal,
            # so they are captured and returned as ``detail`` for the log, not shown.
            sink = io.StringIO()
            with redirect_stdout(sink), redirect_stderr(sink):
                code = _admission_install(
                    root,
                    current[1],
                    print_only=False,
                    sudo_secret="",
                    runner=runner,
                    purpose=PURPOSE_REMOVE,
                )
            detail = sink.getvalue().strip()[:2000]
            anchor = "done" if code == ADMIT_OK else "cancelled"
    elif anchored is False:
        anchor = "cancelled"
    return {
        "device_id": device_id,
        "state": device_state(root, device_id),
        "push_revoked": push_revoked,
        "anchor": anchor,
        "detail": detail,
    }


__all__ = [
    "STATE_PAIRED",
    "STATE_REMOVING",
    "STATE_REVOKED",
    "STATE_UNCONFIRMED",
    "device_state",
    "list_phones",
    "revoke",
]
