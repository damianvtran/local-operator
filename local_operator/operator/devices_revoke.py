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
import logging
import re
from pathlib import Path
from typing import Any, TextIO

from local_operator.operator import devices

logger = logging.getLogger(__name__)

#: A device id this module will act on: ASCII letters and digits, at most 64 (round-1
#: R6). ``str.isalnum`` also accepted non-ASCII digits and letters ("١٢٣"), which no
#: id this computer derives (``devices.new_device_id``: lowercase hex) can contain.
_DEVICE_ID = re.compile(r"[A-Za-z0-9]{1,64}", re.ASCII)


class UnknownDevice(LookupError):
    """``revoke`` was asked to remove an id this computer never issued or listed."""


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
    from local_operator.operator.pair_op import display_name

    rows: dict[str, dict[str, Any]] = {}
    for stored in devices.list_devices(root):
        rows[stored.device_id] = {
            "device_id": stored.device_id,
            # The phone chose this name: sanitised like the pairing card's (round-1
            # R2), so a bidi override cannot reorder the Paired-phones row either.
            "name": display_name(stored.name, limit=64),
            "platform": stored.platform or _platform(linked.get(stored.device_id)),
            "paired_at": stored.issued_at,
        }
    for device_id in devices.revoked_ids(root):
        if device_id in rows:
            continue
        # The certificate is gone (phase 1 deletes it); the revocation record kept its
        # display facts so a ``removing`` row can still say WHICH phone (round-1 Q4).
        kept = devices.revoked_phone(root, device_id)
        paired_at = kept.get("paired_at")
        rows[device_id] = {
            "device_id": device_id,
            "name": display_name(kept.get("name"), limit=64),
            "platform": str(kept.get("platform") or "") or _platform(linked.get(device_id)),
            "paired_at": int(paired_at) if isinstance(paired_at, int) and paired_at else None,
        }
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


def is_known(root: Path, device_id: str) -> bool:
    """Whether ``device_id`` is a phone this computer issued or lists anywhere.

    The five places a phone can be named (round-1 R6/Q3): a stored certificate, a
    pending pairing request, the revoked list (the local record or the anchor's), any
    entry of the installed anchor, or a push row linked to it. An id in none of them
    was never this computer's, so there is nothing to remove -- and "removing" it
    anyway raised an admin prompt and wrote a permanent revocation into the root-owned
    anchor for a phantom device.
    """
    from local_operator.mobile import push_devices

    if devices.read_device(root, device_id) is not None:
        return True
    if any(row.get("device_id") == device_id for row in devices.list_pending(root)):
        return True
    if device_id in devices.revoked_ids(root):
        return True
    from local_operator.operator.trust import load_anchor

    loaded = load_anchor()
    if loaded.usable and loaded.anchor is not None:
        if any(entry.get("device_id") == device_id for entry in loaded.anchor.devices):
            return True
    try:
        rows = push_devices.list_devices(root)["devices"]
    except Exception:  # noqa: BLE001 - an unreadable push store names no phone
        rows = []
    return any(row.get("operator_device_id") == device_id for row in rows)


def revoke(
    root: Path,
    device_id: str,
    *,
    elevate: bool = True,
    runner: Any = None,
    out: TextIO | None = None,
    require_known: bool = False,
) -> dict[str, Any]:
    """Remove ``device_id`` in the three phases above; idempotent; never raises on a prompt.

    Returns ``{device_id, state, push_revoked, anchor: done|cancelled|refused|absent,
    detail}``. ``elevate=False`` runs phases 1-2 only (a caller that cannot show a
    prompt); ``runner`` is the gesture seam the E2E uses (``handlers.admit_device``'s
    own), so CI proves the order without a real sudo.

    ``out`` receives the anchor install's own CLI sentences: the CLI verb passes its
    stderr so they reach the terminal as they happen. A server caller passes nothing;
    the sentences are then written to a stream LOCAL to this call and logged, never
    into the response -- ``detail`` is keyed copy (round-1 Q6), and the process-wide
    ``sys.stdout``/``sys.stderr`` are never swapped from a worker thread (round-1 R3:
    that captured every other thread's output for the length of the OS prompt).

    Raises ``ValueError`` for a malformed id, and -- with ``require_known`` -- raises
    :class:`UnknownDevice` for one this computer never issued or listed
    (:func:`is_known`); both before anything is written or any prompt is raised. The
    desktop's Remove button passes ``require_known`` (it only ever names a listed row).
    The CLI verb does not: an operator revoking a lost phone on a computer that joined
    BEFORE the phone was paired must be able to name an id this computer never saw.
    """
    from local_operator.mobile import push_devices
    from local_operator.operator.handlers import (
        ADMIT_OK,
        _admission_install,
        _installed_admission_anchor,
    )
    from local_operator.operator.pair_handlers import _stage_anchor_revocation

    if not isinstance(device_id, str) or _DEVICE_ID.fullmatch(device_id) is None:
        raise ValueError("bad device id")
    if require_known and not is_known(root, device_id):
        raise UnknownDevice(device_id)
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

            sink = out if out is not None else io.StringIO()
            code = _admission_install(
                root,
                current[1],
                print_only=False,
                sudo_secret="",
                runner=runner,
                purpose=PURPOSE_REMOVE,
                out=sink,
            )
            if out is None and isinstance(sink, io.StringIO) and sink.getvalue().strip():
                # Developer log only: these are the CLI's unkeyed sentences.
                logger.info("phone removal install: %s", sink.getvalue().strip()[:2000])
            anchor = "done" if code == ADMIT_OK else "cancelled"
    elif anchored is False:
        anchor = "cancelled"
    state = device_state(root, device_id)
    if state != STATE_REVOKED:
        from local_operator.operator.pair_op import _copy

        detail = _copy("revoke_unfinished")
    return {
        "device_id": device_id,
        "state": state,
        "push_revoked": push_revoked,
        "anchor": anchor,
        "detail": detail,
    }


__all__ = [
    "STATE_PAIRED",
    "STATE_REMOVING",
    "STATE_REVOKED",
    "STATE_UNCONFIRMED",
    "UnknownDevice",
    "device_state",
    "is_known",
    "list_phones",
    "revoke",
]
