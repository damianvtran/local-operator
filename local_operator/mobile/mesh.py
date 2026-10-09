"""The relay's mesh seam: the peer listing read, and the transfer verb.

WHY THIS MODULE EXISTS. "Sessions and delegation" parity: the daemon must be
able to show the sessions OTHER devices hold (``include_peers`` on
``/api/sessions``) and move a conversation to another device
(``POST /api/sessions/{id}/transfer``). Neither existed under
``local_operator/mobile/`` (verified at ``origin/main`` @
924ccd3a7b7c06ae7ee03dfcb1d9995acb0b32bb: no transfer route among the session
routes, and no ``include_peers`` anywhere in this package), while the desktop
plane has both. This module is the relay half of that work and nothing else.

HOW THIS PROCESS REACHES THE MESH — the structural answer, recorded here
because every future reader asks it: it does NOT hold peer links of its own and
must not grow one. The one process that speaks the mesh is this device's
network RELAY (``network/relay.py``), which publishes a ``0600`` record under
``run/peers/<pid>.json`` and serves a loopback, key-authenticated control
socket; any same-account process reaches the mesh through it — the mechanism
the federated listing (``session.peer_rows``), the move verb
(``network.mobility.request_move``) and every ``lop network`` verb already use.
The desktop plane's ``server/utils/desktop_mesh.py`` is its desktop-shaped
caller. This module is the same seam for the phone plane, and deliberately
composes no control frame of its own: it calls those two entry points — whose
docstrings own the envelope rules and the budgets — and shapes what they
answer for the phone's wire.

The two halves, and the shapes they mirror:

* :func:`remote_session_rows` mirrors ``server.utils.desktop_mesh``'s function
  of the same name: the desktop's FLAT locality fields are the contract
  (Addendum 2 B), and the nested transport ``peer`` block is deliberately NOT
  published. Fields the desktop publishes as a null (``placement``/``origin``/
  ``last_synced_at`` — the federated row this reads carries no owner stamp)
  stay null here for the same reason: a null is no claim, and the peer's home
  device guessed would be a wrong one.
* :func:`request_transfer` hands the move to ``server.utils.desktop_mesh``'s
  own ``transfer``/``transfer_receipt`` pair — the very functions the desktop
  route calls — so the receipt a phone renders (phases, progress, mode,
  ``source_retired``) cannot drift from the one the desktop renders. A second
  caller composing its own ``session_move`` frame would be a second
  implementation of the relay's most destructive local verb.

ZERO-PEER COST IS INHERITED, NOT RE-IMPLEMENTED. ``peer_session_rows``
short-circuits on a device with no relay record before opening any socket, and
``mobility.request_move`` answers a named ``relay_unavailable`` refusal rather
than a silent no-op — so a machine in no mesh answers the read having opened
nothing, which is the property every existing install depends on.

NOTHING HERE RAISES a mesh refusal: the read answers ``[]`` for every reason
there is nothing to show (``peer_session_rows``' own contract), and the write
answers the move's own refusal shapes. The one exception is a caller error
(``parse_transfer_body`` returns a sentence the route refuses with 422), and
the journal's own store failures (``transfer_receipts``), which the route
renders as named refusals rather than smoothing them into "nothing happened".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Collection

#: A mesh device id (``d_`` + hex today) or ``"local"``. PATH-SAFE rather than
#: exact, mirroring ``server/models/desktop_mesh.MESH_ID_PATTERN`` (the
#: renderer's own rule): what matters is that neither can carry ``/``, ``.``
#: or ``%``, because the value reaches the relay's move op.
MESH_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

#: The desktop's ``RequestID`` shape, mirrored: a UUID, because a phone with a
#: retry envelope already mints one per command and a looser id would let two
#: intents share a replay slot.
REQUEST_ID_PATTERN = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")

#: Refusals that did NOT leave a move this device can call settled, so a retry
#: of the same request id must REPLAY the recorded "unconfirmed" answer rather
#: than start a second move for a request whose first attempt may still be
#: running. Everything else is released: a refusal that leaves nothing behind
#: must not answer from a refusal forever (the user frees the session up and
#: presses again).
#:
#: THE SET IS THE DESKTOP ROUTE'S — ``server/routes/desktop_mesh.py``
#: ``_MOVE_UNCONFIRMED_CODES``, from which the desktop's receipt journal takes
#: the same record/release decision. A test pins the two sets equal
#: (``test_mesh_relay``) so a change there cannot silently diverge this plane.
MOVE_UNCONFIRMED_CODES = frozenset({"relay_unavailable", "deadline_exceeded", "peer_unreachable"})

#: The body keys the transfer route accepts, mirroring the desktop's
#: ``TransferSession`` (which forbids extras): a misspelt key on the plane's
#: one destructive mesh route would be a silently dropped intent — ``keeep:
#: true`` must not move a conversation the user asked to copy.
_TRANSFER_KEYS = frozenset({"to", "keep", "wait_s", "request_id"})


def parse_transfer_body(body: object) -> tuple[dict[str, Any] | None, str]:
    """``(fields, "")`` or ``(None, the sentence to refuse with)``.

    The field rules are ``server/models/desktop_mesh.TransferSession``'s,
    restated for a plain-dict JSON boundary (this plane has no pydantic
    models): ``to`` required and path-safe, ``keep`` a strict boolean, ``wait_s``
    a number inside 0..300, ``request_id`` an optional UUID, and NO extra keys.
    Strict rather than coercing, for the desktop rule's reason: a truthy
    integer is not a state the client meant, and on this route an
    under-validated field is a conversation moved when a copy was asked for.
    """
    if not isinstance(body, dict):
        return None, "a JSON object with 'to' is required"
    unknown = sorted(str(key) for key in body if key not in _TRANSFER_KEYS)
    if unknown:
        return None, (
            "unknown field"
            + ("" if len(unknown) == 1 else "s")
            + " "
            + ", ".join(repr(key) for key in unknown)
            + " — this route takes to, keep, wait_s, request_id"
        )
    to = body.get("to")
    if not isinstance(to, str) or not MESH_ID_PATTERN.match(to):
        return None, "'to' must name a device id or 'local'"
    keep = body.get("keep", False)
    if not isinstance(keep, bool):
        return None, "'keep' must be true or false"
    wait_s = body.get("wait_s", 0.0)
    # ``bool`` is excluded explicitly: ``isinstance(True, int)`` is True, and
    # ``wait_s: true`` is not a duration anybody meant.
    if isinstance(wait_s, bool) or not isinstance(wait_s, (int, float)):
        return None, "'wait_s' must be a number of seconds between 0 and 300"
    if not 0.0 <= float(wait_s) <= 300.0:
        return None, "'wait_s' must be between 0 and 300 seconds"
    request_id = body.get("request_id")
    if request_id is not None:
        if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.match(request_id):
            return None, "'request_id' must be a UUID"
    return {
        "to": to,
        "keep": keep,
        "wait_s": float(wait_s),
        "request_id": request_id if isinstance(request_id, str) else None,
    }, ""


def remote_session_rows(
    config_dir: Path | None = None,
    *,
    pinned: Collection[str],
    catalog: object | None = None,
) -> list[dict[str, Any]]:
    """Every session another device holds, as rows the PHONE's list paints.

    THE MIRROR IS ``server.utils.desktop_mesh.remote_session_rows``: same flat
    locality fields, same names, same meanings, and the same refusal to publish
    the nested transport ``peer`` block — a client that grouped by it would
    file every remote row under one heading (Addendum 2 B). The row is a phone
    summary where the phone has its own vocabulary (``session_id``,
    ``section``, ``pinned``, ``conversation_name``, ``mtime``, ``created_at``)
    and carries the transport's ``live_state``/``pending`` pair verbatim — the
    four words and the gate kind the desktop's ``status`` and the catalogue's
    rank already read — never a third spelling of "what is this session doing".

    BINNING IS THE SHARED RULE, NOT A REMOTE AXIS. ``section`` comes from
    ``session.catalog.entry_for``'s ``active`` — pending, unseen or live is
    ACTIVE — the SHARED CONVENTION ``tui/session_sidebar._unpinned_rank``
    states for remote rows: they file into the same bins as local rows, one
    list. ``created_at`` is the peer's ``started`` claim (the only per-row time
    the federated listing carries); a claim that is not a number has already
    been read as the no-claim ``0.0`` inside ``session.peer_rows``
    (``_started_epoch``), so an old-build peer's non-number birth stamp falls
    through the existing bins and sorts last — never a crash, and no
    special-casing of a peer here.

    ONE PROJECTION, ONE STALENESS RULE. The rows come from
    ``session.peer_rows.peer_session_rows`` — the TTL-cached federated
    projection the sidebar's peer heading, the desktop list and the ``/resume``
    guard all read — never from a second dial per call. That read is TTL-cached
    and short-circuits with no relay record, and it is BLOCKING (a loopback
    fan-out), so the daemon hands it to a worker thread; a caller on the event
    loop must do the same.

    ``pinned`` is this device's pin set (the ONE shared pin store, read once
    per request by the caller), exactly as the desktop's remote rows mark from
    its own pin index. ``catalog`` is the injection seam ``peer_session_rows``
    already takes, passed through for tests and future transports; production
    passes nothing and gets this device's own relay.

    Returned rows are sorted by the shared rank so their relative order is the
    same deterministic key the local rows use; the rows themselves are appended
    BELOW the local page by the caller, mirroring the desktop's "page first,
    then the extras" order.
    """
    from local_operator.resume import peer_reason_words
    from local_operator.session.catalog import entry_for
    from local_operator.session.peer_rows import peer_session_rows

    ranked: list[tuple[tuple[int, int, float, str], dict[str, Any]]] = []
    for row in peer_session_rows(config_dir, catalog=catalog):
        entry = entry_for(row, None)
        reachable = bool(getattr(row, "reachable", True))
        ranked.append(
            (
                entry.rank,
                {
                    "session_id": str(getattr(row, "id", "") or ""),
                    # The phone's ordinary bin for this row — the shared
                    # ``active`` rule, so a remote row and a local row cannot
                    # answer "which list is this in" differently.
                    "section": "active" if entry.active else "previous",
                    "pinned": str(getattr(row, "id", "") or "") in pinned,
                    "conversation_name": str(getattr(row, "name", "") or ""),
                    "mtime": float(getattr(row, "mtime", 0.0) or 0.0),
                    "created_at": float(getattr(row, "created_at", 0.0) or 0.0),
                    # The transport's own pair, verbatim (no flattening into a
                    # third vocabulary): ``live_state`` is one of the four
                    # words the rank reads, ``pending`` the gate kind
                    # (``"approval"``/``"answer"``) or None.
                    "live_state": str(getattr(row, "live_state", "") or ""),
                    "pending": getattr(row, "pending", None) or None,
                    # -- the desktop's flat locality fields, same names, same
                    # meanings; present with values on every remote row so a
                    # client merge can settle them ("an absent key is not a
                    # claim").
                    "locality": "remote",
                    "owner_device": str(getattr(row, "owner_device", "") or ""),
                    "owner_device_name": str(getattr(row, "owner_device_name", "") or ""),
                    "reachable": reachable,
                    # Glossed at this boundary, like the desktop's
                    # (``resume.peer_reason_words``): the relay's protocol token
                    # stays in ``lop network peers --json``.
                    "unreachable_reason": (
                        ""
                        if reachable
                        else peer_reason_words(str(getattr(row, "unreachable_reason", "") or ""))
                    ),
                    # Nulls, not guesses: the federated row carries no owner's
                    # stamp, and the desktop publishes the same nulls for the
                    # same reason.
                    "placement": None,
                    "origin": None,
                    "last_synced_at": None,
                },
            )
        )
    ranked.sort(key=lambda item: item[0])
    return [row for _rank, row in ranked]


def request_transfer(
    config_dir: Path | None,
    session_id: str,
    *,
    to: str,
    keep: bool = False,
    wait_s: float = 0.0,
) -> dict[str, Any]:
    """Move (or ``keep``-copy) one conversation; the receipt, or a refusal.

    THE CALL IS THE DESKTOP PLANE'S OWN — ``server.utils.desktop_mesh``'s
    ``transfer`` and ``transfer_receipt``, the two functions its route calls —
    so the move's action choice (recall/offload), its budgets, its phase
    transcript and the shape a renderer consumes are ONE implementation across
    the two planes rather than a copy free to drift. This module contributes
    exactly two things: the ``refused`` marker the route's journal branches on,
    and ``replayed: false`` folded into every first-serve receipt (the
    desktop's response model publishes the same default; the journal replaces
    it with True on a replay).

    BLOCKING, and its caller must run it off the event loop: the relay holds
    this call for the whole move — a retire, a copy and a confirmation — which
    is seconds, and up to ``wait_s`` longer when the source is busy. The bound
    is ``mobility.request_move``'s own per-shape client deadline, deliberately
    not a second timeout here (a caller that gives up first reports its own
    timeout for a move the relay was about to answer).

    A refusal is returned, never raised, in the move's own vocabulary: the
    ``code`` and the sentence compose once, on the relay that can see which
    guard fired, and both travel out verbatim for the phone to render.
    """
    from local_operator.server.utils.desktop_mesh import transfer, transfer_receipt

    result = transfer(
        session_id,
        to=to,
        keep=bool(keep),
        wait_s=float(wait_s or 0.0),
        root=config_dir,
    )
    if result.get("ok"):
        receipt = transfer_receipt(result, session_id=session_id, keep=bool(keep), to=to)
        receipt["replayed"] = False
        return receipt
    return {
        "refused": True,
        "code": str(result.get("code") or "move_refused"),
        "message": str(result.get("message") or "the move was refused"),
        "changed": bool(result.get("changed")),
    }
