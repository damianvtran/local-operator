"""Remote peek: ``net_session_peek`` (owner side) and ``peer_session_peek`` (viewer side).

WHY. ``sessions op=peek peer=<device>`` used to ride the pilot viewer: it bound
the session's runtime and read the newest rows of its synced DISPLAY window.
That served a tail of a LIVE session and nothing else — ``head``/``before_id``/
``around_id``/``query``/``regex``/``digest`` were refused, and a stored session
was refused outright, so a coordinator could not read a remote session's
original request or search its history. This op asks the OWNER to run the same
bounded readers a local peek runs (``session/peek.py``) against its own journal
and send back the window as data; the viewer renders it with the same
``_peek_body``, so a remote peek and a local one on the owner answer the same
rows in the same words.

SECURITY SHAPE (no new authority, no new friction):

* Capability ``view`` — the row ``net_session_history`` carries for the same
  reason: a member that may watch a session here may read the bytes its
  journal would serve anyway. Nothing is written on either device.
* The frame names ``session_id``, so the authoriser's session-scope chokepoint
  (``Authorizer._session_scope``) refuses a session this device does not own
  before the handler runs.
* It starts NOTHING and takes no lease — reading a stored session is the point.
* The readers' own budgets bound the work (``PEEK_MAX_STEPS``, the page-walk
  cap, the 16 MiB search depth), and a peer-supplied regex is additionally
  length- and shape-capped (``session.peek.PEER_REGEX_MAX_CHARS``) because it
  runs on this device's CPU and Python's ``re`` cannot be interrupted.
* SLOW lane: a deep search can take seconds, and the link reader must keep
  serving other frames meanwhile.

COMPATIBILITY. Advertised as :data:`wire.SESSION_PEEK_V1`; the viewer's relay
checks it before asking, and a peer without it answers ``unsupported`` with a
reason so the caller can fall back to the live-display tail and SAY why.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.network import wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:
    from local_operator.network.relay import PeerLink, RelayServer

#: Owner-side deadline for one ``net_session_peek``. The ordinary read is one
#: page (milliseconds); the worst case is the needle search crossing its full
#: 16 MiB depth plus the anchored window read, measured in low seconds on an
#: operator-size journal. 20 s leaves a loaded machine room without letting a
#: caller wait on a wedged disk for the engage budget.
SESSION_PEEK_DEADLINE_S = 20.0

#: Encoded ceiling for one reply. Steps are already clipped by the renderer
#: (``comms.PEEK_STEP_CHARS``) and capped at ``PEEK_MAX_STEPS``, so a real
#: window is tens of KB; the bound exists so a pathological one is a refusal
#: rather than a record ``wire.LinkCrypto.seal`` rejects (which closes the link).
SESSION_PEEK_REPLY_BUDGET_BYTES = 2 * 1024 * 1024

#: The refusal codes from THIS device's relay that mean "this relay predates the
#: op", so the CLI degrades to the display tail instead of failing (the CLI is
#: replaced by ``lop-update`` before a long-lived relay restarts).
OLD_RELAY_CODES = frozenset({"unknown_local_op", "not_implemented"})


def _owner_live_state(session_id: str) -> str:
    """The digest's state words for one of THIS device's sessions.

    The same resolver and the same ``_peek_live_state`` wording the local tool
    uses, so a remote digest's header matches the owner's own. Best effort: a
    registry that cannot be read must not fail a READ, so it degrades to an
    honest "state unknown" rather than claiming ``stored``.
    """
    from local_operator.session.peek import _peek_live_state

    try:
        from local_operator.mobile import peer_send

        record, _candidates, _error = peer_send.resolve_peer_target(
            target=None,
            pid=None,
            session=session_id,
            include_wedged=True,
            require_started=False,
        )
    except Exception:  # noqa: BLE001 — see the docstring: a read never fails on this
        return "state unknown"
    return _peek_live_state(record)


def serve(root: Path, frame: dict[str, Any], *, own_label: str) -> dict[str, Any]:
    """Answer one ``net_session_peek`` frame from ``root``'s own journal.

    Pure apart from file reads so a handler-level test can drive it with the
    same directory a local peek reads. Refusals raise :class:`MeshRefusal` so
    the relay answers in the family's shape.
    """
    import json

    from local_operator.session import peek

    session_id = str(frame.get("session_id") or "")
    if not session_id:
        raise MeshRefusal("protocol_error", "net_session_peek must name a session_id")
    directory = root / "sessions" / session_id
    if not directory.is_dir():
        # The chokepoint already refused ids this device does not own; this is
        # the same fact restated where the answer is composed (``_op_session_
        # history`` makes the identical check).
        raise MeshRefusal("unknown_session", f"{own_label} does not hold a session {session_id}")
    request = peek.request_from_wire(frame)
    if isinstance(request, str):
        raise MeshRefusal("invalid_request", request)
    live_state = _owner_live_state(session_id) if request.digest else ""
    try:
        answer = peek.serve_peek(directory, request, request.given, live_state=live_state)
    except FileNotFoundError:
        raise MeshRefusal(
            "no_transcript",
            f"session {session_id!r} on {own_label} has no transcript to peek at — it "
            "may not have written a first turn yet.",
        ) from None
    if "refusal" in answer:
        raise MeshRefusal("invalid_request", str(answer["refusal"]))
    from local_operator.resume import session_name

    answer["session_id"] = session_id
    # Echoed so the viewer can render the footer's anchor/query words exactly as
    # the local tool does (``_peek_body``'s ``anchor``/``query``).
    answer["request"] = peek.wire_request(request, request.given)
    answer["name"] = session_name(directory) or f"session {session_id}"
    if len(json.dumps(answer).encode("utf-8")) > SESSION_PEEK_REPLY_BUDGET_BYTES:
        raise MeshRefusal(
            "reply_too_large",
            "that window is too large to send in one reply; ask for fewer `steps`.",
        )
    return answer


def make_handler(server: "RelayServer") -> Any:
    """The ``net_session_peek`` peer op (owner side)."""

    def _handle(link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        del link  # ownership was decided at the chokepoint; see the module doc
        return serve(server.root, frame, own_label=server._own_label())

    return _handle


def local_handler(server: "RelayServer") -> Any:
    """The ``peer_session_peek`` local op: ask the owning peer for one window.

    ``{"unsupported": True, "reason": ...}`` — NOT a refusal — when the peer's
    build predates the op, so the caller can fall back to the display tail and
    name the reason. An unreachable peer stays a refusal (``_local_peer_call``'s
    ``peer_unreachable``): blaming the peer's build for a transport failure
    would send the reader to update the wrong thing.
    """
    from local_operator.session import peek

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        peer = str(frame.get("peer") or "")
        session_id = str(frame.get("session_id") or "")
        if not session_id:
            raise MeshRefusal("protocol_error", "peer_session_peek must name a session_id")
        request = peek.request_from_wire(frame)
        if isinstance(request, str):
            raise MeshRefusal("invalid_request", request)
        link = server._ensure_link(server._resolve_peer(peer))
        if link is not None and wire.SESSION_PEEK_V1 not in link.capabilities:
            return {
                "unsupported": True,
                "reason": (
                    f"{server._peer_label(peer)} runs a build without the remote peek "
                    f"({wire.SESSION_PEEK_V1}); update it there to read its journal"
                ),
            }
        fields = peek.wire_request(request, request.given)
        return server._local_peer_call(
            "net_session_peek",
            peer,
            timeout=SESSION_PEEK_DEADLINE_S + _slow_margin(),
            session_id=session_id,
            **fields,
        )

    return _handle


def _slow_margin() -> float:
    from local_operator.network.relay import SLOW_REPLY_MARGIN_S

    return SLOW_REPLY_MARGIN_S


def install(server: "RelayServer") -> None:
    """Register ``net_session_peek`` (slow lane) and its local half."""

    server.register_ops(
        {"net_session_peek": make_handler(server)},
        local_handlers={"peer_session_peek": local_handler(server)},
        slow={"net_session_peek": SESSION_PEEK_DEADLINE_S},
    )
