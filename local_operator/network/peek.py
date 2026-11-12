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
  before the handler runs. The handler ALSO refuses an id that is not one safe
  path component (``store.safe_component``) before it builds a path: the
  chokepoint once read a stray inner ``frame`` key for every op (PR #3831
  review), and a path built from a peer's string must not rest on one guard.
* It starts NOTHING and takes no lease — reading a stored session is the point.
* The readers' own budgets bound the I/O (``PEEK_MAX_STEPS``, the page-walk
  cap, the 16 MiB search depth). Those do NOT bound a peer-supplied regex's
  CPU — one row can backtrack for hours — so the owner runs it through
  ``session.peek.BoundedPeerPattern``: the ``regex`` module's per-call timeout
  (PROCESS CPU seconds), capped at :data:`PEER_REGEX_BUDGET_S` and never past
  what is left of this op's slow-lane deadline. A pattern that runs out is
  refused by name and its worker returns to the pool. The 256-char cap and the
  nested-quantifier pre-check stay as cheap early refusals, not the bound.
* At most ONE peer-regex peek per requesting device is in flight
  (:func:`peer_regex_device`, enforced by the relay before a worker is taken):
  a burst of bad patterns from one member cannot occupy the shared slow lane.
* SLOW lane: a deep search can take seconds, and the link reader must keep
  serving other frames meanwhile.

COMPATIBILITY. Advertised as :data:`wire.SESSION_PEEK_V1`; the viewer's relay
checks it before asking, and a peer without it answers ``unsupported`` with a
reason so the caller can fall back to the live-display tail and SAY why.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.network import wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:
    from local_operator.network.relay import PeerLink, RelayServer

#: Owner-side deadline for one ``net_session_peek``. The ordinary read is one
#: page (milliseconds); the worst case for a literal or well-behaved regex is
#: the needle search crossing its full 16 MiB depth plus the anchored window
#: read, measured in low seconds on an operator-size journal. 20 s leaves a
#: loaded machine room without letting a caller wait on a wedged disk for the
#: engage budget. The relay's deadline only ANSWERS the requester; a peer's
#: regex is what could outlive it, so the regex is cut by its own timeout
#: inside this deadline (:data:`PEER_REGEX_REPLY_RESERVE_S`).
SESSION_PEEK_DEADLINE_S = 20.0

#: Time held back from the slow-lane deadline when a peer's regex is bounded,
#: so the owner's OWN named refusal ("ran out of time") — or the window read
#: after a late hit (one anchored page, milliseconds) — reaches the viewer
#: before the relay's generic overdue answer does.
#:
#: WHAT THE ``regex`` TIMEOUT MEASURES: the PROCESS's CPU time, not the calling
#: thread's and not wall time (PR #3831 review r2-1 / QA Q7: a ``timeout=2``
#: search stopped at 2.0 s of process CPU alone, and at ~0.4-0.5 s of its own
#: thread CPU beside three busy siblings). Two consequences: concurrent peer
#: regexes, and any other CPU the relay burns, SHARE one budget — which only
#: ever stops a search earlier, the safe direction — and the WALL time a
#: search takes stretches on a contended machine (QA r2: 62.8 s wall for 18 s
#: of CPU at load 45-60 on 14 cores). This reserve covers the reply after the
#: search; :data:`PEER_REGEX_BUDGET_S` is what keeps the stretched search
#: itself inside the deadline.
PEER_REGEX_REPLY_RESERVE_S = 2.0

#: The ceiling on what a peer's regex may spend on this device (PR #3831 QA
#: Q5/Q6). The budget used to be the whole remaining deadline (~18 s), so four
#: bad patterns held all four slow workers past the 20 s deadline, and the
#: requester got the relay's generic overdue sentence instead of the owner's
#: named ``regex_timeout``.
#:
#: MEASURED (2026-11-11, this branch, ``regex`` 2026.11.9, load average ~40 on
#: 14 cores): seven legitimate MISSING patterns (a miss walks the full depth —
#: ``INV-\d{4}…``, ``(?i)…\w+``, alternation with ``\b``, ``error.*…``, an
#: anchored one, a character-class run, a URL shape) over three real operator
#: journals of 33-57 MB, each walk crossing the whole 16 MiB scan (16,780,308
#: bytes, ~1,500 searched rows / ~11 MB of row text), 3 runs: n=63. Regex CPU
#: median 0.011 s / max 0.028 s; process CPU for the whole serve median 0.10 s
#: / max 0.45 s; WALL median 0.26 s / max 1.46 s (the per-row deadline is a
#: wall-clock one, so the walk's JSON parsing counts against it too).
#:
#: 4 s is ~2.7x the observed wall max for a legitimate walk, so it is not hit by
#: one, even with siblings sharing the process clock. A catastrophic pattern
#: burns at most 4 s of process CPU; at QA's observed ~3.5x load stretch that is
#: ~14 s wall — still a clear gap under ``SESSION_PEEK_DEADLINE_S`` (20 s), so
#: the owner's named refusal reaches the viewer first.
PEER_REGEX_BUDGET_S = 4.0

#: Encoded ceiling for one reply. Steps are already clipped by the renderer
#: (``comms.PEEK_STEP_CHARS``) and capped at ``PEEK_MAX_STEPS``, so a real
#: window is tens of KB; the bound exists so a pathological one is a refusal
#: rather than a record ``wire.LinkCrypto.seal`` rejects (which closes the link).
SESSION_PEEK_REPLY_BUDGET_BYTES = 2 * 1024 * 1024

#: The refusal codes from THIS device's relay that mean "this relay predates the
#: op", so the CLI degrades to the display tail instead of failing (the CLI is
#: replaced by ``lop-update`` before a long-lived relay restarts).
OLD_RELAY_CODES = frozenset({"unknown_local_op", "not_implemented"})


_UNRESOLVED = object()


def _owner_record(session_id: str) -> Any:
    """The live registry record for one of THIS device's sessions.

    ``None`` when it is not live; :data:`_UNRESOLVED` when the registry could
    not be read. Read on EVERY peek, not only a digest, because the header's
    name needs it too (review r2 n2). The same resolver the local tool uses, so the digest's state
    words and the header's name (the live ``conversation_name`` first, PR #3831
    review m2) match what the owner's own peek prints. Best effort: a registry
    that cannot be read must not fail a READ.
    """
    try:
        from local_operator.mobile import peer_send

        # An EXACT session-id selector with no ``target``, so the team-role
        # vocabulary cannot change the answer (it only gates ``target``); loading
        # the team registry on every peek would be pure cost inside the 4 s peek
        # budget. Named explicitly, and allowlisted in test_peer_send.py like
        # ``request_update._deliver``, so the "every caller passes role_words"
        # source-scan invariant holds.
        record, _candidates, _error = peer_send.resolve_peer_target(
            target=None,
            pid=None,
            session=session_id,
            include_wedged=True,
            require_started=False,
            role_words=(),
        )
    except Exception:  # noqa: BLE001 — see the docstring: a read never fails on this
        return _UNRESOLVED
    return record


def _owner_live_state(record: Any) -> str:
    """The digest's state words: ``_peek_live_state``'s, or an honest
    "state unknown" (never ``stored``) when the registry could not be read."""
    from local_operator.session.peek import _peek_live_state

    if record is _UNRESOLVED:
        return "state unknown"
    return _peek_live_state(record)


def _regex_budget_s() -> float:
    """What a peer's regex may spend: :data:`PEER_REGEX_BUDGET_S`, or less when
    this slow op's remaining deadline minus :data:`PEER_REGEX_REPLY_RESERVE_S`
    is shorter (a frame that queued for a worker). Outside a slow op (a direct
    call) the full deadline is the reference."""
    from local_operator.network.relay import slow_op_remaining_s

    remaining = slow_op_remaining_s()
    if remaining is None:
        remaining = SESSION_PEEK_DEADLINE_S
    return max(0.0, min(PEER_REGEX_BUDGET_S, remaining - PEER_REGEX_REPLY_RESERVE_S))


#: What a member is told when it asks for a second regex peek while its first
#: is still running here (PR #3831 QA Q5). It names the cause and the remedy;
#: the generic "already running N long operations" sentence would blame load.
PEER_REGEX_BUSY_SENTENCE = (
    "a `regex` search from this device is already running on {owner}; wait for "
    "it to finish (at most {budget:g} s of search time) before starting another, "
    "or search literally without `regex`."
)


def peer_regex_device(link: "PeerLink", frame: dict[str, Any]) -> str | None:
    """The device a ``net_session_peek`` frame's regex is charged to, or ``None``.

    WHY (PR #3831 QA Q5). The slow lane is four workers shared by every peer
    and every slow op (broker borrows, moves, syncs). The per-search budget
    bounds how long ONE pattern holds a worker, but five concurrent bad
    patterns from one member still took all four, and whatever arrived next
    queued past its deadline. So the relay admits at most one regex peek per
    requesting device and turns a second one away BEFORE it takes a slot or a
    worker.

    Keyed on the LINK's authenticated ``device_id``, never on a frame field: a
    member cannot spread its burst across names it does not hold. ``regex`` is
    read with the same truthiness ``session.peek.request_from_wire`` uses, so
    every frame that will compile a peer pattern is counted. A literal peek is
    not limited: its cost is bounded by the scan depth alone.
    """
    if str(frame.get("op") or "") != "net_session_peek" or not frame.get("regex"):
        return None
    return str(link.device_id or link.link_id)


def serve(root: Path, frame: dict[str, Any], *, own_label: str) -> dict[str, Any]:
    """Answer one ``net_session_peek`` frame from ``root``'s own journal.

    Pure apart from file reads so a handler-level test can drive it with the
    same directory a local peek reads. Refusals raise :class:`MeshRefusal` so
    the relay answers in the family's shape.
    """
    import json

    from local_operator.network.store import safe_component
    from local_operator.session import peek

    session_id = str(frame.get("session_id") or "")
    if not session_id:
        raise MeshRefusal("protocol_error", "net_session_peek must name a session_id")
    try:
        # Defence in depth (PR #3831 review m1): the path below is built from a
        # peer's string, so it must be ONE component (no separator, no `.`/`..`,
        # not absolute) whatever the chokepoint decided.
        safe_component(session_id)
    except MeshRefusal:
        raise MeshRefusal(
            "invalid_request", "net_session_peek needs a session id without path syntax"
        ) from None
    directory = root / "sessions" / session_id
    if not directory.is_dir():
        # The chokepoint already refused ids this device does not own; this is
        # the same fact restated where the answer is composed (``_op_session_
        # history`` makes the identical check).
        raise MeshRefusal("unknown_session", f"{own_label} does not hold a session {session_id}")
    request = peek.request_from_wire(frame)
    if isinstance(request, str):
        raise MeshRefusal("invalid_request", request)
    record = _owner_record(session_id)
    live_state = _owner_live_state(record) if request.digest else ""
    try:
        answer = peek.serve_peek(
            directory,
            request,
            request.given,
            live_state=live_state,
            regex_budget_s=_regex_budget_s(),
        )
    except peek.PeerRegexTimeout as exc:
        raise MeshRefusal(
            "regex_timeout",
            f"the `regex` query ran out of time on {own_label} (its {exc.budget_s:.1f} s "
            "budget there) before the search finished; simplify it, or search "
            "literally without `regex`.",
        ) from None
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
    live_name = getattr(record, "conversation_name", "") if record is not _UNRESOLVED else ""
    answer["name"] = live_name or session_name(directory) or f"session {session_id}"
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
        started = time.monotonic()
        try:
            return server._local_peer_call(
                "net_session_peek",
                peer,
                timeout=SESSION_PEEK_DEADLINE_S + _slow_margin(),
                session_id=session_id,
                **fields,
            )
        except MeshRefusal as exc:
            # PR #3831 QA Q3: a link that went quiet for the WHOLE peek budget
            # surfaced as "<raw device id> stopped answering before it
            # replied" — read as a liveness failure, though the owner answered
            # the next peek seconds later. Past the deadline, the owner's read
            # overran; say so, by the member's name. A quick failure (no link)
            # is a real reachability fact and stays as it was.
            overran = time.monotonic() - started >= SESSION_PEEK_DEADLINE_S
            if exc.code == "peer_unreachable" and overran:
                raise MeshRefusal(
                    "peer_timeout",
                    f"{server._peer_label(peer)} did not finish this peek within "
                    f"{SESSION_PEEK_DEADLINE_S:g} s — its search ran out of time there; "
                    "narrow the query (or drop `regex`) and try again.",
                ) from None
            raise

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
