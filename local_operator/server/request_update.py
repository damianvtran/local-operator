"""The core of ``POST /v1/desktop/projects/{key}/request-update``.

One job: hand each of a project's linked sessions a check-in message through the
shared peer-send core (:mod:`local_operator.mobile.peer_send`), then describe
what happened per session. Lives beside the route rather than inside it because
the route file's touch is deliberately minimal — the loop is the only part of
this feature that talks to other processes, and it is easier to test and reason
about on its own.

What the loop mirrors, and why (the ``send`` tool is the reference implementation
of "hand a message to a peer", so the two surfaces cannot disagree):

- **mailbox drop + wake**: ``mode="mailbox"``, ``wake=True`` — the same semantics
  as the ``send`` tool's default. Wake may engage a cold session (a runtime
  start), which is deliberate for an explicit check-in request; it is NOT a
  steer, because a check-in must not interrupt a running turn.
- **the resolver's own gates**: an exact session id resolves through
  :func:`resolve_peer_target`, with the same cold fallback the tool uses
  (``session_id_unowned`` + :func:`resolve_cold_session`). The gates that keep a
  never-engaged session from becoming the opening row of someone else's
  conversation are therefore enforced exactly once, in the shared core.

Outcome vocabulary (frozen by the design: three-way, honest per session):

- ``delivered`` — the receive side acknowledged the hand-off.
- ``unconfirmed`` — an ``OSError``-family failure (a dropped socket, a read
  deadline): the message MAY OR MAY NOT have arrived. Never folded into
  ``failed``: the sender must not be told "could not reach" about a message that
  may land, or it invites the duplicate the peer layer warns about.
- ``failed`` — a refusal or a dead end: nothing was delivered. The detail names
  the reason in the peer layer's own vocabulary; the never-engaged case carries
  the one sentence a project owner can act on.

COOLDOWN. Repeated presses on a project that has already asked would queue
identical check-ins, each one a wake and an agent turn. A per-project window
(``REQUEST_UPDATE_COOLDOWN_S``) starts on ``delivered >= 1`` OR
``unconfirmed >= 1`` — deliberately including unconfirmed, because those may
have landed — and while it is active no dial happens; the caller gets the same
sentence the UI shows. The state is deliberately IN-MEMORY and per process: it
is an anti-double-press guard, not a persisted fact, and the store schema does
not grow a field for it. A per-project lock serialises concurrent calls, so a
second press during an in-flight request waits and then re-evaluates against the
stamp the first one wrote.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime
from typing import Any

from local_operator.projects import (
    Project,
    ProjectRegistry,
    build_project_view,
    display_name,
)

#: The wall-clock window a project refuses a repeat request. One minute is about
#: one agent turn: long enough that a double-press cannot queue a second check-in
#: under the first, short enough that an intentional re-ask is not a fight.
REQUEST_UPDATE_COOLDOWN_S = 60.0

#: ``failed`` detail for a linked session that is not (yet) a recipient.
#: Deliberately actionable: the session IS linked, it simply has to be used once
#: before peer messages can land in its history (the unengaged gate in
#: ``peer_send`` — a peer row must never become the opening row of a
#: conversation its owner never started).
NEVER_STARTED_DETAIL = "linked but not yet in use — it becomes a recipient after the first message"

#: ``unconfirmed`` detail. Frozen wording: it states the uncertainty and nothing
#: more, because that is the only true thing this side can say.
UNCONFIRMED_DETAIL = "delivery could not be confirmed"

#: The check-in text, exactly as frozen by the design (``{display}`` and
#: ``{name}`` are substituted per project; ``{today}`` is the LOCAL date). The
#: wording addresses an agent that may not know this project: it names the tool
#: and the operation, forbids starting unrelated work, and allows "nothing
#: changed" — so a session asked about a project it does not own can refuse
#: cleanly instead of inventing progress.
REQUEST_UPDATE_TEMPLATE = (
    'Status check-in for project "{display}" (key: {name}), requested from the Projects view.\n'
    "\n"
    "Please post a progress update for it now:\n"
    '1. Call the `project` tool with op="update", name="{name}" and progress set to ONE dated '
    "line that starts with {today}: what has changed since your last update, what is in flight, "
    "and any blocker. Report only what is true; if nothing changed, say so in that line.\n"
    "2. If the project's status no longer fits (planning, active, qa, validation, paused, done, "
    'archived), change it in the same call with status="<new status>". Leave it unchanged if it '
    'still fits; use "done" only when every requirement is closed.\n'
    "3. Then reply with one short sentence confirming what you posted. Do not start new work "
    "because of this message.\n"
    "\n"
    "If you are not working on this project, reply saying so and do not post an update."
)

#: project id -> (monotonic stamp, wall-clock stamp) of the last request that
#: started a cooldown. Monotonic for the countdown (immune to clock steps),
#: wall-clock for the ISO ``requested_at`` the client shows.
_COOLDOWNS: dict[str, tuple[float, float]] = {}

#: project id -> the lock a concurrent call waits on, so two presses cannot dial
#: the same project at once and the loser re-evaluates against the winner's
#: stamp ("a call during in-flight waits then re-evaluates").
_LOCKS: dict[str, asyncio.Lock] = {}


def reset_cooldowns() -> None:
    """Forget every cooldown stamp. Tests, and any future reload path.

    The map is process-local by design (see the module docstring), so nothing
    outside this process can observe or clear it.
    """
    _COOLDOWNS.clear()


def sender_identity() -> dict[str, Any]:
    """The advisory sender dict a check-in carries to its targets.

    There is no ToolContext here and no sending session to resolve through the
    registry (this runs in the desktop backend's process, not inside a
    conversation), so the dict is synthesized the way the ``send`` tool's
    fallback builds one — with the display name ``"Projects"``, which identifies
    the VIEW the request came from, not a person. The receive side enriches
    only absent or blank fields from its own registry, so this label stands.
    """
    return {"conversation_name": "Projects"}


def compose_text(project: Project, *, today: str | None = None) -> str:
    """The frozen check-in text with this project's substitutions.

    ``{display}`` is the display name (title when set, else name) and ``{name}``
    is the project KEY — kept even when a title exists, because the ``project``
    tool addresses rows by key. ``{today}`` is the local calendar date; it is a
    parameter for tests and defaults to today.
    """
    return REQUEST_UPDATE_TEMPLATE.format(
        display=display_name(project),
        name=project.name,
        today=date.today().isoformat() if today is None else today,
    )


async def request_updates(
    registry: ProjectRegistry,
    project: Project,
    *,
    sender: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run one request-update call. Returns ``(governing sentence, result)``.

    The whole read-dial-write sequence for one project is serialised on the
    project's own lock; everything inside runs in REQUEST order, and the
    response is built only after every dial for this call has settled (the
    route awaits this before responding, so the sends complete even if the UI
    closes).
    """
    sender = sender or sender_identity()
    lock = _LOCKS.setdefault(project.id, asyncio.Lock())
    async with lock:
        stamp = _COOLDOWNS.get(project.id)
        if stamp is not None:
            remaining = REQUEST_UPDATE_COOLDOWN_S - (time.monotonic() - stamp[0])
            if remaining > 0:
                return _cooldown_reply(project, stamp, remaining)
        # The linked-session snapshot is taken OFF the loop and BEFORE any dial:
        # it reads the store and each session's directory (bounded, but blocking
        # filesystem work), and no registry lock is held across the dials below
        # — a slow engage must not block the project store's writers.
        fresh, rows = await asyncio.to_thread(_snapshot, registry, project)
        return await _run_batch(fresh, rows, sender)


def _snapshot(
    registry: ProjectRegistry, project: Project
) -> "tuple[Project, list[tuple[str, str | None]]]":
    """Fresh row + ``(session_id, title)`` per linked session, in link order.

    Re-read (not the caller's copy) because a call that waited on the lock may
    be looking at a list a concurrent link/unlink just changed; the composed
    view is the same one the detail route renders, so titles cannot disagree
    with the Linked sessions list the user just read.
    """
    fresh = registry.get_project(project.id)
    view = build_project_view(fresh, config_dir=registry.config_dir)
    return fresh, [(row["session_id"], row.get("title")) for row in view["sessions"]]


async def _run_batch(
    project: Project, rows: "list[tuple[str, str | None]]", sender: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    display = display_name(project)
    if not rows:
        sentence = f"No linked sessions to ask. Link a session to {display} first."
        return sentence, _payload(project, state="empty")

    text = compose_text(project)
    sessions: list[dict[str, Any]] = []
    for session_id, title in rows:
        # Strictly sequential: one daemon-class connection per target, next
        # target only after this one has meaning. Concurrent dials evict.
        outcome, detail = await _deliver(session_id, text, sender)
        sessions.append(
            {"session_id": session_id, "title": title, "outcome": outcome, "detail": detail}
        )

    delivered = sum(1 for row in sessions if row["outcome"] == "delivered")
    unconfirmed = sum(1 for row in sessions if row["outcome"] == "unconfirmed")
    failed = sum(1 for row in sessions if row["outcome"] == "failed")

    # The window starts on any hand-off that may have LANDED, unconfirmed
    # included: those sessions may already be composing a reply, and a repeat
    # press would double-ask them.
    requested_at: str | None = None
    if delivered or unconfirmed:
        wall = time.time()
        _COOLDOWNS[project.id] = (time.monotonic(), wall)
        requested_at = datetime.fromtimestamp(wall).isoformat()

    sentence = _sent_sentence(
        display,
        sessions,
        total=len(sessions),
        delivered=delivered,
        unconfirmed=unconfirmed,
        failed=failed,
    )
    return sentence, _payload(
        project,
        state="sent",
        requested_at=requested_at,
        counts={
            "total": len(sessions),
            "delivered": delivered,
            "unconfirmed": unconfirmed,
            "failed": failed,
        },
        sessions=sessions,
    )


async def _deliver(session_id: str, text: str, sender: dict[str, Any]) -> tuple[str, str | None]:
    """One target, mapped to ``(outcome, detail)``. Never raises: a batch of N
    sessions reports N outcomes, and one unreachable target must not sink the
    rest."""
    # Imported in-function: `mobile.peer_send` reaches the runtime registry and
    # the engine-adjacent path, and neither belongs at this module's import
    # surface (the server-shape guard reads it; see `server/features.py`).
    from local_operator.mobile.peer_send import (
        DELIVERY_DELIVERED,
        DELIVERY_FAILED,
        DELIVERY_MAILBOX,
        DELIVERY_UNCONFIRMED,
        deliver_peer_message_outcome,
        resolve_cold_session,
        resolve_peer_target,
        session_id_unowned,
    )

    # An EXACT session-id selector (the batch names sessions by id), so the
    # team-role vocabulary cannot change the answer; named explicitly so the
    # "every caller passes role_words" source-scan invariant holds.
    record, _candidates, error = await asyncio.to_thread(
        resolve_peer_target, session=session_id, role_words=()
    )
    cold_id = ""
    if record is None and error and session_id_unowned(error):
        # No live owner: an exact stored id may still be addressable (a closed
        # terminal), which is the same cold fallback the `send` tool runs. The
        # predicate is what keeps a WEDGED or UNENGAGED refusal standing — the
        # live scan reached a session there, and re-asking the store would
        # deliver behind a process that still owns it.
        cold_id = await asyncio.to_thread(resolve_cold_session, session_id) or ""
    if record is None and not cold_id:
        return "failed", _resolver_detail(error)

    try:
        # The OUTCOME face, NOT the receipt-string face: a receipt is not a data
        # channel, and under the delivery-state model an amber result
        # (``mailbox``/``unconfirmed``) RETURNS its sentence rather than raising
        # — so classifying on "did it raise?" calls an unacknowledged dial a
        # delivery. ``.state`` is the peer layer's own settled word; map it.
        outcome = await deliver_peer_message_outcome(
            record,
            session_id=record.session_id if record is not None else cold_id,
            text=text,
            mode="mailbox",
            wake=True,
            sender=sender,
        )
    except RuntimeError as exc:
        # A pre-delivery REFUSAL (the unengaged gate, an engage that could not
        # start): no message id was minted, so there is no state to report. The
        # sentence is the peer layer's own, except the unengaged case, which
        # carries the project-facing sentence.
        return "failed", _refusal_detail(str(exc))
    except (ConnectionError, OSError, ValueError):
        # A transport fault raised OUTSIDE the outcome builder's own
        # classification: the socket or the ack failed, which is NOT the same as
        # "not delivered" — the receive side commits before it acks, so this
        # side cannot know. An `asyncio.TimeoutError` is an `OSError` subclass.
        return "unconfirmed", UNCONFIRMED_DETAIL
    except Exception as exc:  # noqa: BLE001 — one target must not sink the batch
        return "failed", str(exc) or type(exc).__name__

    if outcome.state == DELIVERY_DELIVERED:
        return "delivered", None
    if outcome.state == DELIVERY_MAILBOX:
        # The row is DURABLE in the target's mailbox and it reads the message on
        # its next turn; only the immediate WAKE went unacknowledged. The peer
        # layer's own word for that is "wake unconfirmed" — the DELIVERY is
        # confirmed — so this is a delivery, not the unconfirmed class, whose
        # sentence says the reverse (that delivery could not be confirmed).
        return "delivered", None
    if outcome.state == DELIVERY_UNCONFIRMED:
        return "unconfirmed", UNCONFIRMED_DETAIL
    if outcome.state == DELIVERY_FAILED:
        return "failed", _refusal_detail(outcome.detail)
    # A state a future build added: report it as failed rather than claiming a
    # delivery this build cannot vouch for.
    return "failed", outcome.detail or "the message was not delivered"


def _resolver_detail(error: str) -> str:
    """Map the resolver's refusal about an exact session id to a detail.

    The mapping is by the peer layer's own sentence, not by a second reading of
    state: `peer_send` is the single source of truth for what each refusal
    means, and the strings are pinned by its tests.
    """
    if "has not been engaged yet" in error:
        return NEVER_STARTED_DETAIL
    if "is stale (its pid no longer exists)" in error:
        return "stale"
    if "no session found with session id" in error:
        # Nothing published a record and no session directory exists: the id a
        # project still links is not addressable any more. Reported, never
        # silently dropped.
        return "no longer exists"
    return error or "no target resolved"


def _refusal_detail(message: str) -> str:
    """Map a delivery refusal to a detail; the unengaged gate gets the one
    project-facing sentence (see :data:`NEVER_STARTED_DETAIL`)."""
    if "has not been engaged yet" in message:
        return NEVER_STARTED_DETAIL
    return message


def _cooldown_reply(
    project: Project, stamp: "tuple[float, float]", remaining: float
) -> tuple[str, dict[str, Any]]:
    """The refusal reply, in the UI's own sentence (frozen: the numbers are
    whole seconds, rounded up; the display name is the project's)."""
    left = max(1, math.ceil(remaining))
    ago = max(0, math.ceil(time.monotonic() - stamp[0]))
    display = display_name(project)
    sentence = f"Update already requested {ago} s ago on {display}. Try again in {left} s."
    return sentence, _payload(
        project,
        state="cooldown",
        requested_at=datetime.fromtimestamp(stamp[1]).isoformat(),
        cooldown_remaining_s=left,
    )


def _sent_sentence(
    display: str,
    sessions: "list[dict[str, Any]]",
    *,
    total: int,
    delivered: int,
    unconfirmed: int,
    failed: int,
) -> str:
    """The governing sentence for an attempted batch.

    Mirrors the UI's frozen copy so the two surfaces cannot describe one batch
    two ways; the client composes its own toast from the counts regardless.
    """
    if delivered == total:
        if total == 1:
            return f"Requested an update from 1 session on {display}."
        return f"Requested updates from {total} sessions on {display}."
    if delivered:
        return f"Requested updates from {delivered} of {total} sessions on {display}."
    if unconfirmed and not failed:
        return (
            f"Could not confirm delivery on {display} — the requests may still "
            "reach its sessions."
        )
    if failed == total and all(row["detail"] == NEVER_STARTED_DETAIL for row in sessions):
        # Singularised at N=1: the pluralising forms disagree with themselves
        # ("The 1 linked sessions have not started yet"), and the UI lane pins
        # this branch's wording byte-for-byte, so the two must match.
        if total == 1:
            return (
                "The linked session has not started yet — it becomes a recipient "
                "after its first message."
            )
        return (
            f"The {total} linked sessions have not started yet — they become "
            "recipients after their first message."
        )
    if unconfirmed:
        # A batch that was partly unconfirmed and partly refused: the honest
        # single sentence names the uncertainty rather than asserting delivery
        # failed (the same rule the UI's failure clause follows).
        return (
            "Could not confirm whether the updates were requested — check the "
            "sessions before asking again."
        )
    return f"Could not reach any of the {total} linked sessions on {display}."


def _payload(
    project: Project,
    *,
    state: str,
    requested_at: str | None = None,
    cooldown_remaining_s: int | None = None,
    counts: dict[str, int] | None = None,
    sessions: "list[dict[str, Any]] | None" = None,
) -> dict[str, Any]:
    """The route's ``result`` body — one shape for all three states."""
    return {
        "project": {"id": project.id, "key": project.name, "title": project.title},
        "state": state,
        "requested_at": requested_at,
        "cooldown_remaining_s": cooldown_remaining_s,
        "counts": counts or {"total": 0, "delivered": 0, "unconfirmed": 0, "failed": 0},
        "sessions": sessions or [],
    }
