"""Shared send-side core for peer-to-peer session messaging.

Both entry points that hand a message to another local ``lop`` session ride the
same substrate — the CLI command ``lop send`` (a short-lived child process) and
the in-session ``send`` tool (running inside the sender's own process) — and they
must agree on HOW a target is resolved and WHAT counts as a deliverable body.
Before this module existed that logic lived only in ``cli.py``, so the tool would
have had to re-implement it (and the two would drift). This is the single source
of truth for the send-side decision logic; the transport itself stays in
``peer_client.send_peer_message`` and the receive semantics stay in
``Session.receive_peer_message``.

Kept in ``mobile/`` next to ``peer_client`` because peer messaging is built on the
mobile control-socket + registry substrate (every interactive session publishes a
discovery record and runs an authenticated loopback control server). Import-light
on purpose: it pulls the registry and config path only, never the heavyweight
``Session`` graph, so a tool can import it without dragging the session in.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from local_operator.info.model import format_duration
from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.engagement import (
    session_has_durable_history as _session_has_durable_history,
)
from local_operator.session.runtime.types import PEER_MESSAGE_ID_CAPABILITY

#: How much of a stored session store the send-side fallback scans while
#: hunting for a substring match. Discovery over the WHOLE store would pay one
#: bounded name read per directory (hundreds exist on a well-used machine) on
#: a path whose whole job is to answer quickly; the mtime-ordered scan
#: underneath (:func:`resume._scan_sessions`) visits every directory anyway,
#: so this caps the ROW-BUILDING half, not the scan. A target that matches
#: nothing in the newest window is answered with the no-match error naming
#: ``--session`` — an exact id always resolves, so no message is made
#: undeliverable by the cap, only un-findable by NAME.
STORED_DISCOVERY_LIMIT = 200

#: Peer messaging body cap. Well under the registrant's 1 MB line limit so a huge
#: paste is rejected with a clear message here rather than becoming a silently
#: dropped oversized line on the wire. Shared by the CLI and the tool so both
#: refuse the same size with the same words.
PEER_MESSAGE_MAX_BYTES = 256 * 1024


#: What an unengaged session cannot do, in :func:`unengaged_refusal`'s sentence.
#: The peer-message form is the default; a model switch names its own verb
#: (:data:`MODEL_SWITCH_CAPABILITY`) so a sender is not told a switch was
#: refused because the target "cannot receive peer messages".
PEER_MESSAGE_CAPABILITY = "cannot receive peer messages"
MODEL_SWITCH_CAPABILITY = "cannot be switched remotely"

#: The additive discovery capability a receiver advertises when it understands a
#: sender-minted ``message_id``. IMPORTED, not re-declared: the runtime that
#: advertises it and the sender that gates its retry on it must read ONE
#: spelling, and ``types.py`` is where the wire vocabulary lives (see its note on
#: why the string exists and what gates it).

#: Default for ``send.journal_unconfirmed``: OFF. The configurable notice the
#: operator asked for, defaulting to silence because the tool result already
#: carries the same facts and a second row per send is a second writer for one
#: event (design note B).
#:
#: The registry (``settings_io``) states the same default and
#: ``tests/unit/test_settings_io.py`` pins the two constants together, so a drift
#: is a red test rather than a `/settings` page that lies.
JOURNAL_UNCONFIRMED_DEFAULT = False


def journal_unconfirmed_enabled() -> bool:
    """Whether a settled amber send should also leave a transcript notice.

    Never raises: a malformed config must not fail a send that has already
    happened, and the conservative direction is the SHIPPED default (silence),
    not an exception out of a notice path.
    """
    try:
        from local_operator.config import ConfigManager

        raw = ConfigManager(config_dir()).get_config_value("send", None)
    except Exception:  # noqa: BLE001 — a notice must not break a delivery
        return JOURNAL_UNCONFIRMED_DEFAULT
    if not isinstance(raw, dict):
        return JOURNAL_UNCONFIRMED_DEFAULT
    value = raw.get("journal_unconfirmed")
    return bool(value) if isinstance(value, bool) else JOURNAL_UNCONFIRMED_DEFAULT


#: The sender-minted message id's shape. Validated by the RECEIVER before it is
#: allowed to become a transcript entry id (see
#: ``Session.receive_peer_message``): an id from a build that spells it
#: differently, or a hand-built frame, must not be able to name a row.
PEER_MESSAGE_ID_PREFIX = "peer-"
PEER_MESSAGE_ID_PATTERN = re.compile(r"^peer-[0-9a-f]{32}$")

#: Attempts one live send makes, and the per-attempt socket read deadline. Three
#: attempts of five seconds each is what the failure copy quotes ("no answer
#: within 5s after 3 attempts") and what the 20 s hard budget below bounds.
PEER_SEND_ATTEMPTS = 3
PEER_SEND_DEADLINE_S = 5.0

#: Pauses BETWEEN attempts, indexed by the attempt that just failed. Short on
#: purpose: the retry exists to catch a receiver whose loop turned a moment
#: later, not to back off a load problem, and the sender's tool call is
#: synchronous -- the operator is waiting on it. A ``ControlDialFailed`` (nothing
#: was written) takes the same pauses and skips the probe.
PEER_SEND_RETRY_SLEEPS: tuple[float, ...] = (0.25, 0.5)

#: The hard cap on the whole retry loop. 3 x 5 s of reads plus the two pauses is
#: ~15.8 s, so this is a backstop against a pathological socket rather than a
#: bound the ordinary path can reach.
PEER_SEND_TOTAL_BUDGET_S = 20.0

#: How much of the recipient's transcript TAIL the sender scans to decide whether
#: a timed-out message landed. Bounded because this runs on a synchronous tool
#: call: one megabyte is milliseconds, and every answer it can give is either
#: "committed" (found) or "unknown" (not found) -- never "failed", because a
#: steer still sitting in an in-memory queue is invisible here by construction.
PEER_SEND_PROBE_TAIL_BYTES = 1 << 20

#: The wire states a send can settle in (design note A.1). ``delivered`` and
#: ``failed`` are the two the operator already knew; ``mailbox`` and
#: ``unconfirmed`` exist because "the ack did not arrive" is not the same fact as
#: "the message did not land", and the two were previously painted as one.
DELIVERY_DELIVERED = "delivered"
DELIVERY_MAILBOX = "mailbox"
DELIVERY_UNCONFIRMED = "unconfirmed"
DELIVERY_FAILED = "failed"

WAKE_ACKED = "acked"
WAKE_UNCONFIRMED = "unconfirmed"
WAKE_NOT_REQUESTED = "not_requested"


def mint_peer_message_id() -> str:
    """Mint the sender-owned identity of ONE send, reused by every attempt.

    ``peer-<32 hex>``: a shape the receiver can validate, distinct from the
    user-row admission ids (``CommandReservations``), and stable across the retry
    loop and across the cold-target ``engage_runtime`` errand -- so a re-send of
    the same message is a duplicate to the receiver rather than a second row.
    """
    return f"{PEER_MESSAGE_ID_PREFIX}{uuid.uuid4().hex}"


def valid_peer_message_id(message_id: Any) -> bool:
    """Whether ``message_id`` may become a transcript entry id.

    The receiver-side check, shared with the sender's own mint so the two can
    never disagree about the shape. A malformed or foreign id is not an error:
    the receiver mints its own (old sender, hand-built frame), which is exactly
    today's behaviour.
    """
    return isinstance(message_id, str) and bool(PEER_MESSAGE_ID_PATTERN.match(message_id))


def _attempts_phrase(count: int) -> str:
    """``3 attempts`` / ``1 attempt`` -- the copy is read, so it must agree."""
    return f"{count} attempt" if count == 1 else f"{count} attempts"


@dataclass(frozen=True)
class DeliveryOutcome:
    """What ONE ``send`` actually achieved, and the evidence behind it.

    THE SINGLE SOURCE OF TRUTH for the send state (design note A.2): the tool
    result, the CLI's exit code and the TUI card all read this rather than
    re-deriving a verdict from an exception type, which is how the incident's
    "Sent ... failed" row came to claim a non-delivery the sender could not know.

    ``state`` is one of :data:`DELIVERY_DELIVERED`, :data:`DELIVERY_MAILBOX`,
    :data:`DELIVERY_UNCONFIRMED`, :data:`DELIVERY_FAILED`. ``detail`` is the
    sentence for the SEND's own body (the receiver's ack text for a delivered
    live send, byte-identical to what this module returned before the states
    existed); :attr:`text` composes the model-facing line by prefixing the target.

    ``cause`` is a short machine token, never prose, because it rides the
    persisted ``details`` where a renderer matches on it (``dial_refused``,
    ``peer_refused``, ``spool_write_failed``, ``no_answer``, ``ack_lost``,
    ``queued_busy``). ``route`` says HOW it travelled (``live`` = a dial to a
    running session, ``spool`` = written to a cold session's inbox,
    ``engaged`` = a runtime was started for it).

    ``attempts`` is bounded by :data:`PEER_SEND_ATTEMPTS`, and a target that does
    not advertise :data:`PEER_MESSAGE_ID_CAPABILITY` gets exactly one: the copy
    quotes this number, so it must be the number of tries that really happened.
    """

    state: str
    detail: str
    message_id: str
    wake: str
    attempts: int
    cause: str
    route: str
    target: str

    @property
    def text(self) -> str:
        """The model-facing line, target included.

        The delivered/mailbox/unconfirmed forms keep the ``→ <target>: `` receipt
        shape every send has always had, so the common path stays byte-stable
        (the identity lives in ``details``, not in the text). Only ``failed``
        drops the arrow: its sentence begins with what went wrong, and the target
        sits after "could not deliver to" where a reader looks for it.
        """
        if self.state == DELIVERY_FAILED:
            return (
                f"could not deliver to {self.target}: {self.detail}. "
                f"Nothing was delivered (id {self.message_id}) — fix the cause or retry the send."
            )
        return f"→ {self.target}: {self.detail}"

    @property
    def is_error(self) -> bool:
        """True ONLY for ``failed`` (design note A.1).

        The amber states are not failures: the UI's failed-row jump, its
        "N failed" turn foot and the TUI's error wash all key off this, and a
        delivered message counted as a failure is the defect the states exist to
        remove.
        """
        return self.state == DELIVERY_FAILED

    @property
    def partial(self) -> bool:
        """Whether the result is honest but INCOMPLETE (mailbox, unconfirmed)."""
        return self.state in (DELIVERY_MAILBOX, DELIVERY_UNCONFIRMED)

    @property
    def plain_wake(self) -> bool:
        """The legacy boolean ``details["wake"]`` shape, for old consumers."""
        return self.wake != WAKE_NOT_REQUESTED

    def details(self) -> dict[str, Any]:
        """The ONE nested payload both the live result and the persisted row carry.

        Nested rather than flat because ``details.wake`` is already the
        requested-bool every existing reader (the TUI card, the phone fold) reads,
        and ``outcome`` is owned by the model-switch row -- reusing either would
        make one key mean two things on the same tool result.
        """
        return {
            "state": self.state,
            "message_id": self.message_id,
            "wake": self.wake,
            "attempts": self.attempts,
            "cause": self.cause,
            "route": self.route,
        }


def _mailbox_detail(message_id: str, attempts: int) -> str:
    """Copy for "it landed, the wake did not answer" (design note A.5)."""
    return (
        f"delivered to its mailbox (id {message_id}) — the wake was not "
        f"acknowledged within 5s after {_attempts_phrase(attempts)}. It will read "
        "the message on its next turn; do not send it again."
    )


def _ack_lost_detail(message_id: str, attempts: int) -> str:
    """Copy for a QUIET send whose receipt was lost but whose row is on disk."""
    return (
        f"delivered (id {message_id}) — its receipt was lost after "
        f"{_attempts_phrase(attempts)}, but the message is in its transcript. "
        "Do not send it again."
    )


def _unconfirmed_detail(message_id: str, attempts: int) -> str:
    """Copy for the honest residual: no ack AND no evidence on disk (A.5)."""
    return (
        f"delivery UNCONFIRMED (id {message_id}) — no answer within 5s after "
        f"{_attempts_phrase(attempts)} and the message is not yet in its "
        "transcript. It may still arrive once its loop turns. Check with "
        'sessions(op="peek", …) before resending; sending again may deliver it twice.'
    )


def probe_transcript_for(message_id: str, session_id: str, *, root: "Path | None" = None) -> bool:
    """Whether ``message_id`` is already an entry id in ``session_id``'s transcript.

    The second observation channel the outcome builder has (design note A.1): the
    ack is one signal for "committed", and the recipient's own durable row is
    another, available even when its loop is too busy to answer the socket. Scans
    only the last :data:`PEER_SEND_PROBE_TAIL_BYTES` for the raw id, which is
    enough because a row written by the attempt under test is by definition at the
    tail.

    NOT FOUND MEANS UNKNOWN, NEVER FAILED. A steer still queued in memory, or an
    op still in a stalled socket buffer (the failure the 5 s read deadline's
    sentence was written for), is invisible to a disk scan, and this function
    must not be read as evidence that the message did not land.
    """
    from local_operator.session.runtime.engagement import TRANSCRIPT_FILENAME

    directory = (config_dir() if root is None else root) / "sessions" / session_id
    path = directory / TRANSCRIPT_FILENAME
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False
    needle = message_id.encode("utf-8")
    try:
        with path.open("rb") as handle:
            if size > PEER_SEND_PROBE_TAIL_BYTES:
                handle.seek(size - PEER_SEND_PROBE_TAIL_BYTES)
            chunk = handle.read()
    except OSError:
        return False
    return needle in chunk


def unengaged_refusal(
    label: str,
    *,
    count: int = 1,
    cold: bool = False,
    capability: str = PEER_MESSAGE_CAPABILITY,
) -> str:
    """The ONE sentence that refuses a peer message to a never-engaged session.

    A session that has not run a real turn yet (``SessionRecord.started`` is
    False — the ``/new`` window where the record is published but the owner is
    still composing their first prompt) is not a recipient: a peer row landing
    there becomes the OPENING row of a conversation its owner never started,
    which is the operator-reported symptom this gate exists for.

    Three layers can each stop a delivery — resolution
    (:func:`resolve_peer_target`), delivery (:func:`deliver_peer_message`) and
    the receive-side gate (``session.runtime.server._dispatch``, the one that
    also holds against a sender on an older build) — and they use the SAME
    sentence, because a sender refused at one layer and retried into another
    must not have to learn a second rule. Only ``label`` varies, and it is
    written in the caller's own grammar — ``pid 12345`` where a live record
    gives us the pid the sender typed, ``session 'abc'`` for an id (see
    :func:`delivery_label` and the resolver's own ``f"pid {pid}"`` /
    ``f"session {session!r}"`` forms) — so the target the refusal names is the
    one that was addressed.

    The reason is stated as the FIX, not as a prohibition: the owner sends a
    first message and the session becomes eligible at that moment (the first
    turn publishes the flipped bit the resolver and the receive side both
    read). ``/stop`` never reads this — the kill switch passes
    ``require_started=False``, since a composer window is exactly a session a
    person may need to stop.

    ``count`` pluralises the sentence for a BATCHED refusal: a substring match
    that reached several unengaged sessions refuses them in one line, and
    "every … (4 of them)" with a singular tail disagrees with itself
    (design round 1, D3). ``cold`` swaps the remedy for a target with NO live
    runtime — a stored session: there, "its owner has to send a first message"
    presumes a window the sender cannot open, so the cold tail says the
    conversation has to be opened and used before it becomes a recipient
    (design round 1, D5). Both default to the single live form, which is what
    the exact-address sites want.
    """
    plural = count != 1
    sent = (
        "no user message has been sent in any of them"
        if plural
        else "no user message has been sent in it"
    )
    if cold:
        remedy = (
            "they become recipients once someone opens them and sends a first message"
            if plural
            else "it becomes a recipient once someone opens it and sends a first message"
        )
    else:
        remedy = (
            "their owners have to send a first message"
            if plural
            else "its owner has to send a first message"
        )
    return (
        f"{label} {'have' if plural else 'has'} not been engaged yet ({sent}), "
        f"so {'they' if plural else 'it'} {capability} — {remedy}"
    )


def skipped_clause(skipped: "Sequence[Any]") -> str:
    """The receipt's tail when a name/substring send left matches behind.

    A substring match that found a recipient may still have HELD BACK other
    matches for being unengaged, and the sender typed one command believing it
    reached its needle — so the delivery line says so instead of reporting a
    bare success (design round 1, D1). Returns ``""`` for nothing skipped, so a
    caller appends it unconditionally.

    The wording lives HERE, next to :func:`unengaged_refusal`, because both the
    CLI's print and the ``send`` tool's result text append it: a sender reading
    one surface and then the other must not meet two spellings of the same
    fact, the argument that already keeps the refusal one sentence.
    """
    count = len(skipped)
    if not count:
        return ""
    return f"; {count} match{'es' if count != 1 else ''} skipped (not engaged yet)"


def session_has_durable_history(session_id: str, *, root: "Path | None" = None) -> bool:
    """Whether a session's OWN transcript already holds a real turn.

    The cold half of the unengaged gate: a target with no live record still has
    a durable history to ask about, and that history is what distinguishes a
    session someone has used from a composer window that was never typed in.
    The discriminator (only a plain ``Message`` row counts, so a quiet-dialled
    peer note is not history) is
    :func:`local_operator.session.runtime.engagement.durable_conversation_path`,
    read through that stdlib-only module — NOT through ``session.transcript``,
    which would drag pydantic in and break this module's pinned import contract.

    ``root`` defaults to :func:`config_dir` and exists for the tests, which
    drive the store out of a tmp dir. NEVER raises: an absent, unreadable or
    malformed directory answers False — the conservative unengaged direction,
    which the owner's first real turn immediately corrects.
    """
    return _session_has_durable_history(session_id, root=config_dir() if root is None else root)


def unengaged_label(*, session_id: str, pid: "int | None" = None) -> str:
    """How a refusal NAMES its target: the address the caller actually used.

    ``pid 12345`` when the address was a pid — a live record resolved from one,
    or a dial that arrived at a receiver's control port, which is addressed by
    pid — and ``session 'abc'`` when a session id was typed.

    This is the grammar for the sites that name an ADDRESS: the resolver's
    exact ``pid`` branch, its exact ``session`` branch, ``deliver_peer_message``
    (both its live record and its cold raise) and the receive-side gate. The
    two NEEDLE-shaped refusals — the live substring branch and the stored
    withholding — compose their own label on purpose: neither answers an
    address, they answer a name that reached one row (a pid) or one id, and
    saying ``pid N`` there would hide the needle the sender typed.
    """
    if pid is not None:
        return f"pid {pid}"
    return f"session {session_id!r}"


def resolve_peer_target(
    *,
    target: str | None = None,
    pid: int | None = None,
    session: str | None = None,
    pid_hint: str = "an exact pid",
    session_hint: str = "a session id",
    include_wedged: bool = False,
    require_started: bool = True,
    skipped: "list[Any] | None" = None,
    capability: str = PEER_MESSAGE_CAPABILITY,
) -> "tuple[Any | None, list[Any], str]":
    """Resolve a peer-send target to one live :class:`SessionRecord`.

    Priority: ``pid`` (exact), ``session`` (exact session_id), then the ``target``
    substring matched case-insensitively against conversation_name, then
    session_id, then the cwd basename. An ALL-DIGIT ``target`` is tried as a
    pid first: the picker rows, ``lop sessions`` and every disambiguation
    line present the pid as the thing to retype, and a vocabulary whose
    listed form cannot be typed back is a dead end (found by the ``/stop``
    argument picker: every row it offered failed to resolve). Only when no
    record has that pid does the digit string fall through to the substring
    match, so a session id or name that happens to be numeric still works.

    Only ``live`` records are eligible (a record whose owner has stopped
    reporting for ``HEARTBEAT_TIMEOUT_S`` is not one a plain send should assume
    can be reached; ``stale`` is dead) — unless ``include_wedged``, which the
    kill switch passes: a session that is not answering is exactly the one a
    user needs to be able to STOP, and the stop ladder's signal rungs are built
    for a runtime that will not answer. A send never wants that; a message to
    such a runtime is a message that may sit unread.

    A record that has not run a real turn yet (``started`` False — the ``/new``
    composer window) is not a recipient, and ``require_started`` (default True)
    is where every send path enforces that: an EXACT ``pid``/``session``/all-digit
    address is refused with :func:`unengaged_refusal` naming it, and a SUBSTRING
    match skips it. Both halves matter and they fail differently: a broadcast
    that reached an unengaged session would drive an interrupt into a
    conversation nobody started, while an exact send that resolved to one would
    let a caller who named it explicitly land the first row of that history.
    The refusal for a substring match deliberately does NOT use the
    ``no live session matches`` phrasing: that form is what opens the stored
    fallback (``live_scan_found_nothing``), and a refusal about a session the
    live scan DID reach must never fall through to a stored namesake — the same
    wrong-recipient rule as the target+selector conflict below.

    ``require_started=False`` resolves an unengaged record exactly as a started
    one, and exists for the KILL SWITCH alone (``/stop``, ``lop stop`` in
    ``tui/app.py`` and ``cli.py``): a composer window is a session a person may
    need to stop, and a stop names its target rather than messaging it.
    Delivery is not reachable through it — ``deliver_peer_message`` refuses an
    unengaged target whatever the resolver allowed.

    A session that stops being unengaged needs no re-scan: the first real turn
    flips the record's ``started`` bit (``RuntimeServer.set_record_started``,
    published by the owner process), and this reads the record.

    Note what the wedged refusal is and is not. It is a rule about which target to
    DIAL, not a claim that the owner is broken: the beat is authored by the
    runtime's own event loop, so a busy session can read ``wedged`` too
    (``registry.classify``). The cost of being wrong that way is bounded and
    visible — a sender who addresses the session anyway gets the
    unconfirmed-delivery sentence (:func:`_unconfirmed_detail`) rather than
    a silent loss.

    A selector (``pid``/``session``) alongside a ``target`` substring is REFUSED
    rather than resolved. The two name different sessions, and the precedence
    above would silently prefer the selector — delivering to a session the call
    does not appear to name, and reporting success. That is a wrong-recipient
    hazard, not an ergonomic wart, so it lives here in the shared core: a
    conflict rule that existed only in ``cli.py`` would falsify this module's
    claim to be the single source of truth and leave the ``send`` tool teaching
    a different rule from the command.

    ``capability`` names what an unengaged target cannot do in its refusal;
    the model switch passes :data:`MODEL_SWITCH_CAPABILITY` so its sender reads
    about a switch, not about peer messages.

    Returns ``(record, candidates, error)``: exactly one of ``record`` or
    ``error`` is meaningful; ``candidates`` is populated on an ambiguous substring
    so the caller can list them for disambiguation. The shape is identical for the
    CLI and the tool — only how each SURFACES the outcome differs.

    ``skipped`` is an OUT-PARAMETER, not a fourth tuple element, and it is how a
    substring match's HELD-BACK records reach the sender's receipt: a name that
    reached one engaged recipient may have passed over others, and the caller
    that delivered appends :func:`skipped_clause` so the success line says how
    many were left behind (design round 1, D1). Passing a list is opt-in on
    purpose — the three kill-switch callers pass ``require_started=False`` and
    so never hold anything back, an exact address names one record and can hold
    nothing back either, and a caller that ignores the list keeps exactly
    today's behaviour. This is a courtesy to a sender, never a gate: every
    refusal below fires whether or not anyone is listening.

    The conflict wording uses the caller's own ``pid_hint``/``session_hint``
    grammar for the same reason the "no target given" line does. The CLI never
    reaches these branches in practice — ``cli._bind_send_positionals`` refuses
    first, with a richer message that prints two retypeable command lines, and
    argparse's mutually-exclusive group rejects pid+session at parse time — so
    these are the tool's phrasing. Keeping them here anyway is the point: the
    rule holds for ANY caller, including one added later that has no parser in
    front of it.
    """
    # Before any scan: a conflicting address is refused while the call is still
    # inert, so no registry read and no dial can happen on an ambiguous request.
    if pid is not None and session:
        return None, [], f"pass either {pid_hint} or {session_hint}, not both"
    if (pid is not None or session) and (target or "").strip():
        return (
            None,
            [],
            (
                "pass either a target substring or an exact pid/session, not both — "
                "they name different sessions"
            ),
        )

    scanned = registry.scan(config_dir())
    eligible = ("live", "wedged") if include_wedged else ("live",)
    live = [(rec, state) for rec, state in scanned if state in eligible]

    def resolve_scanned_pid(requested_pid: int) -> tuple[Any | None, list[Any], str]:
        """Resolve an exact pid without reading the registry a second time."""
        for rec, state in scanned:
            if rec.pid == requested_pid:
                if state not in eligible:
                    return None, [], _not_dialable(f"target pid {requested_pid}", rec, state)
                if require_started and not getattr(rec, "started", True):
                    return (
                        None,
                        [],
                        unengaged_refusal(
                            unengaged_label(pid=requested_pid, session_id=rec.session_id),
                            capability=capability,
                        ),
                    )
                return rec, [], ""
        return None, [], f"no session found with pid {requested_pid}"

    if pid is not None:
        return resolve_scanned_pid(pid)

    if session:
        for rec, state in scanned:
            if rec.session_id == session:
                if state not in eligible:
                    return None, [], _not_dialable(f"target session {session}", rec, state)
                if require_started and not getattr(rec, "started", True):
                    # The label is the SESSION ID here, because that is the
                    # address this branch answers: an exact `--session` send
                    # named an id, not a pid.
                    return (
                        None,
                        [],
                        unengaged_refusal(
                            unengaged_label(session_id=session), capability=capability
                        ),
                    )
                return rec, [], ""
        return None, [], f"no session found with session id {session!r}"

    needle_source = (target or "").strip()
    if not needle_source:
        # The hints are the CALLER's own grammar: `lop send` passes `--pid` /
        # `--session` and prints the string a user can retype, while the tool
        # passes its parameter names. Parameterised rather than fixed because
        # the CLI's wording is user-visible and must not drift as a side effect
        # of sharing this code (review round 1, MINOR-2).
        return (
            None,
            [],
            f"no target given (pass a name/substring, {pid_hint}, or {session_hint})",
        )

    if needle_source.isdigit():
        as_pid = int(needle_source)
        if any(rec.pid == as_pid for rec, _state in scanned):
            return resolve_scanned_pid(as_pid)

    needle = needle_source.lower()
    matches: list[Any] = []
    # Live matches held back ONLY because the session has not been engaged yet.
    # Kept rather than dropped so the refusal below can name what it reached and
    # keep the stored fallback closed (see the docstring), and so a DELIVERY can
    # report the held-back count on its receipt through ``skipped``.
    unengaged: list[Any] = []
    for rec, _state in live:
        # A session that has never run a turn is excluded from a
        # BROADCAST/substring match: an interrupt there would drive a turn into
        # a session whose owner has not started it, and a quiet note there would
        # become the opening row of that history. The default True keeps a
        # record that simply lacks the attribute (a test double, a hand-built
        # record) eligible; a pre-field binary's record round-trips through
        # ``from_json``, which reads the ABSENT key as True (old peer behaviour
        # preserved) — see the mixed-version note there. That is exactly why the
        # receive-side gate exists: an older SENDER resolves such a record and
        # dials it, and only the receiver can refuse.
        haystacks = [
            rec.conversation_name or "",
            rec.session_id or "",
            os.path.basename(rec.cwd or ""),
        ]
        if not any(needle in field.lower() for field in haystacks):
            continue
        if require_started and not getattr(rec, "started", True):
            unengaged.append(rec)
            continue
        matches.append(rec)

    if not matches:
        if unengaged:
            # A refusal ABOUT a live session the scan did reach. Deliberately
            # NOT the no-match form: ``live_scan_found_nothing`` reads that
            # phrasing as "try the stored store", and a stored namesake would
            # then be spooled to — a recipient this call never named
            # (BLOCKER-1). A batch states its own count in the label so the
            # sentence's tail can agree with it (design round 1, D3).
            if len(unengaged) == 1:
                label = f"the only live match for {needle_source!r} (pid {unengaged[0].pid})"
            else:
                label = f"{len(unengaged)} live matches for {needle_source!r}"
            return (
                None,
                [],
                unengaged_refusal(label, count=len(unengaged), capability=capability),
            )
        # Distinguish "matched but not live" from "no match at all" so the caller
        # knows whether to wait or to fix the name.
        wedged = [
            rec
            for rec, state in scanned
            if state not in eligible
            and needle
            in (
                f"{rec.conversation_name or ''} {rec.session_id or ''} "
                f"{os.path.basename(rec.cwd or '')}"
            ).lower()
        ]
        if wedged:
            return (
                None,
                [],
                _not_dialable(
                    f"the only match for {needle_source!r} (pid {wedged[0].pid})",
                    wedged[0],
                    "wedged",
                ),
            )
        return None, [], f"no live session matches {needle_source!r}"
    if len(matches) > 1:
        return None, matches, ""
    if skipped is not None:
        # The recipient resolved, so this call is a DELIVERY and the caller is
        # about to print a receipt: hand it the matches the scan held back so
        # that receipt can say a peer was left out (design round 1, D1). Only
        # here — an ambiguous or refused call prints no receipt, so there is
        # nothing for the clause to qualify.
        skipped.extend(unengaged)
    return matches[0], [], ""


def _not_dialable(label: str, record: "Any", state: str) -> str:
    """Why a plain send will not dial this record. Never over-claims.

    ``wedged`` is the record that stopped REPORTING, so the sentence gives the
    measured age — ``registry.classify`` owns that number and the clamp around
    it — and names neither a cause nor a schedule. The wording it replaces
    said "its owner is not responding; try again shortly", which invented a
    timetable: the owner may report again the moment a long turn finishes, and
    it may never, and nothing outside the process can tell the two apart.

    ``stale`` is the pid being gone, and then there is no measurement to quote.
    Shared by all three refusal paths (exact pid, exact session id, substring)
    so one condition reads the same way wherever a user meets it.
    """
    if state == "stale":
        return f"{label} is stale (its pid no longer exists), so nothing can read it"
    age_s = registry.classify(record, check_zombie=False).heartbeat_age_s
    return (
        f"{label} has not reported for {format_duration(age_s)}, so a plain send will "
        f"not dial it; it may report again on its own"
    )


def resolve_cold_session(session: str) -> "str | None":
    """A stored session id addressable even though nothing is running for it.

    ``resolve_peer_target`` matches DISCOVERY RECORDS, which only live sessions
    publish, so before this a note to a session whose terminal was closed had
    no target at all — the very case the quiet mailbox mode is for. An exact
    session id is the only accepted form here on purpose: a substring match
    against on-disk directories needs the conversation names that only the
    picker path reads, which is what :func:`resolve_stored_target` adds.
    Picking the wrong recipient is the one failure this whole path must not
    have.

    Returns the id when its session directory exists, else None.
    """
    if not session or session in (".", "..") or os.path.basename(session) != session:
        return None
    directory = config_dir() / "sessions" / session
    try:
        return session if directory.is_dir() else None
    except OSError:
        return None


class StoredCandidate(NamedTuple):
    """One stored session a substring target could mean.

    The fields mirror the live :class:`SessionRecord`'s identity fields
    (``session_id``/``conversation_name``/``cwd``) so the caller's
    disambiguation code can treat live and stored matches alike. There is no
    ``pid`` — nothing is running — which is exactly why
    :func:`stored_candidate_lines` exists rather than the shared
    :func:`candidate_lines`: a ``pid=`` hint against a stored session is an
    address that can never resolve.
    """

    session_id: str
    conversation_name: str
    cwd: str = ""


#: A stored session's ``cwd`` is deliberately an EMPTY string. There is no
#: cheap on-disk record of where a session was last opened — the only writer
#: is the transcript's ``session`` entry, whose head a bounded scan can miss
#: and whose parse would drag the engine onto an import-guarded path. The
#: exact-id stored resolution (:func:`resolve_cold_session`) already resolves
#: with no cwd and delivers correctly; matching by cwd-basename would be a
#: marginal convenience bought with a fragile scan, so stored candidates match
#: on name and id only.


def live_scan_found_nothing(error: str) -> bool:
    """True only for the live resolver's "no match anywhere" refusal.

    The stored fallback (:func:`resolve_stored_target`'s callers) runs on this
    predicate and NOTHING looser. Every OTHER no-record outcome of
    :func:`resolve_peer_target` is a refusal ABOUT a live session the caller
    did reach — a conflicting ``target``+selector pair, a unique match that is
    wedged, a selector naming a dead session — and each of those must stand as
    the answer: delivering anyway, to a stored session that merely shares the
    name, would send the message to a recipient the call never named
    (review round 1, BLOCKER-1) or spool it behind a wedged process that
    still owns the session (MAJOR-1). ``record is None`` cannot tell those
    apart; only this error form means the live scan genuinely came up empty
    and a stored search is a NEW question rather than a second try at a
    refused one.
    """
    return "no live session matches" in error


def session_id_unowned(error: str) -> bool:
    """True when the live resolver's answer about an exact session id means
    NOTHING OWNS THAT SESSION — the state the stored-exact send exists for.

    Two forms qualify, because they are the two states in which no process is
    behind the conversation at all:

    * the id is UNKNOWN to the scan (``no session found with session id …``):
      nothing published a record, so the store is a NEW question;
    * the record the scan found is STALE (``… is stale (its pid no longer
      exists)``): the process is gone and the sweep is about to reap the
      record, so the conversation is exactly as unowned as the first form, and
      the note must still be spooled for the next runtime that opens it
      (review round 4, MINOR-1 — refusing there turned a one-line cold send
      into "refuse now, retry after the reap").

    Every OTHER form keeps standing as a refusal, and the difference is what
    each one says about the session:

    * WEDGED (``has not reported for …``) — a live pid that did not answer, so
      re-asking the store would spool a note behind the process that still owns
      the conversation (MAJOR-1);
    * the UNENGAGED refusal — a live session that is deliberately not a
      recipient yet, so a cold delivery would route around the gate
      (BLOCKER-1);
    * a CONFLICTING ``target``+selector pair — the caller never named one
      session unambiguously, so there is nothing to look up: the id in
      ``session`` is half of an address that was refused, and spooling to it
      would deliver to a recipient the call did not name (BLOCKER-1's class,
      review round 4 NIT-4).
    """
    return (
        "no session found with session id" in error
        or "is stale (its pid no longer exists)" in error
    )


def resolve_stored_target(
    needle: str,
    *,
    live_ids: "set[str] | None" = None,
    limit: int = STORED_DISCOVERY_LIMIT,
    root: "Path | None" = None,
) -> "tuple[str | None, list[StoredCandidate], str]":
    """Match a substring against STORED sessions the live scan did not claim.

    The fallback half of send-side discovery: :func:`resolve_peer_target`
    searches discovery records, which only RUNNING sessions publish, so a note
    addressed to a session by the name it had before its terminal was closed
    found nothing at all. This searches the session store the same way the
    ``/resume`` picker lists it — :func:`resume.recent_session_rows` for the
    id/name ordering, one bounded read per row, subagent scratch sessions
    excluded — and applies the same case-insensitive substring rule the live
    path does, over name and session id (a stored session has no recoverable
    cwd — see :class:`StoredCandidate`).

    ``live_ids`` is an optional exclusion for callers that know which session
    ids already have a runtime; it exists for the id-collision case, where a
    stored row's id belongs to a live session and a cold delivery would queue
    behind a process that could have been dialled. The CLI and the tool pass
    none: they enter this fallback only on :func:`live_scan_found_nothing`, so
    no live NAME match can have survived to here, and a stored row and its
    live session read their name from the same transcript — a collision that
    differs by name is a rename-timing edge, not a case worth a second
    registry scan on every stored send to catch.

    A row whose session has NO durable history is skipped rather than returned:
    such a session was never engaged, so it is not a recipient either — the
    stored half of the same rule the live resolver applies with
    ``require_started``. The read is one :func:`session_has_durable_history`
    pass per MATCHING row only, and the no-match answer is unchanged.

    Returns ``(session_id, candidates, error)`` shaped like
    :func:`resolve_peer_target`'s triple for symmetry, with one deliberate
    difference: a plain no-match still leaves ``error`` empty. A plain no-match
    is not this function's fact to report — the caller just watched the live
    scan miss, so the refusal it owes the user names BOTH searches ("searched
    live and stored sessions"), a sentence only the caller can say. What is this
    function's fact to report is the OTHER empty answer: rows that DID match and
    were withheld for being unengaged (review round 1, F-4). The caller's
    "no session matches '<needle>'" sentence would be false there — a session
    does answer to that name, it is merely not a recipient yet — so the refusal
    names the withheld row and says why, and the callers pass it through instead
    of composing their own miss. The first or the second element is meaningful,
    never both; the resolved id feeds the EXISTING :func:`resolve_cold_session`
    / :func:`deliver_peer_message` path — this function decides WHO, never HOW a
    message is delivered.
    """
    from local_operator.resume import recent_session_rows

    directory = config_dir() if root is None else root
    try:
        rows = recent_session_rows(directory, limit)
    except Exception:  # noqa: BLE001 — discovery must never refuse a send
        return None, [], ""
    needle_folded = needle.strip().lower()
    if not needle_folded:
        return None, [], ""
    excluded = live_ids or set()
    matches: list[StoredCandidate] = []
    # Rows that answered to the needle and were held back ONLY for being
    # unengaged. Kept so the refusal below can name them: this is not a
    # no-match, and reporting it as one would be a false statement about a
    # session the user can see on the picker.
    withheld: list[StoredCandidate] = []
    for row in rows:
        if row.id in excluded:
            continue
        candidate = StoredCandidate(session_id=row.id, conversation_name=row.name)
        haystacks = [candidate.conversation_name, candidate.session_id]
        if any(needle_folded in field.lower() for field in haystacks):
            # Hold back the never-engaged row instead of resolving onto it: a
            # broadcast must not be able to reach a conversation nobody started
            # even through the stored fallback. The read happens ONLY here, on
            # a row that already matched, so the scan stays one bounded name
            # read per row for everything that does not match.
            if not session_has_durable_history(row.id, root=directory):
                withheld.append(candidate)
                continue
            matches.append(candidate)
    if not matches:
        if withheld:
            # Named in the same grammar the live substring branch uses, with the
            # session id the user would retype rather than a pid a stored
            # session cannot satisfy.
            if len(withheld) == 1:
                label = (
                    f"the only stored match for {needle!r} " f"(session {withheld[0].session_id!r})"
                )
            else:
                label = f"{len(withheld)} stored matches for {needle!r}"
            # ``cold=True``: a stored row has no runtime anyone can type into,
            # so the remedy is the conversation being opened and used, not the
            # owner sending into a window that is already there (D5).
            return None, [], unengaged_refusal(label, count=len(withheld), cold=True)
        return None, [], ""
    if len(matches) > 1:
        return None, matches, ""
    return matches[0].session_id, [], ""


def stored_candidate_lines(
    candidates: "list[StoredCandidate]", *, indent: str = "", prefix: str = "session"
) -> "list[str]":
    """One disambiguation line per ambiguous STORED candidate.

    The stored sibling of :func:`candidate_lines`, which prints ``pid=`` hints
    a stored session can never satisfy — there is no pid. Same layout, same
    caller-controlled separator convention, but the address shown is the
    session id: the one form of a stored session's name that always resolves.
    """
    lines: list[str] = []
    gap = "" if prefix.endswith("=") else " "
    id_w = max(len(c.session_id) for c in candidates)
    for c in candidates:
        name = c.conversation_name or "(unnamed)"
        lines.append(f"{indent}{prefix}{gap}{c.session_id:>{id_w}}  {name}  (not running)")
    return lines


async def _spool_quiet_note(
    session_id: str,
    *,
    text: str,
    mode: str,
    sender: "dict[str, Any]",
    message_id: str = "",
) -> str:
    """Spool one message for a session with NO live runtime. Returns receipt.

    The single spool writer for ``deliver_peer_message``'s cold branch: one
    ``O_APPEND`` row under the session's directory, consumed by the runtime
    child's boot drain (``process._drain_inbox_into``) the next time that
    session opens a runtime — or, for a session that has not been engaged yet,
    by the first real turn (``Session._drain_spooled_peer_inbox``). Only a
    session with durable history ever reaches this writer: a cold target with
    NO history is refused by the caller, so nothing here can become the opening
    row of a conversation nobody started.
    """
    from local_operator.session.runtime.inbox import InboxLine, append_inbox

    directory = config_dir() / "sessions" / session_id
    written = await asyncio.to_thread(
        append_inbox,
        directory,
        InboxLine(
            text=text,
            sender=dict(sender),
            mode=mode,
            written_at=time.time(),
            # The sender's identity rides the spool so the drain can dedupe a
            # row that a crash re-delivered (design note C). Empty for a caller
            # that minted none, which reads exactly as a pre-field row.
            message_id=message_id,
        ),
    )
    if not written:
        raise RuntimeError("could not spool the message for that session")
    # The SAME two receipts the draining runtime answers with, imported rather
    # than restated: a sender deciding whether to re-issue must read one
    # vocabulary whether the target was cold or draining (design round 1, D4).
    from local_operator.session.runtime.inbox import SPOOL_RECEIPT_NOTE

    return SPOOL_RECEIPT_NOTE


class DeliveryFailed(RuntimeError):
    """Nothing durable landed, and the evidence PROVES it (design note A.1).

    Raised by the ``-> str`` wrapper :func:`deliver_peer_message` so its existing
    callers keep their ``except RuntimeError`` arm, and by nothing else: the
    outcome builder RETURNS this state instead of raising, because a caller that
    needs the message id and the cause cannot read them off an exception message.
    """


def peer_message_id_capable(record: "Any") -> bool:
    """Whether ``record``'s build understands a sender-minted ``message_id``.

    The retry contract's gate (design note C's back-compat clause): an OLD
    receiver has no dedupe, so a second attempt of the same message would be a
    second row in its transcript. Without this string the sender makes ONE
    attempt and never probes, which is exactly today's behaviour.

    A record with no ``capabilities`` at all (a test double, a pre-field record)
    answers False -- the conservative side, and the one that cannot duplicate.
    """
    capabilities = getattr(record, "capabilities", None)
    if not isinstance(capabilities, (list, tuple)):
        return False
    return PEER_MESSAGE_ID_CAPABILITY in capabilities


def _live_target_label(record: "Any", session_id: str) -> str:
    """How a receipt names a live peer: what the sender calls it, then its pid."""
    name = getattr(record, "conversation_name", "") or session_id
    return f"{name} (pid {getattr(record, 'pid', '?')})"


def _classify_ack(
    ack: "Any",
    *,
    session_id: str,
    target: str,
    message_id: str,
    wake: bool,
    attempts: int,
    route: str,
) -> DeliveryOutcome:
    """Turn a received ack into an outcome (design note A.3's ack rows).

    ``ack.delivery`` is the receiver's own report; a frame WITHOUT one is a
    receiver from before the carriage, and its contract is unchanged -- it
    commits the row before it acks -- so the ack alone proves delivery. That
    ``None`` branch is what makes a new sender work against an old receiver
    without re-deriving today's semantics.
    """
    detail = str(ack)
    delivery = getattr(ack, "delivery", None)
    wake_state = WAKE_ACKED if wake else WAKE_NOT_REQUESTED
    if not isinstance(delivery, dict):
        return DeliveryOutcome(
            DELIVERY_DELIVERED, detail, message_id, wake_state, attempts, "", route, target
        )
    duplicate = bool(delivery.get("duplicate"))
    committed = bool(delivery.get("committed"))
    queued = bool(delivery.get("queued"))
    if duplicate:
        # A PRIOR ATTEMPT'S ACK WAS THE LOST ONE, and the receiver proved it by
        # answering the re-send with the id it already owns. The row is durable;
        # what the sender cannot know is whether the wake that shared the lost
        # ack ran, so a wake send reports mailbox rather than claiming the turn.
        if wake:
            return DeliveryOutcome(
                DELIVERY_MAILBOX,
                _mailbox_detail(message_id, attempts),
                message_id,
                WAKE_UNCONFIRMED,
                attempts,
                "ack_lost",
                route,
                target,
            )
        return DeliveryOutcome(
            DELIVERY_DELIVERED,
            _ack_lost_detail(message_id, attempts),
            message_id,
            WAKE_NOT_REQUESTED,
            attempts,
            "ack_lost",
            route,
            target,
        )
    if committed or not queued:
        return DeliveryOutcome(
            DELIVERY_DELIVERED, detail, message_id, wake_state, attempts, "", route, target
        )
    # QUEUED IN MEMORY ONLY (a busy steer the receiver has not persisted yet).
    # The ack is honest -- the receiver WILL deliver it -- but nothing durable
    # exists, so the state is unconfirmed and cause names why.
    if probe_transcript_for(message_id, session_id):
        return _probe_found_outcome(
            message_id, target=target, wake=wake, attempts=attempts, route=route
        )
    return DeliveryOutcome(
        DELIVERY_UNCONFIRMED,
        _unconfirmed_detail(message_id, attempts),
        message_id,
        wake_state if wake else WAKE_NOT_REQUESTED,
        attempts,
        "queued_busy",
        route,
        target,
    )


def _probe_found_outcome(
    message_id: str, *, target: str, wake: bool, attempts: int, route: str
) -> DeliveryOutcome:
    """The row IS on disk but no ack explained it (design note A.3's probe row).

    A wake send is ``mailbox`` -- the message landed and the fetch that would
    have run it is unproven. A quiet send is fully ``delivered``: nothing was
    ever going to run, so there is no unacknowledged half.
    """
    if wake:
        return DeliveryOutcome(
            DELIVERY_MAILBOX,
            _mailbox_detail(message_id, attempts),
            message_id,
            WAKE_UNCONFIRMED,
            attempts,
            "no_answer",
            route,
            target,
        )
    return DeliveryOutcome(
        DELIVERY_DELIVERED,
        _ack_lost_detail(message_id, attempts),
        message_id,
        WAKE_NOT_REQUESTED,
        attempts,
        "ack_lost" if attempts > 1 else "no_answer",
        route,
        target,
    )


async def deliver_peer_message_outcome(
    record: "Any | None",
    *,
    session_id: str,
    text: str,
    mode: str,
    wake: bool,
    sender: "dict[str, Any]",
    cwd: str = "",
) -> DeliveryOutcome:
    """Deliver one message and return the HONEST state of that delivery.

    The single source of truth for what a send achieved (see
    :class:`DeliveryOutcome`). It mints the message id once, retries the
    idempotent op against a capable receiver, probes the recipient's transcript
    when the socket stays silent, and classifies what it observed.

    Contracts it keeps rather than replaces: a pre-delivery REFUSAL (an
    unengaged target, cold or live) still raises ``RuntimeError`` with
    :func:`unengaged_refusal`, because no message was ever minted and there is no
    state to report; :func:`deliver_peer_message` stays the ``-> str`` face for
    callers that only want the receipt line.
    """
    message_id = mint_peer_message_id()
    if record is not None:
        # The unengaged gate, at this layer as well as in resolution: a caller
        # can bypass resolution (a hand-built record, a future entry point).
        if not getattr(record, "started", True):
            raise RuntimeError(
                unengaged_refusal(unengaged_label(pid=record.pid, session_id=session_id))
            )
        return await _live_outcome(
            record,
            session_id=session_id,
            message_id=message_id,
            text=text,
            mode=mode,
            wake=wake,
            sender=sender,
        )

    # Cold target: the ONLY signal of engagement is the session's own durable
    # history (see :func:`deliver_peer_message`, which this preserves verbatim).
    if not session_has_durable_history(session_id):
        raise RuntimeError(unengaged_refusal(unengaged_label(session_id=session_id), cold=True))

    target = f"{session_id} (not running)"
    if not wake and mode == "mailbox":
        try:
            detail = await _spool_quiet_note(
                session_id, text=text, mode=mode, sender=sender, message_id=message_id
            )
        except (RuntimeError, OSError):
            # The spool write is SYNCHRONOUS, so a failure here is the one case
            # where a cold send can honestly say nothing landed.
            return DeliveryOutcome(
                DELIVERY_FAILED,
                "its inbox could not be written",
                message_id,
                WAKE_NOT_REQUESTED,
                1,
                "spool_write_failed",
                "spool",
                target,
            )
        return DeliveryOutcome(
            DELIVERY_DELIVERED, detail, message_id, WAKE_NOT_REQUESTED, 1, "", "spool", target
        )

    return await _engaged_outcome(
        session_id,
        target=target,
        message_id=message_id,
        text=text,
        mode=mode,
        wake=wake,
        sender=sender,
        cwd=cwd,
    )


async def _live_outcome(
    record: "Any",
    *,
    session_id: str,
    message_id: str,
    text: str,
    mode: str,
    wake: bool,
    sender: "dict[str, Any]",
) -> DeliveryOutcome:
    """Dial a live peer, retrying the idempotent op inside one hard budget.

    The retry exists because the incident's receiver was ALIVE AND BUSY, not
    gone: its Textual hop could not run inside the sender's 5 s read, and a
    second attempt a moment later is answered. What makes the retry safe is that
    it re-sends the SAME message id, which a capable receiver answers as a
    duplicate; a receiver that does not advertise
    :data:`PEER_MESSAGE_ID_CAPABILITY` gets a single attempt, exactly as before.

    The disk probe runs BETWEEN attempts and short-circuits the loop: the common
    incident shape (a busy terminal that committed the row but could not answer)
    settles at ~5.1 s -- the same latency the sender always paid -- instead of
    spending the whole budget.
    """
    from local_operator.mobile.peer_client import ControlDialFailed, send_peer_message

    target = _live_target_label(record, session_id)
    capable = peer_message_id_capable(record)
    allowed = PEER_SEND_ATTEMPTS if capable else 1
    attempts = 0
    dial_only = True  # no attempt ever reached the wire

    # THE HARD BUDGET (design note C). Three reads plus the two pauses is ~15.8 s,
    # so this can only fire against a pathological socket -- and when it does, the
    # ambiguity is resolved the same way a timeout is: the probe decides between
    # mailbox and unconfirmed, never "failed".
    try:
        async with asyncio.timeout(PEER_SEND_TOTAL_BUDGET_S):
            while attempts < allowed:
                attempts += 1
                try:
                    ack = await send_peer_message(
                        record,
                        text=text,
                        mode=mode,
                        wake=wake,
                        sender=sender,
                        message_id=message_id,
                        deadline_s=PEER_SEND_DEADLINE_S,
                    )
                except ControlDialFailed:
                    # NOTHING WAS WRITTEN, so this attempt cannot have landed.
                    pass
                except (TimeoutError, ConnectionError, OSError):
                    # AMBIGUOUS: the op may already be in the owner's socket buffer.
                    dial_only = False
                    if capable and probe_transcript_for(message_id, session_id):
                        return _probe_found_outcome(
                            message_id, target=target, wake=wake, attempts=attempts, route="live"
                        )
                except RuntimeError as exc:
                    # The peer ANSWERED no (an older registrant, a handle that cannot
                    # receive). Nothing landed, and the peer's own sentence is the
                    # reason.
                    return DeliveryOutcome(
                        DELIVERY_FAILED,
                        str(exc),
                        message_id,
                        _failed_wake(wake),
                        attempts,
                        "peer_refused",
                        "live",
                        target,
                    )
                else:
                    return _classify_ack(
                        ack,
                        session_id=session_id,
                        target=target,
                        message_id=message_id,
                        wake=wake,
                        attempts=attempts,
                        route="live",
                    )
                if attempts >= allowed:
                    break
                await asyncio.sleep(
                    PEER_SEND_RETRY_SLEEPS[min(attempts - 1, len(PEER_SEND_RETRY_SLEEPS) - 1)]
                )
    except TimeoutError:
        # The budget fired mid-loop. Same classification a spent attempt gets.
        dial_only = False

    if dial_only:
        # EVERY attempt failed to open the socket, so nothing was ever written.
        return DeliveryOutcome(
            DELIVERY_FAILED,
            f"pid {getattr(record, 'pid', '?')} refused the connection",
            message_id,
            _failed_wake(wake),
            attempts,
            "dial_refused",
            "live",
            target,
        )
    # Ambiguous: at least one attempt reached the wire and nothing came back, and
    # the probe (re-run after the last attempt) found no row. Not failed -- the
    # op may still sit in a stalled socket buffer -- and not delivered.
    return DeliveryOutcome(
        DELIVERY_UNCONFIRMED,
        _unconfirmed_detail(message_id, attempts),
        message_id,
        WAKE_UNCONFIRMED if wake else WAKE_NOT_REQUESTED,
        attempts,
        "no_answer",
        "live",
        target,
    )


def _failed_wake(wake: bool) -> str:
    """The ``wake`` field for a ``failed`` outcome: no wake ran, by definition."""
    return WAKE_UNCONFIRMED if wake else WAKE_NOT_REQUESTED


async def _engaged_outcome(
    session_id: str,
    *,
    target: str,
    message_id: str,
    text: str,
    mode: str,
    wake: bool,
    sender: "dict[str, Any]",
    cwd: str,
) -> DeliveryOutcome:
    """Start a runtime for a cold target and deliver over its socket.

    ONE engage call, not a retry loop: ``engage_runtime`` already owns the
    spawn/deadline loop (and its lease is what makes a second caller safe), so
    wrapping it in this module's three attempts would multiply spawns rather than
    retries. The message id still rides the errand's existing ``command_id``, so
    the child's receive path dedupes a successor that delivers it twice.
    """
    from local_operator.mobile.peer_client import ControlDialFailed
    from local_operator.session.runtime.launch import PeerMessageErrand, engage_runtime

    try:
        outcome = await engage_runtime(
            session_id,
            cwd or os.path.expanduser("~"),
            PeerMessageErrand(
                text=text, mode=mode, wake=wake, sender=dict(sender), command_id=message_id
            ),
            config_dir=config_dir(),
        )
    except (TimeoutError, ConnectionError, OSError):
        if probe_transcript_for(message_id, session_id):
            return _probe_found_outcome(
                message_id, target=target, wake=wake, attempts=1, route="engaged"
            )
        return DeliveryOutcome(
            DELIVERY_UNCONFIRMED,
            _unconfirmed_detail(message_id, 1),
            message_id,
            WAKE_UNCONFIRMED if wake else WAKE_NOT_REQUESTED,
            1,
            "no_answer",
            "engaged",
            target,
        )
    except ControlDialFailed:
        return DeliveryOutcome(
            DELIVERY_FAILED,
            "no runtime accepted the message",
            message_id,
            _failed_wake(wake),
            1,
            "dial_refused",
            "engaged",
            target,
        )
    except RuntimeError as exc:
        return DeliveryOutcome(
            DELIVERY_FAILED,
            str(exc),
            message_id,
            _failed_wake(wake),
            1,
            "peer_refused",
            "engaged",
            target,
        )
    ack = outcome.detail
    return _classify_ack(
        ack,
        session_id=session_id,
        target=target,
        message_id=message_id,
        wake=wake,
        attempts=1,
        route="engaged",
    )


async def deliver_peer_message(
    record: "Any | None",
    *,
    session_id: str,
    text: str,
    mode: str,
    wake: bool,
    sender: "dict[str, Any]",
    cwd: str = "",
) -> str:
    """Hand one message to a peer session and return the receipt LINE.

    The ``-> str`` face of :func:`deliver_peer_message_outcome`, kept because
    external callers (and their tests) only want a sentence out of this module.
    It preserves the split that predates the delivery states, and still does:

    - a **pre-delivery REFUSAL** (an unengaged target, live or cold) raises
      ``RuntimeError`` carrying :func:`unengaged_refusal` — no message id was
      ever minted, so there is nothing to classify;
    - a **proven non-delivery** raises :class:`DeliveryFailed` (a
      ``RuntimeError``), which is the only arm that may say nothing landed;
    - every other outcome RETURNS its sentence, including the two amber states
      (``mailbox``, ``unconfirmed``) whose whole point is that the sender does
      not know the message failed and must not be told that it did.

    Callers that need the state, the message id or the cause call the outcome
    builder directly — an exception message is not a data channel.
    """
    outcome = await deliver_peer_message_outcome(
        record,
        session_id=session_id,
        text=text,
        mode=mode,
        wake=wake,
        sender=sender,
        cwd=cwd,
    )
    if outcome.state == DELIVERY_FAILED:
        raise DeliveryFailed(outcome.detail)
    return outcome.detail


def candidate_lines(
    candidates: "list[Any]", *, indent: str = "", prefix: str = "pid"
) -> "list[str]":
    """One disambiguation line per ambiguous candidate.

    ``prefix`` names the addressing knob in the caller's own grammar, and it
    carries its own separator: the CLI wants the flag form ``--pid 48213`` (a
    space, so it can be retyped at a shell) and the tool wants the parameter
    form ``pid=48213`` (no space, so a model can copy it into an argument). The
    row content (pid, name, model) is identical so both read the same registry
    truth.
    """
    lines: list[str] = []
    # A prefix that already ends in its own separator (``pid=``) is joined
    # tight; a bare flag name takes the space a shell command needs.
    gap = "" if prefix.endswith("=") else " "
    pid_w = max(len(str(rec.pid)) for rec in candidates)
    for rec in candidates:
        name = rec.conversation_name or rec.session_id
        lines.append(f"{indent}{prefix}{gap}{rec.pid:>{pid_w}}  {name}  ({rec.model_label})")
    return lines


def validate_peer_body(text: str) -> "str | None":
    """Return an error message when ``text`` is not deliverable, else ``None``.

    The two refusals are worded exactly as the CLI has always worded them, because
    the tool now answers with the same strings and a user reading either surface
    should get one vocabulary.
    """
    if not text.strip():
        return "message is empty"
    size = len(text.encode("utf-8"))
    if size > PEER_MESSAGE_MAX_BYTES:
        return f"message is too large ({size} bytes); cap is {PEER_MESSAGE_MAX_BYTES} bytes"
    return None


#: How far up the process tree to look for the owning session. `lop send` is
#: USUALLY a direct child of the TUI, but not always: run from a subagent's
#: bash tool, through a shell wrapper, under nohup, or after a reparent, the
#: session is a grandparent or higher (and a reparented process's ppid is 1).
#: Bounded so a pathological tree cannot turn identity lookup into a walk, and
#: because a session more than a few hops up is not plausibly the sender.
_ANCESTRY_MAX_HOPS = 8


def _parent_pid(pid: int) -> "int | None":
    """The parent of ``pid``, or ``None`` when it cannot be determined.

    Uses ``ps`` because it is the one answer available on both macOS and Linux
    without a dependency; ``/proc`` does not exist on macOS and ``psutil`` is not
    a hard requirement of this package. Every failure mode (no such process, a
    ``ps`` that is missing or slow, unparseable output) degrades to ``None``,
    which simply ends the walk — identity is advisory and must never block a
    send.
    """
    if pid <= 1:
        return None
    try:
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    if not text:
        return None
    try:
        parent = int(text.split()[0])
    except (ValueError, IndexError):
        return None
    return parent if parent > 0 else None


def _record_for_pid(pid: int) -> "Any | None":
    """The LIVE registry record published by ``pid``, or ``None``.

    Only ``live`` records count. ``scan`` also returns ``wedged`` (pid alive but
    the heartbeat has aged out) and ``stale`` (pid gone) entries, and labelling
    a message from one of those is worse than leaving it unlabelled: a pid the
    OS has since reused would attribute the message to whatever session happens
    to hold that number now, and that attribution reaches the model-visible
    provenance envelope, not just the card. ``resolve_peer_target`` filters to
    live for the same reason twenty lines up; enrichment must not be laxer than
    the resolver.

    Never raises — see :func:`resolve_sender_identity` for why every failure
    here has to degrade to "less labelled" rather than propagate.
    """
    try:
        for record, state in registry.scan(config_dir()):
            if record.pid == pid and state == "live":
                return record
    except Exception:
        # Deliberately broad: a scan reads and parses files written by other
        # processes, so it can fail in ways beyond OSError (a torn record
        # surfacing as ValueError, a config-path lookup failing). This runs
        # AHEAD of the transcript write on the receive path, so an escaping
        # exception would drop a message that was already accepted on the wire.
        # Identity is a nicety; delivery is not.
        return None
    return None


def _identity_from_record(pid: int, record: "Any") -> "dict[str, Any]":
    """The advisory sender dict for one registry record."""
    return {
        "pid": pid,
        "session_id": record.session_id,
        "conversation_name": record.conversation_name,
        "model_label": record.model_label,
        "cwd": record.cwd,
    }


async def peer_sender_identity_async(lookup_pid: int) -> "dict[str, Any]":
    """``peer_sender_identity`` off the event loop.

    The walk is blocking work — a registry scan per hop plus a ``ps`` per hop,
    typically ~15 ms but bounded only by the subprocess timeout — and callers
    inside a running loop must not stall it. Matches the ``asyncio.to_thread``
    discipline the rest of this package already uses for registry and
    subprocess work.
    """
    return await asyncio.to_thread(peer_sender_identity, lookup_pid)


def peer_sender_identity(lookup_pid: int) -> "dict[str, Any]":
    """Best-effort identity of the sending session for the peer indicator.

    Looks ``lookup_pid`` up in the registry and copies its conversation/model/
    session id so the target's indicator can name the sender honestly. When no
    record is found the process ANCESTRY is walked upward (bounded by
    :data:`_ANCESTRY_MAX_HOPS`, stopping at pid 1) and the first ancestor that
    published a record wins — the sender pid reported is then that ancestor's,
    because the pid on the card must name the session the reader can go and
    talk to, not the transient shell in between.

    Why the walk: testing only the immediate parent made identity fragile in
    exactly the cases that matter. ``lop send`` invoked from a subagent's bash
    tool, through a shell wrapper, or under ``nohup`` is a grandchild or lower,
    so the lookup missed and the card rendered ``peer message from (pid 1)``:
    no name, no model, nothing to follow in a busy transcript.

    What it does NOT fix: a genuinely REPARENTED sender. Once init has adopted
    the process its chain to the session is gone from the process table, so
    there is nothing left to walk and no amount of hops recovers it — that case
    still arrives pid-only. It is covered on the other side instead:
    :func:`resolve_sender_identity` resolves the sender against the receiver's
    own registry, and the card falls back to the cwd basename. This walk claims
    only the intact-chain cases.

    Blocking (a registry scan and a ``ps`` per hop). Callers on an event loop
    must use :func:`peer_sender_identity_async`.

    When nothing is found we still carry the original pid; the identity is
    advisory, never load-bearing for delivery.

    The pid to start from is the CALLER's decision because the two entry points
    run in different processes relative to the session: ``lop send`` is a
    short-lived CHILD of the TUI, so the CLI starts at ``os.getppid()``, while
    the in-session ``send`` tool runs INSIDE the session, so it starts at
    ``os.getpid()`` and matches on the first hop.
    """
    record = _record_for_pid(lookup_pid)
    if record is not None:
        return _identity_from_record(lookup_pid, record)

    pid = lookup_pid
    for _hop in range(_ANCESTRY_MAX_HOPS):
        parent = _parent_pid(pid)
        if parent is None or parent <= 1:
            break
        record = _record_for_pid(parent)
        if record is not None:
            return _identity_from_record(parent, record)
        pid = parent
    return {"pid": lookup_pid}


def resolve_sender_identity(sender: "dict[str, Any] | None") -> "dict[str, Any]":
    """Fill a RECEIVED sender identity in from the local registry.

    The receive side must not have to trust the sender's self-report: the dict
    arrives over the wire and can be empty or partial (the pid-only case a
    failed ancestry lookup produces). The registry is same-account, local, and
    written by the owning process itself, so for a sender running on this
    machine it is the authoritative answer to "who is pid N" — strictly better
    than whatever the sender chose to claim.

    Only ABSENT or blank fields are filled: a sender that named itself keeps its
    own labels (a session that renamed its conversation mid-flight is right
    about itself), and a sender with no record keeps whatever it supplied.

    Genuinely never raises, and that is load-bearing rather than defensive: this
    runs on the receive path AHEAD of the transcript write, on a message the
    wire has already accepted, so an exception escaping here would DROP a
    delivered message. Every failure degrades to the unenriched dict.

    Cheap enough to call inline (one registry scan, no subprocess) — unlike the
    send-side ancestry walk, which needs a thread.
    """
    # The fallback is built BEFORE the risky region, and the handler only
    # returns it. Doing the conversion inside the handler instead meant the
    # recovery path re-ran the very expression that had just thrown — a
    # non-dict `sender` (the wire hands us whatever JSON decoded to) raised
    # TypeError/ValueError from `dict(...)` in the `try`, then raised it again
    # from the `except`, so the exception escaped to the receive path ahead of
    # the transcript write and dropped a delivered message. A handler that can
    # itself fail is not a guarantee.
    fallback: dict[str, Any] = {}
    try:
        if isinstance(sender, dict):
            fallback = dict(sender)
    except Exception:
        # Even this is guarded: `sender` may be a mapping subclass whose copy
        # misbehaves. An unlabelled message still beats a lost one.
        fallback = {}

    try:
        resolved: dict[str, Any] = dict(fallback)
        pid = resolved.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            return resolved
        record = _record_for_pid(pid)
        if record is None:
            return resolved
        for key, value in _identity_from_record(pid, record).items():
            if not str(resolved.get(key) or "").strip():
                resolved[key] = value
        return resolved
    except Exception:
        # Broad on purpose (see above): a malformed record, an unexpected
        # attribute, or a scan fault must cost the label, never the message.
        return fallback


# ---------------------------------------------------------------------------
# Peer model switch — the send half of ``peer_set_model``
# ---------------------------------------------------------------------------


class PeerModelUnconfirmed(Exception):
    """The switch was sent and no answer came back: it may or may not have landed.

    A class of its own rather than a ``TimeoutError``/``OSError`` so neither
    surface can fold it into "nothing changed". The receiver validates, applies
    and only then acks, so a lost ack can hide an applied switch — the same
    contract :func:`deliver_peer_message_outcome` documents for a message.
    """


def parse_model_selector(selector: str) -> "tuple[str, str] | str":
    """``provider/model_id`` split on the FIRST ``/``, or the refusal sentence.

    Syntax only — the one check a sender can make without the target's config
    (design D3). The split is on the first slash exactly as ``/model`` splits
    it, so ``openrouter/deepseek/deepseek-chat`` keeps its own slash in the id.
    The provider is lower-cased like ``/model`` does; the id keeps its case.
    """
    text = (selector or "").strip()
    provider, sep, model_id = text.partition("/")
    provider = provider.strip().lower()
    model_id = model_id.strip()
    if not sep or not provider or not model_id:
        return f"model must be <provider>/<model-id> (e.g. deepseek/deepseek-flash), not {text!r}"
    return provider, model_id


def older_peer_detail() -> str:
    """What an ``unknown op`` from an older build means to the sender (design D7).

    No address in the sentence: every surface prints the address beside it, and
    a sentence that named the pid again printed it twice (QA Q1, UX U6).
    """
    return (
        "older lop: it cannot switch models remotely; nothing changed — "
        "update it (lop update) or run /model in that session"
    )


#: The advice both unconfirmed-switch receipts end on.
_CHECK_BEFORE_RETRYING = "check `lop sessions` before retrying"


def unconfirmed_switch_detail() -> str:
    """A lost ack AFTER the switch op was sent (design D7).

    FORWARD-LOOKING, not "may or may not have landed" (PR #1587 UX round 2,
    U3): the op can still be sitting unread in a stalled target's socket, so a
    `lop sessions` run straight away shows the old model and the switch applies
    afterwards. Two rows, each inside a 60-column terminal.
    """
    return f"no answer — the switch is unconfirmed and may still apply\n{_CHECK_BEFORE_RETRYING}"


def interrupted_switch_detail() -> str:
    """Ctrl-C while ``lop model`` waited (PR #1587 UX round 1, U1).

    The same honesty as :func:`unconfirmed_switch_detail`: the op may already
    sit in the target's socket buffer, and the target applies before it acks,
    so stopping the wait is not stopping the switch.
    """
    return f"interrupted — the switch is unconfirmed and may still apply\n{_CHECK_BEFORE_RETRYING}"


def interrupted_send_detail() -> str:
    """Ctrl-C while ``lop send`` waited: :func:`_unconfirmed_detail`'s rule.

    Once the dial may have written the op, delivery is unconfirmed rather than
    failed, and a re-send is the duplicate steer or wake that sentence warns of.
    """
    return (
        "interrupted — delivery is UNCONFIRMED: it may still arrive, so do not "
        "send it again unless you know it did not land"
    )


def unreachable_switch_detail(error: BaseException) -> str:
    """The control socket never opened, so the op was never sent (review N3)."""
    # Both facts the reader acts on lead; the socket detail trails (design
    # round 2, D10).
    return f"could not reach that session; nothing changed ({error})"


#: How long the sender waits with NOTHING arriving from the target before it
#: gives up on a switch. ABOVE the TUI host's own app-hop budget
#: (``tui_handle._APP_HOP_TIMEOUT_S``, 10 s), because a target that is busy but
#: succeeding must not read as "no answer" (review N2); the same figure as the
#: attach client's ``ACK_TIMEOUT_S``.
#:
#: AN IDLE BOUND, NOT A TOTAL (PR #1587 review round 1, MINOR-3): it is applied
#: to each socket read (``peer_client._FrameReader.next_line``), and a target
#: that pushes projection frames on the connection resets it, so the whole wait
#: can run longer. No user-facing sentence may quote it as "up to 15s".
PEER_MODEL_ACK_TIMEOUT_S = 15.0

#: How long ``lop model`` stays silent before saying it is still waiting (UX
#: round 3, U11). A healthy target answers in well under a second, but a stopped
#: or wedged one holds the command for at least :data:`PEER_MODEL_ACK_TIMEOUT_S`,
#: and 15 s of nothing reads as a hang.
PEER_MODEL_WAIT_NOTICE_S = 2.0


def waiting_for_switch_detail(record: "Any") -> str:
    """The one line ``lop model`` prints while the target has not answered yet.

    Names the target in the receipt's own ``name (pid N)`` grammar. It promises
    NO total: the ack deadline is an idle bound that the target's own pushes
    reset (see :data:`PEER_MODEL_ACK_TIMEOUT_S`), and "up to 15s" was measured
    running to 20 s (review round 1, MINOR-3). Short on purpose: one row at 60
    columns for a name up to 25 characters (design round 1, D13).
    """
    name = record.conversation_name or record.session_id
    return f"no answer yet from {name} (pid {record.pid})…"


def not_running_detail(session: str) -> str:
    """A stored/closed session: a cold switch has no owner to apply it (D1, D4)."""
    return (
        f"session {session!r} is not running — open it and use /model, or "
        f"`lop --resume {session} --hosting <provider> --model <model-id>`"
    )


async def switch_peer_model(
    record: "Any",
    *,
    provider: str,
    model_id: str,
    sender: "dict[str, Any]",
) -> str:
    """Ask ``record``'s LIVE session to switch model; return its receipt.

    The receipt is the RECEIVER's own sentence (design §2): it validated the
    pair against its own config and credentials, applied it through its own
    switch, and read back what is actually in force. This side adds nothing but
    the address.

    Raises ``RuntimeError`` when the target answered no — a refusal (its
    ``refused: …; still on …``), an unengaged or incapable handle, or an OLDER
    build that does not know the op, which is translated here into
    :func:`older_peer_detail` because its raw ``unknown op`` text says nothing
    about what the sender should do, and when the socket could not be opened
    (nothing was sent, so nothing changed). Raises :class:`PeerModelUnconfirmed`
    only when the op was written and the target's socket then stayed silent for
    :data:`PEER_MODEL_ACK_TIMEOUT_S` — an idle bound per read, not a total, so a
    target that keeps pushing frames can hold the call longer.

    Live, started records only: resolution already refused the rest, and this
    re-checks ``started`` for a caller that bypassed it, exactly as
    :func:`deliver_peer_message` does.
    """
    from local_operator.mobile.peer_client import ControlDialFailed, send_control_op

    if not getattr(record, "started", True):
        label = unengaged_label(pid=record.pid, session_id=record.session_id)
        raise RuntimeError(unengaged_refusal(label, capability=MODEL_SWITCH_CAPABILITY))
    try:
        return await send_control_op(
            record,
            "peer_set_model",
            {"provider": provider, "model_id": model_id, "sender": sender},
            deadline_s=PEER_MODEL_ACK_TIMEOUT_S,
            default_detail=f"switched to {provider}/{model_id}",
            default_error="the switch was refused",
        )
    except ControlDialFailed as exc:
        # Before the ConnectionError arm below, which it subclasses: nothing was
        # sent, so this one CAN say nothing changed.
        raise RuntimeError(unreachable_switch_detail(exc)) from exc
    except RuntimeError as exc:
        # D7: an older registrant's dispatch raises ``unknown op: 'peer_set_model'``.
        # Matched on the prefix AND the op name, so an unrelated refusal that
        # happens to quote a word is not rewritten.
        if str(exc).startswith("unknown op") and "peer_set_model" in str(exc):
            raise RuntimeError(older_peer_detail()) from exc
        raise
    except (ConnectionError, OSError) as exc:
        # After the op frame was written (a dial failure is caught above).
        # TimeoutError is an OSError subclass. Either way there is no
        # acknowledged result — never "nothing changed", because the op may
        # already sit in the target's socket buffer.
        raise PeerModelUnconfirmed(unconfirmed_switch_detail()) from exc


def resolve_switch_target(
    *,
    target: "str | None",
    pid: "int | None",
    session: "str | None",
    pid_hint: str = "an exact pid",
    session_hint: str = "a session id",
) -> "tuple[Any | None, list[Any], str]":
    """Resolve a model-switch address to ONE live, engaged record (design D4).

    ``resolve_peer_target`` with the switch's own refusal wording, plus one
    step for a name only the store answers to. A live record's name can lag a
    rename by a heartbeat, so a just-renamed session's name resolves on disk
    first (UX round 1, U1): the stored id is therefore asked of the LIVE
    registry before it is called "not running". A live owner is switched
    through its record; only a session no process owns gets the "open it and
    use /model" sentence.

    Returns ``resolve_peer_target``'s triple. Blocking (registry and directory
    scans): callers run it off the loop.
    """
    record, candidates, error = resolve_peer_target(
        target=target,
        pid=pid,
        session=session,
        pid_hint=pid_hint,
        session_hint=session_hint,
        capability=MODEL_SWITCH_CAPABILITY,
    )
    if record is not None or candidates:
        return record, candidates, error
    if session and session_id_unowned(error):
        stored = resolve_cold_session(session) or ""
        return None, [], not_running_detail(stored) if stored else error
    if (target or "").strip() and live_scan_found_nothing(error):
        stored_id, stored_candidates, _withheld = resolve_stored_target(target or "")
        if stored_candidates:
            # Several stored namesakes: naming one would pick a recipient the
            # call did not name, so the count is the answer.
            return (
                None,
                [],
                f"{len(stored_candidates)} stored sessions match {target!r} and none is "
                "running — open the one you mean and use /model",
            )
        if stored_id:
            live, _ignored, live_error = resolve_peer_target(
                session=stored_id, session_hint=session_hint, capability=MODEL_SWITCH_CAPABILITY
            )
            if live is not None:
                return live, [], ""
            if not session_id_unowned(live_error):
                # A live owner that refused (unengaged, wedged): its own answer.
                return None, [], live_error
            return None, [], not_running_detail(stored_id)
    return None, [], error


def switch_receipt(record: "Any", detail: str) -> str:
    """How BOTH surfaces print a switch: the target's lines, then the address.

    OUTCOME FIRST (design round 1, D1/D2): the sender's TUI card clips a line
    from the right, and a leading ``pid N 'name':`` prefix spent the cells the
    outcome needed while repeating what the card's own argument lines show. The
    address therefore closes the receipt, in ``lop send``'s grammar
    (``→ name (pid N)``, D4), so one terminal reads one vocabulary.
    """
    name = record.conversation_name or record.session_id
    return f"{detail.rstrip()}\n→ {name} (pid {record.pid})"


def switch_outcome(detail: str) -> str:
    """The machine word for a switch receipt: the card's collapsed-row key (D6).

    Read off the receipt's FIRST WORDS, which ``mobile/peer_model`` composes to
    differ per outcome; a receipt from a target that phrases it differently
    (a future build) maps to ``""`` and the card keeps its argument summary.
    """
    from local_operator.mobile.peer_model import PARTIAL_SWITCH_LEAD

    first = detail.lstrip().split("\n", 1)[0]
    if first.startswith("already on "):
        return "unchanged"
    if first.startswith("pending:"):
        return "pending"
    if first.startswith(("switched to ", "back on ")):
        return "partial" if PARTIAL_SWITCH_LEAD in detail else "switched"
    return ""
