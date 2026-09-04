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
import subprocess
import time
from typing import Any

from local_operator.paths import config_dir
from local_operator.session.runtime import registry

#: Peer messaging body cap. Well under the registrant's 1 MB line limit so a huge
#: paste is rejected with a clear message here rather than becoming a silently
#: dropped oversized line on the wire. Shared by the CLI and the tool so both
#: refuse the same size with the same words.
PEER_MESSAGE_MAX_BYTES = 256 * 1024


def resolve_peer_target(
    *,
    target: str | None = None,
    pid: int | None = None,
    session: str | None = None,
    pid_hint: str = "an exact pid",
    session_hint: str = "a session id",
    include_wedged: bool = False,
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

    Only ``live`` records are eligible (a ``wedged`` owner will not service
    the socket promptly; ``stale`` is dead) — unless ``include_wedged``,
    which the kill switch passes: a wedged session is exactly the one a user
    needs to be able to STOP, and the stop ladder's signal rungs are built
    for an owner that will not answer. A send never wants that; a message to
    a wedged owner is a message nobody reads.

    A selector (``pid``/``session``) alongside a ``target`` substring is REFUSED
    rather than resolved. The two name different sessions, and the precedence
    above would silently prefer the selector — delivering to a session the call
    does not appear to name, and reporting success. That is a wrong-recipient
    hazard, not an ergonomic wart, so it lives here in the shared core: a
    conflict rule that existed only in ``cli.py`` would falsify this module's
    claim to be the single source of truth and leave the ``send`` tool teaching
    a different rule from the command.

    Returns ``(record, candidates, error)``: exactly one of ``record`` or
    ``error`` is meaningful; ``candidates`` is populated on an ambiguous substring
    so the caller can list them for disambiguation. The shape is identical for the
    CLI and the tool — only how each SURFACES the outcome differs.

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

    if pid is not None:
        for rec, state in scanned:
            if rec.pid == pid:
                if state not in eligible:
                    return (
                        None,
                        [],
                        (
                            f"target pid {pid} is {state}, not live "
                            "(its owner is not responding); try again shortly"
                        ),
                    )
                return rec, [], ""
        return None, [], f"no session found with pid {pid}"

    if session:
        for rec, state in scanned:
            if rec.session_id == session:
                if state not in eligible:
                    return None, [], (f"target session {session} is {state}, not live")
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
            return resolve_peer_target(
                pid=as_pid,
                pid_hint=pid_hint,
                session_hint=session_hint,
                include_wedged=include_wedged,
            )

    needle = needle_source.lower()
    matches: list[Any] = []
    for rec, _state in live:
        haystacks = [
            rec.conversation_name or "",
            rec.session_id or "",
            os.path.basename(rec.cwd or ""),
        ]
        if any(needle in field.lower() for field in haystacks):
            matches.append(rec)

    if not matches:
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
                (
                    f"the only match for {needle_source!r} is not responding "
                    f"(pid {wedged[0].pid}); try again shortly"
                ),
            )
        return None, [], f"no live session matches {needle_source!r}"
    if len(matches) > 1:
        return None, matches, ""
    return matches[0], [], ""


def resolve_cold_session(session: str) -> "str | None":
    """A stored session id addressable even though nothing is running for it.

    ``resolve_peer_target`` matches DISCOVERY RECORDS, which only live sessions
    publish, so before this a note to a session whose terminal was closed had
    no target at all — the very case the quiet mailbox mode is for. An exact
    session id is the only accepted form here on purpose: a substring match
    against on-disk directories has no conversation name to match on and could
    silently pick the wrong transcript, and picking the wrong recipient is the
    one failure this whole path must not have.

    Returns the id when its session directory exists, else None.
    """
    if not session or session in (".", "..") or os.path.basename(session) != session:
        return None
    directory = config_dir() / "sessions" / session
    try:
        return session if directory.is_dir() else None
    except OSError:
        return None


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
    """Hand one message to a peer session, running or not. Returns the receipt.

    Three cases, and the split between them is the whole point:

    - **A live record** — dial it, exactly as before.
    - **No runtime, quiet note** (``wake=False`` and mailbox mode) — SPOOL it.
      Starting a runtime here would contradict what the sender asked for:
      ``wake=False`` means "read this on your next turn", not "start one now",
      and a 283 MB process for a note nobody is waiting on is the wrong trade.
      The runtime drains the spool the next time the session opens.
    - **No runtime, but the sender wants attention** (``wake=True``, or a
      steer) — engage a runtime and deliver over its socket. The sender is
      explicitly asking the peer to act, which cannot happen without a process.
    """
    from local_operator.mobile.peer_client import send_peer_message

    if record is not None:
        return await send_peer_message(record, text=text, mode=mode, wake=wake, sender=sender)

    if not wake and mode == "mailbox":
        from local_operator.session.runtime.inbox import InboxLine, append_inbox

        directory = config_dir() / "sessions" / session_id
        written = await asyncio.to_thread(
            append_inbox,
            directory,
            InboxLine(text=text, sender=dict(sender), mode=mode, written_at=time.time()),
        )
        if not written:
            raise RuntimeError("could not spool the message for that session")
        return "spooled (will be read when the session next opens)"

    from local_operator.session.runtime.launch import PeerMessageErrand, engage_runtime

    outcome = await engage_runtime(
        session_id,
        cwd or os.path.expanduser("~"),
        PeerMessageErrand(text=text, mode=mode, wake=wake, sender=dict(sender)),
        config_dir=config_dir(),
    )
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
