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

import os
from typing import Any

from local_operator.mobile import registry
from local_operator.paths import config_dir

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
) -> "tuple[Any | None, list[Any], str]":
    """Resolve a peer-send target to one live :class:`SessionRecord`.

    Priority: ``pid`` (exact), ``session`` (exact session_id), then the ``target``
    substring matched case-insensitively against conversation_name, then
    session_id, then the cwd basename. Only ``live`` records are eligible (a
    ``wedged`` owner will not service the socket promptly; ``stale`` is dead).

    Returns ``(record, candidates, error)``: exactly one of ``record`` or
    ``error`` is meaningful; ``candidates`` is populated on an ambiguous substring
    so the caller can list them for disambiguation. The shape is identical for the
    CLI and the tool — only how each SURFACES the outcome differs.
    """
    scanned = registry.scan(config_dir())
    live = [(rec, state) for rec, state in scanned if state == "live"]

    if pid is not None:
        for rec, state in scanned:
            if rec.pid == pid:
                if state != "live":
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
                if state != "live":
                    return None, [], (f"target session {session} is {state}, not live")
                return rec, [], ""
        return None, [], f"no session found with session id {session!r}"

    needle_source = (target or "").strip()
    if not needle_source:
        # Worded for BOTH callers: the CLI passes a pid via ``--pid`` and the
        # tool via its ``pid`` parameter, so the hint names the concepts, not
        # either caller's flag grammar.
        return None, [], "no target given (pass a name/substring, an exact pid, or a session id)"

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
            if state != "live"
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


def candidate_lines(
    candidates: "list[Any]", *, indent: str = "", prefix: str = "pid"
) -> "list[str]":
    """One disambiguation line per ambiguous candidate.

    ``prefix`` names the addressing knob in the caller's own grammar: the CLI
    surfaces it as the ``--pid`` flag, the tool as the ``pid`` parameter. The row
    content (pid, name, model) is identical so both read the same registry truth.
    """
    lines: list[str] = []
    for rec in candidates:
        name = rec.conversation_name or rec.session_id
        lines.append(f"{indent}{prefix} {rec.pid}  {name}  ({rec.model_label})")
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


def peer_sender_identity(lookup_pid: int) -> "dict[str, Any]":
    """Best-effort identity of the sending session for the peer indicator.

    Looks ``lookup_pid`` up in the registry and copies its conversation/model/
    session id so the target's indicator can name the sender honestly. When the
    record cannot be found we still carry the pid — the identity is advisory,
    never load-bearing for delivery.

    The pid to look up is the CALLER's decision because the two entry points run
    in different processes relative to the session: ``lop send`` is a short-lived
    CHILD of the TUI that spawned it, so the session is ``os.getppid()`` there,
    while the in-session ``send`` tool runs INSIDE the session process, so the
    session is ``os.getpid()``. This function is pid-agnostic so both stay honest.
    """
    sender: dict[str, Any] = {"pid": lookup_pid}
    try:
        for record, _state in registry.scan(config_dir()):
            if record.pid == lookup_pid:
                sender.update(
                    {
                        "session_id": record.session_id,
                        "conversation_name": record.conversation_name,
                        "model_label": record.model_label,
                        "cwd": record.cwd,
                    }
                )
                break
    except OSError:
        # A scan failure never blocks a send: the message still delivers, it is
        # just less labelled.
        pass
    return sender
