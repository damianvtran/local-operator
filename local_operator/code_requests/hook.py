"""The live detection seam: classify a tool result and persist the fact as it happens.

WHY LIVE, when ``scan.py`` can derive everything from the transcript later? Because the
transcript is not a complete record of what a session did. Compaction blanks tool
RESULTS on disk (``session/transcript.py``'s ``_pruned_entry``, made permanent by
``compact_file``), so a URL that only ever appeared in a ``gh pr create`` result can be
gone from the file by the time a scan looks — while a bare ``gh pr create`` result is
around 30 tokens and survives, a compound ``git push && gh pr create`` result does not.
The scan therefore misses some opens, and this seam does not. The scan is still needed
(backfill, resume, forks), which is why the two exist together rather than one instead
of the other.

WHERE IT RUNS. ``Session._run_post_tool_hooks`` — the seam the harness already calls for
every result a tool actually returned (``harness/loop.py``'s ``_apply_post_tool_hooks``;
denied, skipped and unknown-tool results never reach it). Nothing here changes what that
hook RETURNS: this module writes a ledger row and returns no notes, so the tool result
the model sees is byte-identical to what it would have been. The user-facing line about
a tracked row is PR1b's, along with the recommendation trigger.

WHAT IT WRITES. One ``code_request_event.v1`` custom row per detection, appended through
``Transcript.append_custom`` — the same shape ``mesh_credential_binding.v1`` uses, and
deliberately NOT one of the collapsible types: collapsing keeps one newest row per type
and every event is a distinct fact.

SUBAGENT PROPAGATION. A child's open is the parent's business: the operator asked for the
work in the parent's conversation, and a row that only exists in the child's transcript is
invisible to the parent's list. So an ``opened``/``acted`` event is ALSO appended to the
parent's transcript, labelled ``via`` with the job id, label, agent role and the path of
child sessions. Each level re-propagates, so a depth-2 open reads
``via subagent coder › reviewer``. Two constraints:

* the child does NOT also get the parent-relative path of its own event — the child's
  row is its own fact and carries no ``via``;
* propagation is best-effort and never raises: a parent that has already been disposed
  costs the label, not the turn.

COST ON THE HOT PATH. Detection is gated three times before any work happens: the tool
name must be one this module cares about (``bash`` or a bridged ``mcp__`` tool), the
arguments must contain a forge marker, and the result text must look like it carries a
URL or a create's JSON. ``could_matter`` is that gate, and it is a substring test over a
few hundred bytes — measured in the low microseconds, which is what makes running it on
every tool result acceptable. The host context and the MCP server list are loaded off the
event loop and cached; a detection that fires than writes one small row.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from local_operator.code_requests.detect import (
    KIND_ACTED,
    KIND_HINT,
    KIND_OPENED,
    Detection,
    McpServer,
    could_matter,
    detect_tool_result,
    load_mcp_servers,
)
from local_operator.code_requests.refs import (
    EMPTY_CONTEXT,
    HostContext,
    load_host_context,
)

logger = logging.getLogger(__name__)

#: Values under this key are written by the harness for a delegated run: the job id
#: (``Session._job_id``), the label the parent used (``_job_label``), the agent role
#: (``_agent_type``) and the delegation depth. Read with ``getattr`` rather than typed
#: access so this module needs no import of the session runtime — the same reason the
#: ledger is stdlib-only.
_JOB_ID_ATTR = "_job_id"
_JOB_LABEL_ATTR = "_job_label"
_AGENT_TYPE_ATTR = "_agent_type"
_PARENT_ATTR = "_code_request_parent"

#: How many levels an event is propagated up. A guard against a cycle in the child→parent
#: linkage rather than a policy: real delegation is one or two levels deep.
_MAX_PROPAGATION_DEPTH = 4

#: How long the loaded machine context is trusted. It is read once per process minute
#: rather than once per tool result, and a ``gh auth login`` lands within a turn or two.
_CONTEXT_TTL_S = 60.0

_CACHED: dict[str, Any] = {}


def host_context_for(cwd: str | None) -> HostContext:
    """The machine's forge context, cached per cwd for :data:`_CONTEXT_TTL_S`.

    Blocking (a few small file reads and one bounded ``git config``), so the caller
    passes the result of :func:`load_context_async` when it is on the event loop.
    """
    key = cwd or ""
    now = time.monotonic()
    cached = _CACHED.get(key)
    if cached is not None and now - cached[0] < _CONTEXT_TTL_S:
        context: HostContext = cached[1]
        return context
    context = load_host_context(cwd)
    _CACHED[key] = (now, context)
    if len(_CACHED) > 64:
        _CACHED.clear()
        _CACHED[key] = (now, context)
    return context


async def load_context_async(cwd: str | None) -> HostContext:
    """``host_context_for`` on a worker thread, for a caller on the event loop."""
    return await asyncio.to_thread(host_context_for, cwd)


def classify(
    session: Any,
    tool_name: str,
    args: Mapping[str, Any],
    result_text: str,
    *,
    is_error: bool,
    context: HostContext | None = None,
    mcp_servers: Iterable[McpServer] = (),
) -> list[Detection]:
    """Detections for one tool result, or ``[]``. Never raises.

    The gate is deliberately three cheap tests before the classifier: a tool name this
    module does not care about, arguments with no forge word, or a result with no URL-ish
    text all return without tokenising anything.
    """
    try:
        if not could_matter(tool_name, args, result_text):
            return []
        return detect_tool_result(
            tool_name,
            args,
            result_text,
            is_error=is_error,
            context=context if context is not None else EMPTY_CONTEXT,
            mcp_servers=mcp_servers,
        )
    except Exception:  # noqa: BLE001 - a detector must never break a turn
        logger.warning("code-request detection failed for %s", tool_name, exc_info=True)
        return []


def event_details(
    detection: Detection,
    *,
    tool: str = "",
    call_id: str = "",
    at: float | None = None,
) -> dict[str, Any]:
    """The ``code_request_event.v1`` payload details for one detection.

    The shape is the design's, and it is the same one the scanner rebuilds a row from, so
    an event written live and a row derived from the transcript are the same object.
    """
    details: dict[str, Any] = {
        "v": 1,
        "kind": detection.kind,
        "at": round(at if at is not None else time.time(), 3),
    }
    if detection.ref is not None:
        details["ref"] = detection.ref.to_payload()
    evidence: dict[str, Any] = {
        "tool": tool,
        "call_id": call_id,
        "verb": detection.verb,
        "rule": detection.rule,
    }
    if detection.exit is not None:
        evidence["exit"] = detection.exit
    if detection.unverified:
        evidence["unverified"] = True
    details["evidence"] = evidence
    details["act"] = detection.act
    if detection.reason:
        details["reason"] = detection.reason
    if detection.hint:
        details["hint"] = detection.hint
    return details


async def record_detections(
    session: Any,
    detections: Iterable[Detection],
    *,
    tool: str = "",
    call_id: str = "",
    propagate: bool = True,
) -> int:
    """Append one event row per detection. Returns how many were written.

    Never raises: a read-only session directory or a full disk costs the row, not the
    turn — the same contract ``Session._run_post_tool_hooks`` gives the operator's own
    hooks.
    """
    written = 0
    for detection in detections:
        if detection.kind == KIND_HINT:
            # A ``git push`` create-link is recorded as a HINT on the session's row set
            # (the scanner reads it from the transcript) rather than as an event: it
            # names a branch, not a code request, so there is no ref to bind an event to.
            continue
        try:
            transcript = getattr(session, "_transcript", None)
            if transcript is None or getattr(transcript, "directory", None) is None:
                continue
            from local_operator.code_requests.ledger import append_event

            details = event_details(detection, tool=tool, call_id=call_id)
            if await append_event(transcript, details):
                written += 1
            # The CHILD's own row carries no ``via``: ``via`` says "this came from
            # somewhere else", and a child's own list must not read as though its work
            # arrived from another session. The propagation below is the half that does.
            if propagate and detection.kind in (KIND_OPENED, KIND_ACTED):
                via = _via_for(session)
                if via:
                    written += await _propagate_to_parent(session, details, via)
        except Exception:  # noqa: BLE001 - bookkeeping never breaks the turn
            logger.warning("could not record a code-request event", exc_info=True)
    return written


def _via_for(session: Any) -> dict[str, Any] | None:
    """The ``via`` block for a child session, or ``None`` for a top-level one.

    A top-level session's open is its own; only a delegated run needs the label, and the
    label is what makes a parent's row read "via subagent coder".
    """
    job_id = getattr(session, _JOB_ID_ATTR, None)
    if not job_id:
        return None
    # The path starts as this child's own label. Outer levels PREPEND theirs as the event
    # is re-propagated upward, so there is no stored delegation-path attribute to read:
    # one existed as a guess in the first cut and nothing in the tree ever assigned it
    # (review round 1, N1).
    label = str(getattr(session, _JOB_LABEL_ATTR, "") or "")
    path: list[str] = [label] if label else []
    return {
        "job_id": str(job_id),
        "label": label,
        "agent_role": str(getattr(session, _AGENT_TYPE_ATTR, "") or ""),
        "child_session_id": str(getattr(session, "session_id", "") or ""),
        "path": path,
    }


async def _propagate_to_parent(
    session: Any, details: Mapping[str, Any], via: Mapping[str, Any], *, depth: int = 0
) -> int:
    """Append the SAME event to the parent's transcript, and re-propagate upward.

    The parent handle is attached by the delegating side (``harness/subagent.py``) because
    the durable child→parent links are weak: the lane receipt is withdrawn when the runner
    settles and a child's ``origin.json`` historically had no parent id. An event is a
    fact about work the OPERATOR asked for, so it is propagated live to the transcript the
    conversation actually reads.

    EACH LEVEL RE-PROPAGATES, which is what makes a depth-2 open read
    ``via subagent coder › reviewer``: the leaf labels the fact with itself, and every
    level above prepends its own label to the path as it passes the fact along. The depth
    cap is a loop guard, not a policy — a delegation chain that deep does not exist, and
    a cycle in the linkage must not become an infinite write loop.
    """
    parent = getattr(session, _PARENT_ATTR, None)
    if parent is None or depth >= _MAX_PROPAGATION_DEPTH:
        return 0
    try:
        transcript = getattr(parent, "_transcript", None)
        if transcript is None or getattr(transcript, "directory", None) is None:
            return 0
        from local_operator.code_requests.ledger import append_event

        payload = dict(details)
        payload["propagated"] = True
        payload["via"] = dict(via)
        written = 1 if await append_event(transcript, payload) else 0
        parent_via = _via_for(parent)
        if parent_via:
            # The parent's own label goes on the FRONT of the path: the outermost
            # conversation is the one that asked, so it reads first.
            parent_via["path"] = [parent_via["label"]] + list(via.get("path") or ())
            written += await _propagate_to_parent(parent, details, parent_via, depth=depth + 1)
        return written
    except Exception:  # noqa: BLE001 - a disposed parent costs the label, not the turn
        logger.debug("could not propagate a code-request event to the parent", exc_info=True)
        return 0


def attach_parent(child: Any, parent: Any) -> None:
    """Give a child a handle on its parent, for live event propagation.

    Called by the child builder. ``None``/absent is fine — propagation is an enrichment,
    and the scanner still attributes a child's rows through the shared transcript.
    """
    try:
        setattr(child, _PARENT_ATTR, parent)
    except Exception:  # noqa: BLE001 - a frozen or exotic session costs the label only
        logger.debug("could not attach the code-request parent handle", exc_info=True)


def stamp_origin_parent(session_dir: Path, parent_id: str) -> None:
    """Add ``parent=<session_id>`` to a child's ``origin.json``. Best-effort.

    The durable half of the same idea: a later BACKFILL (a scan that predates this
    feature, or a child whose live propagation failed) can attribute a child's rows to
    the parent it came from. ``resume.mark_session_origin`` writes the file, so the
    read-modify-write below keeps whatever the caller already recorded (the subagent
    stamp writes ``label``/``agent``, and losing those would break the ``/resume``
    picker).
    """
    if not parent_id:
        return
    try:
        import json
        import os

        # The file name comes from the module that owns it, so there is one spelling of
        # it in the tree; the import is inside the function because this runs once per
        # delegated child and must not put ``resume`` on the session boot path.
        from local_operator.resume import ORIGIN_NAME

        path = Path(session_dir) / ORIGIN_NAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("parent") == parent_id:
            return
        # WRITTEN THE WAY ``resume.mark_session_origin`` WRITES IT, and for the same
        # documented reason: the marker is bookkeeping ABOUT a session, so the session
        # directory's mtime — which readers use for recency — must not move. That
        # function itself cannot be called here because it REPLACES the payload, and this
        # must MERGE one key into a marker it did not create (the subagent stamp already
        # holds ``label``/``agent``, and losing those breaks the ``/resume`` picker).
        #
        # It is deliberately NOT a tmp+``os.replace``: the guard in
        # ``tests/unit/session/test_no_session_deletion.py`` exists because a rename or
        # replace of a session directory is a deletion by another name, and a marker
        # write has no need to introduce that call shape into a module that never deletes
        # anything (review round 1, Q2). A torn write is survivable by contract — the
        # reader tolerates a truncated marker and reports the session as the user's own.
        try:
            previous = Path(session_dir).stat().st_mtime
        except OSError:
            previous = None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload["parent"] = parent_id
        path.write_text(json.dumps(payload), encoding="utf-8")
        if previous is not None:
            os.utime(session_dir, (previous, previous))
    except Exception:  # noqa: BLE001 - provenance is never a gate
        logger.debug("could not stamp the origin parent for %s", session_dir, exc_info=True)


__all__ = [
    "attach_parent",
    "classify",
    "event_details",
    "host_context_for",
    "load_context_async",
    "load_mcp_servers",
    "record_detections",
    "stamp_origin_parent",
]
