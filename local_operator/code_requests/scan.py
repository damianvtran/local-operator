"""Derive code-request facts from a session's transcript rows. Pure, no I/O.

WHY A SCANNER AT ALL, when the live hook (``hook.py``) already writes an event row
for every create it sees? Because event rows only cover what happened WHILE the
feature existed:

* a session that predates this change has to be backfilled;
* a session RESUMED after a compaction replays its transcript, events and all — but
  compaction blanks tool RESULTS on disk (``transcript._pruned_entry``), so a URL
  that only ever appeared in a ``gh pr create`` result can be gone from the file
  while the event row that recorded it survives (and vice versa, for sessions whose
  create predates the hook);
* a FORK copies its parent's transcript (``fork.py``'s ``COPIED_SIDECARS``), so the
  child inherits the parent's events and must not claim them as its own opens.

So a session's list is derived from two inputs that are both in the transcript: the
``code_request_event.v1`` rows, and the TEXT the session saw.

CLASSIFICATION, and the rules that are not obvious:

* **Events win.** A ref with an ``opened``/``acted`` event is never downgraded by a
  text mention of the same ref; the mention only adds sources and timestamps.
* **Tool-output mentions are COLLECTED but COLLAPSED.** Session ``439818272d84``
  holds 158 distinct PR/MR URLs, most from audit output. They are counted into
  ``tool_output_only`` and are absent from the list unless the caller expands them.
* **A script-created URL is ``unknown``, not ``mentioned``.** A stdout PR URL under
  a command with no recognised verb means *a pull request may have been created by
  this call*, which is the honest word for it (design §A.4). The same ref seen in
  text elsewhere still gets its mention sources.
* **A fork's inherited rows say so.** An event whose ``at`` precedes the fork's
  ``forked_at`` renders as ``inherited``, and the UI is told which parent.
* Rows sort opened → acted → unknown → mentioned → inherited, newest first inside a
  relation — the design's noise budget, so a reader sees the PRs the session
  actually touched above the ones it only read about.

PURE. Nothing here opens a file: the caller streams transcript rows (the scanner's
I/O lives in ``ledger.py``), so every rule in this module is unit-testable against
a list of dicts, and the real-transcript evidence run can drive it over a copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from local_operator.code_requests.detect import (
    KIND_ACTED,
    KIND_HINT,
    KIND_OPENED,
    KIND_UNKNOWN,
    Detection,
    McpServer,
    could_matter,
    detect_tool_result,
)
from local_operator.code_requests.refs import EMPTY_CONTEXT, HostContext, Ref, iter_refs

logger = logging.getLogger(__name__)

#: The transcript custom type the live hook appends, and every reader spells. The
#: ``.v1`` suffix is the shape's own version, so a later payload change is visible in
#: a row rather than implied by the build that wrote it.
EVENT_CUSTOM_TYPE = "code_request_event.v1"

SOURCE_USER = "user"
SOURCE_ASSISTANT = "assistant"
SOURCE_PEER = "peer"
SOURCE_TOOL = "tool"

#: Sources in the order a payload lists them, so the wire is stable for a UI that
#: renders them as chips.
SOURCE_ORDER = (SOURCE_USER, SOURCE_ASSISTANT, SOURCE_PEER, SOURCE_TOOL)

RELATION_OPENED = "opened"
RELATION_ACTED = "acted"
RELATION_MENTIONED = "mentioned"
RELATION_UNKNOWN = "unknown"
RELATION_INHERITED = "inherited"

#: List order, and the whole of the sorting rule: opened first, then acted, then the
#: ones that only might be ours, then mentions, then inherited rows last.
RELATION_ORDER = (
    RELATION_OPENED,
    RELATION_ACTED,
    RELATION_UNKNOWN,
    RELATION_MENTIONED,
    RELATION_INHERITED,
)

#: Custom message types whose ``details`` text is an inbound mention source. All are
#: the operator's own words arriving through the harness rather than remote prose:
#: ``peer_message`` (another session), ``job_result`` (a background job), a wake or
#: monitor delivery. They are counted as ``peer`` so the list can say where a ref was
#: seen without pretending a job wrote the message itself.
_INBOUND_CUSTOM_TYPES = frozenset({"peer_message", "job_result", "wake_prompt", "monitor_prompt"})

#: How many distinct refs a single scan keeps from TOOL OUTPUT only. Session
#: ``439818272d84`` produced 158; the cap exists so a pathological audit dump cannot
#: make one session's row set unbounded, and the overflow is counted rather than
#: silently dropped.
TOOL_MENTION_CAP = 500

#: How many evidence entries one row keeps. The NEWEST are kept: an event's evidence is
#: what a reader audits a row against, and a PR acted on nightly for a month must not
#: grow the row without limit.
MAX_EVIDENCE = 8


@dataclass
class Mention:
    """One source's sightings of a ref: how many, and when the first and last were."""

    source: str
    count: int = 0
    first_at: float = 0.0
    last_at: float = 0.0

    def add(self, at: float) -> None:
        self.count += 1
        if not self.first_at or at < self.first_at:
            self.first_at = at
        if at > self.last_at:
            self.last_at = at

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "count": self.count,
            "first_at": round(self.first_at, 3),
            "last_at": round(self.last_at, 3),
        }


@dataclass
class Row:
    """One code request as this session saw it, before it is written to the index."""

    ref: Ref
    relation: str = RELATION_MENTIONED
    relations: set[str] = field(default_factory=set)
    acts: list[str] = field(default_factory=list)
    mentions: dict[str, Mention] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    via: dict[str, Any] | None = None
    inherited_from: str | None = None
    first_at: float = 0.0
    last_at: float = 0.0
    unknown_reason: str | None = None

    @property
    def tool_only(self) -> bool:
        """Whether every sighting of this ref was tool output.

        The collapsible case: a ``gh pr list`` dump, an audit script's report, a
        dependency's changelog. It cannot be "the session mentioned this PR".
        """
        return bool(self.mentions) and set(self.mentions) == {SOURCE_TOOL}

    def mention(self, source: str, at: float) -> None:
        item = self.mentions.get(source)
        if item is None:
            item = Mention(source=source)
            self.mentions[source] = item
        item.add(at)
        if not self.first_at or at < self.first_at:
            self.first_at = at
        if at > self.last_at:
            self.last_at = at

    def add_evidence(self, item: dict[str, Any]) -> None:
        """Keep the newest :data:`MAX_EVIDENCE` facts about how this row was classified."""
        self.evidence.append(item)
        if len(self.evidence) > MAX_EVIDENCE:
            del self.evidence[: len(self.evidence) - MAX_EVIDENCE]

    def note(self, relation: str, at: float) -> None:
        self.relations.add(relation)
        if relation == RELATION_OPENED:
            self.relation = RELATION_OPENED
        elif relation == RELATION_INHERITED:
            if self.relation != RELATION_OPENED:
                self.relation = RELATION_INHERITED
        elif relation == RELATION_ACTED and self.relation != RELATION_OPENED:
            self.relation = RELATION_ACTED
        elif relation == RELATION_UNKNOWN and self.relation not in (
            RELATION_OPENED,
            RELATION_ACTED,
        ):
            self.relation = RELATION_UNKNOWN
        if at:
            if not self.first_at or at < self.first_at:
                self.first_at = at
            if at > self.last_at:
                self.last_at = at

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.ref.key,
            "ref": self.ref.to_payload(),
            "relation": self.relation,
            "relations": [r for r in RELATION_ORDER if r in self.relations],
            "acts": list(dict.fromkeys(self.acts)),
            "mentions": [
                self.mentions[source].to_payload()
                for source in SOURCE_ORDER
                if source in self.mentions
            ],
            "first_at": round(self.first_at, 3),
            "last_at": round(self.last_at, 3),
        }
        if self.evidence:
            payload["evidence"] = self.evidence
        if self.via:
            payload["via"] = self.via
        if self.inherited_from:
            payload["inherited_from"] = self.inherited_from
        if self.unknown_reason:
            payload["unknown_reason"] = self.unknown_reason
        return payload


@dataclass
class ScanResult:
    """Everything one scan derived: the visible rows, the collapsed ones, and hints.

    ``rows`` is what a list renders by default. ``tool_only_rows`` holds the refs seen
    ONLY in tool output, which are collapsed out of ``rows`` and shown when the caller
    expands them (``?include=mentions_tool``). ``tool_output_only`` is the EXACT count
    of collapsed refs while the stored list is capped at :data:`TOOL_MENTION_CAP`: a
    count and a payload are different promises, and truncating a count would tell a
    reader "158" when the answer is "at least 200".

    ``hints`` are ``git push`` create-links: recorded so a UI can say "this branch has
    an open-PR link", never turned into a row — the link names a branch, not a PR.
    """

    rows: list[Row] = field(default_factory=list)
    tool_only_rows: list[Row] = field(default_factory=list)
    tool_output_only: int = 0
    hints: list[dict[str, Any]] = field(default_factory=list)
    events: int = 0
    #: True when more refs were collapsed than :data:`TOOL_MENTION_CAP` could hold.
    #: The count is then the STORED length, and this flag is what stops it being read
    #: as the exact number of refs the session saw only in tool output.
    tool_only_truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rows": [row.to_payload() for row in self.rows],
            "tool_only_rows": [row.to_payload() for row in self.tool_only_rows],
            "tool_output_only": self.tool_output_only,
            "hints": list(self.hints),
            "events": self.events,
        }
        if self.tool_only_truncated:
            payload["tool_only_truncated"] = True
        return payload


def scan_rows(
    rows: Iterable[Mapping[str, Any]],
    context: HostContext = EMPTY_CONTEXT,
    *,
    mcp_servers: Iterable[McpServer] = (),
    forked_at: float | None = None,
    parent_id: str | None = None,
    tool_mention_cap: int = TOOL_MENTION_CAP,
) -> ScanResult:
    """Derive the visible rows from transcript rows, in order.

    ``forked_at``/``parent_id`` come from the fork's ``origin.json``: an event older
    than the fork is ``inherited`` rather than an open performed here. Both are
    optional so a normal session passes neither.
    """
    result = ScanResult()
    by_key: dict[str, Row] = {}
    calls: dict[str, tuple[str, Mapping[str, Any], float]] = {}
    servers = tuple(mcp_servers)

    def row_for(ref: Ref) -> Row:
        row = by_key.get(ref.key)
        if row is None:
            row = Row(ref=ref)
            by_key[ref.key] = row
        return row

    for entry in rows:
        if not isinstance(entry, Mapping):
            continue
        kind = entry.get("type")
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            continue
        at = _float(entry.get("ts"))
        if kind == "custom":
            _consume_event(payload, row_for, result, at, forked_at=forked_at, parent_id=parent_id)
            continue
        if kind != "message":
            continue
        if payload.get("kind") == "custom":
            _consume_inbound(payload, row_for, at)
            continue
        role = payload.get("role")
        if role == "user":
            for ref in iter_refs(_content_text(payload), context):
                row_for(ref).mention(SOURCE_USER, at)
            continue
        if role == "assistant":
            for ref in iter_refs(_content_text(payload), context):
                row_for(ref).mention(SOURCE_ASSISTANT, at)
            for call in payload.get("tool_calls") or ():
                if not isinstance(call, Mapping):
                    continue
                call_id = call.get("id")
                name = call.get("name")
                args = call.get("arguments")
                if isinstance(call_id, str) and isinstance(name, str) and isinstance(args, Mapping):
                    calls[call_id] = (name, args, at)
            continue
        if role == "tool":
            _consume_tool_row(
                payload, row_for, result, calls, context, servers, by_key, tool_mention_cap
            )
            continue

    for row in by_key.values():
        if row.tool_only and row.relation == RELATION_MENTIONED:
            # Collapsed by default: a ``gh pr list`` dump is not a mention by the
            # session. The stored rows are capped so a scan over a 132 MB journal
            # cannot hold thousands of refs in memory, and the flag says when that
            # cap bound, so the count is never read as exact when it is not.
            result.tool_only_rows.append(row)
            continue
        result.rows.append(row)
    result.rows.sort(key=row_sort_key)
    result.tool_only_rows.sort(key=row_sort_key)
    result.tool_output_only = len(result.tool_only_rows)
    if result.tool_output_only > tool_mention_cap:
        result.tool_only_truncated = True
        result.tool_only_rows = result.tool_only_rows[:tool_mention_cap]
    return result


def row_sort_key(row: Row) -> tuple[int, float]:
    rank = RELATION_ORDER.index(row.relation) if row.relation in RELATION_ORDER else 99
    # Newest first inside a relation, and a stable sort keeps same-instant rows in
    # scan order.
    return (rank, -row.last_at)


def _consume_event(
    payload: Mapping[str, Any],
    row_for: Any,
    result: ScanResult,
    at: float,
    *,
    forked_at: float | None,
    parent_id: str | None,
) -> None:
    if payload.get("custom_type") != EVENT_CUSTOM_TYPE:
        return
    details = payload.get("details")
    if not isinstance(details, Mapping):
        return
    ref = Ref.from_payload(details.get("ref"))
    if ref is None:
        return
    result.events += 1
    row = row_for(ref)
    kind = str(details.get("kind") or "")
    event_at = _float(details.get("at")) or at
    evidence = details.get("evidence")
    if isinstance(evidence, Mapping):
        row.evidence.append(dict(evidence))
    via = details.get("via")
    if isinstance(via, Mapping) and row.via is None:
        row.via = dict(via)
    if forked_at and event_at and event_at < forked_at and kind in (KIND_OPENED, KIND_ACTED):
        row.inherited_from = parent_id or ""
        row.note(RELATION_INHERITED, event_at)
        return
    if kind == KIND_OPENED:
        row.note(RELATION_OPENED, event_at)
    elif kind == KIND_ACTED:
        act = details.get("act")
        if isinstance(act, str) and act and act not in row.acts:
            row.acts.append(act)
        row.note(RELATION_ACTED, event_at)
    elif kind == KIND_UNKNOWN:
        reason = details.get("reason")
        row.unknown_reason = str(reason) if isinstance(reason, str) else None
        row.note(RELATION_UNKNOWN, event_at)


def _consume_inbound(payload: Mapping[str, Any], row_for: Any, at: float) -> None:
    custom_type = payload.get("custom_type")
    if custom_type not in _INBOUND_CUSTOM_TYPES:
        return
    details = payload.get("details")
    if not isinstance(details, Mapping):
        return
    text = details.get("text")
    if not isinstance(text, str) or not text:
        return
    for ref in iter_refs(text):
        row_for(ref).mention(SOURCE_PEER, at)


def _consume_tool_row(
    payload: Mapping[str, Any],
    row_for: Any,
    result: ScanResult,
    calls: Mapping[str, tuple[str, Mapping[str, Any], float]],
    context: HostContext,
    servers: Sequence[McpServer],
    by_key: Mapping[str, Row],
    tool_mention_cap: int,
) -> None:
    """One tool result row: its text is a tool-output mention, its call is classified."""
    text = _content_text(payload)
    at = None
    call_id = payload.get("tool_call_id")
    call = calls.get(call_id) if isinstance(call_id, str) else None
    name = str(payload.get("tool_name") or (call[0] if call else ""))
    args = call[1] if call else {}
    if call is not None:
        at = call[2]
    if text:
        for ref in iter_refs(text, context):
            row_for(ref).mention(SOURCE_TOOL, _event_at(payload, at))
    if not name or not (text or args):
        return
    if not could_matter(name, args, text):
        return
    detections = detect_tool_result(
        name,
        args,
        text,
        is_error=bool(payload.get("is_error")),
        context=context,
        mcp_servers=servers,
    )
    for detection in detections:
        _apply_detection(detection, row_for, result, at, tool=name, call_id=str(call_id or ""))


def _apply_detection(
    detection: Detection,
    row_for: Any,
    result: ScanResult,
    at: float | None,
    *,
    tool: str = "",
    call_id: str = "",
) -> None:
    """Record one detection on its row, with the EVIDENCE a reader audits it against.

    Evidence is not decoration: the risk this feature carries is a false ``opened``, and
    ``rule`` is what makes one traceable back to the rule that fired ("gh-create-stdout"
    vs "script-stdout"). It is capped at the newest :data:`MAX_EVIDENCE` entries by
    :meth:`Row.add_evidence`.
    """
    stamp = at or 0.0
    if detection.kind == KIND_HINT:
        result.hints.append(
            {
                "rule": detection.rule,
                "verb": detection.verb,
                "hint": detection.hint,
                "at": stamp,
            }
        )
        return
    ref = detection.ref
    if ref is None:
        return
    row = row_for(ref)
    evidence: dict[str, Any] = {
        "tool": tool,
        "call_id": call_id,
        "verb": detection.verb,
        "rule": detection.rule,
        "kind": detection.kind,
        "at": round(stamp, 3),
    }
    if detection.exit is not None:
        evidence["exit"] = detection.exit
    if detection.act:
        evidence["act"] = detection.act
    if detection.unverified:
        # The GitHub MCP tool names were never exercised on a live server here, so the
        # claim travels with its own caveat rather than reading like the measured rules.
        evidence["unverified"] = True
    row.add_evidence(evidence)
    if detection.kind == KIND_OPENED:
        row.note(RELATION_OPENED, stamp)
    elif detection.kind == KIND_ACTED:
        if detection.act and detection.act not in row.acts:
            row.acts.append(detection.act)
        row.note(RELATION_ACTED, stamp)
    elif detection.kind == KIND_UNKNOWN:
        # ONLY a ref this session had not seen before. A script that prints a URL
        # already mentioned in the text (or confirmed by an event) is reporting, not
        # creating — and the design's own wording for this rule is "a PR URL that was
        # never seen before". Order is transcript order, so "before" is the state this
        # row holds at this point in the scan.
        seen_before = bool(row.relations & {RELATION_OPENED, RELATION_ACTED}) or bool(
            set(row.mentions) - {SOURCE_TOOL}
        )
        if not seen_before:
            row.unknown_reason = detection.reason or row.unknown_reason
            row.note(RELATION_UNKNOWN, stamp)


def _event_at(payload: Mapping[str, Any], call_at: float | None) -> float:
    """When a tool result landed: the call's own ``ts`` when known, else the row's.

    The tool result row carries no ``ts`` of its own for a streamed call, so the call
    row's timestamp is the honest answer and a zero is never invented.
    """
    return call_at if call_at else 0.0


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _content_text(payload: Mapping[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, Mapping):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


__all__ = [
    "EVENT_CUSTOM_TYPE",
    "Mention",
    "RELATION_ACTED",
    "RELATION_INHERITED",
    "RELATION_MENTIONED",
    "RELATION_OPENED",
    "RELATION_ORDER",
    "RELATION_UNKNOWN",
    "Row",
    "SOURCE_ASSISTANT",
    "SOURCE_ORDER",
    "SOURCE_PEER",
    "SOURCE_TOOL",
    "SOURCE_USER",
    "TOOL_MENTION_CAP",
    "ScanResult",
    "TOOL_MENTION_CAP",
    "row_sort_key",
    "scan_rows",
]
