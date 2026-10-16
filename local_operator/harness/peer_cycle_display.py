"""Specialized display-only proof for finite eval-wrapped peer exchanges.

Nested no_reply is NOT an AgentLoop termination signal. Its actual typed end can
nevertheless close a display exchange once every registered outer/child outcome
succeeded. Only missing nested send DISPLAY rows are stored; raw peer/root bodies
and model messages are never copied or enriched. No Python/stdout text is parsed.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from typing import Any

# Keep live and cold provenance on the harness vocabulary: a function-local
# import in the validator cannot bind the observer, and importing app code here
# would make a body-free display proof depend on the UI it is meant to serve.
from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE

from local_operator.harness.peer_cycles import (
    current_peer_communication_succeeded, is_peer_communication_call,
)
from local_operator.harness.types import (
    FAULT_KEY, OUTPUT_LIMIT_KEY, QUIET_TURN_KEY, AgentEvent,
    Message, MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
    PeerMessageDeliveredEvent, ToolExecutionStartEvent, ToolExecutionEndEvent,
    ProviderTurnStartEvent, ReasoningDeltaEvent, ToolCallComposeEvent, ToolExecutionUpdateEvent,
)

CUSTOM_TYPE = "frontend_peer_cycle_display_v1"
MAX_RECORD_BYTES = 64 * 1024
MAX_SOURCE_IDS = 4096
HOT_RECORDS = 16
# Fixed application namespace, not a session's random view epoch: a restart must
# derive the SAME archive key from a selected durable parent/call identity.
_NAMESPACE = uuid.UUID("10fd2db2-9b97-5e1c-a757-f130fa039753")


def record_id(session_id: str, parent_id: str, quiet_root: str) -> str:
    return uuid.uuid5(_NAMESPACE, json.dumps(
        [session_id, parent_id, quiet_root, CUSTOM_TYPE], separators=(",", ":"))).hex


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def outcome_digest(result: Any) -> str:
    payload = getattr(result, "provider_payload", None)
    details = (payload.get("details") if isinstance(payload, dict)
               else getattr(result, "details", None))
    return digest({"content": [part.model_dump(mode="json") for part in result.content],
                   "details": details, "is_error": result.is_error})


def _successful(result: Any) -> bool:
    details = getattr(result, "details", None)
    if details is None:
        payload = getattr(result, "provider_payload", None)
        details = payload.get("details") if isinstance(payload, dict) else None
    details = details or {}
    return (result.is_error is False and isinstance(details, dict)
            and not any(details.get(key) for key in (
                FAULT_KEY, OUTPUT_LIMIT_KEY, "partial_result", "never_sent", "not_run", "duplicate_suppressed")))


class DisplayCycleAccumulator:
    """One finite candidate and a bounded hot cache; the journal is the archive.

    Event names and root edges are typed producer facts, not inferred from an
    eval's code, a sender label, idle, or later final response. Unknown work
    invalidates the candidate instead of being hidden inside a successful bar.
    """
    def __init__(self, session_id: str, epoch: str, redact: Callable[[str], str] | None = None) -> None:
        self.redact = redact
        self.session_id = session_id
        self.epoch = epoch
        self.records: list[dict[str, Any]] = []
        self._reset()

    def _reset(self, peer_id: str | None = None) -> None:
        self.peer_id = peer_id
        self.roots: dict[str, dict[str, Any]] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.members: list[dict[str, str]] = (
            [{"source_id": f"{peer_id}:0", "role": "peer"}] if peer_id else [])
        self.children: list[dict[str, Any]] = []
        self.quiet: tuple[str, str] | None = None

    def observe(self, event: AgentEvent[Any]) -> dict[str, Any] | None:
        kind = event.type
        if isinstance(event, PeerMessageDeliveredEvent):
            # Delivery shares an event class with receiver-authored bridge
            # cards. Only the declared peer kind (or its legacy empty default)
            # grants ownership; a bridge also ends any earlier peer candidate.
            if event.custom_type not in ("", PEER_MESSAGE_MESSAGE_TYPE):
                self._reset()
                return None
            peer_id = getattr(event, "message_id", None)
            self._reset(peer_id if isinstance(peer_id, str) and peer_id else None)
            return None
        if not self.peer_id:
            return None
        if isinstance(event, (ProviderTurnStartEvent, ReasoningDeltaEvent)):
            # A real provider starts and thinks BEFORE its tool declaration.
            # These typed display scalars grant no ownership, add no members,
            # and never copy reasoning into a record or model message.
            return None
        if isinstance(event, ToolCallComposeEvent):
            # A prediction is not execution. A never-run/replaced prediction
            # stays visible rather than borrowing a later successful outcome.
            if (event.tool_name not in {"eval", "send", "no_reply"}
                    or event.not_run_reason is not None or event.not_run_kind is not None
                    or event.supersedes_tool_call_id is not None):
                self._reset()
            return None
        if isinstance(event, (MessageStartEvent, MessageUpdateEvent, MessageEndEvent)):
            message = getattr(event, "message", None)
            if message is not None and (getattr(message, "role", None) == "user" or (
                    getattr(message, "role", None) == "assistant" and any(
                    getattr(part, "type", None) != "thinking" and (
                        getattr(part, "type", None) != "text" or bool(getattr(part, "text", "")))
                    for part in getattr(message, "content", ())))):
                self._reset()
                return None
            if kind != "message_end" or not isinstance(message, Message) or message.role != "assistant":
                return None
            if not message.tool_calls:
                # Even an empty genuine answer is a boundary, not an implied
                # quiet result or permission to borrow later tool work.
                self._reset()
                return None
            for call in message.tool_calls:
                if (not message.id or call.id in self.calls or
                        not (call.name in {"eval", "no_reply"} or
                             is_peer_communication_call(call.name, call.arguments))):
                    self._reset()
                    return None
                self.roots[call.id] = {"parent_id": message.id, "call_id": call.id,
                    "name": call.name, "arguments_digest": digest(call.arguments)}
            if len(self.roots) > MAX_SOURCE_IDS:
                self._reset()
            return None
        if isinstance(event, ToolExecutionStartEvent):
            call_id, name, arguments = event.tool_call_id, event.tool_name, event.args
            root = self.roots.get(call_id)
            if root is not None:
                if (name != root["name"] or digest(arguments) != root["arguments_digest"]
                        or call_id in self.calls):
                    self._reset()
                    return None
                root_id = call_id
            else:
                # Namespace is accepted ONLY beneath a known, currently active
                # eval root registered by its real assistant MessageEnd.
                parents = [key for key, value in self.calls.items()
                           if value["name"] == "eval" and value["root"] == key
                           and not value.get("result") and call_id.startswith(key + ":")]
                if len(parents) != 1:
                    self._reset()
                    return None
                root_id = parents[0]
                try:
                    uuid.UUID(call_id[len(root_id) + 1:])
                except (ValueError, AttributeError):
                    self._reset()
                    return None
                if call_id in self.calls:
                    self._reset()
                    return None
            if not (name in {"eval", "no_reply"} and root is not None or
                    name == "no_reply" or is_peer_communication_call(name, arguments)):
                self._reset()
                return None
            self.calls[call_id] = {"root": root_id, "name": name,
                "arguments": arguments, "result": None}
            if name != "no_reply":
                self.members.append({"source_id": f"tool:{call_id}:0",
                                     "role": "send" if name == "send" else "carrier"})
            source_ids = {member["source_id"] for member in self.members}
            source_ids.update(f"tool:{key}:0" for key in self.calls)
            source_ids.update(value["parent_id"] for value in self.roots.values())
            if len(source_ids) > MAX_SOURCE_IDS:
                self._reset()
            return None
        if isinstance(event, ToolExecutionUpdateEvent):
            call = self.calls.get(event.tool_call_id)
            update = event.partial_result
            details = update.details
            # Text progress belongs to an ALREADY active permitted call; it
            # cannot certify quiet/success. Eval's exact existing display-only
            # metadata is known, not permission for arbitrary attention/fault
            # payloads or a foreign callback to disappear inside this stretch.
            safe_details = (details is None or details == {} or (
                call is not None and call["name"] == "eval" and isinstance(details, dict)
                and set(details) == {"tool_name", "display"}
                and details["tool_name"] == "eval" and details["display"] is True))
            if (call is None or call["result"] is not None or event.tool_name != call["name"]
                    or not safe_details or any(part.type != "text" for part in update.content)):
                self._reset()
            return None
        if isinstance(event, ToolExecutionEndEvent):
            call_id = event.tool_call_id
            call = self.calls.get(call_id)
            result = event.result
            if (call is None or call["result"] is not None or event.tool_name != call["name"]
                    or result.tool_name != call["name"] or result.tool_call_id != call_id
                    or event.is_error or not _successful(result)
                    or any(getattr(part, "type", None) != "text" for part in result.content)):
                self._reset()
                return None
            if call["name"] == "send" and current_peer_communication_succeeded(result, call_id) is not True:
                self._reset()
                return None
            call["result"] = result
            if call["name"] == "no_reply":
                if (result.details or {}).get(QUIET_TURN_KEY) is not True or self.quiet is not None:
                    self._reset()
                    return None
                self.quiet = (call_id, call["root"])
            return self._seal()
        if kind in {"agent_start", "agent_end"}:
            if kind == "agent_end":
                self._reset()
            return None
        # Streaming thinking and status scalars do not authorize new members.
        # Positive peer/tool events are the ONLY rows admitted above.
        if kind not in {"turn_start", "turn_end"}:
            self._reset()
        return None

    def _seal(self) -> dict[str, Any] | None:
        if (self.quiet is None or any(not value.get("result") for value in self.calls.values())
                or set(self.roots) - set(self.calls)):
            return None
        quiet_id, quiet_root = self.quiet
        # Direct pairs already have a durable raw proof; do not add an archive
        # row/default state field on that unchanged path.
        wrapped = any(value["name"] == "eval" for value in self.calls.values())
        if not wrapped:
            return None
        roots = [dict(value, outcome_digest=outcome_digest(self.calls[key]["result"]))
                 for key, value in self.roots.items()]
        children = []
        for call_id, call in self.calls.items():
            if call_id == call["root"] or call["name"] != "send":
                continue
            # These are the emitted display payloads, not stdout/code or a raw
            # outer eval body. Callers must feed post-redaction events.
            if not callable(self.redact):
                self._reset()
                return None
            # Reuse the loop's recursive JSON string sanitizer with the SAME
            # session result sink. observe_event's raw model_dump is NOT a
            # post-redaction boundary, especially for original tool arguments.
            from local_operator.harness.loop import _scrub_argument_value
            arguments, _changed = _scrub_argument_value(call["arguments"], self.redact)
            result = call["result"].model_dump(mode="json")
            # Identity/type discriminants are producer facts, not text. Masking
            # those strings would destroy a legitimate row's source identity.
            # Only admitted TextContent bodies, details and input values use the
            # same existing credential/shape sink as ordinary result display.
            result["content"] = [dict(part, text=self.redact(part["text"])) for part in result["content"]]
            result["details"], _changed = _scrub_argument_value(result["details"], self.redact)
            children.append({"source_id": f"tool:{call_id}:0", "tool_call_id": call_id,
                "tool_name": "send", "arguments": arguments,
                "result": result, "root_id": call["root"]})
        parent_id = self.roots[quiet_root]["parent_id"]
        record = {"version": 1, "id": record_id(self.session_id, parent_id, quiet_root),
            "session_id": self.session_id, "producer_epoch": self.epoch,
            "peer_id": self.peer_id, "quiet_root": quiet_root, "quiet_call_id": quiet_id,
            "roots": roots, "members": self.members,
            "member_ids": [member["source_id"] for member in self.members],
            "logical_actions": sum(member["role"] == "send" for member in self.members),
            "child_rows": children}
        if len(json.dumps(record, ensure_ascii=False).encode()) > MAX_RECORD_BYTES:
            self._reset()
            return None
        self.records = [*self.records[-(HOT_RECORDS - 1):], record]
        self._reset()
        return record


def indexed_records(transcript: Any, ids: list[str], *, max_bytes: int = 512 * 1024) -> dict[str, Any] | None:
    """Read selected archive IDs using the existing all-row index, never a scan.

    None is UNAVAILABLE. Explicit remaining/unavailable IDs must be propagated
    by a page consumer instead of pretending a capped proof never existed.
    """
    from local_operator.session.transcript import read_indexed_raw_rows
    from local_operator.session.transcript_index import refresh_index
    directory = transcript.directory
    index = refresh_index(directory.parent.parent, directory.name)
    if index is None:
        return None
    return read_indexed_raw_rows(directory, index, ids, max_bytes=max_bytes)


def validate_record(record: Any, messages: Any, session_id: str) -> dict[str, Any] | None:
    """Validate an archive proof against THIS selected durable raw interval.

    A hot typed graph can precede raw persistence; a cold reader cannot. Raw
    parent/call/outcome hashes, complete ordered seats and the actual peer must
    all survive the cut. Missing/corrupt/pruned sources mean no overlay, never a
    plausible success with an invented unavailable payload. Raw details stamps
    are deliberately not an input to this function.
    """
    from local_operator.harness.types import ToolResult
    try:
        if (not isinstance(record, dict) or type(record.get("version")) is not int or record.get("version") != 1
                or type(record.get("logical_actions")) is not int
                or not isinstance(record.get("producer_epoch"), str) or not record["producer_epoch"]
                or record.get("session_id") != session_id
                or len(json.dumps(record, ensure_ascii=False).encode()) > MAX_RECORD_BYTES):
            return None
        ids, members = record["member_ids"], record["members"]
        roots, children = record["roots"], record["child_rows"]
        if (not isinstance(ids, list) or not 2 <= len(ids) <= MAX_SOURCE_IDS
                or any(not isinstance(key, str) or not key for key in ids)
                or len(ids) != len(set(ids))
                or [item["source_id"] for item in members] != ids
                or not isinstance(roots, list) or not roots or not isinstance(children, list)):
            return None
        by_id = {message.id: message for message in messages}
        if len(by_id) != len(messages):
            return None
        peer = by_id.get(record["peer_id"])
        details = getattr(peer, "details", None)
        if (peer is None or getattr(peer, "custom_type", None) != PEER_MESSAGE_MESSAGE_TYPE
                or not isinstance(details, dict)
                or any(details.get(key) for key in ("held", "steer", "is_error", "images", "attachments"))
                or members[0] != {"source_id": f"{peer.id}:0", "role": "peer"}):
            return None
        references = set(ids)
        references.update(f"tool:{root['call_id']}:0" for root in roots)
        references.update(root["parent_id"] for root in roots)
        references.add(f"tool:{record['quiet_call_id']}:0")
        if len(references) > MAX_SOURCE_IDS:
            return None
        root_map = {root["call_id"]: root for root in roots}
        if len(root_map) != len(roots) or record["quiet_root"] not in root_map:
            return None
        if record["id"] != record_id(session_id, root_map[record["quiet_root"]]["parent_id"], record["quiet_root"]):
            return None
        call_order, results, parent_ids = [], {}, set()
        peer_position = next(i for i, message in enumerate(messages) if message is peer)
        final_position = peer_position
        for root in roots:
            parent = by_id.get(root["parent_id"])
            if (not isinstance(parent, Message) or parent.role != "assistant" or parent.content
                    or not parent.tool_calls):
                return None
            parent_ids.add(parent.id)
            calls = [call for call in parent.tool_calls if call.id == root["call_id"]]
            if len(calls) != 1:
                return None
            call = calls[0]
            if (call.name != root["name"] or digest(call.arguments) != root["arguments_digest"]
                    or not (call.name in {"eval", "no_reply"} or is_peer_communication_call(call.name, call.arguments))):
                return None
            matches = [(i, message) for i, message in enumerate(messages)
                       if getattr(message, "role", None) == "tool"
                       and message.tool_call_id == root["call_id"]]
            if len(matches) != 1:
                return None
            index, result = matches[0]
            parent_index = next(i for i, message in enumerate(messages) if message is parent)
            if (not peer_position < parent_index < index or result.tool_name != root["name"]
                    or not _successful(result)
                    or outcome_digest(result) != root["outcome_digest"]
                    or any(getattr(part, "type", None) != "text" for part in result.content)):
                return None
            if call.name == "send" and current_peer_communication_succeeded(result, call.id) is not True:
                return None
            results[call.id] = result
            final_position = max(final_position, index)
        # No genuine USER/final/attention/extra peer/unknown tool inside this
        # finite raw interval. A later real answer is explicitly outside it.
        for message in messages[peer_position + 1:final_position + 1]:
            if getattr(message, "role", None) == "assistant":
                if (message.id not in parent_ids or message.content
                        or any(call.id not in root_map for call in message.tool_calls)):
                    return None
                call_order.extend(call.id for call in message.tool_calls)
            elif getattr(message, "role", None) != "tool" or message.tool_call_id not in root_map:
                return None
        if set(call_order) != set(root_map):
            return None
        child_map = {}
        for child in children:
            call_id, root_id = child["tool_call_id"], child["root_id"]
            if (root_id not in root_map or root_map[root_id]["name"] != "eval"
                    or not call_id.startswith(root_id + ":") or call_id in child_map
                    or child["source_id"] != f"tool:{call_id}:0"
                    or not is_peer_communication_call(child["tool_name"], child["arguments"])):
                return None
            uuid.UUID(call_id[len(root_id) + 1:])
            result = ToolResult.model_validate(child["result"])
            if (not _successful(result) or current_peer_communication_succeeded(result, call_id) is not True
                    or any(getattr(part, "type", None) != "text" for part in result.content)):
                return None
            child_map[call_id] = child
        quiet_id, quiet_root = record["quiet_call_id"], record["quiet_root"]
        if quiet_id == quiet_root:
            result = results[quiet_root]
            payload = getattr(result, "provider_payload", None) or {}
            if result.tool_name != "no_reply" or (payload.get("details") or {}).get(QUIET_TURN_KEY) is not True:
                return None
        elif (root_map[quiet_root]["name"] != "eval" or not quiet_id.startswith(quiet_root + ":")):
            return None
        else:
            uuid.UUID(quiet_id[len(quiet_root) + 1:])
        expected = [members[0]]
        for call_id in call_order:
            name = root_map[call_id]["name"]
            if name != "no_reply":
                expected.append({"source_id": f"tool:{call_id}:0", "role": "carrier" if name == "eval" else "send"})
            expected.extend({"source_id": child["source_id"], "role": "send"}
                            for child in children if child["root_id"] == call_id)
        if expected != members or record["logical_actions"] != sum(item["role"] == "send" for item in members):
            return None
        return record
    except (AttributeError, KeyError, TypeError, ValueError, StopIteration):
        return None


def _indexed_root_result(index: Any, call_id: str, parent_id: str, cut: int | None,
                         reference_budget: int) -> tuple[str | None, int]:
    """Disambiguate settled reuse by EXACT parent, never latest/hash-picking.

    The prior owner must already have exactly one result before this parent;
    otherwise a late prior callback could masquerade as the new completion.
    The existing inverse lists are bounded BEFORE sorting or traversing them.
    """
    owners = index.source_views.owners.get(call_id, [])
    results = index.source_views.results.get(call_id, [])
    count = len(owners) + len(results)
    if not 0 < count <= reference_budget or not owners or not results:
        return None, min(count, reference_budget)
    if (len(set(owners)) != len(owners) or len(set(results)) != len(results)
            or parent_id not in owners
            or any(key not in index.journal_rows or key in index.source_views.duplicates for key in [*owners, *results])):
        return None, count
    ordered = sorted(owners, key=index.journal_rows.__getitem__)
    position = ordered.index(parent_id)
    parent_seq = index.journal_rows[parent_id]
    if cut is not None and parent_seq > cut:
        return None, count
    if position:
        prior_seq = index.journal_rows[ordered[position - 1]]
        prior = [key for key in results if prior_seq < index.journal_rows[key] < parent_seq]
        if len(prior) != 1:
            return None, count
    next_seq = index.journal_rows[ordered[position + 1]] if position + 1 < len(ordered) else None
    selected = [key for key in results if parent_seq < index.journal_rows[key]
                and (next_seq is None or index.journal_rows[key] < next_seq)
                and (cut is None or index.journal_rows[key] <= cut)]
    return (selected[0] if len(selected) == 1 else None), count


def validate_indexed_record(transcript: Any, record: Any, session_id: str, *,
                            index: Any = None, cut: int | None = None,
                            max_bytes: int = 512 * 1024) -> tuple[dict[str, Any] | None, int]:
    """Validate ONE finite raw interval through existing offsets/dependencies.

    The existing index supplies root results and journal order; no source->proof
    index is introduced. At most4096 existing facts/512 IDs per read and512KiB
    total are inspected. Bookkeeping is classified by the index's existing
    source_class; rendered CustomMessages are still decoded and checked.
    """
    from local_operator.session.transcript import read_indexed_raw_rows, _entry_to_message
    from local_operator.session.transcript_index import refresh_index
    if not isinstance(record, dict) or not 0 < max_bytes <= 512 * 1024:
        return None, 0
    directory = transcript.directory
    index = index or refresh_index(directory.parent.parent, directory.name)
    if index is None or not index.source_views.origins_complete:
        return None, 0
    try:
        peer = record["peer_id"]
        roots = record["roots"]
        if (not isinstance(roots, list) or not 0 < len(roots) <= MAX_SOURCE_IDS
                or len(json.dumps(record, ensure_ascii=False).encode()) > MAX_RECORD_BYTES):
            return None, 0
        result_ids = []
        reference_budget = MAX_SOURCE_IDS
        for root in roots:
            result_id, inspected = _indexed_root_result(
                index, root["call_id"], root["parent_id"], cut, reference_budget)
            reference_budget -= inspected
            if result_id is None:
                return None, 0
            result_ids.append(result_id)
        start = index.journal_rows[peer]
        end = max(index.journal_rows[key] for key in result_ids)
        if start > end or end - start + 1 > MAX_SOURCE_IDS or cut is not None and end > cut:
            return None, 0
        facts = index.source_projection_facts[start:end + 1]
        if len(facts) != end - start + 1:
            return None, 0
        ids = []
        for fact in facts:
            kind = fact["source_class"]
            if kind == "bookkeeping":
                continue
            if (kind not in {"plain_message", "custom_message"}
                    or any(flag in fact["flags"] for flag in ("global_unknown", "incomplete", "opaque_causal"))
                    or index.source_views.prunes.get(fact["id"])):
                return None, 0
            ids.append(fact["id"])
        rows, used = [], 0
        for offset in range(0, len(ids), 512):
            if used >= max_bytes:
                return None, used
            response = read_indexed_raw_rows(directory, index, ids[offset:offset + 512], max_bytes=max_bytes - used)
            if response is None:
                return None, used
            used += response["bytes"]
            if response["missing"] or response["remaining"] or response["unavailable"]:
                return None, used
            rows.extend(response["rows"])
        messages = [_entry_to_message(row) for row in rows]
        if any(message is None for message in messages):
            return None, used
        return validate_record(record, messages, session_id), used
    except (AttributeError, KeyError, TypeError, ValueError):
        return None, 0


def discover_records(transcript: Any, messages: Any, session_id: str, *, max_bytes: int = 512 * 1024) -> dict[str, Any] | None:
    """Discover deterministic IDs from selected raw roots using the existing index.

    Historical proofs are NOT limited by the hot16 cache. Only existing archive
    IDs are read, in <=512-ID batches within one response's byte allowance.
    Remaining/unavailable source is explicit; no whole-journal fallback exists.
    """
    from local_operator.session.transcript import read_indexed_raw_rows
    from local_operator.session.transcript_index import refresh_index
    candidates = []
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        candidates.extend(record_id(session_id, message.id, call.id)
                          for call in message.tool_calls if call.name in {"eval", "no_reply"})
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None
    directory = transcript.directory
    index = refresh_index(directory.parent.parent, directory.name)
    if index is None:
        return {"records": [], "remaining": [], "unavailable": candidates, "bytes": 0}
    cut_rows = [index.journal_rows.get(message.id) for message in messages]
    if not cut_rows or any(row is None for row in cut_rows):
        return {"records": [], "remaining": [], "unavailable": candidates, "bytes": 0}
    # The complete-page guard above refuses missing seats; expose that exact
    # invariant to the type checker rather than asserting away optional rows.
    cut = max(row for row in cut_rows if row is not None)
    existing = [key for key in candidates if key in index.journal_rows
                and index.journal_rows[key] <= cut]
    if not existing:
        return None
    records, remaining, unavailable, used = [], [], [], 0
    for start in range(0, len(existing), 512):
        batch = existing[start:start + 512]
        if used >= max_bytes:
            remaining.extend(existing[start:])
            break
        response = read_indexed_raw_rows(directory, index, batch, max_bytes=max_bytes - used)
        if response is None:
            unavailable.extend(existing[start:])
            break
        used += response["bytes"]
        remaining.extend(response["remaining"])
        unavailable.extend([*response["unavailable"], *response["missing"]])
        for row in response["rows"]:
            payload = row.payload
            record = payload.get("details") if payload.get("custom_type") == CUSTOM_TYPE else None
            if row.type != "custom" or not isinstance(record, dict) or record.get("id") != row.id:
                unavailable.append(row.id)
            elif used >= max_bytes:
                remaining.append(row.id)
            else:
                validated, read_bytes = validate_indexed_record(
                    transcript, record, session_id, index=index, cut=cut, max_bytes=max_bytes - used)
                used += read_bytes
                if validated is None:
                    unavailable.append(row.id)
                else:
                    records.append(validated)
    return {"records": records, "remaining": remaining, "unavailable": unavailable, "bytes": used}


def attach_page_records(window: Any, transcript: Any, *, max_wire_bytes: int) -> Any:
    """Account optional proof metadata inside the existing page/cut byte budget.

    An unavailable proof leaves the original rows visible. Metadata overflow is
    explicit, never a truncated child payload certified as complete. The small
    fallback names counts if even the unavailable-ID list cannot fit.
    """
    if window.status != "ok" or not window.messages:
        return window
    generation = transcript._history_generation
    base_bytes = len(json.dumps(window.model_dump(mode="json")).encode())
    budget = max(0, max_wire_bytes - 8192 - base_bytes)
    envelope = discover_records(transcript, window.messages, window.conversation_id,
                                max_bytes=min(512 * 1024, max(1, budget)))
    if envelope is None:
        return window
    if transcript._history_generation != generation or generation != window.history_generation:
        envelope = {"records": [], "remaining": [], "unavailable": [],
                    "status": "source_changed", "bytes": 0}
    kept = []
    remaining = list(envelope["remaining"])
    unavailable = list(envelope["unavailable"])
    for record in envelope["records"]:
        trial = {"records": [*kept, record], "remaining": remaining,
                 "unavailable": unavailable, "status": "partial" if remaining or unavailable else "complete"}
        if len(json.dumps(trial).encode()) <= budget:
            kept.append(record)
        else:
            remaining.append(record["id"])
    payload = {"records": kept, "remaining": remaining, "unavailable": unavailable,
               "status": envelope.get("status", "partial" if remaining or unavailable else "complete")}
    if len(json.dumps(payload).encode()) > budget:
        payload = {"records": [], "status": "metadata_budget",
                   "remaining_count": len(kept) + len(remaining), "unavailable_count": len(unavailable)}
    # The ordinary page reserves 8192 bytes; this constant-size refusal fits
    # that envelope without dropping or pretending to have rendered any body.
    window.peer_cycle_display_records = payload
    return window


def merge_page_records(previous: Any, current: Any) -> dict[str, Any] | None:
    """Retain proof for the existing loaded window across compatible head pages.

    Caller owns session/epoch/generation compatibility. This is an ephemeral
    bounded metadata union, not a journal index or body cache. Conflicting IDs
    refuse rather than silently choosing whichever page arrived last.
    """
    if previous is None and current is None:
        return None
    records: dict[str, dict[str, Any]] = {}
    conflict: set[str] = set()
    remaining: set[str] = set()
    unavailable: set[str] = set()
    incomplete = False
    for envelope in (previous, current):
        if not isinstance(envelope, dict):
            continue
        incomplete = incomplete or envelope.get("status") in {"source_changed", "metadata_budget"} or bool(envelope.get("source_metadata_incomplete"))
        remaining.update(envelope.get("remaining", ()))
        unavailable.update(envelope.get("unavailable", ()))
        for record in envelope.get("records", ()):
            key = record.get("id") if isinstance(record, dict) else None
            if not isinstance(key, str) or key in conflict:
                continue
            if key in records and records[key] != record:
                conflict.add(key)
                records.pop(key)
                unavailable.add(key)
            else:
                records[key] = record
    kept, source_ids, used = [], set(), 0
    for key, record in records.items():
        ids = record.get("member_ids", ())
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            unavailable.add(key)
            continue
        cost = len(json.dumps(record).encode())
        if len(source_ids.union(ids)) > MAX_SOURCE_IDS or used + cost > 512 * 1024:
            remaining.add(key)
            continue
        source_ids.update(ids)
        used += cost
        kept.append(record)
        remaining.discard(key)
    return {"records": kept, "remaining": sorted(remaining), "unavailable": sorted(unavailable),
            "status": "partial" if remaining or unavailable or incomplete else "complete",
            **({"source_metadata_incomplete": True} if incomplete else {})}


async def persist_record(transcript: Any, record: dict[str, Any]) -> bool:
    """Idempotent archive append; a collision cannot overwrite another record."""
    key = record["id"]
    if transcript.has_entry(key):
        existing = indexed_records(transcript, [key], max_bytes=MAX_RECORD_BYTES + 1024)
        if existing is None or existing["remaining"] or existing["unavailable"] or len(existing["rows"]) != 1:
            return False
        row = existing["rows"][0]
        return row.type == "custom" and row.payload == {"custom_type": CUSTOM_TYPE, "details": record}
    await transcript.append_custom(CUSTOM_TYPE, record, entry_id=key, preserve_mtime=True)
    return True
