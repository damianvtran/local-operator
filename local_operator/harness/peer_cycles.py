"""Additive display-only closure for a completed, finite peer exchange.

The default quiet-group scanner intentionally needs two receipts. A positively
completed peer/send/no_reply cycle is a DIFFERENT proof, not a fake USER scope
or a change to that scanner's threshold. Only IDs/typed outcome facts are held;
no message bodies or model-facing metadata are copied into this projection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.harness.types import (
    FAULT_KEY,
    OUTPUT_LIMIT_KEY,
    QUIET_TURN_KEY,
    Message,
    ToolResult,
)

# Match the existing ownership proof's finite source-ID budget: 65 small
# cycles use only 130 IDs and must not disappear at an arbitrary cycle count.
MAX_PEER_CYCLE_ROWS = 4096
MAX_PEER_CYCLE_CERTIFICATE_IDS = 4096
MAX_PENDING_PEER_CYCLES = 16
ViewIdentity = tuple[str | None, int]


@dataclass(frozen=True)
class PeerCycleRow:
    source_id: str
    kind: Literal["peer", "send", "carrier", "boundary"]
    session_id: str | None
    view_epoch: int
    successful: bool = False
    never_sent: bool = False
    attention: bool = False
    sender: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PeerCycleStart:
    session_id: str
    view_epoch: int
    quiet_call_id: str
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class PeerCycleCertificate(PeerCycleStart):
    owner_kind: Literal["peer_cycle", "peer_cycle_display"] = "peer_cycle"

    @property
    def first_peer_source_id(self) -> str:
        return self.member_ids[0]

    @property
    def last_member_source_id(self) -> str:
        return self.member_ids[-1]


def remember_peer_cycle_certificate(
    certificates: dict[str, PeerCycleCertificate],
    certificate: PeerCycleCertificate,
) -> None:
    """Bound unattached proof by member IDs, never by bodies or cycle count.

    Attached rows/bars retain immutable certificates independently: evicting a
    cache entry cannot erase their closed facts or reader preference. Windows
    over this ID budget conservatively lack older, still-unattached proof.
    """
    certificates[certificate.quiet_call_id] = certificate
    while (
        sum(len(item.member_ids) for item in certificates.values()) > MAX_PEER_CYCLE_CERTIFICATE_IDS
    ):
        certificates.pop(next(iter(certificates)))


def is_peer_communication_call(tool_name: Any, arguments: Any) -> bool:
    """Qualify the real communication arm, not the overloaded ``send`` name.

    ``send(model=...)`` mutates a peer's model; its success is not evidence
    that the incoming message was answered. Only original typed arguments
    distinguish those acts. Missing/unknown arguments refuse; body text and
    sender labels never provide communication authority.
    """
    return (
        tool_name == "send"
        and isinstance(arguments, Mapping)
        and isinstance(arguments.get("message"), str)
        and arguments.get("model") is None
    )


def _valid_rows(
    records: Sequence[PeerCycleRow], identity: ViewIdentity, *, display: bool = False
) -> bool:
    session_id, epoch = identity
    if (
        not isinstance(session_id, str)
        or not session_id.strip()
        or type(epoch) is not int
        or epoch < 0
    ):
        return False
    if not 2 <= len(records) <= MAX_PEER_CYCLE_ROWS:
        return False
    ids = [record.source_id for record in records]
    if any(not isinstance(source_id, str) or not source_id.strip() for source_id in ids) or len(
        set(ids)
    ) != len(ids):
        return False
    allowed_actions = {"send", "carrier"} if display else {"send"}
    if records[0].kind != "peer" or records[-1].kind not in allowed_actions:
        return False  # An unacknowledged later peer cannot ride an earlier send.
    return all(
        (record.session_id, record.view_epoch) == identity and not record.attention and
        # A carrier is produced ONLY by the validated display overlay. The raw
        # adapter never assigns it, so default scanners keep their splitters.
        (
            record.kind == "peer"
            or (
                record.kind in allowed_actions
                and record.successful is True
                and not record.never_sent
            )
        )
        for record in records
    )


def begin_peer_cycle(
    records: Sequence[PeerCycleRow], identity: ViewIdentity, quiet_call_id: str
) -> PeerCycleStart | None:
    if (
        not isinstance(quiet_call_id, str)
        or not quiet_call_id.strip()
        or not _valid_rows(records, identity)
    ):
        return None
    return PeerCycleStart(
        str(identity[0]), identity[1], quiet_call_id, tuple(record.source_id for record in records)
    )


def _result_details(result: Any) -> Mapping[str, Any]:
    details = getattr(result, "details", None)
    if isinstance(details, Mapping):
        return details
    payload = getattr(result, "provider_payload", None)
    details = payload.get("details") if isinstance(payload, Mapping) else None
    return details if isinstance(details, Mapping) else {}


def current_peer_communication_succeeded(result: Any, call_id: str | None = None) -> bool | None:
    """Read a CURRENT typed end, never painted success or a historical receipt.

    Live ToolResult stores details directly; durable Message stores them under
    provider_payload. Unknown/untyped/manual states carry no execution proof.
    The builtin's duplicate_suppressed arm explicitly describes an older send,
    so it cannot certify that THIS source member communicated anything.
    """
    if isinstance(result, ToolResult):
        details = result.details if isinstance(result.details, Mapping) else {}
    elif isinstance(result, Message) and result.role == "tool":
        payload = result.provider_payload
        details = payload.get("details") if isinstance(payload, Mapping) else None
        details = details if isinstance(details, Mapping) else {}
    else:
        return None
    actual_id = result.tool_call_id
    if (
        result.tool_name != "send"
        or not isinstance(actual_id, str)
        or not actual_id
        or call_id is not None
        and call_id != actual_id
    ):
        return False
    if result.is_error is not False:
        return False
    return details.get(FAULT_KEY) is None and not any(
        details.get(key)
        for key in (
            "duplicate_suppressed",
            "not_run",
            "never_sent",
            "partial_result",
            OUTPUT_LIMIT_KEY,
        )
    )


def close_peer_cycle(
    start: PeerCycleStart | None,
    records: Sequence[PeerCycleRow],
    identity: ViewIdentity,
    call_id: str,
    tool_name: str,
    result: Any,
) -> PeerCycleCertificate | None:
    """A marker alone is not closure: require its real call/name/current view.

    The exact start inventory also refuses a late result after another peer,
    answer, page reorder or session/view change. Consumers pop starts once.
    """
    if (
        start is None
        or (start.session_id, start.view_epoch) != identity
        or start.quiet_call_id != call_id
    ):
        return None
    if (
        tool_name != "no_reply"
        or getattr(result, "tool_name", None) != "no_reply"
        or getattr(result, "tool_call_id", None) != call_id
    ):
        return None
    details = _result_details(result)
    if (
        getattr(result, "is_error", None) is not False
        or details.get(QUIET_TURN_KEY) is not True
        or details.get(FAULT_KEY) is not None
        or details.get(OUTPUT_LIMIT_KEY)
        or details.get("partial_result")
    ):
        return None
    if (
        not _valid_rows(records, identity)
        or tuple(record.source_id for record in records) != start.member_ids
    ):
        return None
    return PeerCycleCertificate(start.session_id, start.view_epoch, call_id, start.member_ids)


def peer_cycle_sources_match(
    records: Sequence[PeerCycleRow], certificate: PeerCycleCertificate | None
) -> bool:
    """Shared finite-source proof; importing it never imports a renderer."""
    return (
        isinstance(certificate, PeerCycleCertificate)
        and certificate.owner_kind in {"peer_cycle", "peer_cycle_display"}
        and isinstance(certificate.quiet_call_id, str)
        and bool(certificate.quiet_call_id.strip())
        and _valid_rows(
            records,
            (certificate.session_id, certificate.view_epoch),
            display=certificate.owner_kind == "peer_cycle_display",
        )
        and tuple(record.source_id for record in records) == certificate.member_ids
    )


class PeerCycleReplay:
    """Metadata-only ordered prepass, fed beside the existing replay index walk.

    Raw `details.peer_cycle` is NEVER read. Actual quiet calls and their typed
    results alone mint local certificates; a carrier-only assistant is not a
    painted assistant row, while ANY other assistant (even empty) is a boundary.
    The candidate and outstanding calls are bounded and contain no body cache.
    """

    def __init__(self, identity: ViewIdentity) -> None:
        self.identity = identity
        self.records: list[PeerCycleRow] = []
        self.calls: dict[str, str] = {}
        self.quiet: dict[str, PeerCycleStart] = {}
        self._overflowed = False

    def _reset(self) -> None:
        self.records.clear()
        self.calls.clear()
        self.quiet.clear()
        self._overflowed = False

    def _append(self, record: PeerCycleRow) -> bool:
        self.records.append(record)
        if len(self.records) > MAX_PEER_CYCLE_ROWS:
            self._reset()
            self._overflowed = True
            return False
        return True

    def reset_pending(self) -> None:
        """Retire the accumulating candidate, keeping closed certificates.

        A non-peer typed delivery ends any pending peer exchange without
        fabricating a Message/tool event. Already-closed certificates live in
        the caller's registry and are deliberately untouched; this clears only
        the in-flight rows/calls/quiet start for THIS replay.
        """
        self._reset()

    def delivered_peer(self, message_id: str, sender: Mapping[str, Any]) -> None:
        """Seed the live adapter from the actual delivery ID, without a fake Message.

        The replay path still reads the real custom row. Live consumers already
        have a typed delivery and must not fabricate a model-history row just
        to feed this metadata-only prepass. Overflow remains fail-closed.
        """
        if not isinstance(message_id, str) or not message_id:
            self._reset()
        elif not self._overflowed:
            self._append(PeerCycleRow(f"{message_id}:0", "peer", *self.identity, sender=sender))

    def step_live_result(self, result: Any) -> PeerCycleCertificate | None:
        """Feed one ACTUAL typed tool result from the live event stream.

        The live session fans tool outcomes out as ``ToolExecutionEndEvent``;
        it does NOT re-emit the persisted ``role="tool"`` Message, so a replay
        fed only ``MessageEndEvent`` could never settle a live call and every
        live cycle stayed uncertified (the phone's live/replay divergence).
        The arm below is the SAME one the history walk uses, so the two paths
        cannot drift; the result object itself is the authority, exactly as in
        the durable case.
        """
        return self._step_tool(result)

    def _step_tool(self, message: Any) -> PeerCycleCertificate | None:
        call_id = getattr(message, "tool_call_id", None)
        if not isinstance(call_id, str):
            self._reset()
            return None
        name = self.calls.pop(call_id, None)
        if name == "send" and getattr(message, "tool_name", None) == "send":
            success = current_peer_communication_succeeded(message, str(call_id)) is True
            self.records = [
                (
                    replace(record, successful=success)
                    if record.source_id == f"tool:{call_id}:0"
                    else record
                )
                for record in self.records
            ]
        elif name == "no_reply":
            close = close_peer_cycle(
                self.quiet.pop(call_id, None),
                self.records,
                self.identity,
                str(call_id),
                str(getattr(message, "tool_name", "")),
                message,
            )
            self._reset()
            return close
        else:
            self._reset()
        if len(self.calls) > MAX_PEER_CYCLE_ROWS or len(self.quiet) > MAX_PENDING_PEER_CYCLES:
            self._reset()
            self._overflowed = True
        return None

    def step(self, message: Any) -> PeerCycleCertificate | None:
        role = getattr(message, "role", None)
        custom_type = getattr(message, "custom_type", None)
        if self._overflowed:
            # Refuse the whole oversized interval, not merely its old prefix.
            # A later peer cannot rebase it into a deceptively complete suffix.
            # Clearing on a quiet/presentation boundary grants no certificate.
            calls = getattr(message, "tool_calls", None) or []
            peer_details = getattr(message, "details", None)
            same_peer_interval = (
                custom_type == PEER_MESSAGE_MESSAGE_TYPE
                and isinstance(peer_details, Mapping)
                and not any(
                    peer_details.get(key)
                    for key in ("held", "steer", "is_error", "images", "attachments")
                )
            )
            same_interval = (
                same_peer_interval
                or role == "tool"
                and getattr(message, "tool_name", None) == "send"
                or role == "assistant"
                and not getattr(message, "content", None)
                and calls
                and all(
                    is_peer_communication_call(
                        getattr(call, "name", None), getattr(call, "arguments", None)
                    )
                    or getattr(call, "name", None) == "no_reply"
                    for call in calls
                )
            )
            if not same_interval:
                self._reset()
            return None
        if custom_type == PEER_MESSAGE_MESSAGE_TYPE:
            details = getattr(message, "details", None)
            message_id = getattr(message, "id", None)
            # The actual mailbox producer uses attribution="user" for this
            # typed transport receipt. It is NOT a genuine Message.role=user
            # opener/steer; those still reset the candidate in the role arm.
            if (
                not isinstance(details, Mapping)
                or not isinstance(message_id, str)
                or not message_id
                or any(
                    details.get(key)
                    for key in ("held", "steer", "is_error", "images", "attachments")
                )
            ):
                self._reset()
                return None
            sender = details.get("sender")
            if not self._append(
                PeerCycleRow(
                    f"{message_id}:0",
                    "peer",
                    *self.identity,
                    sender=sender if isinstance(sender, Mapping) else {},
                )
            ):
                return None
        elif role == "assistant":
            calls = getattr(message, "tool_calls", None) or []
            # Media and visible prose always remain outside the fold. Reading
            # emptiness here is carrier classification, never quiet-code parsing.
            if getattr(message, "content", None) or not calls:
                self._reset()
                return None
            for call in calls:
                name, call_id = getattr(call, "name", None), getattr(call, "id", None)
                if not isinstance(call_id, str) or not call_id or call_id in self.calls:
                    self._reset()
                    return None
                if name == "send" and is_peer_communication_call(
                    name, getattr(call, "arguments", None)
                ):
                    self.calls[call_id] = name
                    if not self._append(PeerCycleRow(f"tool:{call_id}:0", "send", *self.identity)):
                        return None
                elif name == "no_reply":
                    start = begin_peer_cycle(self.records, self.identity, call_id)
                    if start is not None:
                        self.quiet[call_id] = start
                        self.calls[call_id] = name
                else:
                    self._reset()
                    return None
        elif role == "tool":
            return self._step_tool(message)
        else:
            self._reset()
        if len(self.calls) > MAX_PEER_CYCLE_ROWS or len(self.quiet) > MAX_PENDING_PEER_CYCLES:
            self._reset()
            self._overflowed = True
        return None
