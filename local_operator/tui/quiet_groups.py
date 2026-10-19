"""The quiet-group definition (quiet-turn design §5 + §8 S5, rev 2), derived
for the TUI's own record vocabulary.

WHAT A GROUP IS. A quiet group is a client-derived fold over a run's delivery
receipts: a maximal run of >= 2 receipt rows with nothing the reader can see
between them. A `user` row, a compaction statement, a notice, visible assistant
prose — every row a reader must be able to read — splits it; tool rows (the
quiet `no_reply` call included, when it is on hand at all) sit inside. ONE
receipt is not a group at all — its row keeps its ordinary card.

There is no wire kind and no capability flag: the group is a PURE function of
the rows on hand, and each surface derives it for itself. The single definition
is kept from drifting across surfaces by the shared parity fixture
(``quiet-groups.parity.json``, copied byte-identical and hash-pinned by this
module's test — the same cross-client pattern ``format.parity.json`` and
``spend-context.parity.json`` use), which this module's test replays case by
case.

THE TUI MAPPING, stated once (what this surface can and cannot express):

- Receipt rows exist for three families here: peer deliveries
  (``PeerMessageBlock``), wake receipts (``WakeBlock``) and monitor deltas
  (``MonitorDeltaBlock``). All three are real, production-firing triggers.
- In the DEFAULT standalone projection, the ``job`` family has NO receipt row:
  a delivered job result
  paints nothing (the turn it opened carries its answer), and a HELD one paints
  a generic notice — once it is a notice row, no field distinguishes it from
  any other notice — so it splits, which is the safe direction: an
  unclassifiable receipt stays visible rather than hiding inside a group. The
  fixture's job-only case is named and skipped on this surface for exactly
  that reason.
- No per-row timestamp reaches the rows this surface holds (a transcript
  message carries no entry time of its own, and the store's ``ts`` — the
  ``{entry id: entry ts}`` join the display window ships to wire viewers — is
  not joined back onto the local replay), so ``first_ts``/``last_ts`` come back
  null and the TUI states no span. The fields stay in the shape because they
  are part of the shared cross-client contract the fixture pins — a surface
  whose records carry times (the desktop's entry-time join) states them from
  this same derivation.

KEY STABILITY. ``key`` is ``qg:<first row id>``: rows only append at the tail,
so the key never moves while the group grows, and a press on the bar survives
the append. A closed group's facts cannot move either — later appends land
outside its boundary — so no latch state is needed here: the derivation is
pure per call, which is the latch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from local_operator.harness.rows import is_quiet_turn_name
from local_operator.tui.internal_work_membership import WorkMembership

#: The family word's set (design §5): the trigger kinds that can compose a
#: group, plus ``mixed`` for more than one. ``job`` is kept because this is the
#: shared contract's own union — see the module doc for why this surface cannot
#: reach it.
QuietGroupFamily = Literal["peer", "wake", "monitor", "job", "mixed"]


@dataclass(frozen=True)
class QuietGroupSender:
    """ONE QUIET GROUP'S sender entry: the identity line and how many receipts
    this sender contributed."""

    #: The identity ladder's own spelling — quoted conversation name,
    #: ``basename/``, a short session id, or ``pid N``.
    label: str
    count: int


@dataclass(frozen=True)
class QuietGroup:
    """ONE QUIET GROUP (design §5). Every field is derived from the rows on
    hand; nothing here is ever sent."""

    #: ``qg:<first row id>`` — stable across appends (see the module doc).
    key: str
    #: The family the trigger rows compose; ``mixed`` when more than one.
    family: QuietGroupFamily
    #: Internal trigger count; one receipt requires at least three real tools.
    count: int
    #: First trigger's instant; null when the span's head is cut OR the rows
    #: carry no times (this surface's state — see the module doc).
    first_ts: float | None
    #: Last trigger's instant; null for the same reasons as ``first_ts``.
    last_ts: float | None
    #: Peer-family senders, by count: the top 2, then one aggregate entry whose
    #: label is ``<N> more`` for the remaining distinct senders and whose count
    #: is their receipts. Empty for every other family (the shape is
    #: peer-only).
    senders: tuple[QuietGroupSender, ...]
    #: Non-quiet tool rows in the group.
    actions: int
    #: Of ``actions``, the rows whose outcome is a genuine error.
    failed: int
    #: It is the tail and no later visible row exists (the group may still
    #: grow).
    open: bool
    #: Every row of the group, in order.
    row_ids: tuple[str, ...]
    # Mixed summaries describe actual tool work separately from receipt count.
    mixed_actions: bool = False
    reply_requested: int = 0
    steered: int = 0


@dataclass(frozen=True)
class QuietGroupRecord:
    """The row shape this derivation reads.

    It carries only the fields the definition consumes, per kind: ``text`` for
    an assistant row's visible-prose test, ``sender`` for a peer receipt's
    identity, ``tool_name``/``tool_state`` for a ledger row. ``ts`` is the one
    field this surface cannot fill (see the module doc), and it is optional so
    a client whose records DO carry times maps them onto the same derivation —
    the fixture's own contract ("Timestamps are ms epoch, copied verbatim into
    firstTs/lastTs").

    ``kind`` is the surface's positive classification. This surface can say:
    ``peer``, ``wake``, ``monitor`` (its three receipt kinds), ``bridge`` (an
    org member's bridge receipt — a SPLITTER by this module's own rule below,
    never a member: §13.4 forbids overloading it onto the peer family's word),
    ``ask`` (an ask receipt — a question to the reader, classified as a
    boundary), ``tool`` (a ledger row), ``assistant`` (a prose row), and
    ``inside`` — its own kind for
    a row that is neither a trigger nor countable work but may sit inside a
    group (an image a folded tool row produced). Everything else — ``user``, a
    notice, a compaction statement, any unknown kind — is a splitter, which is
    the derivation's default and the safe direction.
    """

    kind: str
    id: str
    ts: float | None = None
    text: str = ""
    sender: Mapping[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    tool_state: str = ""
    expects_reply: bool | None = None
    steer: bool | None = None
    # Optional local source proof; the standalone/pinned corpus supplies none.
    work: WorkMembership | None = None


#: The kinds the BASE vocabulary already classifies as interior — a row every
#: reader may have hidden. Named ONCE and read by both the base checks and the
#: work gate below: two copies of this list are how the work arm came to rescue
#: a FAILED ``tool`` row (the deny-list omitted ``tool``/``inside``, so a tool
#: row carrying ``work`` was classified by its membership role, and a
#: failed/stopped call — ``role=boundary`` — split the very run it must fold
#: into, contradicting the shared parity fixture's own failed-call case).
WORK_ARM_EXEMPT_KINDS: frozenset[str] = frozenset(
    {"tool", "peer", "hub", "wake", "monitor", "inside"}
)


def quiet_family_of(record: QuietGroupRecord) -> QuietGroupFamily | None:
    """The family this row contributes when it is a trigger, or null.

    The default projection positively classifies peer/wake/monitor rows only.
    Optional LOCAL owned-work metadata admits a source-backed delivered job
    result inside a genuine user/tool interval. Generic INFO notices never
    acquire that authority. The hash-pinned standalone corpus supplies none.
    """
    # The work arm ADMITS what the base vocabulary cannot classify (a delivered
    # job result inside a genuine user/tool interval) and never re-classifies a
    # row the base already knows. A receipt kind keeps its base family: the
    # internal-work view calls a hub STEER a boundary because it demands
    # attention, and reading that role here turned the steer into a run
    # SPLITTER on a row that the fold has always been able to hide, breaking
    # the ordering/prefix runs (test_hub_message ordered-repair, seated-picture
    # and human-prefix cases). The role belongs to the membership view, not to
    # this fold's boundary rule.
    if record.work is not None and record.kind not in WORK_ARM_EXEMPT_KINDS:
        if record.work.role != "receipt":
            return None
        if record.work.family in ("peer", "hub"):
            return "peer"
        if record.work.family == "job_result":
            return "job"
        if record.work.family == "wake":
            return "wake"
        if record.work.family == "monitor":
            return "monitor"
        return None
    if record.kind in ("peer", "hub"):
        return "peer"
    if record.kind == "wake":
        return "wake"
    if record.kind == "monitor":
        return "monitor"
    return None


def is_quiet_group_trigger(record: QuietGroupRecord) -> bool:
    """A trigger row: the group's >= 2 unit (the definition's countable rows)."""
    return quiet_family_of(record) is not None


def group_splitter_of(record: QuietGroupRecord) -> bool:
    """Does this row SPLIT a quiet group?

    The boundary vocabulary is this surface's own visibility list: every row a
    reader must be able to see is a boundary, so a group can never hide one.
    A ``user`` row splits, a compaction statement splits, visible assistant
    prose splits, and a notice splits — this surface cannot tell a monitor
    prompt or a job result from a generic notice once it is a notice row.
    Tool rows sit inside, and so do the two rows that paint nothing a reader
    would miss: an empty or still-streaming assistant row, and this surface's
    ``inside`` kind (an image a folded tool row produced).
    """
    # Same gate as `quiet_family_of`: the work arm speaks only for rows the
    # base vocabulary does not classify, so a receipt kind keeps its base
    # reading (not a splitter) whatever the membership view calls its role.
    if record.work is not None and record.kind not in WORK_ARM_EXEMPT_KINDS:
        return record.work.role == "boundary"
    if record.kind in WORK_ARM_EXEMPT_KINDS:
        return False
    if record.kind == "assistant":
        return bool(record.text.strip())
    return True


def is_quiet_turn_call(record: QuietGroupRecord) -> bool:
    """Whether a tool row is the quiet-turn tool's (``no_reply``).

    Core owns the literal (``harness/rows.py``) and both folds already hide the
    pair (S1), so a matching row should not arrive here at all — the exclusion
    is kept because the shared fixture replays the pairs, and because a
    mixed-build runtime that predates the fold would otherwise have its quiet
    call counted as an action.
    """
    return record.kind == "tool" and is_quiet_turn_name(record.tool_name)


def is_failed_call(record: QuietGroupRecord) -> bool:
    """Is this tool row a genuine FAILURE? Only ``tool_state == "failed"``.

    This surface maps a settled card's ``error`` mark onto ``failed`` and keeps
    ``interrupted`` its own state, so a call the user stopped is not counted as
    a failure — the same exclusion the relay fold and the desktop's
    ``isFailedCall`` make.
    """
    return record.kind == "tool" and record.tool_state == "failed"


#: The sender ladder's separators. Split on BOTH rather than one: the desktop
#: runs on Windows too, where a bare ``/`` split keeps the whole path as a name
#: (local-operator-ui ``receipt-row-model.ts`` makes the same call).
_TRAILING_SEPARATORS = re.compile(r"[\\/]+$")
_PATH_SEPARATOR = re.compile(r"[\\/]")


def quiet_sender_label(sender: Mapping[str, Any]) -> str:
    """The identity a peer receipt contributes to a group's sender summary.

    The spelling mirrors the relay-web port's ``quietSenderLabel`` (itself the
    desktop's ``peerIdentity`` ladder): a name the peer CHOSE (the conversation
    name) is quoted, the ladder's guesses (a cwd basename, marked with its
    trailing slash, then a short session id) are not, and a senderless receipt
    lands on ``another session`` — the same vocabulary ``harness/comms.py``
    uses, kept identical so a reader meeting one in the summary and one in a
    card does not think they are two different states. Total over partial
    senders: every missing field falls to the ladder's next rung rather than
    raising.
    """
    name = str(sender.get("conversation_name") or "")
    if name:
        return f'"{name}"'
    cwd = _TRAILING_SEPARATORS.sub("", str(sender.get("cwd") or ""))
    if cwd:
        base = _PATH_SEPARATOR.split(cwd)[-1]
        # A trailing slash says "this is a directory", which is the only
        # thing that distinguishes a guessed name from a chosen one when
        # both are unquoted.
        if base:
            return f"{base}/"
    session_id = str(sender.get("session_id") or "")
    if session_id:
        # A short prefix: a full ULID is 26 cells of entropy that helps no
        # reader and pushes the count off the row.
        return session_id[:8]
    pid = sender.get("pid")
    return f"pid {pid}" if pid else "another session"


def quiet_group_of_segment(
    records: Sequence[QuietGroupRecord],
    span: tuple[int, int],
    *,
    span_head_loaded: bool = True,
    open: bool | None = None,
) -> QuietGroup | None:
    """The quiet group a single span IS, or None when the span is not one.

    THE SPAN MUST BE THE WHOLE GROUP (design §5): its neighbours must be
    splitters or the list's edges (``quiet_group_stretches`` builds spans that
    way), and no splitter may sit inside. A span that merely OVERLAPS a group —
    a caller that sliced inside one — refuses here instead of stating a count
    over part of something: the caller degrades to the rows' ordinary cards,
    which is the safe direction. The shared fixture pins the refusal (its
    sub-span case).

    ``span_head_loaded`` (default True) is the caller's own head-cut verdict
    for this span: false nulls the times and leaves the count a minimum — the
    "at least N" rule, which the caller states in its own vocabulary. ``open``
    (default: the span reaches the list's end) is the tail fact the
    growing-group rule reads.
    """
    span_from, span_to = span
    for i in range(span_from, span_to + 1):
        if group_splitter_of(records[i]):
            return None
    if span_from > 0 and not group_splitter_of(records[span_from - 1]):
        return None
    if span_to + 1 < len(records) and not group_splitter_of(records[span_to + 1]):
        return None
    triggers = [
        records[i] for i in range(span_from, span_to + 1) if is_quiet_group_trigger(records[i])
    ]
    # Mixed internal work may fold with one incoming receipt only when three
    # REAL tools are present. Receipt-only groups retain their existing minimum;
    # a tool-only span is never eligible as a side effect of this extension.
    real_actions = sum(
        records[i].kind == "tool" and not is_quiet_turn_call(records[i])
        for i in range(span_from, span_to + 1)
    )
    members = [records[i].work for i in range(span_from, span_to + 1)]
    owned = members[0] if members else None
    contextual = owned is not None and all(
        member is not None
        and member.scope_id == owned.scope_id
        and member.stretch_id == owned.stretch_id
        and member.role != "boundary"
        for member in members
    )
    # Contextual user work has the existing action threshold; standalone quiet
    # groups keep their exact receipt-only / mixed-work admission vocabulary.
    if len(triggers) < 2 and not ((triggers or contextual) and real_actions >= 3):
        return None
    first = records[span_from] if contextual else triggers[0]
    last = triggers[-1] if triggers else records[span_to]
    actions = 0
    failed = 0
    # Peer senders keep their first-appearance order for the tie-break below.
    sender_counts: dict[str, int] = {}
    for i in range(span_from, span_to + 1):
        record = records[i]
        if record.kind == "tool" and not is_quiet_turn_call(record):
            actions += 1
            if is_failed_call(record):
                failed += 1
            continue
        if record.kind in ("peer", "hub"):
            label = quiet_sender_label(record.sender)
            sender_counts[label] = sender_counts.get(label, 0) + 1
    family: QuietGroupFamily | None = None
    mixed = False
    for trigger in triggers:
        trigger_family = quiet_family_of(trigger)
        if family is None:
            family = trigger_family
        elif family != trigger_family:
            mixed = True
    family_final: QuietGroupFamily = "mixed" if mixed else (family or "mixed")
    senders: list[QuietGroupSender] = []
    if family_final == "peer":
        entries = list(sender_counts.items())
        order = {label: index for index, (label, _) in enumerate(entries)}
        entries.sort(key=lambda entry: (-entry[1], order[entry[0]]))
        top = [QuietGroupSender(label=label, count=count) for label, count in entries[:2]]
        if len(entries) > 2:
            rest = sum(count for _, count in entries[2:])
            top.append(QuietGroupSender(label=f"{len(entries) - 2} more", count=rest))
        senders.extend(top)
    return QuietGroup(
        key=f"qg:{records[span_from].id}",
        family=family_final,
        count=len(triggers),
        first_ts=first.ts if span_head_loaded else None,
        last_ts=last.ts if span_head_loaded else None,
        senders=tuple(senders),
        actions=actions,
        failed=failed,
        open=open if open is not None else (span_to == len(records) - 1),
        row_ids=tuple(records[i].id for i in range(span_from, span_to + 1)),
        mixed_actions=real_actions >= 3
        and (contextual or len(triggers) < 2 or any(record.kind == "hub" for record in triggers)),
        reply_requested=sum(record.expects_reply is True for record in triggers),
        steered=sum(record.steer is True for record in triggers),
    )


def quiet_group_stretches(records: Sequence[QuietGroupRecord]) -> list[tuple[int, int]]:
    """Every candidate span between splitters, in order.

    The list's own edges are legitimate bounds — that is how a head-cut
    leading span forms — and the caller states the minimum in its own
    vocabulary (see ``quiet_group_of_segment``). A stretch with fewer than two
    triggers is not a group; scoring is the caller's call, because only the
    caller knows whether a span's outside neighbours are splitters.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(len(records) + 1):
        if i < len(records) and not group_splitter_of(records[i]):
            continue
        if i > start:
            spans.append((start, i - 1))
        start = i + 1
    return spans


def quiet_groups_of(records: Sequence[QuietGroupRecord]) -> list[QuietGroup]:
    """Every quiet group in a records list, in order: the stretches between
    splitters, each scored by ``quiet_group_of_segment`` (so a stretch with
    fewer than two triggers yields none)."""
    groups: list[QuietGroup] = []
    for span in quiet_group_stretches(records):
        group = quiet_group_of_segment(records, span)
        if group is not None:
            groups.append(group)
    return groups


def quiet_group_label(family: QuietGroupFamily) -> str:
    """The family's word for a group bar: family plural, ``Messages`` for a mix
    (design §5's copy table; sole author of these words)."""
    if family == "peer":
        return "Peer messages"
    if family == "wake":
        return "Wake messages"
    if family == "monitor":
        return "Monitor messages"
    if family == "job":
        return "Job results"
    return "Messages"
