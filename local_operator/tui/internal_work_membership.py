"""Presentation-only internal work ownership, equivalent to the desktop projection.

This vocabulary is intentionally separate from the hash-pinned quiet-group
corpus: default quiet thresholds and standalone receipt policy do not change.
Adapters supply typed source facts, never a guessed meaning from rendered text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from local_operator.harness.rows import is_quiet_turn_name

Role = Literal["action", "receipt", "bridge", "boundary"]
Attention = Literal["none", "required", "unknown"]
Family = Literal[
    "tool", "peer", "hub", "job_result", "compaction", "subagent_completion", "wake", "monitor"
]


@dataclass(frozen=True)
class WorkRecord:
    """Local projection input; no model context or protocol fields are added."""

    source_id: str
    kind: str
    tool_name: str = ""
    custom_type: str = ""
    provisional: bool = False
    streaming: bool = False
    final: bool = False
    is_error: bool = False
    stopped: bool = False
    never_sent: bool = False
    not_run: bool = False
    real_tool: bool = True
    material: bool = False
    level: str = ""
    producer: str = ""
    job_id: str | None = None
    launched_job_id: str | None = None
    launched_job_ids: tuple[str, ...] = ()
    status: str | None = None
    held: bool = False
    unknown_held: bool = False
    expects_reply: bool | None = None
    steer: bool | None = None
    direction: str = ""


@dataclass(frozen=True)
class WorkMembership:
    source_id: str
    scope_id: str | None
    stretch_id: str | None
    role: Role
    family: Family | None
    attention: Attention
    owner_evidence: str
    job_id: str | None = None
    outcome: Literal["completed", "failed", "unknown"] | None = None


def _id(value: str | None) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class WorkProjector:
    """Incremental adapter over the SAME policy; reset with the session/view epoch.

    No transcript body is cached. An overflow of typed launch owners refuses
    result correlation rather than forgetting a conflict and guessing ownership.
    """

    def __init__(self) -> None:
        self.scope: str | None = None
        self.has_tool = False
        self.stretch: str | None = None
        self.owners: dict[str, str | None] = {}
        self.complete = True

    def step(self, record: WorkRecord) -> WorkMembership:
        if record.kind == "user":
            self.scope = None if record.provisional else _id(record.source_id)
            self.has_tool = False
            self.stretch = None
        role: Role = "boundary"
        family: Family | None = None
        attention: Attention = "none"
        job_id: str | None = None
        outcome: Literal["completed", "failed", "unknown"] | None = None
        if record.kind == "tool":
            family = "tool"
            # An executed material/error row authorises no new prompt, but is
            # still real work under this USER; only its presentation must split.
            if (
                self.scope is not None
                and record.real_tool
                and not record.not_run
                and not record.never_sent
                and not record.stopped
                and not is_quiet_turn_name(record.tool_name)
            ):
                self.has_tool = True
                for candidate in (record.launched_job_id, *record.launched_job_ids):
                    launched = _id(candidate)
                    if launched:
                        old = self.owners.get(launched, self.scope)
                        if launched in self.owners or len(self.owners) < 4096:
                            self.owners[launched] = self.scope if old == self.scope else None
                        else:
                            self.complete = False
            if (
                record.tool_name == "ask"
                or record.is_error
                or record.stopped
                or record.never_sent
                or record.not_run
                or record.material
            ):
                attention = "required"
            elif not record.real_tool:
                attention = "unknown"
            elif is_quiet_turn_name(record.tool_name):
                role = "bridge"
            else:
                role = "action"
                if self.scope is not None:
                    self.has_tool = True

        elif record.kind in ("peer", "wake"):
            family = "peer" if record.kind == "peer" else "wake"
            role = "receipt"
        elif record.kind == "compaction":
            family = "compaction"
            role = "bridge"
        elif record.kind == "custom" and record.custom_type == "hub_message":
            family = "hub"
            job_id = _id(record.job_id)
            if (
                record.expects_reply is False
                and record.steer is False
                and record.direction in ("to_parent", "to_child")
            ):
                role = "receipt"
            else:
                attention = "required" if record.expects_reply or record.steer else "unknown"
        elif record.kind == "custom" and record.custom_type == "job_result":
            family = "job_result"
            job_id = _id(record.job_id)
            outcome = (
                "completed"
                if record.status == "completed"
                else "failed" if record.status == "failed" else "unknown"
            )
            if record.level in ("warning", "error") or outcome == "failed" or record.held:
                attention = "required"
            elif (
                record.unknown_held
                or (self.scope is not None and not self.complete)
                or (record.status is not None and record.status != "completed")
            ):
                attention = "unknown"
            elif not job_id or record.producer != "job_result":
                attention = "unknown"
            else:
                role = "receipt"
        elif record.kind == "notice" and record.producer == "subagent_completion":
            family = "subagent_completion"
            job_id = _id(record.job_id)
            outcome = (
                "completed"
                if record.status == "completed"
                else "failed" if record.status == "failed" else "unknown"
            )
            if record.level != "info" or outcome == "failed":
                attention = "required"
            elif outcome == "completed" and job_id:
                role = "bridge"
            else:
                attention = "unknown"
        elif record.kind == "custom" and record.custom_type == "monitor_prompt":
            family = "monitor"
            attention = "unknown"
        conflict = (
            self.scope is not None
            and job_id is not None
            and job_id in self.owners
            and self.owners[job_id] != self.scope
        )
        if (
            self.scope is not None
            and job_id is not None
            and not self.complete
            and attention == "none"
        ):
            attention = "unknown"
        authorised = (
            self.scope is not None and self.has_tool and not conflict and attention == "none"
        )
        if not authorised or role == "boundary":
            role = "boundary"
            self.stretch = None
        elif self.stretch is None:
            self.stretch = record.source_id
        evidence = (
            "conflicting-job-owner"
            if conflict
            else (
                "attention"
                if attention != "none"
                else (
                    "missing-user"
                    if self.scope is None
                    else (
                        "no-prior-tool"
                        if not self.has_tool
                        else (
                            "visible-boundary"
                            if role == "boundary"
                            else "loaded-user-tool-interval"
                        )
                    )
                )
            )
        )
        member = WorkMembership(
            record.source_id,
            self.scope,
            self.stretch,
            role,
            family,
            attention,
            evidence,
            job_id,
            outcome,
        )
        if record.kind == "assistant" and record.final and not record.streaming:
            self.scope = None
            self.has_tool = False
            self.stretch = None
        return member


def project_internal_work(records: Sequence[WorkRecord]) -> dict[str, WorkMembership]:
    """Pure replay of the incremental policy over the loaded source window."""
    projector = WorkProjector()
    return {record.source_id: projector.step(record) for record in records}


def owned_work_clauses(
    records: Sequence[WorkRecord], memberships: dict[str, WorkMembership], *, partial: bool = False
) -> tuple[str, ...] | None:
    """Only distinct source members, no wire totals/nested eval estimates/time."""
    counts: dict[str, int] = {}
    seen: set[str] = set()
    owner: tuple[str, str] | None = None
    for record in records:
        if record.source_id in seen:
            continue
        seen.add(record.source_id)
        member = memberships.get(record.source_id)
        if (
            member is None
            or member.scope_id is None
            or member.stretch_id is None
            or member.role == "boundary"
        ):
            return None
        current_owner = (member.scope_id, member.stretch_id)
        if owner is not None and owner != current_owner:
            return None
        owner = current_owner
        label: str | None = None
        if member.role == "action":
            label = (
                "shell actions"
                if record.tool_name == "bash"
                else (
                    "subagent launches"
                    if record.tool_name == "task"
                    else (
                        "messages sent"
                        if record.tool_name in ("hub", "send")
                        else f"{record.tool_name} actions"
                    )
                )
            )
        elif member.role == "receipt":
            label = (
                "job results"
                if member.family == "job_result"
                else (
                    "wake receipts"
                    if member.family == "wake"
                    else "monitor receipts" if member.family == "monitor" else "incoming messages"
                )
            )
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    singular = {
        "messages sent": "message sent",
        "subagent launches": "subagent launch",
        "incoming messages": "incoming message",
    }
    return (
        tuple(
            f"{count}{'+' if partial else ''} "
            + (
                singular.get(label, label[:-1] if label.endswith("s") else label)
                if count == 1
                else label
            )
            for label, count in counts.items()
        )
        if owner is not None
        else None
    )


@dataclass(frozen=True)
class WorkGroupIdentity:
    key: str
    ids: tuple[str, ...]
    anchor_id: str
    session_id: str | None
    view_generation: int
    user_id: str | None
    stretch_id: str
    expanded: bool = False
    known_user_id: str | None = None
    mixed_owners: bool = False
    # Completed peer cycles have no genuine USER authority. Their finite quiet
    # call is an additional owner discriminator, never a synthetic user ID.
    owner_kind: Literal["user", "peer_cycle"] = "user"
    closure_id: str | None = None


@dataclass(frozen=True)
class WorkGroupSnapshot:
    groups: tuple[WorkGroupIdentity, ...]
    owners: dict[str, tuple[int, int]]
    complete: bool
    reason: str | None = None


def capture_work_groups(
    groups: Sequence[WorkGroupIdentity], source_ids: Sequence[str]
) -> WorkGroupSnapshot:
    """The existing view's complete old owners AND pinned inventory; <=4096 IDs.

    Not a global registry. A snapshot lives only through one source-change/page
    transaction. Closed groups remain owners; only OPEN preferences have a128 cap.
    """

    def refuse(reason: str) -> WorkGroupSnapshot:
        return WorkGroupSnapshot((), {}, False, reason)

    if len(source_ids) > 4096:
        return refuse("prior-id-limit")
    if len(set(source_ids)) != len(source_ids):
        return refuse("duplicate-prior-id")
    source_set = set(source_ids)
    if sum(group.expanded for group in groups) > 128:
        return refuse("prior-open-limit")
    if len({group.key for group in groups}) != len(groups):
        return refuse("prior-key-collision")
    owners: dict[str, tuple[int, int]] = {}
    for ordinal, group in enumerate(groups):
        for order, source_id in enumerate(group.ids):
            if source_id in owners:
                return refuse("duplicate-prior-owner")
            if source_id not in source_set:
                return refuse("missing-prior-source")
            owners[source_id] = (ordinal, order)
    for source_id in source_ids:
        owners.setdefault(source_id, (-1, -1))
    return WorkGroupSnapshot(tuple(groups), owners, True)


def retained_work_group(
    previous: WorkGroupSnapshot,
    current: WorkGroupIdentity,
    outside_ids: Sequence[str],
    *,
    current_ids_unique: bool = False,
) -> int | None:
    """ONE ordered owner with its opaque anchor, not any-overlap open inheritance.

    A legacy qg may learn its USER once the real head loads, but only inside the
    same KNOWN actual session/local view generation. Known-owner changes refuse.
    Callers compute all replacements against the same snapshot before removing
    any bar; ownership outside the candidate detects a split, including pinned
    source promotions. Explicit close is read from the retained owner's flag.
    """
    if (
        not previous.complete
        or not current.session_id
        or not current.session_id.strip()
        or type(current.view_generation) is not int
        or current.view_generation < 0
    ):
        return None
    if not current_ids_unique and (
        len(current.ids) > 4096 or len(set(current.ids)) != len(current.ids)
    ):
        return None
    matched: set[str] = set()
    owner: int | None = None
    last_order = -1
    for source_id in current.ids:
        old = previous.owners.get(source_id)
        if old is None:
            continue
        ordinal, order = old
        if (
            source_id in matched
            or ordinal < 0
            or (owner is not None and ordinal != owner)
            or order <= last_order
        ):
            return None
        matched.add(source_id)
        owner, last_order = ordinal, order
    if owner is None:
        return None
    old = previous.groups[owner]
    if (
        old.session_id != current.session_id
        or old.view_generation != current.view_generation
        or not old.session_id
    ):
        return None
    if old.mixed_owners or current.mixed_owners:
        return None
    if old.owner_kind == "peer_cycle" or current.owner_kind == "peer_cycle":
        # A default qg can acquire its first real closure certificate, but only
        # over its ENTIRE exact inventory, with no known USER owner. This is
        # certification of the same bar, never an any-overlap expansion merge.
        legacy_close = (
            old.owner_kind == "user"
            and current.owner_kind == "peer_cycle"
            and old.user_id is None
            and old.known_user_id is None
            and old.closure_id is None
            and current.user_id is None
            and bool(current.closure_id)
            and old.ids == current.ids
            and old.anchor_id == current.anchor_id
            and old.stretch_id == current.stretch_id
        )
        if not legacy_close and (
            old.owner_kind != current.owner_kind
            or not old.closure_id
            or old.closure_id != current.closure_id
            or old.user_id is not None
            or current.user_id is not None
            or old.stretch_id != current.stretch_id
            or old.ids != current.ids
        ):
            return None
    if old.user_id is None:
        if current.user_id is not None and old.known_user_id not in (None, current.user_id):
            return None
        if current.user_id is not None and any(source_id not in matched for source_id in old.ids):
            return None  # The one upgrade requires ALL old ordered members.
        if not old.key.startswith("qg:") or (
            current.user_id is None and old.stretch_id != current.stretch_id
        ):
            return None
    elif old.user_id != current.user_id or old.stretch_id != current.stretch_id:
        return None
    if not old.anchor_id or old.anchor_id not in matched:
        return None
    for source_id in outside_ids:
        prior = previous.owners.get(source_id)
        if prior is not None and prior[0] == owner and source_id not in matched:
            return None
    return owner
