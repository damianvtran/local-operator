"""Live, host-owned receipt actions. Transcript flags never create authority.

Wire contract (schema 1): a saved ToolResult may carry ``profile_action`` with
``{schema, kind:'profile_revert', action_id, label:'Revert', profile_kind,
profile_name, receipt_id}``. The authenticated session read ``profile_action_status``
accepts those four identity fields plus the trusted row's ``tool_call_id``;
its result contains ONLY ``{action_id, state}``. Frontends enable a button only
for a fresh ``available`` response, never for a persisted verification flag.

Activation reuses authority-gated ``approval_answer`` with request_id=action_id,
approved=True, remember=False. A timeout is PENDING/UNCONFIRMED until the live
read reconciles it. Completed records are cached: duplicates never replay an
inverse. The process owns registration; restart loses immediate-owner records.
These are discipline, not a same-uid filesystem security boundary.
"""

from __future__ import annotations

import copy
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from local_operator.definition_transaction import (
    DefinitionTransactionConflict,
    atomic_edit,
    commitment,
    definition_lock,
)

ProfileActionState = Literal["available", "pending", "succeeded", "stale", "failed", "unavailable"]


class ProfileActionReply(TypedDict):
    action_id: str
    state: ProfileActionState


@runtime_checkable
class ProfileActionStatusCapability(Protocol):
    """Additive owner capability; older handles need not implement receipt reads.

    Discovery is not authorization: the transport applies its existing view and
    operator checks before dispatch. Explicit keys cannot become transport kwargs.
    """

    async def profile_action_status(
        self,
        *,
        action_id: str,
        receipt_id: str,
        profile_kind: str,
        profile_name: str,
        tool_call_id: str,
    ) -> ProfileActionReply: ...


class ProfileAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = Field(default=1, alias="schema")
    kind: Literal["profile_revert"] = "profile_revert"
    action_id: str = Field(pattern=r"^prv_[A-Za-z0-9_-]{32}$")
    label: Literal["Revert"] = "Revert"
    profile_kind: Literal["agent", "team"]
    profile_name: str = Field(min_length=1, max_length=128)
    receipt_id: str = Field(min_length=1, max_length=160)


def signed_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    """Stable identity only; callers must verify before registering authority."""
    from local_operator.profile_approval import _proposal

    p = _proposal(record)
    token = commitment([record["request_digest"], p.session_id, p.receipt_id]).split(":")[1][:32]
    return ProfileAction(
        action_id="prv_" + token,
        profile_kind=p.profile_kind,
        profile_name=p.name,
        receipt_id=p.receipt_id,
    ).model_dump(by_alias=True)


@dataclass
class _BoundAction:
    descriptor: ProfileAction
    tool_call_id: str
    target_id: str
    inverse: dict[str, Any]
    after: dict[str, Any]
    backup_path: Path
    backup_commitment: str
    state: ProfileActionState = "available"


def restore_bound_profile(
    root: Path, profile_kind: str, target_id: str, expected: dict[str, Any], inverse: dict[str, Any]
) -> None:
    """The shared CAS/inverse core for a granted tool revert or owner gesture.

    Authority is the caller's responsibility. This layer restores only authored
    fields, under the same write boundary as competing supported writers, and
    rolls metadata/briefs back together on an ordinary fault.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.profile_approval import (
        AgentImage,
        TeamImage,
        snapshot,
        write_image,
    )
    from local_operator.teams import TeamRegistry

    model = AgentImage if profile_kind == "agent" else TeamImage
    for image in (expected, inverse):
        model.model_validate(image)
        if image["id"] != target_id or image["name"] != expected["name"]:
            raise ValueError("the bound inverse does not match its target")
    with definition_lock(root):
        if commitment(snapshot(root, profile_kind, target_id)) != commitment(expected):
            raise DefinitionTransactionConflict("the saved profile changed; nothing was restored")
        registry = (
            AgentRegistry(root, migrate=False) if profile_kind == "agent" else TeamRegistry(root)
        )
        with atomic_edit(
            root,
            owner=registry,
            read=lambda: snapshot(root, profile_kind, target_id),
            restore=lambda image: write_image(
                root, profile_kind, target_id, image, registry=registry
            ),
        ):
            write_image(root, profile_kind, target_id, inverse, registry=registry)
            if commitment(snapshot(root, profile_kind, target_id)) != commitment(inverse):
                raise DefinitionTransactionConflict("revert did not restore its bound snapshot")


def _action_binding(session_id: str, record: _BoundAction) -> dict[str, Any]:
    return {
        "action": record.descriptor.model_dump(by_alias=True),
        "session_id": session_id,
        "tool_call_id": record.tool_call_id,
        "target_id": record.target_id,
        "backup_commitment": record.backup_commitment,
    }


def _prepare_action(root: Path, session_id: str, record: _BoundAction) -> None:
    from local_operator.definition_transaction import prepare

    prepare(
        root,
        transaction_id="revert_" + record.descriptor.action_id,
        before=record.after,
        after=record.inverse,
        binding=_action_binding(session_id, record),
    )


def prepare_signed_revert(root: Path, verified: dict[str, Any]) -> None:
    """Called only by the authenticated save producer, never by recovery/read."""
    from local_operator.profile_approval import _proposal

    p = _proposal(verified)
    if p.before is None:
        return
    descriptor = ProfileAction.model_validate(signed_descriptor(verified))
    payload = {
        "schema": 1,
        "kind": p.profile_kind,
        "name": p.name,
        "id": p.target_id,
        "approval_receipt_id": p.receipt_id,
        "before": p.before,
        "base_commitment": p.base_commitment,
        "after_commitment": p.after_commitment,
    }
    record = _BoundAction(
        descriptor,
        p.tool_call_id,
        p.target_id,
        p.before,
        p.after,
        root / "sessions" / p.session_id / "profile-backups" / (p.receipt_id + ".json"),
        commitment(payload),
    )
    _prepare_action(root, p.session_id, record)


class ProfileActions:
    """One Session's private registrations; no file/transcript loader exists."""

    def __init__(self, session_id: str, root: Path | None = None) -> None:
        self.session_id = session_id
        self._root = Path(root) if root is not None else None
        self._records: dict[str, _BoundAction] = {}
        self._roots: dict[str, Path] = {}

    def register(
        self,
        *,
        root: Path,
        profile_kind: str,
        profile_name: str,
        target_id: str,
        tool_call_id: str,
        receipt_id: str,
        before: dict[str, Any],
        after: dict[str, Any],
        backup_path: Path,
        backup_commitment: str,
    ) -> dict[str, Any]:
        # Called ONLY by host receipt code holding the original before image.
        # Reading an editable backup here would turn a file into registration
        # authority; its commitment is captured while writing the trusted backup.
        from local_operator.profile_approval import AgentImage, TeamImage

        model = AgentImage if profile_kind == "agent" else TeamImage
        for image in (before, after):
            model.model_validate(image)
            if image["id"] != target_id or image["name"] != profile_name:
                raise ValueError("receipt action does not match its target")
        if profile_kind not in ("agent", "team"):
            raise ValueError("unknown profile kind")
        descriptor = ProfileAction(
            action_id="prv_" + secrets.token_urlsafe(24),
            profile_kind=profile_kind,
            profile_name=profile_name,
            receipt_id=receipt_id,
        )
        action_id = descriptor.action_id
        self._records[action_id] = _BoundAction(
            descriptor,
            tool_call_id,
            target_id,
            copy.deepcopy(before),
            copy.deepcopy(after),
            backup_path,
            backup_commitment,
        )
        self._roots[action_id] = Path(root)
        _prepare_action(Path(root), self.session_id, self._records[action_id])
        return descriptor.model_dump(by_alias=True)

    def _recover_signed(self, identity: dict[str, str]) -> None:
        # No immediate-owner/transcript/backup loader exists. Recovery is ONLY
        # from a verified signed proposal with its exact host session and row.
        if self._root is None:
            return
        from local_operator.network import approvals as A
        from local_operator.network.types import MeshRefusal
        from local_operator.network.wire import crockford
        from local_operator.profile_approval import _proposal

        try:
            receipt = identity["receipt_id"]
            request_id = "req_" + crockford(bytes.fromhex(receipt[:16]))
            record = A._find_by_request_id(request_id, self._root)
            if record is None or record.get("state") != A.STATE_CONNECTED:
                return
            verified = A.verify_for_run(record["approval_id"], root=self._root)
            from local_operator.network.credentials import desktop_approval

            if desktop_approval.assurance()[0] != desktop_approval.ASSURANCE_PRESENCE:
                return
            p = _proposal(verified)
            descriptor = ProfileAction.model_validate(signed_descriptor(verified))
            if (
                p.before is None
                or p.session_id != self.session_id
                or any(
                    (
                        descriptor.action_id != identity["action_id"],
                        p.receipt_id != receipt,
                        p.tool_call_id != identity["tool_call_id"],
                        p.profile_kind != identity["profile_kind"],
                        p.name != identity["profile_name"],
                    )
                )
            ):
                return
            with definition_lock(self._root):
                if descriptor.action_id in self._records:
                    return  # A concurrent recovery already owns the record.
                payload = {
                    "schema": 1,
                    "kind": p.profile_kind,
                    "name": p.name,
                    "id": p.target_id,
                    "approval_receipt_id": p.receipt_id,
                    "before": p.before,
                    "base_commitment": p.base_commitment,
                    "after_commitment": p.after_commitment,
                }
                path = (
                    self._root
                    / "sessions"
                    / p.session_id
                    / "profile-backups"
                    / (p.receipt_id + ".json")
                )
                self._records[descriptor.action_id] = _BoundAction(
                    descriptor,
                    p.tool_call_id,
                    p.target_id,
                    copy.deepcopy(p.before),
                    copy.deepcopy(p.after),
                    path,
                    commitment(payload),
                )
                self._roots[descriptor.action_id] = self._root
        except (MeshRefusal, ValueError, KeyError, OSError, TypeError):
            return

    def _consume(self, action_id: str, record: _BoundAction, *, mutate: bool = True):
        from local_operator.definition_transaction import transact
        from local_operator.profile_approval import snapshot, write_image

        root = self._roots[action_id]

        def write(image):
            if not mutate:
                raise DefinitionTransactionConflict("receipt could not be applied")
            write_image(root, record.descriptor.profile_kind, record.target_id, image)

        return transact(
            root,
            transaction_id="revert_" + action_id,
            before=record.after,
            after=record.inverse,
            read=lambda: snapshot(root, record.descriptor.profile_kind, record.target_id),
            write=write,
            binding=_action_binding(self.session_id, record),
            one_shot=True,
        )

    def _settle(self, action_id: str, record: _BoundAction) -> ProfileActionState:
        import json

        from local_operator.definition_transaction import transact
        from local_operator.profile_approval import snapshot, write_image

        root = self._roots[action_id]
        try:
            raw = json.loads(
                (root / "profile-transactions" / ("revert_" + action_id + ".json")).read_text()
            )
            if (
                not isinstance(raw, dict)
                or raw.get("schema") != 1
                or raw.get("before") != commitment(record.after)
                or raw.get("after") != commitment(record.inverse)
                or raw.get("binding") != _action_binding(self.session_id, record)
            ):
                return "unavailable"
            state = raw.get("state")
            if state == "complete":
                return "succeeded"
            if state == "failed":
                return "failed"
            if state == "applying":
                # Only an already-after image can finish uncertain bookkeeping.
                # A before-looking row could be ABA and must never re-execute.
                if commitment(
                    snapshot(root, record.descriptor.profile_kind, record.target_id)
                ) != commitment(record.inverse):
                    return "pending"
                transact(
                    root,
                    transaction_id="revert_" + action_id,
                    before=record.after,
                    after=record.inverse,
                    read=lambda: snapshot(root, record.descriptor.profile_kind, record.target_id),
                    write=lambda image: write_image(
                        root, record.descriptor.profile_kind, record.target_id, image
                    ),
                    binding=_action_binding(self.session_id, record),
                    one_shot=True,
                )
                return "succeeded"
            if state != "prepared":
                return "unavailable"
            if record.state == "pending":
                return "pending"
            return (
                "available"
                if commitment(snapshot(root, record.descriptor.profile_kind, record.target_id))
                == commitment(record.after)
                else "stale"
            )
        except (OSError, ValueError, DefinitionTransactionConflict):
            return "unavailable"

    def status(
        self,
        *,
        action_id: str,
        receipt_id: str,
        profile_kind: str,
        profile_name: str,
        tool_call_id: str,
    ) -> ProfileActionReply:
        if action_id not in self._records:
            self._recover_signed(
                dict(
                    action_id=action_id,
                    receipt_id=receipt_id,
                    profile_kind=profile_kind,
                    profile_name=profile_name,
                    tool_call_id=tool_call_id,
                )
            )
        record = self._records.get(action_id)
        if record is None or (
            record.descriptor.receipt_id != receipt_id
            or record.descriptor.profile_kind != profile_kind
            or record.descriptor.profile_name != profile_name
            or record.tool_call_id != tool_call_id
        ):
            return {"action_id": action_id, "state": "unavailable"}
        with definition_lock(self._roots[action_id]):
            record.state = self._settle(action_id, record)
        return {"action_id": action_id, "state": record.state}

    def knows(self, action_id: str) -> bool:
        return action_id.startswith("prv_")

    def activate(self, action_id: str, *, approved: bool, remember: bool) -> ProfileActionReply:
        record = self._records.get(action_id)
        if record is None:
            return {"action_id": action_id, "state": "unavailable"}
        if remember:
            raise ValueError("receipt actions never enable session approval")
        root = self._roots[action_id]
        with definition_lock(root):
            # Serialized by the same per-root reentrant lock as the actual
            # write, so two simultaneous authenticated gestures apply once.
            record.state = self._settle(action_id, record)
            if record.state != "available":
                return {"action_id": action_id, "state": record.state}
            if not approved:
                record.state = "unavailable"
                return {"action_id": action_id, "state": record.state}
            record.state = "pending"
            started = False
            import json

            from local_operator.profile_approval import snapshot

            try:
                current = snapshot(root, record.descriptor.profile_kind, record.target_id)
                if commitment(current) != commitment(record.after):
                    record.state = "stale"
                    return {"action_id": action_id, "state": record.state}
                if (
                    commitment(json.loads(record.backup_path.read_text()))
                    != record.backup_commitment
                ):
                    raise ValueError("the receipt backup changed")
                started = True
                outcome = self._consume(action_id, record)
                record.state = "succeeded" if outcome["state"] == "complete" else "failed"
            except (OSError, ValueError, DefinitionTransactionConflict):
                if not started:
                    # A rejected backup has not touched the profile. Consume
                    # that terminal rejection too, without executing an inverse.
                    try:
                        self._consume(action_id, record, mutate=False)
                    except (OSError, ValueError, DefinitionTransactionConflict):
                        pass
                record.state = self._settle(action_id, record)
            return {"action_id": action_id, "state": record.state}
