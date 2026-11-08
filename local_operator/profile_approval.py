"""Durable profile proposals; only the existing verified signer can approve.

A queued request carries exact bounded authored values, not a replaceable path
or executable command. The typed apply continuation works without the filing
Session. Journal/backup files are bookkeeping: authority always comes from
verify_for_run over this exact signed request, rechecked under the write lock.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_operator.definition_transaction import commitment, definition_lock, transact
from local_operator.network import approval_ops
from local_operator.network import approvals as A
from local_operator.network.types import MeshRefusal
from local_operator.teams import TeamMember

OP = "profile_save"
Token = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_-]+$")]
Digest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Text = Annotated[str, Field(max_length=100_000)]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Anchor(_Closed):
    key_id: str = Field(min_length=1, max_length=160)
    spki_fp: str = Field(min_length=1, max_length=160)
    statement_digest: str = Field(min_length=1, max_length=160)


class AgentImage(_Closed):
    id: Token
    name: str = Field(min_length=1, max_length=128)
    label: str = Field(max_length=256)
    description: str = Field(max_length=16_000)
    tags: list[Annotated[str, Field(max_length=512)]] = Field(max_length=128)
    categories: list[Annotated[str, Field(max_length=128)]] = Field(max_length=32)
    model: str | None = Field(max_length=512)
    instructions: Text


class TeamImage(_Closed):
    id: Token
    name: str = Field(min_length=1, max_length=128)
    label: str = Field(max_length=256)
    description: str = Field(max_length=16_000)
    manager: str = Field(min_length=1, max_length=128)
    members: list[TeamMember] = Field(max_length=64)
    aliases: list[Annotated[str, Field(max_length=128)]] = Field(max_length=32)
    instructions: Text
    project: Text


class ProfileProposal(_Closed):
    version: Literal[1] = 1
    anchor: Anchor
    session_id: Token
    tool_call_id: Token
    receipt_id: Token
    profile_kind: Literal["agent", "team"]
    scope: Literal["global"] = "global"
    operation: Literal["create", "update", "revert"]
    target_id: Token
    name: str = Field(min_length=1, max_length=128)
    base_commitment: Digest
    before: dict[str, Any] | None
    after: dict[str, Any]
    after_commitment: Digest
    created_at: float
    expires_at: float

    @model_validator(mode="after")
    def _bind(self):
        model = AgentImage if self.profile_kind == "agent" else TeamImage
        if len(json.dumps(self.model_dump(mode="json"))) > 400_000:
            raise ValueError("profile proposal exceeds bounded inline size")
        for image in (self.before, self.after):
            if image is not None:
                checked = model.model_validate(image).model_dump(mode="json")
                if (
                    checked != image
                    or checked["id"] != self.target_id
                    or checked["name"] != self.name
                ):
                    raise ValueError("profile image does not match its exact target")
        if (self.before is None) != (self.operation == "create"):
            raise ValueError("only create carries an absence precondition")
        if self.base_commitment != commitment(self.before) or self.after_commitment != commitment(
            self.after
        ):
            raise ValueError("profile image commitment mismatch")
        if not all(math.isfinite(v) for v in (self.created_at, self.expires_at)):
            raise ValueError("invalid approval window")
        if not 0 < self.expires_at - self.created_at <= 86400:
            raise ValueError("approval expiry must be fixed within one day")
        return self


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


#: ``Token``'s character set and bound, spelled once so :func:`request_token`
#: cannot drift from the field it has to satisfy.
_TOKEN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_TOKEN_MAX = 160


def request_token(value: str) -> str:
    """A ``Token``-shaped spelling of a HOST-supplied identity, stable per input.

    WHY THIS EXISTS (measured incident, 2026-11-08). A nested ``eval`` tool call
    is minted with the id ``"<outer id>:<uuid4 hex>"`` (``harness/loop.py``'s
    ``dispatch_tool`` bridge), so ``tool("team", op="update", ...)`` from inside
    a Python cell carried a COLON into :func:`build`. ``Token`` forbids it,
    ``model_validate`` raised ``ValidationError``, and the seam's blanket
    ``except Exception`` reported that as ``error`` — "the confirmation channel
    failed" — for a request whose only defect was its spelling. The save was
    impossible from inside ``eval`` and nothing said so.

    Normalising here is safe because this id is RECORDED (card, journal, applied
    tool result) and never matched against a live call; it is not authority. It
    runs in :func:`build`, the one constructor every producer goes through.
    ``session_id`` is deliberately NOT normalised: ``applied_tool_result``
    compares it verbatim against the conversation that asks for the receipt, so
    rewriting it would refuse a legitimate apply.

    An id that already satisfies ``Token`` comes back BYTE-IDENTICAL, so no
    existing request id, receipt or journal changes. Anything else keeps a
    readable sanitised prefix plus a digest of the raw value, so ids that differ
    only in characters the pattern forbids cannot collide.
    """
    if value and len(value) <= _TOKEN_MAX and set(value) <= _TOKEN_CHARS:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    prefix = "".join(char if char in _TOKEN_CHARS else "-" for char in value)
    return prefix[: _TOKEN_MAX - len(digest) - 1] + "-" + digest


def snapshot(root: Path, kind: str, target_id: str) -> dict[str, Any] | None:
    """Fresh bounded authored image; runtime state is never replayed by a save."""
    TokenModel = AgentImage if kind == "agent" else TeamImage
    if not target_id or any(c in target_id for c in "/\\."):
        raise _refuse("profile_target_invalid", "invalid profile target identity")
    with definition_lock(root):
        if kind == "agent":
            from local_operator.agents import AgentRegistry

            registry = AgentRegistry(root, migrate=False)
            try:
                row = registry.get_agent(target_id)
            except KeyError:
                return None
            data = {
                key: getattr(row, key) for key in AgentImage.model_fields if key != "instructions"
            }
            data["instructions"] = registry.get_agent_system_prompt(target_id)
        elif kind == "team":
            from local_operator.teams import read_team_snapshot

            row = read_team_snapshot(root / "teams", target_id)
            if row is None:
                return None
            data = {key: row.model_dump(mode="json")[key] for key in TeamImage.model_fields}
        else:
            raise _refuse("profile_kind_invalid", "unknown profile kind")
        return TokenModel.model_validate(data).model_dump(mode="json")


def build(
    root: Path,
    *,
    profile_kind: str,
    operation: str,
    target_id: str,
    name: str,
    session_id: str,
    tool_call_id: str,
    before: Any,
    after: Any,
) -> ProfileProposal:
    trio = A.local_anchor_trio(root)
    if trio is None:
        raise _refuse(
            "approval_anchor_unavailable", "set up local operator authority before filing this save"
        )
    now = time.time()
    # The harness may hand us an id ``Token`` cannot carry (see
    # :func:`request_token`); the RECORD keeps a Token-shaped derivation rather
    # than crashing the filing on a colon.
    tool_call_id = request_token(tool_call_id)
    receipt = commitment([session_id, tool_call_id, profile_kind, target_id, before, after]).split(
        ":"
    )[1]
    return ProfileProposal.model_validate(
        dict(
            anchor={key: trio[key] for key in A.ANCHOR_KEYS},
            session_id=session_id,
            tool_call_id=tool_call_id,
            receipt_id=receipt,
            profile_kind=profile_kind,
            operation=operation,
            target_id=target_id,
            name=name,
            before=before,
            after=after,
            base_commitment=commitment(before),
            after_commitment=commitment(after),
            created_at=now,
            expires_at=now + 86400.0,
        )
    )


def _request_id(proposal: ProfileProposal) -> str:
    from local_operator.network.wire import crockford

    return "req_" + crockford(bytes.fromhex(proposal.receipt_id[:16]))


def file_request(root: Path, proposal: ProfileProposal) -> dict[str, Any]:
    """File only a proposal: no profile, backup, signature or model turn changes."""
    proposal = ProfileProposal.model_validate(proposal.model_dump(mode="json"))
    anchor = proposal.anchor.model_dump(mode="json")
    # Filing may create approval/journal bookkeeping, never the authored row
    # or a backup. Repeat filing must NOT recreate a deleted journal or revive
    # a denied/expired approval: a new user decision is needed for that.
    from local_operator.definition_transaction import prepare

    with definition_lock(root):
        existing = A._find_by_request_id(_request_id(proposal), root)
        record = A.create_request(
            kind=OP,
            request_id=_request_id(proposal),
            requested_by={
                "surface": "profile-tool",
                "session_id": proposal.session_id,
                "tool_call_id": proposal.tool_call_id,
            },
            machine={"key_id": anchor["key_id"], "spki_fp": anchor["spki_fp"]},
            what={"anchor": anchor, OP: proposal.model_dump(mode="json")},
            credential_ref=None,
            created_at=proposal.created_at,
            expires_at=proposal.expires_at,
            root=root,
        )

        if existing is None:
            prepare(
                root,
                transaction_id=proposal.receipt_id,
                before=proposal.before,
                after=proposal.after,
                read=lambda: snapshot(root, proposal.profile_kind, proposal.target_id),
                binding=_binding(record),
            )
        return record


def _binding(record: Mapping[str, Any]) -> dict[str, Any]:
    """Exact request/target binding; only the verified caller supplies authority."""
    p = _proposal(record)
    return {
        "approval_id": record["approval_id"],
        "request_digest": record["request_digest"],
        "target_id": p.target_id,
        "profile_kind": p.profile_kind,
        "scope": p.scope,
        "session_id": p.session_id,
        "tool_call_id": p.tool_call_id,
        "receipt_id": p.receipt_id,
        "expires_at": p.expires_at,
    }


def _proposal(record: Mapping[str, Any]) -> ProfileProposal:
    try:
        what = record["what"]
        if (
            record["kind"] != OP
            or set(what) != {"anchor", OP}
            or record.get("credential_ref") is not None
        ):
            raise ValueError("incorrect profile approval envelope")
        p = ProfileProposal.model_validate(what[OP])
        anchor = p.anchor.model_dump(mode="json")
        if what["anchor"] != anchor or record.get("machine") != {
            "key_id": anchor["key_id"],
            "spki_fp": anchor["spki_fp"],
        }:
            raise ValueError("profile approval subject mismatch")
        if record.get("requested_by") != {
            "surface": "profile-tool",
            "session_id": p.session_id,
            "tool_call_id": p.tool_call_id,
        }:
            raise ValueError("profile filing identity mismatch")
        if record.get("created_at") != p.created_at or record.get("expires_at") != p.expires_at:
            raise ValueError("profile approval window mismatch")
        if record.get("request_id") != _request_id(p):
            raise ValueError("profile request identity mismatch")
        return p
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(
            "profile_payload_invalid", "this profile card is not an exact bounded proposal"
        ) from exc


def disclosure(record: dict[str, Any]) -> dict[str, Any]:
    p = _proposal(record)
    before = p.before or {}
    changed = [
        key for key in p.after if key not in {"id", "name"} and before.get(key) != p.after[key]
    ]
    return {
        "version": 1,
        "operation": OP,
        "approval_id": str(record.get("approval_id") or ""),
        "profile_kind": p.profile_kind,
        "scope": p.scope,
        "profile_operation": p.operation,
        "target_id": p.target_id,
        "profile_name": p.name,
        "session_id": p.session_id,
        "tool_call_id": p.tool_call_id,
        "receipt_id": p.receipt_id,
        "changed_fields": changed,
        "before": p.before,
        "after": p.after,
        "base_commitment": p.base_commitment,
        "after_commitment": p.after_commitment,
        "expires_at": p.expires_at,
        "signing_principal": p.anchor.key_id,
    }


def expected(record: Mapping[str, Any]) -> dict[str, Any]:
    p = _proposal(record)
    return {
        "request_digest": A.request_digest(record),
        "receipt_id": p.receipt_id,
        "session_id": p.session_id,
        "target_id": p.target_id,
        "scope": p.scope,
    }


def _write(root: Path, p: ProfileProposal, image: dict[str, Any] | None) -> None:
    write_image(root, p.profile_kind, p.target_id, image)


def write_image(
    root: Path,
    profile_kind: str,
    target_id: str,
    image: dict[str, Any] | None,
    *,
    registry: Any = None,
) -> None:
    """Shared authored-only write; callers own authority, identity, CAS and lock."""
    # Caller holds the transaction boundary and has reverified authority. This
    # writes only authored columns; conversation state/avatar/credentials stay.
    if profile_kind == "agent":
        from local_operator.agents import AgentData, AgentEditFields, AgentRegistry

        r = registry if registry is not None else AgentRegistry(root, migrate=False)
        current = snapshot(root, "agent", target_id)
        if image is None:
            if current is not None:
                r.delete_agent(target_id)
            return
        fields = {key: None for key in AgentEditFields.model_fields}
        fields.update(
            {key: value for key, value in image.items() if key not in {"id", "instructions"}}
        )
        if current is None:
            # Preallocated identity is signed; create_agent's random id cannot
            # replace it. Preserve its ordinary defaults via AgentData schema.
            row = AgentData.model_validate(dict(image, id=target_id))
            r.save_agent(row)
        else:
            r.update_agent(target_id, AgentEditFields(**fields))
        r.set_agent_system_prompt(target_id, image["instructions"])
    else:
        from local_operator.teams import Team, TeamEditFields, TeamRegistry

        r = registry if registry is not None else TeamRegistry(root)
        current = snapshot(root, "team", target_id)
        if image is None:
            if current is not None:
                r.delete_team(target_id)
            return
        if current is None:
            r.save_team(Team.model_validate(image))
        else:
            fields = {key: value for key, value in image.items() if key != "id"}
            r.update_team(target_id, TeamEditFields(**fields))


def _backup(root: Path, p: ProfileProposal, before: Any) -> None:
    if before is None:
        return
    from local_operator.definition_transaction import _durable_json

    path = root / "sessions" / p.session_id / "profile-backups" / (p.receipt_id + ".json")
    payload = {
        "schema": 1,
        "kind": p.profile_kind,
        "name": p.name,
        "id": p.target_id,
        "approval_receipt_id": p.receipt_id,
        "before": before,
        "base_commitment": p.base_commitment,
        "after_commitment": p.after_commitment,
    }
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise _refuse(
                "profile_backup_conflict", "this receipt's backup changed; nothing was applied"
            )
        return
    _durable_json(path, payload)


def apply(approval_id: str, expected_block: dict[str, Any], root: Path) -> dict[str, Any]:
    record = A._load_raw(approval_id, root)
    p = _proposal(record)
    if expected_block != expected(record):
        raise _refuse("profile_expected_mismatch", "this is not the reviewed profile operation")
    verified = A.verify_for_run(approval_id, root=root)
    _proposal(verified)
    from local_operator.network.credentials import desktop_approval

    if desktop_approval.assurance()[0] != desktop_approval.ASSURANCE_PRESENCE:
        raise _refuse(
            "profile_presence_required",
            "a presence-backed operator key is required; "
            "file-only signatures cannot execute profile saves",
        )
    trio = A.local_anchor_trio(root)
    if trio is None or any(str(trio[k]) != getattr(p.anchor, k) for k in ("key_id", "spki_fp")):
        raise _refuse(
            "profile_anchor_mismatch", "this save belongs to another local operator anchor"
        )
    # Authentication happens even for already-completed requests. Mutable
    # state/journals alone never turn an unsigned request into an operation.
    if time.time() >= p.expires_at:
        raise _refuse("profile_expired", "this profile approval window has expired")
    run_id = A.new_run_id()
    already_complete = record.get("state") == A.STATE_CONNECTED
    if already_complete:
        # The approval's state is editable bookkeeping. It cannot replace an
        # execution lease for a first write, even with a valid old signature.
        path = root / "profile-transactions" / (p.receipt_id + ".json")
        if not path.exists():
            raise _refuse("profile_missing_execution", "no completed profile transaction exists")
        journal = json.loads(path.read_text())
        if (
            journal.get("state") != "complete"
            or journal.get("after") != p.after_commitment
            or journal.get("before") != p.base_commitment
        ):
            raise _refuse(
                "profile_missing_execution", "no matching completed profile transaction exists"
            )
    if not already_complete:
        A.begin_run(approval_id, run_id=run_id, runner_pid=os.getpid(), root=root)
    try:
        with definition_lock(root):
            A.verify_for_run(approval_id, root=root)
            if p.operation == "create":
                from local_operator.agents import AgentRegistry
                from local_operator.teams import TeamRegistry

                if p.profile_kind == "agent":
                    existing = AgentRegistry(root, migrate=False).get_agent_by_name(p.name)
                else:
                    existing = TeamRegistry(root).get_team_by_name(p.name)
                if existing is not None and str(existing.id) != p.target_id:
                    raise _refuse("profile_name_conflict", "the requested name is now in use")
            # Never prepare here: a missing queued journal is a refusal even
            # after restart, not permission to reconstruct a grant from files.
            result = transact(
                root,
                transaction_id=p.receipt_id,
                binding=_binding(record),
                before=p.before,
                after=p.after,
                read=lambda: snapshot(root, p.profile_kind, p.target_id),
                write=lambda image: _write(root, p, image),
                backup=lambda image: _backup(root, p, image),
            )
        if not already_complete:
            A.mark_connected(
                approval_id, run_id=run_id, step=OP, detail=p.after_commitment, root=root
            )
        result = {
            "state": "complete",
            "approval_id": approval_id,
            "receipt_id": p.receipt_id,
            "profile_kind": p.profile_kind,
            "profile_name": p.name,
            "target_id": p.target_id,
            "after_commitment": p.after_commitment,
            "backup_id": p.receipt_id if p.before else None,
            "reconciled": bool(result.get("reconciled")),
        }
        if p.before is not None:
            from local_operator.profile_actions import (
                prepare_signed_revert,
                signed_descriptor,
            )

            prepare_signed_revert(root, record)
            result["profile_action"] = signed_descriptor(record)
        _publish(root, p, result)
        return result
    except Exception as exc:
        if not already_complete:
            try:
                A.mark_failed(approval_id, run_id=run_id, step=OP, detail=str(exc), root=root)
            except MeshRefusal:
                pass  # A concurrent deny/expiry is terminal, never overwritten.
        if isinstance(exc, MeshRefusal):
            raise
        raise _refuse("profile_apply_conflict", str(exc)) from exc


def _publish(root: Path, p: ProfileProposal, result: dict[str, Any]) -> None:
    # A durable neutral result, not a user prompt and never an authorization
    # witness. A live filing runtime may consume it without resurrecting a turn.
    from local_operator.definition_transaction import _durable_json

    directory = root / "sessions" / p.session_id
    if directory.is_dir():
        _durable_json(directory / "profile-results" / (p.receipt_id + ".json"), result)
        from local_operator.session.runtime.inbox import (
            SOURCE_PEER,
            InboxLine,
            append_for_successor,
        )

        append_for_successor(
            root,
            directory,
            InboxLine(
                text=f"Profile save {p.receipt_id}: {p.profile_kind} {p.name!r} saved.",
                sender={"source": "profile_save"},
                custom_type="profile_save_result",
                message_id=p.receipt_id,
                wake=False,
                source=SOURCE_PEER,
            ),
            raise_reader=False,
        )
        _notify_live(root, p)


_live_publications: set[Any] = set()


def _notify_live(root: Path, p: ProfileProposal) -> None:
    """Best-effort existing mailbox transport; the spool is the crash fallback."""
    import asyncio

    async def deliver():
        from local_operator.mobile.peer_client import send_peer_message
        from local_operator.session.runtime.registry import scan

        for record, _ in scan(root):
            if record.session_id != p.session_id:
                continue
            try:
                await send_peer_message(
                    record,
                    text="Profile approval result available.",
                    mode="mailbox",
                    wake=False,
                    sender={"source": "profile_save"},
                    message_id=p.receipt_id,
                    custom_type="profile_save_result",
                    deadline_s=2,
                )
            except (RuntimeError, ConnectionError, OSError, TimeoutError):
                pass  # An absent/busy owner leaves the durable inbox untouched.
            return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(deliver())
    else:
        task = loop.create_task(deliver())
        _live_publications.add(task)
        task.add_done_callback(_live_publications.discard)


def applied_tool_result(root: Path, receipt_id: str, session_id: str):
    """Resolve a hint into a native receipt only from the verified completed save.

    Inbox text, mutable result files and peer-provided props grant nothing. The
    signed proposal and completed exact-bound transaction determine every byte.
    """
    from local_operator.harness.types import TextContent, ToolResult
    from local_operator.network import approvals as A
    from local_operator.network.wire import crockford
    from local_operator.profile_actions import signed_descriptor
    from local_operator.tools.agent_tool import _instruction_diff

    if len(receipt_id) != 64:
        raise ValueError("unknown profile receipt")
    request_id = "req_" + crockford(bytes.fromhex(receipt_id[:16]))
    record = A._find_by_request_id(request_id, root)
    if record is None or record.get("state") != A.STATE_CONNECTED:
        raise ValueError("profile save is not complete")
    verified = A.verify_for_run(record["approval_id"], root=root)
    p = _proposal(verified)
    if p.receipt_id != receipt_id or p.session_id != session_id:
        raise ValueError("profile receipt belongs to another conversation")
    transaction = json.loads((root / "profile-transactions" / (p.receipt_id + ".json")).read_text())
    if (
        transaction.get("state") != "complete"
        or transaction.get("binding") != _binding(verified)
        or transaction.get("before") != p.base_commitment
        or transaction.get("after") != p.after_commitment
    ):
        raise ValueError("profile save has no matching completed transaction")
    before = p.before or {}
    fields = [
        key for key in p.after if key not in {"id", "name"} and before.get(key) != p.after[key]
    ]
    values = {key: (before.get(key), p.after[key]) for key in fields}
    if p.profile_kind == "agent":

        from local_operator.agent_profiles import profile_from_definition

        def view(image):
            data = {
                "id": p.target_id,
                "name": p.name,
                "label": "",
                "description": "",
                "tags": [],
                "model": "",
                "hosting": "",
                "instructions": "",
            } | image
            return profile_from_definition(
                name=data["name"],
                label=data["label"],
                description=data["description"],
                tags=data["tags"],
                model=data["model"],
                hosting=data["hosting"],
                instructions=data["instructions"],
                agent_id=data["id"],
            )

        left, right = view(before), view(p.after)
        names = {
            "label": "label",
            "description": "description",
            "instructions": "instructions",
            "model": "model",
            "effort": "effort",
            "tools": "tools",
            "may_delegate": "delegate",
            "saver": "saver",
            "action_class": "class",
        }
        values = {
            label: (getattr(left, key), getattr(right, key))
            for key, label in names.items()
            if getattr(left, key) != getattr(right, key)
        }
        if before.get("categories") != p.after.get("categories"):
            values["categories"] = (before.get("categories"), p.after.get("categories"))
        fields = list(values)
    diff = []
    # Policy settings are projected from agent tags: display names such as
    # tools/delegate/class are not raw image keys. Use the same value pairs for
    # both projected agents and raw team fields so publication cannot fail after
    # a valid signed mutation has already committed.
    for field, (old, new) in values.items():
        if field == "instructions":
            value, _ = _instruction_diff(
                str(old or ""), str(new), fromfile="before", tofile="after"
            )
            diff.extend(["instructions:", *value.splitlines()])
        else:
            diff.append(f"{field}: {old!r} → {new!r}")
    from local_operator.profile_scope import profile_diff_summary

    details = {
        "status": "applied",
        "profile_kind": p.profile_kind,
        "profile_name": p.name,
        "scope": "global",
        "receipt_id": p.receipt_id,
        "fields": fields,
        "diff": diff,
        "approval_id": record["approval_id"],
        "request_digest": record["request_digest"],
    }
    if p.before is not None:
        details["profile_action"] = signed_descriptor(verified)
    text = f"Saved {p.profile_kind} {p.name!r} in your account after signed approval."
    summary = profile_diff_summary("\n".join(diff))
    if summary:
        text += "\n" + summary
    return ToolResult(
        tool_call_id=p.tool_call_id,
        tool_name=p.profile_kind,
        content=[TextContent(text=text)],
        details=details,
    )


class _ProfileOp:
    op = OP
    kind = OP
    file_only_signing = False
    cli_approvable = False

    def intent(self, record):
        try:
            return _proposal(record).model_dump(mode="json")
        except MeshRefusal:
            return None

    disclosure = staticmethod(disclosure)
    expected = staticmethod(expected)

    def apply(self, approval_id: str, expected: dict[str, Any], root: Path) -> dict[str, Any]:
        return apply(approval_id, expected, root)

    def generic_refusal(self, record, approval_id):
        return _refuse(
            "profile_typed_approval_required",
            "Review the exact profile card and approve with a presence-backed signer.",
        )


# SHIPPED AS DISCIPLINE, NOT A BOUNDARY (operator posture, 2026-11-04). The
# signed card plus the unconsumed ``prepared`` record stop an agent from silently
# rewriting account-wide definitions; they do NOT stop a same-uid shell, which
# can reach this state. Sandboxing is the tracked follow-up.
approval_ops.register(_ProfileOp())
