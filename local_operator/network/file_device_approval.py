"""Dormant P1: owner-local presence approval of one authenticated mesh key.

This is NOT file serving, a login attestation, or a new credential-consent domain.
No CLI, tool, relay op or HTTP mint is registered. A future reviewed local UI must
call ``file_request`` from the actual incoming owner relay link. Until that flow
exists, the provider is deliberately dormant. Existing PAD/attachment authority
never consults this state. P2 issuance, strict per-read audit and P3 serving are
separate prerequisites; a returned context is not permission to disclose bytes.

Actual incoming-listener capture publishes an owner-local durable pending shelf,
never tokens. Desktop workers use their own pinned native AuthStore and fresh
existing authenticated local status; no process-local map is positive authority.
Relay restart/reconnect invalidates unapplied candidates, not completed standing
associations. A negative journal/sentinel separate from the signed current slot
keeps later revocation dominant across selective current-file restore and card
pruning. Coordinated rollback of ALL owner-private witnesses remains outside the
existing trusted-owner filesystem boundary; signatures are not freshness.
The config store has the existing owner-account filesystem boundary, not protection
against an attacker running arbitrary code as that same OS account.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from secrets import token_hex
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.i18n.text import tr
from local_operator.network import approval_ops
from local_operator.network import approvals as A
from local_operator.network import identity, store
from local_operator.network.credentials import placement
from local_operator.network.credentials.consent import MAX_REVISION as _MAX_REVISION
from local_operator.network.types import MeshRefusal
from local_operator.network.wire import b64u, unb64u
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.oauth.radient import verify_canonical_account

OP = "file_device_approval"
PURPOSE = "future_live_mentioned_file_access"
DECISION_DOMAIN = b"lop-file-device-approval-v1\x00"
_COPY = tr("wire.errors.desktop")


def _refuse(code: str) -> MeshRefusal:
    # Reuse the audited generic refusal envelope; identifiers are protocol data,
    # not new English copy or a user-facing file/credential disclosure flow.
    return MeshRefusal(code, _COPY(_wd.desktop_refusal(error=code)))


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _exact_version(cls, value):
        # JSON booleans compare equal to integer literals in Python. Do not let
        # that equality turn a malformed schema/version into accepted metadata.
        if isinstance(value, dict):
            for field in ("schema", "version"):
                if field in value and (type(value[field]) is not int or value[field] != 1):
                    raise ValueError("file_approval_schema_invalid")
        return value


class MeshKey(_Closed):
    device_id: str
    algorithm: Literal["ed25519"] = "ed25519"
    public_key: str

    @model_validator(mode="after")
    def _full_key(self):
        try:
            raw = unb64u(self.public_key)
        except (ValueError, TypeError):
            raise ValueError("file_approval_key_invalid") from None
        if (
            len(raw) != 32
            or b64u(raw) != self.public_key
            or identity.device_id_for(raw) != self.device_id
        ):
            raise ValueError("file_approval_key_invalid")
        return self


class Anchor(_Closed):
    key_id: str = Field(min_length=1)
    spki_fp: str = Field(min_length=1)
    statement_digest: str = Field(min_length=1)


class NativeAccount(_Closed):
    hub: str
    account_id: str = Field(min_length=1)
    credential_id: int = Field(gt=0)
    credential_created_at: int = Field(gt=0)
    login_incarnation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical(self):
        if self.hub != DEFAULT_RADIENT_API_BASE_URL or self.account_id.strip() != self.account_id:
            raise ValueError("file_approval_account_invalid")
        return self


class FileIntent(_Closed):
    version: Literal[1] = 1
    purpose: Literal["future_live_mentioned_file_access"] = PURPOSE
    action: Literal["approve", "revoke"]
    owner: MeshKey
    requester: MeshKey
    network_id: str = Field(min_length=1)
    network_epoch: int = Field(gt=0)
    owner_added_at: float = Field(gt=0)
    requester_added_at: float = Field(gt=0)
    owner_added_by: str
    requester_added_by: str
    owner_added_via: str
    requester_added_via: str
    account: NativeAccount
    anchor: Anchor
    assurance: Literal["installed_presence"] = "installed_presence"
    authority_epoch: str = Field(pattern=r"^[a-f0-9]{32}$")
    expected_generation: int = Field(ge=0, le=_MAX_REVISION)
    generation: int = Field(gt=0, le=_MAX_REVISION)
    request_id: str = Field(min_length=1)
    nonce: str = Field(pattern=r"^[a-f0-9]{32}$")
    relay_instance: str = Field(min_length=1)
    link_id: str = Field(min_length=1)
    transcript_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: float
    expires_at: float

    @model_validator(mode="after")
    def _coherent(self):
        store.safe_component(self.network_id)
        if (
            self.owner.device_id == self.requester.device_id
            or self.generation != self.expected_generation + 1
            or not self.request_id.startswith("req_")
            or not 0 < self.expires_at - self.created_at <= A.DEFAULT_EXPIRY_S
        ):
            raise ValueError("file_approval_intent_invalid")
        return self


def _anchor(root: Path) -> Anchor:
    from local_operator.network.enrollment import bound_profile
    from local_operator.operator import LEVEL_OPERATOR_PRESENCE, authority_level_load
    from local_operator.operator.sign import resolve_backend_name
    from local_operator.operator.trust import anchor_trio, is_presence_backend

    bound_profile(root)
    level, loaded = authority_level_load()
    if (
        level != LEVEL_OPERATOR_PRESENCE
        or not loaded.usable
        or not loaded.root_owned
        or loaded.anchor is None
        or not is_presence_backend(loaded.anchor.backend)
        or resolve_backend_name(root) != loaded.anchor.backend
    ):
        raise _refuse("file_approval_presence_required")
    expected = Anchor.model_validate(anchor_trio(loaded.anchor))
    local = A.local_anchor_trio(root)
    if local is None or any(local.get(k) != v for k, v in expected.model_dump().items()):
        raise _refuse("file_approval_anchor_changed")
    return expected


def _native(auth: AuthStore, credential_id: int, root: Path) -> NativeAccount:
    if not isinstance(auth, AuthStore) or Path(auth._db_path).resolve() != root / "auth.db":
        raise _refuse("file_approval_native_account_required")
    selected = auth.get_credential(credential_id)
    selected_incarnation = auth.native_login_incarnation(credential_id)
    # A passed AuthStore can still be attached to an unlinked old SQLite inode.
    # Observe the current profile-local file without migrations or resolver
    # cascades; a deleted/recreated database is never an offline-positive row.
    current = AuthStore.open_read_only(root / "auth.db")
    try:
        row = current.get_credential(credential_id)
        incarnation = current.native_login_incarnation(credential_id)
    finally:
        current.close()
    if (
        selected is None
        or row is None
        or selected_incarnation != incarnation
        or any(
            getattr(selected, key) != getattr(row, key)
            for key in ("id", "provider", "credential_type", "created_at", "disabled_cause")
        )
        or any(
            selected.data.get(key) != row.data.get(key)
            for key in ("api_endpoint", "account_id", "mesh_origin")
        )
    ):
        raise _refuse("file_approval_native_account_changed")
    if (
        row is None
        or row.disabled_cause is not None
        or row.provider != "radient"
        or row.credential_type != "oauth"
        or "mesh_origin" in row.data
        or not incarnation
    ):
        raise _refuse("file_approval_native_account_required")
    hub, account_id = row.data.get("api_endpoint"), row.data.get("account_id")
    if not isinstance(hub, str) or not isinstance(account_id, str):
        raise _refuse("file_approval_native_account_required")
    try:
        return NativeAccount(
            hub=hub,
            account_id=account_id,
            credential_id=row.id,
            credential_created_at=row.created_at,
            login_incarnation=incarnation,
        )
    except ValidationError:
        raise _refuse("file_approval_native_account_required") from None


def _link_root(link: Any) -> Path:
    from local_operator.network.relay import PeerLink

    if not isinstance(link, PeerLink):
        raise _refuse("file_approval_listener_required")
    return Path(link.server.root).resolve()


def _mesh(link: Any, root: Path) -> dict[str, Any]:
    # Importing the relay is restricted to an actual candidate construction,
    # never provider registration: this module must not turn a generic approval
    # lookup into a relay bootstrap or a new default identity/key mint.
    from local_operator.network.relay import PeerLink

    if not isinstance(link, PeerLink) or Path(link.server.root).resolve() != root:
        raise _refuse("file_approval_listener_required")
    result = link.result
    if result.role != "listener" or result.phase != "member" or not result.peer_public_key:
        raise _refuse("file_approval_listener_required")
    with link.server._links_lock:
        if link.server.links.get(link.link_id) is not link or not link.alive or link.closing:
            raise _refuse("file_approval_link_changed")
    current = store.load(result.network_id, root)
    owner = identity.load(root)
    member = next((m for m in current.members if m.device_id == result.peer_device_id), None)
    local_member = next(
        (m for m in current.members if owner and m.device_id == owner.device_id), None
    )
    if (
        current.trust != "active"
        or current.epoch != result.epoch
        or result.epoch != link.epoch
        or result.network_id != link.network_id
        or result.peer_device_id != link.device_id
        or owner is None
        or owner.public_key != link.server.identity.public_key
        or current.self_device_id != owner.device_id
        or member is None
        or local_member is None
        or not member.active
        or member.suspect
        or not local_member.active
        or local_member.suspect
        or member.device_id in current.removed_ids
        or local_member.device_id in current.removed_ids
        or member.public_key != result.peer_public_key
        or local_member.public_key != owner.public_key
        or "view" not in member.capabilities
        or "view" not in local_member.capabilities
    ):
        raise _refuse("file_approval_member_changed")
    return {
        "owner": MeshKey(device_id=owner.device_id, public_key=owner.public_key),
        "requester": MeshKey(device_id=member.device_id, public_key=result.peer_public_key),
        "network_id": current.network_id,
        "network_epoch": current.epoch,
        "owner_added_at": float(local_member.added_at),
        "requester_added_at": float(member.added_at),
        "owner_added_by": local_member.added_by,
        "requester_added_by": member.added_by,
        "owner_added_via": local_member.added_via,
        "requester_added_via": member.added_via,
    }


@dataclass(frozen=True)
class _Candidate:
    # A filing-only local capability. It is never the authority a desktop worker
    # needs: its exact commitment is handed off through the private disk shelf.
    root: Path
    link: Any
    auth: AuthStore
    intent: FileIntent


@dataclass(frozen=True)
class _Checked:
    root: Path
    intent: FileIntent
    digest: str


_FILING: ContextVar[_Candidate | None] = ContextVar("file_device_approval_filing", default=None)
_CHECKED: ContextVar[_Checked | None] = ContextVar("file_device_approval_checked", default=None)
_MUTATING: ContextVar[bool] = ContextVar("file_device_approval_mutating", default=False)


def _membership(intent: FileIntent, root: Path) -> None:
    current = store.load(intent.network_id, root)
    own = identity.load(root)
    peer = next((m for m in current.members if m.device_id == intent.requester.device_id), None)
    owner = next((m for m in current.members if m.device_id == intent.owner.device_id), None)
    if (
        own is None
        or own.public_key != intent.owner.public_key
        or own.device_id != intent.owner.device_id
        or current.self_device_id != own.device_id
        or current.trust != "active"
        or current.epoch != intent.network_epoch
        or peer is None
        or owner is None
        or not peer.active
        or not owner.active
        or peer.suspect
        or owner.suspect
        or peer.device_id in current.removed_ids
        or owner.device_id in current.removed_ids
        or peer.public_key != intent.requester.public_key
        or owner.public_key != intent.owner.public_key
        or peer.added_at != intent.requester_added_at
        or owner.added_at != intent.owner_added_at
        or peer.added_by != intent.requester_added_by
        or owner.added_by != intent.owner_added_by
        or peer.added_via != intent.requester_added_via
        or owner.added_via != intent.owner_added_via
        or "view" not in peer.capabilities
        or "view" not in owner.capabilities
    ):
        raise _refuse("file_approval_member_changed")


def _local_relay_record(root: Path):
    return store.find_own_relay(root)


def _status_current(intent: FileIntent, root: Path) -> None:
    # This is the already-authenticated SAME-ROOT local control plane, not a
    # peer status/label or a new issuer. Missing additive fields deny only here.
    from local_operator.network import relay

    own = _local_relay_record(root)
    if (
        own is None
        or own.instance_id != intent.relay_instance
        or own.device_id != intent.owner.device_id
    ):
        raise _refuse("file_approval_link_changed")
    reply = relay.control_request(own, "net_status")
    detail = reply.get("detail") if isinstance(reply, dict) and reply.get("op") == "ack" else None
    if (
        not isinstance(detail, dict)
        or detail.get("instance_id") != intent.relay_instance
        or detail.get("device_id") != intent.owner.device_id
    ):
        raise _refuse("file_approval_status_unknown")
    links = detail.get("links")
    rows = (
        [r for r in links if isinstance(r, dict) and r.get("link_id") == intent.link_id]
        if isinstance(links, list)
        else []
    )
    if len(rows) != 1:
        raise _refuse("file_approval_link_changed")
    row = rows[0]
    if (
        row.get("role") != "listener"
        or row.get("phase") != "member"
        or row.get("alive") is not True
        or row.get("closing") is not False
        or type(row.get("epoch")) is not int
        or row["epoch"] != intent.network_epoch
        or row.get("device_id") != intent.requester.device_id
        or row.get("network_id") != intent.network_id
    ):
        raise _refuse("file_approval_status_unknown")


@contextmanager
def _own_auth(root: Path, *, writable: bool = False) -> Iterator[AuthStore]:
    path = root / "auth.db"
    if not path.is_file() or path.is_symlink():
        raise _refuse("file_approval_native_account_required")
    auth = AuthStore(path, config_dir=root) if writable else AuthStore.open_read_only(path)
    try:
        yield auth
    finally:
        auth.close()


def _local_context(intent: FileIntent, root: Path, auth: AuthStore) -> None:
    if (
        _anchor(root) != intent.anchor
        or _native(auth, intent.account.credential_id, root) != intent.account
    ):
        raise _refuse("file_approval_context_changed")
    _membership(intent, root)


async def _canonical(intent: FileIntent, auth: AuthStore, root: Path) -> None:
    _local_context(intent, root, auth)
    try:
        data = await auth.ensure_oauth_fresh_or_raise(intent.account.credential_id)
        if not isinstance(data, dict):
            raise _refuse("file_approval_account_unverified")
        account = await verify_canonical_account(
            data.get("access") or data.get("access_token") or ""
        )
    except Exception as error:
        raise _refuse("file_approval_account_unverified") from error
    if account != intent.account.account_id:
        raise _refuse("file_approval_account_changed")
    _local_context(intent, root, auth)


def _canonical_sync(intent: FileIntent, auth: AuthStore, root: Path) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_canonical(intent, auth, root))
    else:
        raise _refuse("file_approval_worker_required")


def _shelf(record: Mapping[str, Any], root: Path) -> "_ShelfRow":
    intent = _intent(record)
    journal = _load_journal(root)
    if _load(root).authority_epoch != journal.authority_epoch:
        raise _refuse("file_approval_epoch_changed")
    row = journal.candidates.get(intent.request_id)
    digest = A.request_digest(record)
    if row is None or row.intent != intent or row.digest != digest:
        raise _refuse("file_approval_provenance_lost")
    if journal.authority_epoch != intent.authority_epoch:
        raise _refuse("file_approval_epoch_changed")
    fence = journal.fences.get(_association(intent))
    if (
        fence
        and fence.generation > intent.expected_generation
        and not (
            fence.generation == intent.generation
            and fence.request_id == intent.request_id
            and fence.digest == digest
        )
    ):
        raise _refuse("file_approval_generation_changed")
    return row


def prepare_current(record: Mapping[str, Any], root: Path, *, canonical: bool = False) -> _Checked:
    # All external work is OUTSIDE mutation locks. The disk shelf is authority
    # for provenance; this short-lived receipt merely closes IO/lock ordering.
    root = root.resolve()
    row = _shelf(record, root)
    intent = row.intent
    _status_current(intent, root)
    with _own_auth(root, writable=canonical) as auth:
        _local_context(intent, root, auth)
        if canonical:
            _canonical_sync(intent, auth, root)
        _status_current(intent, root)
        _local_context(intent, root, auth)
    _shelf(record, root)
    checked = _Checked(root, intent, A.request_digest(record))
    _CHECKED.set(checked)
    return checked


@contextmanager
def mutation_context(
    record: Mapping[str, Any], root: Path, *, canonical: bool = False
) -> Iterator[None]:
    prior = _CHECKED.get()
    prepare_current(record, root, canonical=canonical)
    token = _MUTATING.set(True)
    try:
        yield
    finally:
        _MUTATING.reset(token)
        _CHECKED.set(prior)


@contextmanager
def decision_lock(approval_id: str, root: Path | None) -> Iterator[None]:
    if root is None:
        raise _refuse("file_approval_provenance_lost")
    record = A._load_raw(approval_id, root)
    with mutation_context(record, root):
        with A._record_lock(approval_id, root):
            yield


def _association(intent: FileIntent) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "owner": intent.owner.model_dump(),
                "requester": intent.requester.model_dump(),
                "network_id": intent.network_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class _Slot(_Closed):
    generation: int = Field(gt=0)
    status: Literal["active", "revoked"]
    intent: FileIntent
    record: dict[str, Any]
    applied_at: float

    @model_validator(mode="after")
    def _bind(self):
        if self.generation != self.intent.generation or self.status != (
            "active" if self.intent.action == "approve" else "revoked"
        ):
            raise ValueError("file_approval_state_invalid")
        if not self.intent.created_at <= self.applied_at < self.intent.expires_at:
            raise ValueError("file_approval_state_invalid")
        return self


class _Document(_Closed):
    schema_version: Literal[1] = Field(default=1, alias="schema")
    authority_epoch: str = Field(pattern=r"^[a-f0-9]{32}$")
    slots: dict[str, _Slot]

    @model_validator(mode="after")
    def _bind(self):
        for key, slot in self.slots.items():
            if (
                key != _association(slot.intent)
                or slot.intent.authority_epoch != self.authority_epoch
            ):
                raise ValueError("file_approval_state_invalid")
        return self


def _path(root: Path) -> Path:
    return root / "network" / "file-device-approvals.json"


@contextmanager
def _locked(root: Path, *, initialize: bool = False) -> Iterator[None]:
    path = _path(root)
    try:
        if initialize:
            store._ensure_private_directory_durable(path.parent)
        if not path.parent.is_dir() or path.parent.is_symlink() or path.is_symlink():
            raise _refuse("file_approval_state_missing")
        with store._write_lock(path):
            with placement._cross_process_lock(
                path.with_name(".file-device-approvals.lock"), root=root
            ):
                yield
    except OSError:
        # Unsupported host durability and path/lock failures are a refusal, not
        # a weaker file-only fallback or an OS exception containing local paths.
        raise _refuse("file_approval_state_unavailable") from None


def _load(root: Path) -> _Document:
    try:
        path = _path(root)
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("file_approval_state_redirected")
        return _Document.model_validate(store._read_private_json_durable(path))
    except (OSError, ValueError, ValidationError):
        raise _refuse("file_approval_state_unavailable") from None


def _write(root: Path, document: _Document) -> None:
    try:
        store._write_private_json_durable(
            _path(root), document.model_dump(mode="json", by_alias=True)
        )
        if _load(root) != document:
            raise OSError("file_device_state_unconfirmed")
    except (OSError, ValueError):
        # A rename may have committed. No rollback fiction and no automatic new
        # epoch/generation: the SAME signed request must reconcile the slot.
        raise _refuse("file_approval_write_unconfirmed") from None


# Same finite control-plane metadata envelope as finite_journal (not its
# credential authority): <=64 association fences and <=64 pending candidates,
# one dominating denial per fence (therefore <=64, below its 256-deny bound),
# and <=256KiB TOTAL encoded metadata. Capacity denies; live denies never evict.
_METADATA_LIMIT = 262144
_RECORD_LIMIT = 64


class _ShelfRow(_Closed):
    intent: FileIntent
    digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    state: Literal["prepared", "consumed"] = "prepared"


class _Fence(_Closed):
    generation: int = Field(ge=1, le=_MAX_REVISION)
    phase: Literal["pending", "complete"]
    action: Literal["approve", "revoke"]
    request_id: str
    approval_id: str
    digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    deny_generation: int = Field(ge=0, le=_MAX_REVISION)
    deny_reason: Literal["revoked"] | None

    @model_validator(mode="after")
    def _coherent(self):
        if self.deny_generation > self.generation or bool(self.deny_generation) != (
            self.deny_reason == "revoked"
        ):
            raise ValueError("file_approval_journal_invalid")
        if self.action == "revoke" and self.deny_generation != self.generation:
            raise ValueError("file_approval_journal_invalid")
        return self


class _Journal(_Closed):
    schema_version: Literal[1] = Field(default=1, alias="schema")
    authority_epoch: str = Field(pattern=r"^[a-f0-9]{32}$")
    revision: int = Field(ge=1, le=_MAX_REVISION)
    candidates: dict[str, _ShelfRow]
    fences: dict[str, _Fence]

    @model_validator(mode="after")
    def _coherent(self):
        if len(self.candidates) > _RECORD_LIMIT or len(self.fences) > _RECORD_LIMIT:
            raise ValueError("file_approval_journal_capacity")
        for key, row in self.candidates.items():
            if key != row.intent.request_id or row.intent.authority_epoch != self.authority_epoch:
                raise ValueError("file_approval_journal_invalid")
        for key in self.fences:
            if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("file_approval_journal_invalid")
        if (
            len(
                json.dumps(self.model_dump(mode="json", by_alias=True), ensure_ascii=False).encode()
            )
            > _METADATA_LIMIT
        ):
            raise ValueError("file_approval_journal_capacity")
        return self


def _journal_path(root: Path) -> Path:
    return _path(root).with_name("file-device-journal.json")


def _sentinel_path(root: Path) -> Path:
    return _path(root).with_name("file-device-journal.initialized.json")


def _load_journal(root: Path) -> _Journal:
    # A FILE positive has no permissive legacy genesis. Readers NEVER create or
    # reconstruct either witness, even when an old signed current slot exists.
    try:
        path = _journal_path(root)
        if path.parent.is_symlink() or path.is_symlink() or _sentinel_path(root).is_symlink():
            raise ValueError("file_approval_journal_redirected")
        if path.stat(follow_symlinks=False).st_size > _METADATA_LIMIT:
            raise ValueError("file_approval_journal_capacity")
        journal = _Journal.model_validate(
            store._read_private_json_durable(path, max_bytes=_METADATA_LIMIT)
        )
        marker = store._read_private_json_durable(_sentinel_path(root), max_bytes=1024)
        if (
            not isinstance(marker, dict)
            or type(marker.get("schema")) is not int
            or marker != {"schema": 1, "purpose": OP, "authority_epoch": journal.authority_epoch}
        ):
            raise ValueError("file_approval_journal_invalid")
        return journal
    except (OSError, ValueError, ValidationError):
        raise _refuse("file_approval_journal_unavailable") from None


def _publish_journal(root: Path, journal: _Journal, *, expected_revision: int) -> _Journal:
    current = _load_journal(root)
    if type(expected_revision) is not int or not 1 <= expected_revision <= _MAX_REVISION:
        raise _refuse("file_approval_journal_conflict")
    if current.revision != expected_revision or current.authority_epoch != journal.authority_epoch:
        raise _refuse("file_approval_journal_conflict")
    try:
        candidate = _Journal.model_validate(
            {
                **journal.model_dump(mode="json", by_alias=True),
                "revision": expected_revision + 1,
            }
        )
        store._write_private_json_durable(
            _journal_path(root), candidate.model_dump(mode="json", by_alias=True)
        )
        if _load_journal(root) != candidate:
            raise OSError("file_approval_journal_unconfirmed")
        return candidate
    except (OSError, ValueError, ValidationError):
        raise _refuse("file_approval_write_unconfirmed") from None


def _initialize(root: Path, epoch: str) -> tuple[_Document, _Journal]:
    paths = (_path(root), _journal_path(root), _sentinel_path(root))
    if any(path.exists() or path.is_symlink() for path in paths):
        document, journal = _load(root), _load_journal(root)
        if document.authority_epoch != journal.authority_epoch:
            raise _refuse("file_approval_epoch_changed")
        return document, journal
    # Journal -> sentinel -> EMPTY projection. An interrupted pair is unknown,
    # not a license to rebuild grants from approval cards or an old snapshot.
    journal = _Journal(authority_epoch=epoch, revision=1, candidates={}, fences={})
    try:
        store._write_private_json_durable(
            _journal_path(root), journal.model_dump(mode="json", by_alias=True)
        )
        store._write_private_json_durable(
            _sentinel_path(root), {"schema": 1, "purpose": OP, "authority_epoch": epoch}
        )
        document = _Document(authority_epoch=epoch, slots={})
        _write(root, document)
        return document, _load_journal(root)
    except (OSError, ValueError):
        raise _refuse("file_approval_write_unconfirmed") from None


def _save_candidate(root: Path, intent: FileIntent, digest: str) -> None:
    journal = _load_journal(root)
    # Candidate expiry reuses the existing signed card window. It is NOT a
    # transaction fence: pruning it cannot remove a retained denial or grant.
    candidates = {
        key: row
        for key, row in journal.candidates.items()
        if row.state == "prepared" and row.intent.expires_at > time.time()
    }
    # Completed candidate rows are not authority and can be reclaimed. Old API
    # requests then deny; their durable association/denial fences never prune.
    candidates[intent.request_id] = _ShelfRow(intent=intent, digest=digest)
    try:
        updated = _Journal.model_validate(
            {
                **journal.model_dump(mode="json", by_alias=True),
                "candidates": {k: v.model_dump(mode="json") for k, v in candidates.items()},
            }
        )
    except ValidationError:
        raise _refuse("file_approval_journal_capacity") from None
    _publish_journal(root, updated, expected_revision=journal.revision)


def _active_fence(journal: _Journal, slot: _Slot) -> None:
    fence = journal.fences.get(_association(slot.intent))
    if (
        fence is None
        or fence.phase != "complete"
        or fence.action != "approve"
        or fence.generation != slot.generation
        or fence.deny_generation >= slot.generation
        or fence.request_id != slot.intent.request_id
        or fence.approval_id != slot.record.get("approval_id")
        or fence.digest != slot.record.get("request_digest")
    ):
        raise _refuse("file_approval_negative_fence")


def _intent(record: Mapping[str, Any]) -> FileIntent:
    try:
        what = record.get("what")
        if (
            record.get("kind") != A.KIND_FILE_DEVICE_APPROVAL
            or not isinstance(what, Mapping)
            or set(what) != {OP, "anchor"}
        ):
            raise ValueError("file_approval_intent_invalid")
        intent = FileIntent.model_validate(what[OP])
        if (
            what["anchor"] != intent.anchor.model_dump()
            or record.get("request_id") != intent.request_id
            or record.get("created_at") != intent.created_at
            or record.get("expires_at") != intent.expires_at
            or record.get("credential_ref") is not None
            or record.get("requested_by") != {"surface": OP}
            or record.get("machine") != intent.owner.model_dump()
            or "device" in record
        ):
            raise ValueError("file_approval_intent_invalid")
        return intent
    except (ValidationError, TypeError, ValueError, KeyError):
        raise _refuse("file_approval_intent_invalid") from None


def require_filing_shape(record: Mapping[str, Any], root: Path | None) -> None:
    intent = _intent(record)
    candidate = _FILING.get()
    if (
        root is None
        or candidate is None
        or candidate.root != Path(root).resolve()
        or candidate.intent != intent
    ):
        raise _refuse("file_approval_listener_required")
    _local_context(intent, candidate.root, candidate.auth)
    facts = _mesh(candidate.link, candidate.root)
    if (
        any(facts[k] != getattr(intent, k) for k in facts)
        or candidate.link.server.instance_id != intent.relay_instance
    ):
        raise _refuse("file_approval_link_changed")
    if (
        candidate.link.result.link_id != intent.link_id
        or candidate.link.result.transcript_hash != intent.transcript_hash
    ):
        raise _refuse("file_approval_link_changed")
    _shelf(record, candidate.root)


def require_current(record: Mapping[str, Any], root: Path | None) -> None:
    intent = _intent(record)
    if root is None or A.request_digest(record) != record.get("request_digest"):
        raise _refuse("file_approval_intent_changed")
    root = Path(root).resolve()
    checked = _CHECKED.get()
    if (
        checked is None
        or checked.root != root
        or checked.intent != intent
        or checked.digest != record.get("request_digest")
    ):
        raise _refuse("file_approval_preflight_required")
    _shelf(record, root)
    with _own_auth(root) as auth:
        _local_context(intent, root, auth)
    if not intent.created_at <= time.time() < intent.expires_at:
        raise _refuse("file_approval_expired")


def require_write(record: Mapping[str, Any], root: Path | None) -> None:
    intent = _intent(record)
    state = record.get("state")
    if state == A.STATE_REQUESTED:
        require_filing_shape(record, root)
    elif state in {A.STATE_APPROVED, A.STATE_CONNECTING, A.STATE_CONNECTED}:
        if not _MUTATING.get():
            raise _refuse("file_approval_typed_required")
        require_current(record, root)
        if root is None or not isinstance(record.get("signature"), dict):
            raise _refuse("file_approval_evidence_invalid")
        applied_at = record.get("decided_at")
        if not isinstance(applied_at, (int, float)):
            raise _refuse("file_approval_evidence_invalid")
        checked = _Slot(
            generation=intent.generation,
            status="active" if intent.action == "approve" else "revoked",
            intent=intent,
            record=_evidence(record),
            applied_at=applied_at,
        )
        _verify_evidence(checked, Path(root).resolve())
    elif state not in {A.STATE_DENIED, A.STATE_EXPIRED, A.STATE_WITHDRAWN, A.STATE_FAILED}:
        raise _refuse("file_approval_state_invalid")


def require_signing(request_id: str, digest: str, root: Path) -> None:
    record = A._find_by_request_id(request_id, root)
    if (
        record is None
        or record.get("state") != A.STATE_REQUESTED
        or record.get("request_digest") != digest
        or A.request_digest(record) != digest
    ):
        raise _refuse("file_approval_intent_changed")
    prepare_current(record, root, canonical=True)
    require_current(record, root)


async def file_request(
    link: Any,
    auth: AuthStore,
    credential_id: int,
    *,
    action: Literal["approve", "revoke"] = "approve",
) -> dict[str, Any]:
    """Trusted listener-side internal issuer; no raw-body or public mint.

    Candidate preparation is NOT a mutation fence. Merely filing a new card
    cannot suspend a previously completed association. Only apply publishes the
    negative PENDING checkpoint. External account IO precedes all file locks.
    """
    root = _link_root(link)
    anchor = _anchor(root)
    mesh = _mesh(link, root)
    account = _native(auth, credential_id, root)
    now = time.time()
    draft = FileIntent(
        **mesh,
        account=account,
        anchor=anchor,
        action=action,
        authority_epoch=token_hex(16),
        expected_generation=0,
        generation=1,
        request_id=A.new_request_id(),
        nonce=token_hex(16),
        relay_instance=link.server.instance_id,
        link_id=link.result.link_id,
        transcript_hash=link.result.transcript_hash,
        created_at=now,
        expires_at=now + A.DEFAULT_EXPIRY_S,
    )
    await _canonical(draft, auth, root)
    if _mesh(link, root) != mesh:
        raise _refuse("file_approval_link_changed")
    with _locked(root, initialize=True):
        document, journal = _initialize(root, draft.authority_epoch)
        prior = document.slots.get(_association(draft))
        fence = journal.fences.get(_association(draft))
        generation = max(prior.generation if prior else 0, fence.generation if fence else 0)
        intent = FileIntent.model_validate(
            {
                **draft.model_dump(),
                "authority_epoch": document.authority_epoch,
                "expected_generation": generation,
                "generation": generation + 1,
            }
        )
        preview = {
            "schema": A.SCHEMA,
            "kind": A.KIND_FILE_DEVICE_APPROVAL,
            "request_id": intent.request_id,
            "requested_by": {"surface": OP},
            "machine": intent.owner.model_dump(),
            "what": {OP: intent.model_dump(mode="json"), "anchor": intent.anchor.model_dump()},
            "credential_ref": None,
            "created_at": intent.created_at,
            "expires_at": intent.expires_at,
        }
        _save_candidate(root, intent, A.request_digest(preview))
    candidate = _Candidate(root, link, auth, intent)
    token = _FILING.set(candidate)
    try:
        return A.create_request(
            kind=A.KIND_FILE_DEVICE_APPROVAL,
            request_id=intent.request_id,
            requested_by={"surface": OP},
            machine=intent.owner.model_dump(),
            what=preview["what"],
            credential_ref=None,
            created_at=intent.created_at,
            expires_at=intent.expires_at,
            root=root,
        )
    finally:
        _FILING.reset(token)


def _evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "kind",
            "request_id",
            "approval_id",
            "requested_by",
            "machine",
            "what",
            "credential_ref",
            "created_at",
            "expires_at",
            "request_digest",
            "decided_at",
            "signature",
        )
    }


def _verify_evidence(slot: _Slot, root: Path) -> None:
    record = slot.record
    if _intent(record) != slot.intent or A.request_digest(record) != record.get("request_digest"):
        raise _refuse("file_approval_evidence_invalid")
    trio = A.local_anchor_trio(root)
    if trio is None or _anchor(root) != slot.intent.anchor:
        raise _refuse("file_approval_anchor_changed")
    signature = record.get("signature")
    decided = record.get("decided_at")
    if (
        not isinstance(signature, dict)
        or set(signature) != {"by", "alg", "sig", "key_id", "cert"}
        or signature.get("alg") != "ES256"
        or signature.get("cert") is not None
        or signature.get("by") != "operator:" + slot.intent.anchor.key_id
        or signature.get("key_id") != slot.intent.anchor.key_id
        or not isinstance(decided, (int, float))
        or isinstance(decided, bool)
        or not slot.intent.created_at <= decided < slot.intent.expires_at
    ):
        raise _refuse("file_approval_evidence_invalid")
    A._verify_decision_signature(
        record,
        decision="approve",
        decided_at=decided,
        signature_hex=signature.get("sig", ""),
        spki=trio["spki"],
    )


def expected(record: Mapping[str, Any]) -> dict[str, Any]:
    intent = _intent(record)
    return {"request_id": intent.request_id, "request_digest": A.request_digest(record)}


def _finish_journal(root: Path, pending: _Journal, intent: FileIntent) -> _Journal:
    key = _association(intent)
    current = _load_journal(root)
    if current != pending or current.fences[key].phase != "pending":
        raise _refuse("file_approval_journal_conflict")
    fence = current.fences[key].model_copy(update={"phase": "complete"})
    row = current.candidates[intent.request_id].model_copy(update={"state": "consumed"})
    finished = current.model_copy(
        update={
            "fences": {**current.fences, key: fence},
            "candidates": {**current.candidates, intent.request_id: row},
        }
    )
    try:
        return _publish_journal(root, finished, expected_revision=current.revision)
    except (MeshRefusal, OSError, ValueError):
        # Mirror finite_journal's negative compensation. Never infer rollback
        # from a failed completion acknowledgement; retain the SAME pending
        # checkpoint at a higher revision. Total loss/inability to write every
        # witness is not claimed recoverable or arbitrary-snapshot resistant.
        latest = _load_journal(root)
        compensation = pending.model_copy(update={"revision": latest.revision})
        _publish_journal(root, compensation, expected_revision=latest.revision)
        raise _refuse("file_approval_write_unconfirmed") from None


def apply(approval_id: str, expected_block: dict[str, Any], root: Path) -> dict[str, Any]:
    """Pending-before-projection CAS; every desktop worker opens its own context."""
    import os

    root = root.resolve()
    record = A._load_raw(approval_id, root)
    if expected(record) != expected_block:
        raise _refuse("file_approval_intent_changed")
    intent = _intent(record)
    with mutation_context(record, root, canonical=True):
        if record.get("state") in {A.STATE_CONNECTING, A.STATE_CONNECTED}:
            run_id = str((A._active_run(record) or {}).get("run_id") or "")
        else:
            run_id = A.new_run_id()
            A.begin_run(approval_id, run_id=run_id, runner_pid=os.getpid(), root=root)
        # Last authenticated local status read immediately before mutation.
        # No HTTP, control socket or OS prompt occurs under either lock below.
        _status_current(intent, root)
        with A._record_lock(approval_id, root):
            record = A.verify_for_run(approval_id, root=root)
            require_current(record, root)
            if expected(record) != expected_block or record.get("state") not in {
                A.STATE_CONNECTING,
                A.STATE_CONNECTED,
            }:
                raise _refuse("file_approval_not_approved")
            with _locked(root):
                document, journal = _load(root), _load_journal(root)
                if (
                    document.authority_epoch != intent.authority_epoch
                    or journal.authority_epoch != intent.authority_epoch
                ):
                    raise _refuse("file_approval_epoch_changed")
                key = _association(intent)
                prior, fence = document.slots.get(key), journal.fences.get(key)
                same = bool(
                    fence
                    and fence.generation == intent.generation
                    and fence.request_id == intent.request_id
                    and fence.approval_id == approval_id
                    and fence.digest == record["request_digest"]
                    and fence.action == intent.action
                )
                if same and fence is not None and fence.phase == "complete":
                    if (
                        prior is None
                        or prior.record != _evidence(record)
                        or prior.generation != intent.generation
                    ):
                        # A completed positive cannot reconstruct a missing or
                        # rewound current projection from an old approved card.
                        raise _refuse("file_approval_state_unavailable")
                    _verify_evidence(prior, root)
                    if prior.status == "active":
                        _active_fence(journal, prior)
                    _finish_record(approval_id, record, run_id, root)
                    return {"status": prior.status, "generation": prior.generation}
                if not same:
                    high = fence.generation if fence else 0
                    if high != intent.expected_generation or (prior and prior.generation > high):
                        raise _refuse("file_approval_generation_changed")
                    # Reserve begin/finish/negative-compensation revisions; do
                    # not wrap the established exact-JSON integer domain.
                    if journal.revision + 3 > _MAX_REVISION:
                        raise _refuse("file_approval_journal_capacity")
                    denied = max(
                        fence.deny_generation if fence else 0,
                        intent.generation if intent.action == "revoke" else 0,
                    )
                    fence = _Fence(
                        generation=intent.generation,
                        phase="pending",
                        action=intent.action,
                        request_id=intent.request_id,
                        approval_id=approval_id,
                        digest=record["request_digest"],
                        deny_generation=denied,
                        deny_reason="revoked" if denied else None,
                    )
                    journal = _publish_journal(
                        root,
                        journal.model_copy(update={"fences": {**journal.fences, key: fence}}),
                        expected_revision=journal.revision,
                    )
                # PENDING (and a revoke's g2 denial) is durable BEFORE this
                # mutable projection can change. Any interrupted writer denies.
                require_current(record, root)
                slot = _Slot(
                    generation=intent.generation,
                    status="active" if intent.action == "approve" else "revoked",
                    intent=intent,
                    record=_evidence(record),
                    applied_at=time.time(),
                )
                _verify_evidence(slot, root)
                if (
                    prior is None
                    or prior.record != slot.record
                    or prior.generation != slot.generation
                ):
                    _write(
                        root, document.model_copy(update={"slots": {**document.slots, key: slot}})
                    )
                _finish_journal(root, journal, intent)
                _finish_record(approval_id, record, run_id, root)
                return {"status": slot.status, "generation": slot.generation}


def _finish_record(approval_id: str, record: Mapping[str, Any], run_id: str, root: Path) -> None:
    # Caller already holds record -> file-domain locks in that fixed order.
    # Authority was durably saved first. A bookkeeping failure can therefore be
    # indeterminate, never a claim that the association remained unchanged.
    if record.get("state") != A.STATE_CONNECTED:
        try:
            A._mark_connected_locked(approval_id, run_id=run_id, step=OP, root=root)
        except OSError:
            raise _refuse("file_approval_write_unconfirmed") from None


def revoke(approval_id: str, expected_block: dict[str, Any], root: Path) -> dict[str, Any]:
    """Apply only the separately signed revoke action, never reinterpret approval."""
    if _intent(A._load_raw(approval_id, root)).action != "revoke":
        raise _refuse("file_approval_action_mismatch")
    return apply(approval_id, expected_block, root)


@dataclass(frozen=True, slots=True)
class _VerifiedFileContext:
    """Owner-internal snapshot, never a raw slot or permission to disclose bytes.

    Construct only after signature/current-context/negative-journal verification
    under the file-domain lock. The intent contains frozen scalar-only models;
    neither the mutable signed record nor the journal maps escape that lock.
    """

    association: str
    authority_epoch: str
    generation: int
    request_digest: str
    journal_revision: int
    deny_generation: int
    intent: FileIntent


async def _verified_context(link: Any, auth: AuthStore, credential_id: int) -> _VerifiedFileContext:
    """Metadata-only current-context check; future P3 still needs its own guards.

    Activation request expiry does not expire standing association; context and
    revocation do. This validates the retained signed evidence even after card
    pruning, but never reconstructs a missing positive from that history.
    """
    root = _link_root(link)
    mesh = _mesh(link, root)
    native = _native(auth, credential_id, root)
    anchor = _anchor(root)
    try:
        data = await auth.ensure_oauth_fresh_or_raise(credential_id)
        if not isinstance(data, dict):
            raise _refuse("file_approval_account_unverified")
        account_id = await verify_canonical_account(
            data.get("access") or data.get("access_token") or ""
        )
    except Exception as error:
        raise _refuse("file_approval_account_unverified") from error
    if account_id != native.account_id or _native(auth, credential_id, root) != native:
        raise _refuse("file_approval_account_changed")
    with _locked(root):
        document = _load(root)
        journal = _load_journal(root)
        if journal.authority_epoch != document.authority_epoch:
            raise _refuse("file_approval_epoch_changed")
        # Use the exact current key association, not a device-id/previous-id alias.
        for key, slot in document.slots.items():
            intent = slot.intent
            if (
                all(mesh[k] == getattr(intent, k) for k in mesh)
                and intent.account == native
                and intent.anchor == anchor
            ):
                if slot.status != "active":
                    raise _refuse("file_approval_revoked")
                _verify_evidence(slot, root)
                _active_fence(journal, slot)
                if (
                    _mesh(link, root) != mesh
                    or _native(auth, credential_id, root) != native
                    or _anchor(root) != anchor
                ):
                    raise _refuse("file_approval_context_changed")
                return _VerifiedFileContext(
                    association=key,
                    authority_epoch=document.authority_epoch,
                    generation=slot.generation,
                    request_digest=slot.record["request_digest"],
                    journal_revision=journal.revision,
                    deny_generation=journal.fences[key].deny_generation,
                    intent=intent,
                )
    raise _refuse("file_approval_not_active")


async def authorized_context(link: Any, auth: AuthStore, credential_id: int) -> dict[str, Any]:
    """Preserve the existing metadata API; this projection grants no file access."""
    context = await _verified_context(link, auth, credential_id)
    return {
        "association": context.association,
        "authority_epoch": context.authority_epoch,
        "generation": context.generation,
        "request_digest": context.request_digest,
    }


class FileDeviceApprovalOp:
    op = OP
    kind = A.KIND_FILE_DEVICE_APPROVAL
    file_only_signing = False
    cli_approvable = False

    def intent(self, record: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return _intent(record).model_dump(mode="json")
        except MeshRefusal:
            return None

    def disclosure(self, record: dict[str, Any]) -> dict[str, Any]:
        intent = _intent(record)
        from local_operator.paths import config_dir

        root = config_dir().resolve()
        prepare_current(record, root)
        require_current(record, root)
        # No label-only synopsis: the future reviewed disclosure must show every
        # committed fact and obtain the existing server recomputed digest.
        return {"op": OP, "kind": self.kind, "intent": intent.model_dump(mode="json")}

    def expected(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return expected(record)

    def apply(self, approval_id: str, expected: dict[str, Any], root: Path) -> Any:
        return apply(approval_id, expected, root)

    def generic_refusal(self, record: Mapping[str, Any], approval_id: str) -> MeshRefusal | None:
        return _refuse("file_approval_typed_required")


approval_ops.register(FileDeviceApprovalOp())
