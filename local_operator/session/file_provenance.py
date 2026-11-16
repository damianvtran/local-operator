"""Owner-live, metadata-only file provenance; never a file-serving authority.

Capture happens at native admission/executor completion, not transcript replay or
inventory time. A frozen ticket is process-private until the owning transcript
joins it to exact retained bytes. Only committed private records can later be
read by another process. Losing a ticket cannot be repaired by statting history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

logger = logging.getLogger(__name__)
DIRECTORY = ".file-provenance"
SENTINEL = ".file-provenance.initialized.json"
RECEIPT = "file_provenance_v1"
NATIVE_TYPES = frozenset({"file_provenance_native_call_v1", "file_provenance_native_v1"})
FRESH = object()
CONTINUATION = object()
# Only the actual native executor sets this context. Async registration carries
# it privately; persisted roster dictionaries and wait/peek replays cannot.
EXECUTION_SOURCE: ContextVar[dict[str, Any] | None] = ContextVar(
    "file_provenance_execution", default=None
)


def admitted_ticket(session: Any, text: str, command_id: str, admission: Any, injected: bool):
    """Continue only an original live handle; a cold spool is not fresh input."""
    if injected:
        return None
    if admission is CONTINUATION:
        take = getattr(session, "_take_file_continuation", None)
        return take(command_id) if callable(take) else None
    capture = getattr(session, "_capture_file_input", None)
    return capture(text, command_id) if callable(capture) and valid_admission(admission) else None


@dataclass(frozen=True, slots=True)
class Candidate:
    """A literal parser result, not a directory scan or an expanded expression."""

    reference: str
    locator: str


@dataclass(frozen=True, slots=True)
class FrozenFile:
    reference_digest: str
    locator: str
    path: str
    root: str
    cwd: str
    home: str
    root_identity: tuple[int, ...]
    identity: tuple[int, int, int, int, int, int]


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_ctime_ns,
        info.st_mtime_ns,
        info.st_size,
        info.st_nlink,
    )


def _capture_file(candidate: Candidate, *, cwd: Path, home: Path, config_root: Path) -> FrozenFile:
    """Freeze through S1's descriptor checks, without reading any target bytes.

    Original components matter: collapsing link/../target would attest a different
    lookup. Dynamic shell expansion is likewise not evidence of a literal name.
    Caller handles an unknown candidate without delaying the ordinary message.
    """
    from local_operator.network import files
    from local_operator.supplements.denylist import is_sensitive_for_supplements

    raw = candidate.reference
    # Raw `..` is refused with its own reason rather than collapsed lexically:
    # the user/tool named a traversal, and normalizing it away would attest a
    # different name than the one actually written.
    if ".." in Path(raw).parts:
        raise ValueError("file_provenance_unsafe_components")
    if any(c in raw for c in ("$", "`", "*", "?", "\x00")):
        raise ValueError("file_provenance_unproven_reference")
    if raw.startswith("~/"):
        path = home / raw[2:]
    elif raw.startswith("~"):
        raise ValueError("file_provenance_unproven_reference")
    else:
        path = Path(raw) if Path(raw).is_absolute() else cwd / raw
    # Relocated config roots must not rely on the ambient/default denylist root.
    if not any(path == allowed or allowed in path.parents for allowed in (cwd, home)):
        raise ValueError("file_provenance_outside_owner_context")
    if path == config_root or config_root in path.parents or is_sensitive_for_supplements(path):
        raise ValueError("file_provenance_sensitive")
    scope = files._read_scope(Path(path.anchor), path.parent, "file_provenance")
    source = files._Source(
        path=path,
        scope=scope,
        name=path.name,
        mime="application/octet-stream",
        label="file_provenance",
        attachment="",
    )
    fd, info = files._open_source(source, "file_provenance", "owner")
    try:
        before = _identity(info)
        if before != _identity(os.fstat(fd)) or info.st_nlink != 1:
            raise ValueError("file_provenance_identity_changed")
        return FrozenFile(
            hashlib.sha256(raw.encode()).hexdigest(),
            candidate.locator,
            str(path),
            str(path.parent),
            str(cwd),
            str(home),
            tuple(_directory_identity(path.parent)),
            before,
        )
    finally:
        os.close(fd)


_SEAL = object()


class UserAdmission:
    """Native checked-command context, never a wire or human/account assertion."""

    __slots__ = ("locality", "_seal")

    def __init__(self, seal: object, locality: str):
        if seal is not _SEAL:
            raise TypeError("file_provenance_native_admission_required")
        self.locality, self._seal = locality, seal

    def __reduce__(self):
        raise TypeError("file_provenance_admission_not_serializable")


def user_admission(authority: Any, operation: str = "prompt") -> UserAdmission | bool:
    # Called only by native runtime dispatch AFTER its existing control-key and
    # capability checks. Incoming source/role/owner_typed JSON is never consulted.
    if authority.locality not in {"local", "remote"} or (
        authority.locality == "remote" and operation not in authority.capabilities
    ):
        return False
    return UserAdmission(_SEAL, str(authority.locality))


def valid_admission(value: Any) -> bool:
    return value is None or isinstance(value, UserAdmission) and value._seal is _SEAL


@dataclass(frozen=True, slots=True)
class FrozenOrigin:
    """Owner/store context observed at native capture, never rebound at commit."""

    owner: str
    config_root: tuple[int, int]
    session: str
    incarnation: str | None
    birth: float | None


class CaptureTicket:
    """Nonserialized native handle. JSON/role/source labels cannot recreate it."""

    __slots__ = (
        "nonce",
        "source",
        "files",
        "unknown",
        "context",
        "origin_id",
        "call_id",
        "job_id",
        "outer_call_id",
        "_published",
        "_seal",
    )

    def __init__(
        self,
        seal: object,
        *,
        source: str,
        files: tuple[FrozenFile, ...],
        unknown: tuple[str, ...],
        context: FrozenOrigin | None = None,
    ):
        if seal is not _SEAL:
            raise TypeError("file_provenance_native_ticket_required")
        self.nonce = secrets.token_hex(16)
        self.source, self.files, self.unknown, self._seal = source, files, unknown, seal
        self.context = context
        self.origin_id: str | None = None
        self.call_id: str | None = None
        self.job_id: str | None = None
        self.outer_call_id: str | None = None
        self._published = False

    def __reduce__(self):
        raise TypeError("file_provenance_ticket_not_serializable")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def text_candidates(text: str) -> Iterable[Candidate]:
    """Share inventory's pure lexical leaves, never its stat-at-scan callbacks.

    Assistant explicit references intentionally use the same leaf grammar as
    user text here; inventory's prose-promotion policy is not provenance policy.
    """
    from local_operator.session_files.scan import (
        _FILE_URL,
        _PROSE_PATH,
        _REFERENCE_HEAD,
    )
    from local_operator.supplements.candidates import _MD_LINK

    for match in _MD_LINK.finditer(text):
        if "://" not in match["target"] and not match["target"].startswith("#"):
            yield Candidate(match["target"], f"markdown:{match.start()}:{match.end()}")
    for match in re.finditer(r"`((?:\./)?[^`\s]+\.[A-Za-z0-9]+)`", text):
        yield Candidate(match[1], f"quoted:{match.start()}:{match.end()}")
    for grammar, pattern in (
        ("reference", _REFERENCE_HEAD),
        ("file_url", _FILE_URL),
        ("literal", _PROSE_PATH),
    ):
        for match in pattern.finditer(text):
            raw = match.group(1) if grammar != "file_url" else match.group(0)
            if raw.startswith("file://"):
                parsed = urlsplit(raw)
                if parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
                    continue
                raw = unquote(parsed.path)
            yield Candidate(raw.rstrip(".,;:"), f"{grammar}:{match.start()}:{match.end()}")


def argument_candidates(tool: str, args: Mapping[str, Any]) -> Iterable[Candidate]:
    from local_operator.session_files.scan import _MENTION_ARG_KEYS

    for key in _MENTION_ARG_KEYS:
        value = args.get(key)
        values = value if isinstance(value, list) else [value]
        for index, item in enumerate(values):
            if isinstance(item, str) and item:
                yield Candidate(item, f"argument:{key}:{index}")
    if tool == "bash" and isinstance(args.get("command"), str):
        from local_operator.supplements.candidates import (
            SHELL_SCAN_MAX_CHARS,
            _shell_tokens,
            bash_effects,
            literal_bash_inputs,
        )

        command = args["command"]
        if len(command) > SHELL_SCAN_MAX_CHARS:
            return
        # Existing deterministic output grammar includes known literal cwd/env
        # effects. Never use its sweep_dirs: discovering a file is not naming it.
        raw_tokens = _shell_tokens(command)
        if raw_tokens is None:
            return
        unsafe = [
            token.split("=", 1)[-1]
            for token in raw_tokens
            if ".." in Path(token.split("=", 1)[-1]).parts
        ]
        if unsafe:
            for token in unsafe:
                yield Candidate(token, "shell_unsafe_lexical")
            return
        for index, output in enumerate(bash_effects(command).outputs):
            yield Candidate(output, f"shell_output:{index}")
        for index, path in enumerate(literal_bash_inputs(command)):
            yield Candidate(path, f"shell_literal:{index}")
    elif tool == "eval" and isinstance(args.get("code"), str):
        from local_operator.supplements.candidates import literal_eval_paths

        for index, path in enumerate(literal_eval_paths(args["code"])):
            yield Candidate(path, f"eval_literal:{index}")


def native_result_candidates(tool: str, details: Any) -> Iterable[Candidate]:
    # These are explicit native path contracts, not arbitrary stdout/web text.
    # Generated digest-only media has no outside-PAD path to attest; its existing
    # attachment authority stays separate and is never converted to a raw path.
    if tool not in {"write", "edit", "browser"} or not isinstance(details, dict):
        return
    if isinstance(details.get("path"), str):
        yield Candidate(details["path"], "result:path")
    if tool == "browser" and isinstance(details.get("files"), list):
        for index, item in enumerate(details["files"]):
            path = item.get("path") if isinstance(item, dict) else item
            if isinstance(path, str):
                yield Candidate(path, f"result:files:{index}")


def _payload_candidates(payload: Mapping[str, Any]) -> Iterable[Candidate]:
    if payload.get("role") in {"tool", "tool_result"}:
        yield from native_result_candidates(
            str(payload.get("tool_name", "")), payload.get("details")
        )
        return  # arbitrary tool/web text is not owner adoption
    content = payload.get("content", [])
    if isinstance(content, str):
        yield from text_candidates(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("text"), str):
                yield from text_candidates(block["text"])
            if block.get("type") in {"tool_call", "toolCall"}:
                args = block.get("arguments", block.get("args", {}))
                if isinstance(args, dict):
                    yield from argument_candidates(str(block.get("name", "")), args)
    for call in payload.get("tool_calls") or ():
        if isinstance(call, dict) and isinstance(call.get("arguments"), dict):
            yield from argument_candidates(str(call.get("name", "")), call["arguments"])
    # Native host custom events contain actual structured facts. Their issuing
    # handle is still required; arbitrary imported dictionaries never call this.
    details = payload.get("details")
    if isinstance(details, dict) and isinstance(details.get("file_arguments"), dict):
        yield from argument_candidates(str(details.get("tool", "")), details["file_arguments"])


class OwnerProvenance:
    """One owning Session's short-lived handles, not a consumer authority map.

    Fully committed proof verification uses only disk. Fresh admissions queued
    in this runtime may reuse a frozen handle; cold spool delivery never recaptures.
    """

    def __init__(self, transcript: Any, *, config_root: Path, cwd: Path, home: Path):
        self.transcript = transcript
        self.config_root, self.cwd, self.home = config_root, cwd, home
        self.messages: dict[str, list[CaptureTicket]] = {}
        self.admissions: dict[str, CaptureTicket] = {}
        self.calls: dict[str, tuple[Any, CaptureTicket | None]] = {}
        self.call_sources: dict[str, str] = {}
        self.call_references: dict[str, set[tuple[str, str]]] = {}
        self.native_calls: dict[str, tuple[Any, CaptureTicket]] = {}

    def capture(
        self, source: str, candidates: Iterable[Candidate], *, cwd: Path | None = None
    ) -> CaptureTicket:
        files, unknown, seen = [], [], set()
        try:
            context = _capture_origin(self.transcript, self.config_root)
        except Exception:
            logger.debug("file provenance owner context unavailable", exc_info=True)
            return CaptureTicket(
                _SEAL, source=source, files=(), unknown=("file_provenance_owner_context_unknown",)
            )
        try:
            for candidate in candidates:
                key = (candidate.reference, candidate.locator)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    files.append(
                        _capture_file(
                            candidate,
                            cwd=cwd or self.cwd,
                            home=self.home,
                            config_root=self.config_root,
                        )
                    )
                except Exception as failure:
                    # No ordinary prompt/tool/transcript failure may be caused by
                    # provenance. Crucially this outcome is never restatted later.
                    code = getattr(failure, "code", str(failure))
                    unknown.append(
                        code if re.fullmatch(r"[a-z_]+", code) else "file_provenance_unproven"
                    )
        except Exception:
            # A failing pure parser/producer is UNKNOWN, not a failed prompt and
            # not permission to keep a partially enumerated positive capture.
            logger.debug("file provenance candidate capture unavailable", exc_info=True)
            files.clear()
            unknown.append("file_provenance_capture_unavailable")
        return CaptureTicket(
            _SEAL, source=source, files=tuple(files), unknown=tuple(unknown), context=context
        )

    def admit(self, text: str, command_id: str) -> CaptureTicket:
        if command_id not in self.admissions:
            self.admissions[command_id] = self.capture("user", text_candidates(text))
        return self.admissions[command_id]

    def continuation(self, command_id: str) -> CaptureTicket | None:
        return self.admissions.pop(command_id, None)

    def bind(self, message_id: str, ticket: Any) -> None:
        if isinstance(ticket, CaptureTicket) and ticket._seal is _SEAL:
            self.messages.setdefault(message_id, []).append(ticket)

    def capture_message(self, message: Any) -> None:
        from local_operator.session.transcript import encode_message_payload

        payload = encode_message_payload(message)
        self.bind(message.id, self.capture("assistant", _payload_candidates(payload)))
        for call in message.tool_calls:
            self.call_sources[call.id] = message.id
            self.call_references[call.id] = {
                (hashlib.sha256(c.reference.encode()).hexdigest(), c.locator)
                for c in argument_candidates(call.name, call.arguments)
            }

    def take(self, messages: Iterable[Any]) -> dict[str, list[CaptureTicket]]:
        bindings = {}
        for message in messages:
            if message.id in self.messages:
                bindings[message.id] = self.messages.pop(message.id)
            call_id = getattr(message, "tool_call_id", None)
            if call_id:
                self.call_sources.pop(call_id, None)
                self.call_references.pop(call_id, None)
                self.native_calls.pop(call_id, None)
                self.calls.pop(call_id, None)
        return bindings


def _private_read(path: Path) -> dict[str, Any]:
    from local_operator.network.store import _read_private_json_durable

    _directory_identity(path.parent)
    value = _read_private_json_durable(path, max_bytes=512 * 1024)
    if not isinstance(value, dict):
        raise ValueError("file_provenance_metadata_invalid")
    return value


def _private_write(path: Path, data: dict[str, Any]) -> None:
    from local_operator.network.store import _write_private_json_durable

    _write_private_json_durable(path, data)


def _directory_identity(path: Path) -> list[int]:
    from local_operator.network import files, sync

    scope = files._read_scope(Path(path.anchor), path, "file_provenance")
    fd = sync.open_directory(scope, scope.relative_in(path), "file_provenance")
    try:
        info = os.fstat(fd)
        return [info.st_dev, info.st_ino]
    finally:
        os.close(fd)


def _capture_origin(transcript: Any, config_root: Path) -> FrozenOrigin:
    from local_operator.network.identity import load
    from local_operator.session.creation import _stored

    owner = load(config_root)
    if owner is None:
        raise ValueError("file_provenance_owner_unknown")
    directory = transcript.directory
    incarnation = None
    if (directory / DIRECTORY).exists() or (directory / SENTINEL).exists():
        incarnation = _meta(directory, config_root)["incarnation"]
    else:
        previous = transcript._latest_custom_entries.get(RECEIPT)
        if (
            previous is not None
            and previous.payload.get("details", {}).get("source_session") == directory.name
        ):
            raise ValueError("file_provenance_initialized_state_missing")
    identity = _directory_identity(config_root)
    # Only a canonical stored birth is frozen. A deferred first-turn directory
    # has none yet (its filesystem fallback reads 0.0 on Linux), and that birth
    # is legitimately bound once by initialization in the same commit; freezing
    # the fallback would deny every first-ever prompt rather than a rebinding.
    return FrozenOrigin(
        owner.public_key,
        (identity[0], identity[1]),
        directory.name,
        incarnation,
        _stored(directory),
    )


def _matches_origin(context: Any, meta: Mapping[str, Any]) -> bool:
    return isinstance(context, dict) and (
        context.get("owner") == meta["owner"]
        and list(context.get("config_root", ())) == meta["config_root"]
        and context.get("session") == meta["session"]
        and (context.get("incarnation") is None or context["incarnation"] == meta["incarnation"])
        and (context.get("birth") is None or context["birth"] == meta["birth"])
    )


def _meta(directory: Path, config_root: Path) -> dict[str, Any]:
    from local_operator.network.identity import load
    from local_operator.session.creation import session_created_at

    owner = load(config_root)
    if owner is None:
        raise ValueError("file_provenance_owner_unknown")
    meta = _private_read(directory / DIRECTORY / "meta.json")
    marker = _private_read(directory / SENTINEL)
    if (
        type(meta.get("schema")) is not int
        or meta["schema"] != 1
        or marker != {"schema": 1, "incarnation": meta.get("incarnation")}
        or type(marker.get("schema")) is not int
        or meta.get("owner") != owner.public_key
        or meta.get("session") != directory.name
        or meta.get("directory") != _directory_identity(directory)
        or meta.get("config_root") != _directory_identity(config_root)
        or meta.get("store_directory") != _directory_identity(directory / DIRECTORY)
        or (directory / DIRECTORY).stat(follow_symlinks=False).st_mode & 0o077
        or meta.get("birth") != session_created_at(directory)
        or not isinstance(meta.get("incarnation"), str)
    ):
        raise ValueError("file_provenance_incarnation_unknown")
    return meta


def _initialize(transcript: Any, config_root: Path) -> dict[str, Any]:
    """Materialize only for a genuine live ticket, never for a reader/viewer.

    Deferred sessions need their first source birth and lifetime writer lock
    here. Failure to acquire exclusivity denies only the proof, not the journal.
    """
    from local_operator.network.identity import load
    from local_operator.session import compression
    from local_operator.session.creation import ensure_session_created_at

    directory = transcript.directory
    if transcript._read_only:
        raise ValueError("file_provenance_not_owner")
    directory.mkdir(parents=True, exist_ok=True)
    _directory_identity(directory)
    if not transcript._writer_lock_held:
        if not compression.acquire_writer(directory):
            raise ValueError("file_provenance_writer_unknown")
        transcript._writer_lock_held = True
    if (directory / DIRECTORY).exists() or (directory / SENTINEL).exists():
        return _meta(directory, config_root)
    previous = transcript._latest_custom_entries.get(RECEIPT)
    if (
        previous is not None
        and previous.payload.get("details", {}).get("source_session") == directory.name
    ):
        # Reuse the writer's existing constant-time bookkeeping index; never
        # walk history to reconstruct positive provenance after state loss.
        raise ValueError("file_provenance_initialized_state_missing")
    owner = load(config_root)
    if owner is None:
        raise ValueError("file_provenance_owner_unknown")
    birth = ensure_session_created_at(directory, transcript._created_at or time.time())
    transcript._created_at = birth
    from local_operator.network.store import _ensure_private_directory_durable

    _ensure_private_directory_durable(directory / DIRECTORY)
    meta = {
        "schema": 1,
        "incarnation": secrets.token_hex(16),
        "owner": owner.public_key,
        "store_directory": _directory_identity(directory / DIRECTORY),
        "session": directory.name,
        "directory": _directory_identity(directory),
        "config_root": _directory_identity(config_root),
        "birth": birth,
    }
    # Partial initialization remains damaged/unknown. Do not repair a directory
    # or sentinel left by a previous attempt, and never recover from transcript.
    _private_write(directory / SENTINEL, {"schema": 1, "incarnation": meta["incarnation"]})
    _private_write(directory / DIRECTORY / "meta.json", meta)
    return _meta(directory, config_root)


def prepare_batch(
    transcript: Any, rows: list[Any], bindings: Mapping[str, list[CaptureTicket]]
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Pending metadata + hidden receipt in the SAME ordered source-row batch.

    Called inside the existing transcript commit worker/order lock. A malformed,
    redacted-away or unmatched literal cannot use its premask original to mint.
    """
    from local_operator.session.transcript import ENTRY_CUSTOM, TranscriptEntry

    pending = []
    if not any(
        isinstance(ticket, CaptureTicket) and ticket._seal is _SEAL and ticket.files
        for tickets in bindings.values()
        for ticket in tickets
    ):
        return rows, pending
    try:
        meta = _initialize(transcript, transcript.directory.parent.parent)
        for row in rows:
            for ticket in bindings.get(row.id, []):
                if (
                    not isinstance(ticket, CaptureTicket)
                    or ticket._seal is not _SEAL
                    or ticket.context is None
                    or not _matches_origin(asdict(ticket.context), meta)
                    or ticket._published
                ):
                    continue
                ticket._published = True
                origin = (
                    next((r for r in rows if r.id == ticket.origin_id), None)
                    if ticket.origin_id
                    else None
                )
                if ticket.origin_id and origin is None:
                    continue
                retained = set()
                for source_payload in (
                    [origin.payload, row.payload] if origin is not None else [row.payload]
                ):
                    retained.update(
                        (hashlib.sha256(c.reference.encode()).hexdigest(), c.locator)
                        for c in _payload_candidates(source_payload)
                    )
                    if source_payload.get("custom_type") in {
                        "file_provenance_native_v1",
                        "file_provenance_native_call_v1",
                    }:
                        retained.update(
                            (r["digest"], r["locator"])
                            for r in source_payload.get("details", {}).get("references", [])
                        )
                for frozen in ticket.files:
                    if (frozen.reference_digest, frozen.locator) not in retained:
                        continue
                    opaque = secrets.token_hex(16)
                    proof = {
                        "schema": 1,
                        "id": opaque,
                        "state": "pending",
                        "nonce": ticket.nonce,
                        "incarnation": meta["incarnation"],
                        "owner": meta["owner"],
                        "session": meta["session"],
                        "source_class": ticket.source,
                        # JSON-normal form: the pending readback compares the
                        # parsed record for equality, so a tuple here would
                        # never equal its own durable list and never commit.
                        "capture_context": {
                            **asdict(ticket.context),
                            "config_root": list(ticket.context.config_root),
                        },
                        "event_id": row.id,
                        "origin_id": ticket.origin_id,
                        "call_id": ticket.call_id,
                        "job_id": ticket.job_id,
                        "outer_call_id": ticket.outer_call_id,
                        "file": json.loads(json.dumps(asdict(frozen))),
                    }
                    proof["metadata_digest"] = _digest(proof)
                    _private_write(transcript.directory / DIRECTORY / (opaque + ".json"), proof)
                    pending.append(proof)
        if pending:
            receipt = TranscriptEntry(
                secrets.token_hex(16),
                time.time(),
                ENTRY_CUSTOM,
                {
                    "custom_type": RECEIPT,
                    "details": {
                        "hidden": True,
                        "source_session": meta["session"],
                        "proofs": [
                            {
                                "id": p["id"],
                                "event_id": p["event_id"],
                                "source_class": p["source_class"],
                                "event_digest": hashlib.sha256(
                                    next(r for r in rows if r.id == p["event_id"])
                                    .to_json()
                                    .encode()
                                ).hexdigest(),
                                "metadata_digest": p["metadata_digest"],
                            }
                            for p in pending
                        ],
                    },
                },
            )
            rows.append(receipt)
        return rows, pending
    except Exception:
        logger.debug("file provenance pending publication unavailable", exc_info=True)
        return rows, []


def finish_batch(transcript: Any, rows: list[Any], pending: list[dict[str, Any]]) -> None:
    """Finalize only after the exact source/receipt fsync, without target restat.

    This runs in the same cancellation-shielded worker as the ordinary write.
    A failed finalization cannot roll back an already durable conversation.
    """
    if not pending:
        return
    try:
        offset = transcript.path.stat().st_size - sum(
            len((row.to_json() + "\n").encode()) for row in rows
        )
        spans = {}
        for row in rows:
            raw = (row.to_json() + "\n").encode()
            spans[row.id] = {
                "id": row.id,
                "offset": offset,
                "length": len(raw),
                "digest": hashlib.sha256(raw).hexdigest(),
            }
            offset += len(raw)
        for proof in pending:
            path = transcript.directory / DIRECTORY / (proof["id"] + ".json")
            if _private_read(path) != proof:
                continue
            committed = {
                **proof,
                "state": "committed",
                "source": spans[proof["event_id"]],
                "receipt": spans[rows[-1].id],
            }
            if proof["origin_id"] is not None:
                committed["origin"] = spans[proof["origin_id"]]
            committed["proof_digest"] = _digest(committed)
            _private_write(path, committed)
    except Exception:
        logger.debug("file provenance finalization unavailable", exc_info=True)


def lookup_proof(config_root: Path, source_session: str, opaque_id: str) -> dict[str, Any] | None:
    """Point-read completed owner metadata; UNKNOWN is None, never a bytes grant.

    Exact recorded row spans avoid a history scan/rebuild. Like the existing raw
    indexed-row reader, one row above its 512 KiB read budget is unavailable.
    No missing/pending proof is repaired and no caller-supplied path is accepted.
    """
    from local_operator.network import files, sync

    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_session) or not re.fullmatch(
        r"[0-9a-f]{32}", opaque_id
    ):
        return None
    directory = config_root / "sessions" / source_session
    try:
        meta = _meta(directory, config_root)
        proof = _private_read(directory / DIRECTORY / (opaque_id + ".json"))
        digest = proof.get("proof_digest")
        if (
            type(proof.get("schema")) is not int
            or proof["schema"] != 1
            or proof.get("state") != "committed"
            or proof.get("id") != opaque_id
            or proof.get("incarnation") != meta["incarnation"]
            or proof.get("owner") != meta["owner"]
            or proof.get("session") != source_session
            or not _matches_origin(proof.get("capture_context"), meta)
            or digest != _digest({k: v for k, v in proof.items() if k != "proof_digest"})
        ):
            return None
        path = directory / "transcript.jsonl"
        scope = files._read_scope(Path(path.anchor), directory, "file_provenance")
        fd = sync.open_for_read(path, scope, nonblock=True)
        try:
            source_rows = {}
            for name in (
                ("source", "receipt", "origin")
                if proof.get("origin_id") is not None
                else ("source", "receipt")
            ):
                span = proof[name]
                if (
                    type(span["offset"]) is not int
                    or type(span["length"]) is not int
                    or span["offset"] < 0
                    or not 0 < span["length"] <= 512 * 1024
                ):
                    return None
                raw = os.pread(fd, span["length"], span["offset"])
                if (
                    not raw.endswith(b"\n")
                    or hashlib.sha256(raw).hexdigest() != span["digest"]
                    or json.loads(raw)["id"] != span["id"]
                ):
                    return None
                source_rows[name] = json.loads(raw)
            if proof.get("call_id") is not None:
                source_payload = source_rows["source"]["payload"]
                source_call = source_payload.get("tool_call_id") or source_payload.get(
                    "details", {}
                ).get("call_id")
                if source_call != proof["call_id"]:
                    return None
                if "origin" in source_rows:
                    origin_payload = source_rows["origin"]["payload"]
                    if not (
                        origin_payload.get("details", {}).get("call_id") == proof["call_id"]
                        or any(
                            c.get("id") == proof["call_id"]
                            for c in origin_payload.get("tool_calls", [])
                        )
                    ):
                        return None
                elif proof["source_class"] != "tool_call":
                    return None
        finally:
            os.close(fd)
        frozen = proof["file"]
        now = _capture_file(
            Candidate(frozen["path"], frozen["locator"]),
            cwd=Path(frozen["cwd"]),
            home=Path(frozen["home"]),
            config_root=config_root,
        )
        if (
            list(now.identity) != list(frozen["identity"])
            or list(now.root_identity) != frozen["root_identity"]
        ):
            return None
        return proof
    except Exception:
        return None
