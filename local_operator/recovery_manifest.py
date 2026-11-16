"""Lossless dry-run classification, manifest and restore for LEGACY recovery records.

WHY THIS EXISTS. Legacy ``.update-operations/<id>.json`` records cannot be
proved terminal (see :mod:`local_operator.fenced_operation`): any loaded legacy
build may still rewrite them. Unblocking a full namespace therefore needs an
explicit OPERATOR abandonment decision, and that decision needs an exact,
reviewable inventory first. This module produces that inventory and a restore
path for it; it never decides, retires, moves or deletes anything.

CONSTRAINTS, deliberately enforced here:

* The classifier and manifest are PURE reads: no lock, no directory creation,
  no liveness probe, no write. Every fact about an entry (bytes, SHA-256, mode,
  mtime, parsed fields) comes from ONE bounded no-follow read of ONE descriptor,
  so the hash and the classification describe the same bytes.
* The candidate class is ``legacy_abandonment_candidate`` -- never "terminal",
  "complete" or "retirable". Every candidate stays PROTECTED; the label only
  names what an operator could choose to abandon. Phase and PID are never used
  as proof; phase is used only to EXCLUDE (a prepared candidate is a live ref).
* Anything referenced by ``current``/an installed pointer, a prepared
  candidate, a generation with an install attempt or v2 locator, and anything
  unknown, unsafe, unreadable or ambiguous is EXCLUDED, with its reason.
* :func:`capture_content` copies bytes into a content-addressed store OUTSIDE
  the root and never touches the source. :func:`restore_manifest` recreates a
  missing record exactly (bytes, mode, mtime) and never overwrites or deletes:
  an existing different file refuses the whole restore, an identical one is a
  no-op.
* Both run ONLY from an explicit manifest whose digest the caller passes as the
  reviewed ``approved_sha256``, and both preflight every entry before writing:
  bytes must match the reviewed hashes, and every path must resolve (after
  ``..`` and symlink resolution) inside the LEGACY ``.update-operations/`` of
  the reviewed root (never ``.update-operations-v2/``: that namespace is only
  written through the fenced protocol). Restore must target the reviewed
  manifest's own ``root``, which must already exist. Any failure refuses the
  run before anything is written or locked.

Callers are tests over synthetic roots today; nothing in the CLI invokes this,
and nothing discovers a live root implicitly: capture uses the reviewed
manifest's ``root``, and restore's ``root`` argument must equal it.
A live dry run is a separate, explicitly authorised read-only step.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from local_operator import install_transaction as _tx

MANIFEST_SCHEMA = "update-recovery-manifest.v1"
CANDIDATE_CLASS = "legacy_abandonment_candidate"
CANDIDATE_LABEL = "ABANDONMENT CANDIDATE - finality unproven; still protected"
#: Phases whose candidate generation is a prepared/in-flight ref. Exclusion only.
_PREPARED_PHASES = frozenset({"queued", "preparing", "prepared"})
#: Bounded scan: never iterate an unbounded directory in a dry run.
DEFAULT_MAX_ENTRIES = 4096
_V2_SIDECAR = _tx.V2_SIDECAR_NAME
#: The repo's one platform question for owner checks (see ``secrets/protocol.py``):
#: ``os.getuid`` does not exist off POSIX, and a Windows ``st_uid`` is always 0,
#: so ownership is only compared where a uid means something. Elsewhere the
#: other refusals (symlink, type, identity, lock) still apply.
_UID_IS_MEANINGFUL = os.name == "posix"
_V2_DIR = _tx.V2_OPERATION_DIR


class ManifestError(_tx.InstallTransactionError):
    """A manifest/restore precondition failed; nothing was changed."""


def _is_hex_id(identifier: str) -> bool:
    return len(identifier) == 32 and all(c in "0123456789abcdef" for c in identifier)


def _read_exact(path: Path) -> tuple[bytes, os.stat_result] | None:
    """One bounded, no-follow, non-blocking read; same refusals as ``read_record``."""
    descriptor: int | None = None
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return None
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        sample = os.fstat(descriptor)
        if (
            not stat.S_ISREG(sample.st_mode)
            or sample.st_nlink != 1
            or sample.st_size > _tx.MAX_RECOVERY_RECORD_BYTES
            or (os.name == "posix" and sample.st_uid != os.getuid())
        ):
            return None
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            data = stream.read(_tx.MAX_RECOVERY_RECORD_BYTES + 1)
        if len(data) > _tx.MAX_RECOVERY_RECORD_BYTES or len(data) != sample.st_size:
            # Grew or shrank while read: the stat and the bytes disagree.
            return None
        return data, sample
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _parse(data: bytes) -> dict[str, Any] | None:
    # The SAME byte/encoding/nesting bounds as the live reader, so the dry run
    # can never classify a record the reader would refuse.
    return _tx.parse_record_bytes(data)


def _referenced_generations(root: Path) -> tuple[set[str], list[str]]:
    """Generation NAMES held by top-level pointers (``current`` and the like).

    Any top-level symlink resolving into ``generations/`` is an installed/current
    ref. An unreadable pointer, or a non-symlink ``current``, makes the whole
    root uncertain: the caller then excludes every record.
    """
    names: set[str] = set()
    uncertain: list[str] = []
    store = root / "generations"
    current = root / "current"
    if (current.exists() or current.is_symlink()) and not current.is_symlink():
        uncertain.append("current_is_not_a_pointer")
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return names, ["root_unreadable"]
    for entry in entries:
        if not entry.is_symlink():
            continue
        try:
            target = Path(os.readlink(entry))
        except OSError:
            uncertain.append(f"pointer_unreadable:{entry.name}")
            continue
        resolved = Path(os.path.normpath(target if target.is_absolute() else root / target))
        if resolved.parent == store:
            names.add(resolved.name)
        elif store in resolved.parents:
            names.add(resolved.relative_to(store).parts[0])
    return names, uncertain


def _generation_name(root: Path, value: object) -> tuple[str | None, bool]:
    """(name, certain). Empty means no generation; foreign paths are uncertain."""
    if value in (None, ""):
        return None, True
    if not isinstance(value, str):
        return None, False
    path = Path(os.path.normpath(value))
    store = root / "generations"
    if path.parent != store:
        return None, False
    return path.name, True


def _exclusion(record: dict[str, Any], root: Path, referenced: set[str]) -> str | None:
    if record.get("phase") in _PREPARED_PHASES:
        return "prepared_ref"
    names: list[str] = []
    for field in ("candidate", "current"):
        name, certain = _generation_name(root, record.get(field))
        if not certain:
            return f"uncertain_{field}"
        if name is not None:
            names.append(name)
    for name in names:
        if name in referenced:
            return "current_or_installed_ref"
        generation = root / "generations" / name
        for marker in (_tx.ATTEMPT_NAME, _V2_SIDECAR):
            if (generation / marker).exists() or (generation / marker).is_symlink():
                return "generation_attempt_or_v2_locator"
    return None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def dry_run_manifest(root: Path, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> dict[str, Any]:
    """Classify every legacy namespace entry and describe how to restore candidates.

    Pure: reads only. The result is the whole decision input an operator would
    review; ``mutation_performed`` is always ``False``.
    """
    root = Path(root).expanduser().absolute()
    directory = root / _tx.OPERATION_DIR
    referenced, uncertain_root = _referenced_generations(root)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    phases: dict[str, int] = {}
    complete = True
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        uncertain_root.append("operation_directory_unsafe")
    elif directory.exists():
        for index, path in enumerate(sorted(directory.iterdir())):
            if index >= max_entries:
                complete = False
                break
            relpath = f"{_tx.OPERATION_DIR}/{path.name}"
            identifier = path.name[: -len(".json")] if path.name.endswith(".json") else ""
            if not _is_hex_id(identifier) or path.name != f"{identifier}.json":
                excluded.append({"relpath": relpath, "reason": "unknown_namespace_entry"})
                continue
            read = _read_exact(path)
            record = _parse(read[0]) if read is not None else None
            if read is None or record is None or record.get("operation_id") != identifier:
                excluded.append({"relpath": relpath, "reason": "unreadable_or_unsafe_record"})
                continue
            data, sample = read
            phase = str(record.get("phase", ""))
            phases[phase] = phases.get(phase, 0) + 1
            reason = (
                "root_uncertain:" + ",".join(uncertain_root)
                if uncertain_root
                else _exclusion(record, root, referenced)
            )
            if reason is not None:
                excluded.append({"relpath": relpath, "reason": reason})
                continue
            digest = hashlib.sha256(data).hexdigest()
            mode = stat.S_IMODE(sample.st_mode)
            name, _ = _generation_name(root, record.get("candidate"))
            sidecar = (
                name is not None
                and (root / "generations" / name / ".update-operation.json").exists()
            )
            candidates.append(
                {
                    "path": str(path),
                    "relpath": relpath,
                    "operation_id": identifier,
                    "class": CANDIDATE_CLASS,
                    "label": CANDIDATE_LABEL,
                    "finality": "unproven",
                    "protected": True,
                    "phase": phase,
                    # A retained legacy sidecar is exactly the re-entry path that
                    # makes finality unprovable; surface it for the operator.
                    "legacy_sidecar_reentry_possible": bool(sidecar),
                    "sha256": digest,
                    "bytes": len(data),
                    "mtime_ns": sample.st_mtime_ns,
                    "mode": mode,
                    "restore": {
                        "op": "restore_file",
                        "relpath": relpath,
                        "content_sha256": digest,
                        "bytes": len(data),
                        "mode": mode,
                        "mtime_ns": sample.st_mtime_ns,
                        "precondition": "absent_or_identical",
                    },
                }
            )
    v2_directory = root / _V2_DIR
    if v2_directory.exists() or v2_directory.is_symlink():
        excluded.append({"relpath": _V2_DIR, "reason": "new_protocol_namespace_out_of_scope"})
    body = {
        "schema": MANIFEST_SCHEMA,
        "root": str(root),
        "candidates": candidates,
    }
    return {
        **body,
        "generated_at": time.time(),
        "dry_run": True,
        "mutation_performed": False,
        "scan_complete": complete,
        "root_uncertain": uncertain_root,
        "excluded": excluded,
        "phase_histogram_informational_only": phases,
        "manifest_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


#: The ONLY directory a manifest may read from or restore into: the LEGACY
#: namespace. Checked by name AND after resolution, so neither ``..`` nor a
#: symlinked namespace directory can redirect capture/restore elsewhere. The v2
#: namespace is deliberately excluded for BOTH directions (review round 1, M3):
#: restoring there would plant a v2 record without ``start_operation``, the
#: fence or the shared cap, and capture never needs it because
#: :func:`dry_run_manifest` only ever emits legacy candidates.
_NAMESPACES = (_tx.OPERATION_DIR,)


def _relpath_ok(relpath: object) -> bool:
    if not isinstance(relpath, str):
        return False
    parts = relpath.split("/")
    return (
        len(parts) == 2
        and parts[0] in _NAMESPACES
        and parts[1].endswith(".json")
        and _is_hex_id(parts[1][: -len(".json")])
    )


def _verify_manifest(manifest: dict[str, Any], approved_sha256: str) -> list[dict[str, Any]]:
    """Accept only the EXPLICIT manifest a reviewer approved, byte for byte.

    ``approved_sha256`` is the digest the reviewer signed off, passed by the
    caller -- never taken from the manifest itself, so an edited manifest that
    recomputed its own digest is still refused. Every entry must name a record
    inside one of the operation namespaces and carry a complete restore
    transaction.
    """
    body = {key: manifest.get(key) for key in ("schema", "root", "candidates")}
    if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(body["candidates"], list):
        raise ManifestError("manifest schema is unknown; nothing was changed")
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    if digest != manifest.get("manifest_sha256"):
        raise ManifestError("manifest digest does not match its entries; nothing was changed")
    if not isinstance(approved_sha256, str) or digest != approved_sha256:
        raise ManifestError("manifest is not the reviewed manifest; nothing was changed")
    entries: list[dict[str, Any]] = []
    for entry in body["candidates"]:
        restore = entry.get("restore") if isinstance(entry, dict) else None
        if not isinstance(restore, dict):
            raise ManifestError("manifest restore transaction is malformed; nothing was changed")
        relpath = restore.get("relpath")
        if not _relpath_ok(relpath) or entry.get("relpath", relpath) != relpath:
            raise ManifestError("manifest path is outside the operation namespaces")
        content = restore.get("content_sha256")
        if (
            not isinstance(content, str)
            or len(content) != 64
            or any(c not in "0123456789abcdef" for c in content)
            or content != entry.get("sha256", content)
            or not isinstance(restore.get("bytes"), int)
            or not isinstance(restore.get("mtime_ns"), int)
            or not isinstance(restore.get("mode"), int)
            or not 0 <= restore["mode"] <= 0o777
        ):
            raise ManifestError("manifest restore transaction is malformed; nothing was changed")
        entries.append(entry)
    return entries


def _contained(root: Path, relpath: str) -> tuple[Path, Path]:
    """(namespace directory, target) proved inside ``root`` after resolution.

    The namespace directory may be absent (restore creates it later); if it
    exists it must be a real, owned directory whose realpath is exactly
    ``<canonical root>/<namespace>``. An existing target must not be a symlink.
    """
    canonical = _tx.canonical_root(root)
    namespace, name = relpath.split("/")
    directory = canonical / namespace
    target = directory / name
    if directory.exists() or directory.is_symlink():
        info = directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or (_UID_IS_MEANINGFUL and info.st_uid != os.getuid())
            or Path(os.path.realpath(directory)) != directory
        ):
            raise ManifestError(f"operation directory escapes the install root: {namespace}")
        if target.is_symlink() or Path(os.path.realpath(target)).parent != directory:
            raise ManifestError(f"manifest target escapes its operation directory: {relpath}")
    return directory, target


def _write_exclusive(
    directory: Path,
    name: str,
    data: bytes,
    mode: int,
    mtime_ns: int,
    *,
    expected: tuple[int, int] | None = None,
) -> bool:
    """Publish ``name`` only if absent: temp + hard link, never replace.

    Every operation goes through ONE directory descriptor opened with
    ``O_NOFOLLOW`` (``openat``-style, as ``_persist_operation`` does), so a
    directory swapped after the path checks cannot redirect the write: a symlink
    refuses at open, and ``expected`` (device, inode) from the preflight refuses
    a different real directory. Mode and mtime are set on the file descriptor.

    Crash window (review round 1, N2): a crash between ``link`` and the staging
    unlink leaves the published record with ``st_nlink == 2`` plus a
    ``.restore.tmp`` name. That FAILS CLOSED: ``read_record`` refuses a
    multiply-linked record, and the leftover counts toward the shared cap until
    an operator removes it, so nothing is silently trusted or freed.
    """
    try:
        handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ManifestError(f"cannot open {directory} without following links") from exc
    staged = f".{uuid.uuid4().hex}.restore.tmp"
    try:
        info = os.fstat(handle)
        if (_UID_IS_MEANINGFUL and info.st_uid != os.getuid()) or (
            expected is not None and (info.st_dev, info.st_ino) != expected
        ):
            raise ManifestError(f"directory changed after its checks: {directory}")
        descriptor = os.open(
            staged,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=handle,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.utime(stream.fileno(), ns=(mtime_ns, mtime_ns))
                os.fsync(stream.fileno())
            try:
                os.link(staged, name, src_dir_fd=handle, dst_dir_fd=handle)
            except FileExistsError:
                return False
            return True
        finally:
            # Our own staging name only; the published name is the second link.
            try:
                os.unlink(staged, dir_fd=handle)
            except FileNotFoundError:
                pass
            os.fsync(handle)
    finally:
        os.close(handle)


def _identity(directory: Path) -> tuple[int, int] | None:
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def capture_content(
    manifest: dict[str, Any], store: Path, *, approved_sha256: str
) -> dict[str, Any]:
    """Copy each candidate's exact bytes into ``store/<sha256>``; source untouched.

    All-or-nothing preflight: every entry's on-disk bytes must still hash to the
    reviewed manifest, and every source path must resolve inside the root's
    operation namespaces, before the store receives a single byte. The store
    must lie outside the root (it would otherwise add files to the install root).
    """
    entries = _verify_manifest(manifest, approved_sha256)
    root = _tx.canonical_root(Path(manifest["root"]))
    store = Path(os.path.realpath(Path(store).expanduser().absolute()))
    if store == root or root in store.parents:
        raise ManifestError("content store must be outside the recovery root")
    staged: list[tuple[str, str, bytes, int]] = []
    for entry in entries:
        restore = entry["restore"]
        _directory, source = _contained(root, restore["relpath"])
        recorded = entry.get("path")
        if recorded is not None and Path(os.path.realpath(str(recorded))) != source:
            raise ManifestError(f"manifest path disagrees with its namespace: {recorded}")
        read = _read_exact(source)
        if read is None or hashlib.sha256(read[0]).hexdigest() != restore["content_sha256"]:
            raise ManifestError(
                f"on-disk bytes no longer match the reviewed manifest: {restore['relpath']}"
            )
        staged.append((restore["relpath"], restore["content_sha256"], read[0], read[1].st_mtime_ns))
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    report: dict[str, list[str]] = {"captured": [], "already": []}
    for relpath, digest, data, mtime_ns in staged:
        existing = _read_exact(store / digest)
        if existing is not None and hashlib.sha256(existing[0]).hexdigest() == digest:
            report["already"].append(relpath)
        elif _write_exclusive(store, digest, data, 0o600, mtime_ns):
            report["captured"].append(relpath)
        else:
            raise ManifestError(f"content store holds different bytes for {digest}")
    return report


def _restore_plan(
    entries: list[dict[str, Any]], store: Path, root: Path
) -> tuple[list[tuple[dict[str, Any], Path, Path, bytes]], list[str]]:
    """Check EVERY entry without writing; return (to write, already present).

    Run twice: once lock-free (so a refusal creates nothing, not even
    ``install.lock``) and again under the root transaction, because the first
    pass is only advice until the lock is held.
    """
    plan: list[tuple[dict[str, Any], Path, Path, bytes]] = []
    already: list[str] = []
    for entry in entries:
        restore = entry["restore"]
        relpath = restore["relpath"]
        directory, target = _contained(root, relpath)
        content = _read_exact(store / restore["content_sha256"])
        if (
            content is None
            or hashlib.sha256(content[0]).hexdigest() != restore["content_sha256"]
            or len(content[0]) != restore["bytes"]
        ):
            raise ManifestError(f"stored bytes do not match the reviewed manifest: {relpath}")
        if target.exists():
            present = _read_exact(target)
            if present is None or present[0] != content[0]:
                raise ManifestError(f"a different record already exists: {relpath}")
            already.append(relpath)
            continue
        plan.append((restore, directory, target, content[0]))
    return plan, already


def restore_manifest(
    manifest: dict[str, Any], store: Path, root: Path, *, approved_sha256: str
) -> dict[str, Any]:
    """Recreate missing records exactly; never overwrite, never delete.

    ``root`` must be the reviewed manifest's own root after resolution (review
    round 1, M2): a manifest approved for one install root may not recreate
    records in another. The root must already exist; it is never created.
    Every entry is preflighted lock-free first (Q2), so a refusal, or a run with
    nothing to write, leaves the root byte-for-byte untouched. Only then is the
    existing root transaction taken, the preflight repeated under it, and the
    missing records written.
    """
    entries = _verify_manifest(manifest, approved_sha256)
    store = Path(store).expanduser().absolute()
    root = _tx.canonical_root(root)
    if root != _tx.canonical_root(Path(manifest["root"])):
        raise ManifestError("restore root is not the reviewed manifest's root; nothing was changed")
    try:
        root_info = root.lstat()
    except FileNotFoundError as exc:
        raise ManifestError("restore root does not exist; nothing was created") from exc
    if not stat.S_ISDIR(root_info.st_mode) or (
        _UID_IS_MEANINGFUL and root_info.st_uid != os.getuid()
    ):
        raise ManifestError("restore root is unsafe; nothing was changed")
    plan, already = _restore_plan(entries, store, root)
    if not plan:
        return {"restored": [], "already_present": already}
    report: dict[str, list[str]] = {"restored": [], "already_present": []}
    with _tx.transaction(root, operation="restore recovery manifest") as held:
        held.validate(root)
        plan, report["already_present"] = _restore_plan(entries, store, root)
        for restore, directory, target, data in plan:
            if not directory.exists() and not directory.is_symlink():
                directory.mkdir(mode=0o700)
            _contained(root, restore["relpath"])
            if not _write_exclusive(
                directory,
                target.name,
                data,
                restore["mode"],
                restore["mtime_ns"],
                expected=_identity(directory),
            ):
                raise ManifestError(f"record appeared during restore: {restore['relpath']}")
            check = _read_exact(target)
            if (
                check is None
                or hashlib.sha256(check[0]).hexdigest() != restore["content_sha256"]
                or stat.S_IMODE(check[1].st_mode) != restore["mode"]
                or check[1].st_mtime_ns != restore["mtime_ns"]
            ):
                raise ManifestError(f"restored record failed verification: {restore['relpath']}")
            report["restored"].append(restore["relpath"])
        held.validate(root)
    return report
