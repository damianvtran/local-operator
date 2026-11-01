"""Stage only one CI pytest invocation's existing timeout logs, never test scratch.

The workflow supplies the same validated identity to the controller and this CLI.
No PID discovery, historical-directory glob, new handler, or process inspection
is involved. These are diagnostic timeout files, not a promise of fatal stacks.
Their existing text can contain fixture paths: retain for one day, never capture
locals, environment or argv. The console reporter intentionally stays reduced.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

ENV_INVOCATION = "LOCAL_OPERATOR_SHARD_STALL_INVOCATION"
ENV_REGISTRY = "LOCAL_OPERATOR_SHARD_STALL_REGISTRY"
REGISTRATION_FAILURE = ".retention-registration-failed"
_ID = re.compile(r"ci-([0-9]+)-([0-9]+)-([0-9]+\.[0-9]+)-([0-9]+)", re.ASCII)
_WORKER = re.compile(r"worker-[0-9]+\.log", re.ASCII)
FILE_CAP = 1024 * 1024
TOTAL_CAP = 8 * FILE_CAP
COUNT_CAP = 32
# Bound even the manifest/directory scan; an omitted tail has unknown size/count,
# rather than pretending we inspected files beyond this metadata budget.
SCAN_CAP = 128


def provenance(invocation: str) -> dict[str, str]:
    """Reject paths and arbitrary identities before touching the filesystem."""
    match = _ID.fullmatch(invocation)
    if match is None or len(invocation) > 100:
        raise ValueError("invalid CI invocation identity")
    return dict(zip(("run", "attempt", "python", "shard"), match.groups()))


def _directory(path, *, dir_fd=None):
    # Descriptor-relative descent pins the directory even if its name is swapped;
    # O_NOFOLLOW rejects a substituted root/run symlink instead of following it.
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)


def note_registration_failure(directory: Path) -> None:
    """Deny retention on lifecycle failure without disabling the independent timer.

    This exact, empty marker is denial-only: no contents are captured or trusted
    as worker provenance. It also overrides any receipt left by a failed final
    registration write. Never replace unknown files. If even the original run
    directory is unwritable, persistence is best-effort; absence of a receipt
    still cannot authorize a log export. No console/timer behavior changes.
    """
    directory_fd = marker_fd = None
    try:
        directory_fd = _directory(directory)
        marker_fd = os.open(
            REGISTRATION_FAILURE,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
    except OSError:
        pass
    finally:
        for fd in (marker_fd, directory_fd):
            if fd is not None:
                os.close(fd)


def registered_handle(path: Path, registry: Path):
    """Open/register a worker writer at configure time, before test code runs.

    The workflow allocates this registry outside the fixture/dump directory.
    Workers consume its environment handoff, pin source inode/device from their
    actual writer handle, and keep that handle for the existing timer. This
    excludes ordinary fixture-created numeric files and path replacements; it
    is NOT authentication against hostile Python in the same uid/process that
    can introspect handles, discover/modify the registry, or rewrite an inode.
    """
    provenance(path.parent.name)
    if not _WORKER.fullmatch(path.name):
        raise ValueError("invalid worker name")
    run_fd = registry_fd = fd = None
    try:
        run_fd = _directory(path.parent)
        registry_fd = _directory(registry)
        fd = os.open(
            path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=run_fd
        )
        info, directory = os.fstat(fd), os.fstat(run_fd)
        receipt = {
            "schema": 1,
            "invocation": path.parent.name,
            "name": path.name,
            "device": info.st_dev,
            "inode": info.st_ino,
            "directory_device": directory.st_dev,
            "directory_inode": directory.st_ino,
        }
        receipt_fd = os.open(
            path.name + ".json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=registry_fd,
        )
        with os.fdopen(receipt_fd, "w", encoding="utf-8") as target:
            json.dump(receipt, target)
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = None
        return handle
    finally:
        for opened in (fd, registry_fd, run_fd):
            if opened is not None:
                os.close(opened)


def _registrations(registry_fd, invocation, run_fd, manifest):
    """Bounded metadata-only allowlist, never a scan of fixture worker files."""
    owners = []
    directory = os.fstat(run_fd)
    with os.scandir(registry_fd) as entries:
        for index, entry in enumerate(entries):
            if index == SCAN_CAP:
                manifest["directory_tail"] = {
                    "status": "omitted-cap",
                    "count": None,
                    "original_bytes": None,
                    "output_bytes": 0,
                }
                break
            name = entry.name.removesuffix(".json")
            if not entry.name.endswith(".json") or not _WORKER.fullmatch(name):
                continue
            fd = None
            try:
                fd = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=registry_fd
                )
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4096:
                    raise ValueError("unsafe registration")
                data = os.read(fd, 4097)
                if len(data) > 4096:
                    raise ValueError("oversize registration")
                receipt = json.loads(data)
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("schema") != 1
                    or receipt.get("invocation") != invocation
                    or receipt.get("name") != name
                    or any(
                        type(receipt.get(key)) is not int
                        for key in ("device", "inode", "directory_device", "directory_inode")
                    )
                    or receipt["directory_device"] != directory.st_dev
                    or receipt["directory_inode"] != directory.st_ino
                ):
                    raise ValueError("registration mismatch")
                owners.append((name, receipt))
            except (OSError, ValueError, UnicodeError) as exc:
                # Do not reflect malformed registration contents/paths into logs.
                manifest["registration_errors"] += 1
                if isinstance(exc, OSError) and exc.errno not in (errno.ELOOP, errno.ENOTDIR):
                    manifest["registration_io_errors"] += 1
            finally:
                if fd is not None:
                    os.close(fd)
    return sorted(owners)


def _copy_worker(run_fd, name, output, budget, receipt):
    entry = {"name": name, "status": "rejected", "original_bytes": None, "output_bytes": 0}
    fd = None
    try:
        # O_NONBLOCK also prevents a malicious FIFO from parking this always-step.
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=run_fd)
        info = os.fstat(fd)
        entry["original_bytes"] = info.st_size
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            # A hard link can alias unrelated scratch just as a symlink can.
            entry["reason"] = "not-regular-or-hardlinked"
            return entry
        if (info.st_dev, info.st_ino) != (receipt["device"], receipt["inode"]):
            entry["reason"] = "writer-inode-mismatch"
            return entry
        if budget <= 0:
            entry.update(status="omitted-cap", reason="file-count-or-total-bytes")
            return entry
        # A size snapshot bounds a concurrently growing writer too. We never
        # modify, truncate or unlink the original worker file.
        data = bytearray()
        remaining = min(budget, info.st_size)
        while remaining:
            part = os.read(fd, min(65536, remaining))
            if not part:
                break
            data.extend(part)
            remaining -= len(part)
        raw = bytes(data)
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # A byte cap may split the final UTF-8 codepoint, not corrupt input.
            if exc.reason == "unexpected end of data" and len(raw) < info.st_size:
                raw = raw[: exc.start]
                raw.decode("utf-8")
            else:
                entry["reason"] = "invalid-utf8-in-retained-prefix"
                return entry
        if b"\x00" in raw:
            entry["reason"] = "binary-in-retained-prefix"
            return entry
        with (output / name).open("xb") as target:
            target.write(raw)
        after = os.fstat(fd)
        entry.update(
            status="truncated" if len(raw) < info.st_size else "retained",
            output_bytes=len(raw),
            observed_after_bytes=after.st_size,
            changed_during_copy=(
                info.st_size != after.st_size or info.st_mtime_ns != after.st_mtime_ns
            ),
            sha256_retained_prefix=hashlib.sha256(raw).hexdigest(),
            timeout_marker="present" if b"Timeout (" in raw else "absent-in-retained-prefix",
        )
    except FileNotFoundError:
        entry.update(status="missing", reason="disappeared-before-open")
    except OSError as exc:
        # No exception text: attacker-controlled paths must not become a second
        # channel. Boundary rejection and actual I/O failure remain distinct.
        entry.update(
            status="rejected" if exc.errno in (errno.ELOOP, errno.ENOTDIR) else "error",
            reason="unsafe-or-unreadable",
        )
    finally:
        if fd is not None:
            os.close(fd)
    return entry


class _RegistrationFailed(Exception):
    """A denial-only lifecycle marker forbids all retention for this invocation."""


def stage(
    root: Path,
    invocation: str,
    output: Path,
    *,
    file_cap=FILE_CAP,
    total_cap=TOTAL_CAP,
    count_cap=COUNT_CAP,
    registry: Path | None = None,
) -> dict[str, Any]:
    """Create a fresh isolated artifact with bounded files and a truthful manifest.

    Root is the workflow's trusted dump base, not a test-provided source path.
    Only its exact invocation child and registered writer inodes are eligible.
    Limits may be lowered for synthetic controls, never raised above policy.

    Exit 0: missing (healthy, including cleaned unfired files), collected, or
    limited (explicit policy truncation/omission, never a completeness claim).
    Exit 2: rejected (boundary/protocol), error (I/O), or partial (some retained
    bytes plus a failure). A missing explicitly handed registry is a protocol
    error; no handoff at all (e.g. setup skipped) is healthy absence. A persisted
    registration-failure marker yields error/no logs even if a receipt remains.
    Optional retention failure never disables the independent report/hard timer;
    its fallback writer is not registered for export.
    Failed/partial artifacts are manifested locally but
    NOT published: the workflow emits an upload path only after a zero exit.
    A failed stage does not authorize an upload of any pre-existing directory.
    """
    source = provenance(invocation)
    if not (0 < file_cap <= FILE_CAP and 0 < total_cap <= TOTAL_CAP and 0 < count_cap <= COUNT_CAP):
        raise ValueError("limits must be positive and within retention policy")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 2,
        "invocation": invocation,
        "provenance": source,
        "source_protocol": "pre-test-worker-writer-inode-v1",
        "status": "missing",
        "limits": {
            "file_bytes": file_cap,
            "total_bytes": total_cap,
            "files": count_cap,
            "directory_entries": SCAN_CAP,
        },
        "files": [],
        "output_bytes": 0,
        "registration_errors": 0,
        "registration_io_errors": 0,
        "notice": (
            "Timeout-file prefixes only; marker presence is not proof of a complete stack "
            "or fatal crash. Text may include fixture paths. No locals/env/argv capture added."
        ),
    }
    root_fd = run_fd = registry_fd = None
    try:
        root_fd = _directory(root)
        run_fd = _directory(invocation, dir_fd=root_fd)
        manifest["status"] = "collected"
        try:
            os.stat(REGISTRATION_FAILURE, dir_fd=run_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            # Mere presence is denial-only, including symlink/nonregular markers.
            # Never follow/read it or scan/export fallback worker files.
            raise _RegistrationFailed
        if registry is None:
            eligible = []
        else:
            try:
                registry_fd = _directory(registry)
            except FileNotFoundError as exc:
                # An absent worker log is healthy; losing a registry path that
                # the pre-test workflow explicitly handed us is protocol failure.
                raise OSError(errno.EIO, "registry handoff missing") from exc
            eligible = _registrations(registry_fd, invocation, run_fd, manifest)
        retained = 0
        for name, receipt in eligible:
            budget = (
                min(file_cap, total_cap - manifest["output_bytes"]) if retained < count_cap else 0
            )
            entry = _copy_worker(run_fd, name, output, budget, receipt)
            manifest["files"].append(entry)
            manifest["output_bytes"] += entry["output_bytes"]
            if entry["status"] in ("retained", "truncated"):
                retained += 1
        if not eligible:
            if "directory_tail" in manifest:
                manifest.update(status="omitted-cap", reason="directory-scan-limit")
            else:
                manifest.update(status="missing", reason="no-current-invocation-worker-files")
    except _RegistrationFailed:
        manifest.update(status="error", reason="worker-registration-failed")
    except FileNotFoundError:
        manifest.update(status="missing", reason="current-invocation-directory-or-registry-missing")
    except OSError as exc:
        manifest.update(
            status="rejected" if exc.errno in (errno.ELOOP, errno.ENOTDIR) else "error",
            reason="unsafe-or-unreadable-source-directory",
        )
    finally:
        for fd in (registry_fd, run_fd, root_fd):
            if fd is not None:
                os.close(fd)
    failures = manifest["registration_errors"] or any(
        item["status"] in ("rejected", "error") for item in manifest["files"]
    )
    if failures:
        has_io_error = manifest["registration_io_errors"] or any(
            item["status"] == "error" for item in manifest["files"]
        )
        manifest["status"] = (
            "partial" if manifest["output_bytes"] else ("error" if has_io_error else "rejected")
        )
    elif manifest["status"] == "collected" and (
        "directory_tail" in manifest
        or any(item["status"] in ("truncated", "omitted-cap") for item in manifest["files"])
    ):
        manifest["status"] = "limited"
    manifest["exit_code"] = 2 if manifest["status"] in ("partial", "rejected", "error") else 0
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", default=None)
    args = parser.parse_args(argv)
    try:
        manifest = stage(
            args.root,
            args.invocation,
            args.output,
            registry=Path(args.registry) if args.registry else None,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"timeout-file staging refused: {type(exc).__name__}\n")
    print(
        f"timeout-file artifact: {manifest['status']}; "
        f"{manifest['output_bytes']} bytes; see manifest.json"
    )
    return manifest["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
