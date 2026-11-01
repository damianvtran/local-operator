"""Stage only one CI pytest invocation's existing timeout logs, never test scratch.

The workflow supplies the same validated identity to the controller and this CLI.
No PID discovery, historical-directory glob, new handler, or process inspection
is involved. These are diagnostic timeout files, not a promise of fatal stacks.
Their existing text can contain fixture paths: retain for one day, never capture
locals, environment or argv. The console reporter intentionally stays reduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

ENV_INVOCATION = "LOCAL_OPERATOR_SHARD_STALL_INVOCATION"
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


def _copy_worker(run_fd, name, output, budget):
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
    except OSError:
        # No exception text: an attacker-controlled path must not become a
        # secondary channel for exporting scratch names or filesystem content.
        entry["reason"] = "unsafe-or-unreadable"
    finally:
        if fd is not None:
            os.close(fd)
    return entry


def stage(
    root: Path,
    invocation: str,
    output: Path,
    *,
    file_cap=FILE_CAP,
    total_cap=TOTAL_CAP,
    count_cap=COUNT_CAP,
) -> dict[str, Any]:
    """Create a fresh isolated artifact with bounded files and a truthful manifest.

    Root is the workflow's trusted dump base, not a test-provided source path.
    Only its exact invocation child and numeric worker filenames are eligible.
    Limits may be lowered for synthetic controls, never raised above policy.
    """
    source = provenance(invocation)
    if not (0 < file_cap <= FILE_CAP and 0 < total_cap <= TOTAL_CAP and 0 < count_cap <= COUNT_CAP):
        raise ValueError("limits must be positive and within retention policy")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "invocation": invocation,
        "provenance": source,
        "status": "missing",
        "limits": {
            "file_bytes": file_cap,
            "total_bytes": total_cap,
            "files": count_cap,
            "directory_entries": SCAN_CAP,
        },
        "files": [],
        "output_bytes": 0,
        "notice": (
            "Timeout-file prefixes only; marker presence is not proof of a complete stack "
            "or fatal crash. Text may include fixture paths. No locals/env/argv capture added."
        ),
    }
    root_fd = run_fd = None
    try:
        root_fd = _directory(root)
        run_fd = _directory(invocation, dir_fd=root_fd)
        manifest["status"] = "collected"
        eligible = []
        with os.scandir(run_fd) as entries:
            for index, entry in enumerate(entries):
                if index == SCAN_CAP:
                    manifest["directory_tail"] = {
                        "status": "omitted-cap",
                        "count": None,
                        "original_bytes": None,
                        "output_bytes": 0,
                    }
                    break
                if _WORKER.fullmatch(entry.name):
                    eligible.append(entry.name)
        retained = 0
        for name in sorted(eligible):
            budget = (
                min(file_cap, total_cap - manifest["output_bytes"]) if retained < count_cap else 0
            )
            entry = _copy_worker(run_fd, name, output, budget)
            manifest["files"].append(entry)
            manifest["output_bytes"] += entry["output_bytes"]
            if entry["status"] in ("retained", "truncated"):
                retained += 1
        if not eligible:
            if "directory_tail" in manifest:
                manifest.update(status="omitted-cap", reason="directory-scan-limit")
            else:
                manifest.update(status="missing", reason="no-current-invocation-worker-files")
    except FileNotFoundError:
        manifest["reason"] = "current-invocation-directory-missing"
    except OSError:
        manifest.update(status="rejected", reason="unsafe-or-unreadable-source-directory")
    finally:
        for fd in (run_fd, root_fd):
            if fd is not None:
                os.close(fd)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = stage(args.root, args.invocation, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"timeout-file staging refused: {type(exc).__name__}\n")
    print(
        f"timeout-file artifact: {manifest['status']}; "
        f"{manifest['output_bytes']} bytes; see manifest.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
