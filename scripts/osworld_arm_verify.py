#!/usr/bin/env python3
"""Verify an installed OSWorld arm against the source commit it claims.

Run with the ARM interpreter (``"$VENV/bin/python3.12" -I -B <this file>``):
every fact here is a property of that interpreter, so reading it from the
caller's would answer about the wrong environment.

Three separate questions, deliberately not conflated:

1. SOURCE IDENTITY. The harness's pinned core modules, hashed in the running
   interpreter (``source_pin.compute_core_source_hashes``), must equal the bytes
   of the same files in the tree extracted from the claimed commit. This is the
   stable identity a wheel archive hash cannot give: zip member timestamps move
   between two builds of one commit.
2. SELECTOR IDENTITY. The adapter's ``package_digest`` -- exactly the value a
   selector records and the worker checks -- plus the adapter version and file
   count. This is a THIS-INSTALL identity: it hashes the wheel RECORD, and the
   RECORD contains entries the installer writes (``direct_url`` carrying the
   wheel path, ``uv_installer``/``uv_cache`` carrying timestamps), so two
   installations of the same wheel from the same bytes give different digests.
   ``package_digest`` is therefore deliberately NOT offered as a drift
   detector.
2b. CROSS-BUILD IDENTITY. ``source_record_digest`` re-hashes the same RECORD
   rows with the installer-written entries removed, so it IS comparable between
   two installs of one commit. Reported alongside, never instead of, the
   selector digest; ``discovery.distribution_digest`` keeps its existing
   semantics because the adapter guard and the recorded arms depend on them.
2c. IMPORT PROVENANCE. Imports must come from inside the venv, not from a
   stray PYTHONPATH or a system install; a build whose modules load from
   somewhere else is not the arm it claims to be.
3. THE INSTALLED SET, enumerated so the caller can compare it against its own
   pinned plan. Deciding whether a difference is acceptable is the caller's
   job, not this script's -- but the set is emitted whole, never summarised
   into a count.

NO GUEST, provider, model or network is touched. The report is written BEFORE
the exit code is decided, refusals included: a refusal without its evidence is
not actionable, and an operator cannot tell an absent report from a crash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
import sysconfig
from pathlib import Path
from typing import Any

# RECORD rows the INSTALLER writes. Their content is a path or a timestamp, not
# a property of the wheel, so they are excluded from the cross-build identity
# while staying part of the selector's ``distribution_digest``.
_INSTALLER_ROWS = frozenset({"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"})
_INSTALLER_SUFFIXES = tuple(f"/{name}" for name in _INSTALLER_ROWS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--source", required=True, type=Path, help="tree extracted from the commit")
    parser.add_argument("--commit", required=True, help="the commit the tree came from")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    from local_operator.evaluation.adapters.discovery import distribution_digest
    from local_operator.evaluation.source_pin import (
        CORE_SOURCE_MODULES,
        compute_core_source_hashes,
    )

    refusals: list[str] = []
    prefix = Path(sys.prefix).resolve()
    hashes = compute_core_source_hashes()
    origins: dict[str, str] = {}
    module_rows: dict[str, dict[str, Any]] = {}
    for key, module_name in CORE_SOURCE_MODULES:
        module = importlib.import_module(module_name)
        origin = Path(module.__file__ or "").resolve()
        origins[module_name] = str(origin)
        relative = Path(*module_name.split(".")).with_suffix(".py")
        committed = args.source / relative
        matches: bool | None = None
        if not origin.is_relative_to(prefix):
            refusals.append(f"{module_name} imports from {origin}, outside the arm venv {prefix}")
        elif not committed.is_file():
            refusals.append(f"{relative} is missing from the source tree of {args.commit}")
        else:
            matches = hashes[key] == _sha256(committed)
            if not matches:
                refusals.append(f"core module {key} hash differs from {args.commit}")
        module_rows[key] = {
            "module": module_name,
            "origin": str(origin),
            "hash": hashes[key],
            "matches_commit": matches,
        }

    adapter: dict[str, Any] = {}
    installed: dict[str, str] = {}
    try:
        distribution = importlib.metadata.distribution("lop-osworld-v2-adapter")
        record_rows = (distribution.read_text("RECORD") or "").splitlines()
        dist_info = next(
            Path(sysconfig.get_paths()["purelib"]).glob("lop_osworld_v2_adapter-*.dist-info"),
            None,
        )
        adapter = {
            "distribution": distribution.metadata["Name"],
            "version": distribution.version,
            "dist_info": str(dist_info) if dist_info else None,
            # What the selector pins and the worker re-verifies. THIS INSTALL only.
            "package_digest": distribution_digest(distribution),
            "record_rows": len([row for row in record_rows if row.strip()]),
            # Comparable across builds: installer-written rows removed.
            "source_record_digest": _source_record_digest(record_rows),
            "installer_rows_excluded": sorted(_INSTALLER_ROWS),
        }
    except importlib.metadata.PackageNotFoundError:
        refusals.append(f"the adapter is not installed in {sysconfig.get_paths()['purelib']}")
    installed = _installed_versions()

    report: dict[str, Any] = {
        "commit": args.commit,
        "python": sys.version.split()[0],
        "prefix": str(prefix),
        "core_source": module_rows,
        "core_module_digest": hashlib.sha256(
            "".join(
                f"{key}:{value['hash']}\n" for key, value in sorted(module_rows.items())
            ).encode()
        ).hexdigest(),
        "harness": {
            "distribution": "local-operator",
            "version": _version_or_none("local-operator"),
            "origin": origins.get("local_operator.sdk"),
        },
        "adapter": adapter,
        "installed_distributions": installed,
        "distribution_count": len(installed),
        "refusals": refusals,
        "verified": not refusals,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Written BEFORE the verdict: a refusal that leaves no report is unusable.
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if refusals:
        for reason in refusals:
            print(f"REFUSED: {reason}", file=sys.stderr)
        return 1
    print(
        "VERIFIED: "
        f"{len(module_rows)} core modules match {args.commit[:12]}; "
        f"adapter {adapter['version']} package_digest "
        f"{adapter['package_digest'][:12]} (this install), source_record_digest "
        f"{adapter['source_record_digest'][:12]} (cross-build); "
        f"{len(installed)} distributions installed"
    )
    return 0


def _source_record_digest(record_rows: list[str]) -> str:
    """sha256 over RECORD rows minus the installer-written entries.

    Same algorithm shape as ``discovery.distribution_digest`` (canonical rows,
    sorted) so the two digests differ only by what is excluded, and a reader can
    see why one moves between installs and the other does not.
    """

    kept: list[str] = []
    for row in record_rows:
        if not row.strip():
            continue
        if row.endswith("/RECORD") or row == "RECORD":
            continue
        if any(part in row for part in _INSTALLER_SUFFIXES):
            continue
        kept.append(row)
    canonical = "\n".join(sorted(kept))
    return hashlib.sha256(f"adapter-source-record-v1\n{canonical}".encode()).hexdigest()


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions[name] = distribution.version
    return dict(sorted(versions.items()))


def _version_or_none(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
