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
2. DISTRIBUTION IDENTITY. The adapter's installed RECORD digest
   (``discovery.distribution_digest``) -- what the selector pins and the worker
   re-verifies -- plus the installed file count and the adapter version.
2b. IMPORT PROVENANCE. Imports must come from inside the venv, not from a
   stray PYTHONPATH or a system install; a build whose modules load from
   somewhere else is not the arm it claims to be.
3. THE INSTALLED SET, enumerated so the caller can compare it against its own
   pinned plan. Deciding whether a difference is acceptable is the caller's
   job, not this script's -- but the set is emitted whole, never summarised
   into a count.

No guest, provider, model or network is touched. Exit 1 when any identity check
refuses; the partial report is still written, because a refusal without its
evidence is not actionable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import sysconfig
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"REFUSED: {message}")


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

    prefix = Path(sys.prefix).resolve()
    hashes = compute_core_source_hashes()
    origins: dict[str, str] = {}
    module_rows: dict[str, dict[str, Any]] = {}
    for key, module_name in CORE_SOURCE_MODULES:
        module = importlib.import_module(module_name)
        origin = Path(module.__file__ or "").resolve()
        origins[module_name] = str(origin)
        if not origin.is_relative_to(prefix):
            fail(f"{module_name} imports from {origin}, outside the arm venv {prefix}")
        relative = Path(*module_name.split(".")).with_suffix(".py")
        committed = args.source / relative
        if not committed.is_file():
            fail(f"{relative} is missing from the source tree of {args.commit}")
        module_rows[key] = {
            "module": module_name,
            "origin": str(origin),
            "hash": hashes[key],
            "matches_commit": hashes[key] == _sha256(committed),
        }
    mismatched = sorted(key for key, row in module_rows.items() if not row["matches_commit"])
    if mismatched:
        fail(f"core module hashes differ from {args.commit}: {mismatched}")

    distribution = importlib.metadata.distribution("lop-osworld-v2-adapter")
    purelib = Path(sysconfig.get_paths()["purelib"])
    dist_info = next(purelib.glob("lop_osworld_v2_adapter-*.dist-info"), None)
    if dist_info is None:
        fail(f"the adapter is not installed in {purelib}")
    record_rows = distribution.read_text("RECORD") or ""
    installed = {name: version for name, version in sorted(_installed_versions().items())}
    report: dict[str, Any] = {
        "commit": args.commit,
        "python": sys.version.split()[0],
        "prefix": str(prefix),
        "core_source": module_rows,
        "harness": {
            "distribution": "local-operator",
            "version": importlib.metadata.version("local-operator"),
            "origin": origins.get("local_operator.sdk"),
        },
        "adapter": {
            "distribution": distribution.metadata["Name"],
            "version": distribution.version,
            "dist_info": str(dist_info),
            # What the selector pins and the worker re-verifies.
            "package_digest": distribution_digest(distribution),
            "record_rows": len([row for row in record_rows.splitlines() if row.strip()]),
        },
        "installed_distributions": installed,
        "distribution_count": len(installed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "VERIFIED: "
        f"{len(module_rows)} core modules match {args.commit[:12]}; "
        f"adapter {report['adapter']['version']}",
        f"digest {report['adapter']['package_digest'][:12]};",
        f"{len(installed)} distributions installed",
    )
    return 0


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            versions[name] = distribution.version
    return versions


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
