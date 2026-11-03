"""Build or verify the experimental-only content inventory, never the core kit.

Why: mutable source names cannot identify a reproducible method revision. Hash
all distributed experimental files, plus the shared public loadout byte inputs.
Constraints: an inventory is NOT a frozen preregistration, runtime attestation,
paid-run authorization or evidence of quality. No secrets/environment captured.
The inventory excludes itself, caches and generated runtime logs; run only on a
clean publication folder, not a working directory containing runtime artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "EXPERIMENTAL-MANIFEST.json"


def inventory() -> dict:
    files = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == NAME:
            continue
        if path.is_symlink():
            raise ValueError("Symlinks cannot freeze portable inputs")
        rel = path.relative_to(HERE).as_posix()
        files[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    shared = {}
    for name in ("public_mcp_schemas.json", "public_user_instructions.md", "public_skills.sha256"):
        path = HERE.parent / "public_loadout" / name
        shared["../public_loadout/" + name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    return {"status": "experimental-draft-not-run-authorization", "files": files, "shared_inputs": shared}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="Explicitly replace experimental inventory only")
    args = parser.parse_args()
    actual = inventory()
    path = HERE / NAME
    if args.write:
        path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        print("Wrote experimental draft inventory; this is not a runtime freeze")
    else:
        expected = json.loads(path.read_text())
        if expected != actual:
            raise SystemExit("Experimental inventory differs; inspect changes before explicit regeneration")
        print("Experimental and shared input hashes match")


if __name__ == "__main__":
    main()
