"""Durable per-import manifest and the target transcript prefix checksum.

The manifest is the ONLY durable claim that a target session belongs to a
foreign source. It is written atomically, carries the source fingerprint that
authorises a no-op re-run, and carries ``capture_memory=false`` so automatic
memory capture never re-mines an imported conversation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_operator.harness_import import store

MANIFEST_NAME = "harness-import.json"
INDEX_NAME = "index.json"


def manifest_path(session_dir: Path) -> Path:
    return Path(session_dir) / MANIFEST_NAME


def read_manifest(session_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(session_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_manifest(session_dir: Path, payload: dict[str, Any]) -> None:
    store.atomic_write_text(
        manifest_path(session_dir),
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def prefix_sha256(session_dir: Path) -> str | None:
    """Hash the target transcript exactly as it stands, or ``None`` if absent."""
    from local_operator.session import compression

    # The DECOMPRESSED bytes, so the hash is an identity for the conversation
    # rather than for whichever codec happens to hold it: an import that
    # compares a pre-compression hash must still match after a compression pass.
    path = compression.transcript_for_read(Path(session_dir))
    if path is None:
        return None
    return store.sha256_file(path)


def read_index(config: Path) -> dict[str, Any]:
    try:
        value = json.loads((store.imports_root(config) / INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"v": 1, "sessions": {}}
    return value if isinstance(value, dict) else {"v": 1, "sessions": {}}


def write_index(config: Path, payload: dict[str, Any]) -> None:
    store.atomic_write_text(
        store.imports_root(config) / INDEX_NAME,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
