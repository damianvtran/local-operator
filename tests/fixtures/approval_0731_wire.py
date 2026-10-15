"""Minimal frozen OLD-wire contract, NOT an installed-old executor or vendored engine.

Source: Local Operator0.73.1 generation20261015T112141Z, approvals.py SHA256
 dd18f6956deea1f9abc9d6d1f0faee253bb2eca6aa3651839599c0f3a22558c1.
Actual isolated old-process probes supplement this portable CI contract. It models
only old load/list/pre-sign ID checks, immutable digest and decision framing. It
cannot certify old lifecycle/SSH behavior; no CI dependency on a local install.
Unknown-kind legacy-domain positive controls are mandatory: kind alone was not a
security boundary. Old pre-sign admission does NOT check a request digest; the
new envelope's absent top-level ID protects copied pending cards before presence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from local_operator.network.wire import canonical_json
from local_operator.operator.verify import verify_signature

OLD_DOMAIN = b"lop-approval-v1\0"


def immutable(record: dict[str, Any]) -> dict[str, Any]:
    # The frozen0.73.1 helper excludes approval_id/schema/version/request_id;
    # its two window fields ARE included. New commitment must bind the rest.
    result = {
        key: record.get(key)
        for key in ("kind", "requested_by", "what", "credential_ref", "created_at", "expires_at")
    }
    key = "machine" if record.get("kind") == "local_authority" else "device"
    result[key] = record.get(key)
    return result


def digest(record: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(immutable(record))).hexdigest()


def payload(record: dict[str, Any]) -> bytes:
    fields = (
        str(record["kind"]),
        str(record["request_id"]),
        str(record["request_digest"]),
        "approve",
        f"{float(record['decided_at']):.6f}",
    )
    return OLD_DOMAIN + b"".join(
        len(v.encode("utf-8", "surrogatepass")).to_bytes(4, "big")
        + v.encode("utf-8", "surrogatepass")
        for v in fields
    )


def load(directory: Path, aid: str) -> dict[str, Any]:
    try:
        data = json.loads((directory / (aid + ".json")).read_text())
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("approval_id") != aid:
        raise ValueError("unknown_approval")
    return data


def rows(directory: Path) -> list[dict[str, Any]]:
    result = []
    for path in directory.glob("ap_*.json"):
        try:
            record = load(directory, path.stem)
        except ValueError:
            continue
        result.append(record)
    return result


def presign(directory: Path, aid: str, signer: Any) -> None:
    # Intentionally no digest check: do not falsely promise domains prevent
    # old signing when an owner deliberately unwraps/rebuilds a pending record.
    signer(load(directory, aid))


def verifies(record: dict[str, Any], spki: bytes) -> bool:
    if digest(record) != record.get("request_digest"):
        return False
    try:
        sig = bytes.fromhex(record["signature"]["sig"])
    except (ValueError, KeyError, TypeError):
        return False
    return verify_signature(spki=spki, message=payload(record), signature=sig)
