#!/usr/bin/env python3
"""Scan, checksum, verify and package ONLY the public benchmark kit.

Why: recursively zipping a developer workspace can leak request bodies, auth
state or private instructions. This tool binds an explicit file inventory to
SHA256s, rejects symlinks and sensitive file classes, and scans text before any
ZIP is written. Constraints: heuristic secret detection is not a legal/privacy
proof; the curated metering schema is an additional data-minimization boundary.
In a Git checkout packaging requires every inventory file to be tracked. A
standalone extracted kit uses the signed-by-provenance (not cryptographically
signed) inventory instead. No subprocess invokes a harness or remote service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import zipfile
from pathlib import Path
from typing import NoReturn, TypeAlias

ROOT = Path(__file__).resolve().parent
CONTROL = {"MANIFEST.json", "SHA256SUMS"}
SKIP = {".git", ".venv", "__pycache__", ".mplconfig", "kit-src"}
BLOCKED = {".env", "auth.json", "credentials.json", "tokens.json", "auth.db"}
PATTERNS = {
    "home path": r"(?<![A-Za-z0-9])/(?:Users|home)/(?!user(?:/|\b))[A-Za-z0-9_.-]+",
    "email address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "bearer value": r"(?i)bearer\s+[A-Za-z0-9_.-]{24,}",
    # Shape-based detection deliberately includes all-zero offline dummies:
    # push protection does not grant a dummy/low-entropy exemption. Do not
    # match arbitrary 40-hex strings, since public commit SHAs are provenance.
    "provider secret": (
        "(?:sk-(?:ant-|or-v1-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30"
        ",}|glpat-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|sbp_[A-Za-z0-9]{40}|"
        "ntn_[A-Za-z0-9]{46}|api-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9"
        "a-fA-F]{12}|api-[0-9a-fA-F]{32,40})"
    ),
}
REQUEST_FIELDS = {
    "ts",
    "run_id",
    "path",
    "status",
    "latency_s",
    "stream",
    "requested_model",
    "served_model",
    "n_tools",
    "n_messages",
    "tools_bytes",
    "system_bytes",
    "auto_cache",
    "usage",
    # Measurement provenance (S13, R20 MINOR 3). WHY: ``n_messages`` counts the
    # prefill-continuation turn the meter injected, and a charge can be a stated
    # bill, a token derivation or a bill-free rejection. Without these flags a
    # published snapshot cannot show which rows were shaped or how a row was
    # priced, which is the reproducibility property the kit is for. Constraints:
    # booleans, one status, the provider's own error type/code and closed maps of
    # numbers -- no prompt, message, body or error text is published.
    "prefill_continuation",
    "settled_from_tokens",
    "settled_no_charge",
    "upstream_status",
    "upstream_error_type",
    "upstream_error_code",
    "cost_derivation",
    "incomplete",
}
USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cache_creation",
    "cost",
}
# A top-level key allowlist alone can hide private objects inside allowed
# metadata. Each container and leaf is therefore closed recursively; arbitrary
# dictionaries/lists/strings cannot masquerade as token counters. Null usage
# is the one documented exception for discovery/incomplete responses.
Schema: TypeAlias = type | tuple[type, ...] | dict[str, "Schema"]
USAGE_SCHEMA: dict[str, Schema] = {k: int for k in USAGE_FIELDS - {"cache_creation", "cost"}}
USAGE_SCHEMA.update(
    cost=(int, float),
    cache_creation={"ephemeral_5m_input_tokens": int, "ephemeral_1h_input_tokens": int},
)
REQUEST_SCHEMA: dict[str, Schema] = {
    k: str for k in ["run_id", "path", "requested_model", "served_model"]
}
REQUEST_SCHEMA.update(
    {k: int for k in ["status", "n_tools", "n_messages", "tools_bytes", "system_bytes"]}
)
REQUEST_SCHEMA.update(
    ts=(int, float), latency_s=(int, float), stream=bool, auto_cache=bool, usage=USAGE_SCHEMA
)
# The arithmetic behind a token-priced row (meter_proxy.settled_charge): the
# counts, the frozen card rates used and the resulting USD. Closed at every
# level so the charge can be recomputed from the published record without any
# request or response text; a new rate key must be added here deliberately.
TOKEN_COUNT_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)
RATE_KEYS = ("prompt", "completion", "input_cache_read", "input_cache_write")
COST_DERIVATION_SCHEMA: dict[str, Schema] = {
    "tokens": {k: int for k in TOKEN_COUNT_KEYS},
    "rates_usd_per_m": {k: (int, float) for k in RATE_KEYS},
    "usd": (int, float),
}
REQUEST_SCHEMA.update(
    {
        k: bool
        for k in (
            "prefill_continuation",
            "settled_from_tokens",
            "settled_no_charge",
            "incomplete",
        )
    },
    upstream_status=int,
    # The provider's own error type and code, copied verbatim into the row by
    # meter_proxy.no_charge_rejection; a code may be numeric, so both scalars.
    upstream_error_type=str,
    upstream_error_code=(str, int),
    cost_derivation=COST_DERIVATION_SCHEMA,
)


def validate_projection(value, schema, path="request"):
    """Reject unknown nested fields and non-scalar leaves without echoing data.

    Why: metadata is public only within the exact metering contract. A private
    subtree is unsafe even when its text does not match a secret/PII regex.
    Constraints: no coercion or stringification; booleans are not token counts.
    """
    if value is None and path == "request.usage":
        return
    if isinstance(schema, dict):
        if type(value) is not dict or not set(value) <= set(schema):
            fail(f"Metering projection contains unexpected fields or container at {path}")
        for key, item in value.items():
            validate_projection(item, schema[key], path + "." + key)
    else:
        types = schema if isinstance(schema, tuple) else (schema,)
        if type(value) not in types or (
            isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0)
        ):
            fail(f"Metering projection contains invalid leaf at {path}")


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def inventory(root):
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(x in SKIP for x in rel.parts) or p.suffix == ".zip":
            continue
        if p.is_symlink():
            fail(f"Symlinks are not publishable: {rel}")
        if p.is_file() and rel.as_posix() not in CONTROL:
            files.append(rel.as_posix())
    return files


def scan(root, files):
    """Scan curated files; never print a suspected secret value."""
    for rel in files:
        p = root / rel
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            fail(f"Unsafe inventory member: {rel}")
        if p.name in BLOCKED or any(
            x in {"bodies", "auth", "credentials", "node_modules"} for x in Path(rel).parts
        ):
            fail(f"Private file class: {rel}")
        if p.suffix not in {".png"}:
            text = p.read_text()
            for name, pat in PATTERNS.items():
                # The upstream ISO currency-document URL has a path that
                # resembles a home directory; permit only that exact URL path.
                for m in re.finditer(pat, text):
                    if (
                        name == "email address"
                        and m.group()
                        in {"@".join(["name", "company.com"]), "@".join(["team", "example.com"])}
                        and rel == "public_loadout/public_mcp_schemas.json"
                    ):
                        continue  # Verbatim upstream schema examples, not identities.
                    line = text[
                        max(0, text.rfind("\n", 0, m.start()) + 1) : (
                            text.find("\n", m.end()) if "\n" in text[m.end() :] else len(text)
                        )
                    ]
                    if (
                        name == "home path"
                        and ("http://" in line or "https://" in line)
                        and "/".join(["", "home", "standards", ""]) in m.group()
                    ):
                        continue
                    fail(f"{name} indicator in {rel}; inspect locally, value withheld")
    for line in (root / "v1-snapshot/requests.jsonl").read_text().splitlines():
        q = json.loads(line)
        validate_projection(q, REQUEST_SCHEMA)
    for t in json.loads((root / "v1-snapshot/tasks.json").read_text()):
        if set(t) != {"id", "_domain", "_tier"}:
            fail("Task manifest must contain only identity/domain/tier, not private fixtures")
    return {
        "status": "PASS",
        "files_scanned": len(files),
        "raw_bodies": 0,
        "auth_databases": 0,
        "note": (
            "Heuristic scan plus explicit request/task allowlists; public sche"
            "ma example emails/dummy credentials are not real identities"
        ),
    }


def create_manifest(root):
    files = inventory(root)
    scan(root, files)
    records = [
        dict(
            path=f,
            bytes=(root / f).stat().st_size,
            sha256=hashlib.sha256((root / f).read_bytes()).hexdigest(),
        )
        for f in files
    ]
    (root / "MANIFEST.json").write_text(
        json.dumps(
            dict(
                format_version=1,
                classification="EXPLORATORY v1; no current-performance claim",
                files=records,
            ),
            indent=2,
        )
        + "\n"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in records)
        + f"{hashlib.sha256((root/'MANIFEST.json').read_bytes()).hexdigest()}  MANIFEST.json\n"
    )
    return files


def verify(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    files = []
    for r in manifest["files"]:
        rel = r["path"]
        if Path(rel).is_absolute() or ".." in Path(rel).parts or rel in files:
            fail("Invalid/duplicate inventory path")
        p = root / rel
        if (
            not p.is_file()
            or p.stat().st_size != r["bytes"]
            or hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]
        ):
            fail(f"Checksum mismatch: {rel}")
        files.append(rel)
    # Check both representations rather than treating SHA256SUMS as decoration.
    expected = (
        "".join(f"{r['sha256']}  {r['path']}\n" for r in manifest["files"])
        + f"{hashlib.sha256((root/'MANIFEST.json').read_bytes()).hexdigest()}  MANIFEST.json\n"
    )
    if (root / "SHA256SUMS").read_text() != expected:
        fail("SHA256SUMS does not match manifest")
    return files


def pack(root, files, dest):
    paths = files + sorted(CONTROL)
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if result.returncode == 0:
        repo = Path(result.stdout.strip())
        prefix = root.relative_to(repo).as_posix() + "/"
        tracked = subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "-z", "--", prefix], text=True
        ).split("\0")
        missing = [p for p in paths if prefix + p not in tracked]
        if missing:
            fail(
                f"Kit files not tracked yet ({len(missing)}); "
                "coordinator must stage/review the kit before Git-checkout packaging"
            )
    if dest.exists():
        fail("Refusing to overwrite an existing ZIP")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in sorted(paths):
            info = zipfile.ZipInfo("harness_cost/" + rel, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, (root / rel).read_bytes())
    print(
        json.dumps(
            dict(
                zip=dest.name,
                members=len(paths),
                bytes=dest.stat().st_size,
                git_tracking_checked=result.returncode == 0,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["manifest", "verify", "scan", "zip"])
    parser.add_argument("--output", type=Path, default=Path("harness-cost-kit.zip"))
    args = parser.parse_args()
    files = create_manifest(ROOT) if args.action == "manifest" else verify(ROOT)
    print(json.dumps(scan(ROOT, files)))
    if args.action == "zip":
        pack(ROOT, files, args.output)


if __name__ == "__main__":
    main()
