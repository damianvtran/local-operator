#!/usr/bin/env python3
"""Deterministic isolation probes for the Slice-0 POC container.

Why this is a script and not a prompt
------------------------------------
docs/design/remote-cloud-agents.md §9.2 runs the probes from the entrypoint
BEFORE the agent starts, because a script is a deterministic instrument and a
prompt is not. Every isolation claim this POC makes in §7.1 has to be made by
something the model cannot talk out of.

What it never does
------------------
It never prints, logs, or writes the model key. Probe 4c reads the key from an
INHERITED DESCRIPTOR — the entrypoint pipes it in; it is never an argv value and
never an environment variable — and reports only counts, the value's LENGTH and a
truncated SHA-256 of it. That holds for a placeholder value too: the report has to
be safe to publish on the day the value IS a credential.

Two modes
---------
Default: run the probes and write probes.json.
``--scan-dir``: no probes; count occurrences of the key under a directory, so
the entrypoint's pre-upload rescan (step 8) reuses probe 4c's own code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

#: Filesystem roots a full-disk scan must not descend into. /proc holds the
#: environment of running processes, /sys and /dev hold kernel and device
#: pseudo-files; none of them is "the filesystem" probe 4c asks about.
SCAN_SKIP_DIRS: frozenset[str] = frozenset({"/proc", "/sys", "/dev"})

#: Targets that MUST be unreachable behind an egress rule of tcp/443 only. The
#: IMDS address is here because §7.1 claims Fargate exposes no EC2 IMDS.
BLOCKED_TARGETS: tuple[tuple[str, int], ...] = (
    ("example.com", 80),
    ("github.com", 22),
    ("1.1.1.1", 53),
    ("portquiz.net", 8080),
    ("169.254.169.254", 80),
)

#: The positive control. Without it, "everything failed" is also what a broken
#: security group looks like, and a cut-off run would read as a hardened one.
ALLOWED_TARGETS: tuple[tuple[str, int], ...] = (("github.com", 443),)

CONNECT_TIMEOUT_SECONDS = 5.0

DENIED_CODES = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "AuthorizationError",
        "UnauthorizedOperation",
    }
)


def _read_key(fd: int) -> str:
    """Read the model key from an inherited descriptor.

    Never argv and never the environment: argv is readable from
    /proc/<pid>/cmdline by anything in the container, and the entrypoint has
    already removed the environment variable by the time this runs.
    """
    with os.fdopen(fd, "rb", closefd=False) as stream:
        return stream.read().decode("utf-8", "replace").strip()


def _denied_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    inner = response.get("Error")
    if not isinstance(inner, Mapping):
        return None
    code = str(inner.get("Code", ""))
    return code if code in DENIED_CODES else None


def _expect_denied(call: Callable[[], Any]) -> str:
    """Run ``call`` and describe the outcome as an evidence string.

    ``"denied: AccessDenied"`` is the expected result for an empty task role.
    ``"ALLOWED"`` means the probe failed and the caller reports that.
    """
    try:
        result = call()
    except (ClientError, BotoCoreError) as error:
        code = _denied_code(error)
        if code is not None:
            return f"denied: {code}"
        return f"unexpected error: {type(error).__name__}: {error}"[:300]
    return f"ALLOWED (probe failed): {result!r}"[:300]


def _container_credentials() -> tuple[dict[str, Any], str]:
    """Fetch this task's role credentials from the ECS credentials endpoint.

    The endpoint is link-local (169.254.170.2) and served by the Fargate agent,
    which is why the security group's 443-only egress rule does not block it.
    Probe 4a exists to check that claim rather than restate it (§9.2 step 1).
    """
    relative = os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "")
    url = f"http://169.254.170.2{relative}"
    with urllib.request.urlopen(url, timeout=CONNECT_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode()), url


def probe_credentials(model_secret_arn: str, region: str) -> dict[str, Any]:
    """Probe 4a: the credentials endpoint resolves, and its role can do nothing."""
    detail: dict[str, Any] = {}
    try:
        payload, url = _container_credentials()
    except Exception as error:  # noqa: BLE001 — any failure here IS the finding
        detail["endpoint"] = "unreachable"
        detail["error"] = f"{type(error).__name__}: {error}"
        return {"name": "4a_creds_endpoint", "pass": False, "detail": detail}
    detail["endpoint_url"] = url
    # The secret access key and session token are NEVER recorded, not even hashed:
    # the role identity is the evidence probe 4a needs.
    detail["role_arn"] = payload.get("RoleArn")
    detail["expiration"] = payload.get("Expiration")
    session = boto3.Session(
        aws_access_key_id=payload.get("AccessKeyId"),
        aws_secret_access_key=payload.get("SecretAccessKey"),
        aws_session_token=payload.get("Token"),
        region_name=region,
    )
    try:
        identity = session.client("sts").get_caller_identity()
        detail["caller_identity"] = identity.get("Arn")
        identity_ok = True
    except (ClientError, BotoCoreError) as error:
        detail["caller_identity"] = f"{type(error).__name__}: {error}"[:300]
        identity_ok = False
    detail["s3_list_buckets"] = _expect_denied(lambda: session.client("s3").list_buckets())
    detail["ecs_list_clusters"] = _expect_denied(lambda: session.client("ecs").list_clusters())
    if model_secret_arn:
        detail["secretsmanager_get_secret_value"] = _expect_denied(
            lambda: session.client("secretsmanager").get_secret_value(SecretId=model_secret_arn)
        )
    else:
        detail["secretsmanager_get_secret_value"] = "not probed: no secret ARN was passed"
    denied = [
        detail["s3_list_buckets"],
        detail["ecs_list_clusters"],
        detail["secretsmanager_get_secret_value"],
    ]
    passed = identity_ok and all(value.startswith("denied:") for value in denied)
    return {"name": "4a_creds_endpoint", "pass": passed, "detail": detail}


def _connect(host: str, port: int) -> tuple[bool, str]:
    """Try one TCP connect; return (connected, human-readable detail)."""
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SECONDS):
            elapsed = (time.monotonic() - started) * 1000
            return True, f"connected in {elapsed:.0f} ms"
    except OSError as error:
        elapsed = (time.monotonic() - started) * 1000
        return False, f"failed in {elapsed:.0f} ms: {type(error).__name__}"


def probe_egress() -> dict[str, Any]:
    """Probe 4b: only tcp/443 leaves the task, and there is no IMDS to reach."""
    detail: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    for host, port in BLOCKED_TARGETS:
        connected, note = _connect(host, port)
        blocked[f"{host}:{port}"] = f"REACHABLE (probe failed): {note}" if connected else note
    detail["must_be_unreachable"] = blocked
    allowed: dict[str, str] = {}
    for host, port in ALLOWED_TARGETS:
        connected, note = _connect(host, port)
        allowed[f"{host}:{port}"] = note if connected else f"UNREACHABLE (probe failed): {note}"
    detail["must_be_reachable"] = allowed
    passed = all(not value.startswith("REACHABLE") for value in blocked.values()) and all(
        not value.startswith("UNREACHABLE") for value in allowed.values()
    )
    return {"name": "4b_egress", "pass": passed, "detail": detail}


def _result(name: str, passed: bool | None, detail: dict[str, Any], note: str) -> dict[str, Any]:
    detail["note"] = note
    return {"name": name, "pass": passed, "detail": detail}


def _needles(key: str, prefix_chars: int) -> dict[str, bytes]:
    """The byte strings probe 4c searches for, keyed by NAME.

    "value" is always the EXACT string this run was handed, so a mock run's
    placeholder is searched for like any other value. "prefix" is the first
    ``prefix_chars`` characters and is included only when the caller asks for it
    (a real key, where a leaked prefix is itself worth detecting). Names, never
    needles, are what reaches the report.
    """
    needles = {"value": key.encode()}
    if prefix_chars and len(key) > prefix_chars:
        needles["prefix"] = key[:prefix_chars].encode()
    return needles


def probe_no_secret_on_disk(key: str, prefix_chars: int) -> dict[str, Any]:
    """Probe 4c: the key is not on the filesystem.

    What is recorded is a COUNT per needle plus the value's LENGTH and a truncated
    DIGEST of it. The value and the prefix themselves are never written, logged or
    returned — and that is true for a placeholder too, because code that is careful
    about a secret only on the days it holds one is code that leaks on the day it
    does.

    pass=None survives only for the case that cannot happen in a deployed task: no
    value injected at all. A placeholder is a value, so a mock run is a real verdict.
    """
    if not key:
        return _result(
            "4c_no_secret_on_disk",
            None,
            {"key_present_in_run": False},
            "no value was injected in this run; nothing to search for",
        )
    counts, scanned = _scan_for_needles(Path("/"), _needles(key, prefix_chars), SCAN_SKIP_DIRS)
    total = sum(counts.values())
    detail = {
        "files_scanned": scanned,
        "matches_by_needle": counts,
        "key_length_bytes": len(key),
        "key_sha256_first8": hashlib.sha256(key.encode()).hexdigest()[:8],
        "key_present_in_run": True,
    }
    needles_tested = "+".join(sorted(counts))
    return _result(
        "4c_no_secret_on_disk",
        total == 0,
        detail,
        f"{total} file match(es) over {scanned} files scanned for needle(s) {needles_tested}",
    )


def probe_key_absent_from_child_env(key: str) -> dict[str, Any]:
    """Probe 4c-env: a child spawned AFTER the unset cannot see the key.

    The entrypoint removes LOP_POC_MODEL_KEY from its own environment before
    anything the model can influence starts; this checks that from the far side by
    spawning a real child and counting how many of ITS environment entries contain
    the key's bytes. Zero is the only acceptable count.

    Separate from 4c because it can fail independently, and the two failures want
    different fixes: nothing on disk but the variable still exported means every bash
    and eval child the model starts is handed the key.
    """
    if not key:
        return _result(
            "4c_env_no_key_in_child_env",
            None,
            {"key_present_in_run": False},
            "no value was injected in this run; nothing to check",
        )
    completed = subprocess.run(["/usr/bin/env"], capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line]
    needle = key.encode()
    # NAMES of the offending entries, never their values: a variable called
    # LOP_POC_MODEL_KEY is the finding, and printing it is not a leak.
    names = [line.split("=", 1)[0] for line in lines if needle in line.encode()]
    detail = {
        "child_environment_entries": len(lines),
        "entries_containing_key": len(names),
        "entry_names": names[:10],
        "key_present_in_run": True,
    }
    return _result(
        "4c_env_no_key_in_child_env",
        not names,
        detail,
        f"{len(names)} of {len(lines)} child environment entries contain the key",
    )


def _scan_for_needles(
    root: Path, needles: dict[str, bytes], skip_dirs: frozenset[str]
) -> tuple[dict[str, int], int]:
    """Count, per needle NAME, the regular files whose bytes contain that needle.

    ONE pass for every needle: a walk per needle reads the whole filesystem twice,
    and this scan is already the slowest thing in the container's pre-agent phase.
    Chunked with only the longest needle's worth of tail carried over, so a needle
    straddling a chunk boundary is still found and no file is held in memory.
    """
    counts: dict[str, int] = dict.fromkeys(needles, 0)
    longest = max(len(needle) for needle in needles.values())
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _: None):
        dirnames[:] = [name for name in dirnames if os.path.join(dirpath, name) not in skip_dirs]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                if not stat.S_ISREG(os.stat(path, follow_symlinks=False).st_mode):
                    continue
                found: set[str] = set()
                with open(path, "rb") as stream:
                    carry = b""
                    while True:
                        chunk = stream.read(1 << 20)
                        if not chunk:
                            break
                        window = carry + chunk
                        for label, needle in needles.items():
                            if label not in found and needle in window:
                                found.add(label)
                        if len(found) == len(needles):
                            break
                        carry = chunk[-(longest - 1) :] if longest > 1 else b""
                for label in found:
                    counts[label] += 1
                scanned += 1
            except OSError:
                continue
    return counts, scanned


def _is_writable(path: Path) -> bool:
    try:
        with open(path, "wb") as stream:
            stream.write(b"probe")
        path.unlink()
        return True
    except OSError:
        return False


def probe_platform() -> dict[str, Any]:
    """The platform facts the isolation claims stand on, each checked live."""
    checks = {
        "uname_m_is_aarch64": os.uname().machine == "aarch64",
        "uid_is_10001": os.getuid() == 10001,
        "root_is_read_only": not _is_writable(Path("/lop-poc-probe-root")),
        "usr_is_read_only": not _is_writable(Path("/usr/lop-poc-probe-usr")),
        "workspace_is_writable": _is_writable(Path("/workspace/lop-poc-probe-ws")),
    }
    detail: dict[str, Any] = {"checks": checks}
    passed = all(checks.values())
    note = (
        "aarch64, uid 10001, read-only root, writable workspace"
        if passed
        else "at least one platform check failed"
    )
    return _result("4d_platform", passed, detail, note)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice-0 isolation probes")
    parser.add_argument("--out", type=Path, help="where to write probes.json")
    parser.add_argument(
        "--key-fd", type=int, default=0, help="inherited descriptor carrying the key"
    )
    parser.add_argument("--model-secret-arn", default="", help="ARN probe 4a must fail to read")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "ca-central-1"))
    parser.add_argument(
        "--key-prefix-chars",
        type=int,
        default=0,
        help=(
            "also search for the first N characters of the key; the entrypoint asks "
            "for 8 on a real-key run and 0 on a mock run, where the injected value "
            "is a placeholder rather than a credential"
        ),
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        help="rescan mode: count files under this directory containing the key, then exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    key = _read_key(args.key_fd)
    if args.scan_dir is not None:
        return _main_scan(args.scan_dir, key, args.key_prefix_chars)
    probes = [
        probe_credentials(args.model_secret_arn, args.region),
        probe_egress(),
        probe_no_secret_on_disk(key, args.key_prefix_chars),
        probe_key_absent_from_child_env(key),
        probe_platform(),
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probes": probes,
        "failed": [probe["name"] for probe in probes if probe["pass"] is False],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"failed": report["failed"]}, sort_keys=True))
    return 1 if report["failed"] else 0


def _main_scan(root: Path, key: str, prefix_chars: int) -> int:
    """Rescan mode, step 8: how many files under ``root`` carry the key.

    The same needles and the same scanner as probe 4c, so "the pre-upload rescan
    passed" and "probe 4c passed" cannot disagree about what was searched for.
    """
    if not key:
        print(json.dumps({"scanned_files": 0, "match_count": 0, "matches_by_needle": {}}))
        return 0
    counts, scanned = _scan_for_needles(root, _needles(key, prefix_chars), frozenset())
    total = sum(counts.values())
    print(
        json.dumps(
            {"scanned_files": scanned, "match_count": total, "matches_by_needle": counts},
            sort_keys=True,
        )
    )
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
