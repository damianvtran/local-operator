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
import shutil
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
    """Try one TCP connect to a target, under ONE deadline for the whole target.

    The budget is the spec's "within 5 s" for the TARGET, not for each address it
    resolves to. A dual-stack name whose families both black-hole used to cost two
    full timeouts — measured: example.com:80 reported 10 009 ms for a connection that
    was correctly refused. The verdict was right and the number was wrong, which is
    the kind of instrument that gets quoted forward.
    """
    started = time.monotonic()
    deadline = started + CONNECT_TIMEOUT_SECONDS
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        elapsed = (time.monotonic() - started) * 1000
        return False, f"failed in {elapsed:.0f} ms: {type(error).__name__}"
    last = "no address resolved"
    for family, socktype, proto, _canonical, sockaddr in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            last = "deadline exhausted"
            break
        with socket.socket(family, socktype, proto) as probe:
            probe.settimeout(remaining)
            try:
                probe.connect(sockaddr)
            except OSError as error:
                last = type(error).__name__
                continue
        elapsed = (time.monotonic() - started) * 1000
        return True, f"connected in {elapsed:.0f} ms"
    elapsed = (time.monotonic() - started) * 1000
    return False, f"failed in {elapsed:.0f} ms: {last}"


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


#: Every path a ``ps`` could be reached by. ``busybox ps`` and ``toybox ps`` matter
#: because both implement it without a ``ps`` file of their own.
_PS_PATHS = (
    "/bin/ps",
    "/usr/bin/ps",
    "/sbin/ps",
    "/usr/sbin/ps",
    "/usr/local/bin/ps",
    "/usr/local/sbin/ps",
)
_PS_MULTIPLEXERS = ("busybox", "toybox")


def probe_ps_absent() -> dict[str, Any]:
    """Probe 4f: no ``ps`` the agent's own tool children could spawn.

    WHY A PROBE AND NOT ONLY THE DOCKERFILE GUARD. Two spawn sites in the product hand
    the CALLER's environment to ``ps``: ``tools/group_reaper.py`` runs ``ps -o lstart=
    -p <pid>`` with ``env={**os.environ, "LC_ALL": "C"}`` (reached from the bash tool's
    group registration and the teardown reaper) and ``memory_guard._default_runner``
    runs ``ps -axo pid=,pgid=,rss=`` with no ``env=`` at all, on every guarded command's
    tick. Either child is a child of the process that holds the key, so its own
    ``/proc/<pid>/environ`` would carry it — and the model's same-uid bash child could
    read that file, which is the read path SEC-1 closed. So "no process's initial
    environment carries the key" is CONDITIONAL on there being no ``ps`` to run, and
    this probe is what makes the condition visible: add procps to this image for any
    reason and 4f fails, instead of the closure quietly reopening. 4e cannot catch it —
    the guard's reads last tens of milliseconds against 4e's one-second samples.

    Closing it product-side (passing a filtered environment at those two sites) is
    deferred work recorded in the PR thread: this POC runs the RELEASED wheel and does
    not patch product code, which is what makes the condition worth stating.
    """
    on_path = shutil.which("ps")
    present = [path for path in _PS_PATHS if os.path.exists(path)]
    multiplexers = [name for name in _PS_MULTIPLEXERS if shutil.which(name)]
    passed = on_path is None and not present and not multiplexers
    detail = {
        "which_ps": on_path,
        "checked_paths": list(_PS_PATHS),
        "paths_present": present,
        "multiplexers_present": multiplexers,
    }
    note = (
        ""
        if passed
        else (
            "a `ps` is reachable, so the two inherited-environment spawn sites "
            "(tools/group_reaper.py, memory_guard._default_runner) can carry the key into a "
            "child's initial environment"
        )
    )
    return _result("4f_no_ps", passed, detail, note)


def _comm_of(pid: str) -> str:
    """The process's `comm`, which is a NAME and never a value."""
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "?"


def _procfs_available() -> bool:
    """Whether this host has procfs, i.e. whether the watcher can observe at all.

    A named predicate rather than an inline ``isdir`` so the BLOCKED path is testable
    on a host that HAS procfs — which is the host that matters, because a probe whose
    blocked case has never been exercised is a probe whose blocked case does not work.
    """
    return os.path.isdir("/proc")


def _environ_blobs() -> list[tuple[str, bytes, str]]:
    """Every readable process's INITIAL environment: (pid label, blob, comm).

    THE BLOB IS THE ENVIRONMENT ALONE, NEVER A COMMAND LINE, and procfs is the only
    source this probe will use. Two measured reasons: ``/proc/<pid>/environ`` is
    exactly the environment image the kernel copied at ``exec``, and the Darwin
    substitute is not — ``ps -Eww -ax`` on this host printed only ARGV, so a key
    sitting in some process's command line matched it (a false positive) while the
    child started with the key in its environment did not (a false negative). An
    instrument that can only return the wrong answer is worse than one that says it
    cannot observe, so the watcher reports BLOCKED where there is no procfs.
    """
    if not _procfs_available():
        return []
    blobs: list[tuple[str, bytes, str]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as handle:
                blobs.append((entry, handle.read(), _comm_of(entry)))
        except OSError:
            continue
    return blobs


def _proc_environ_scan(
    needles: dict[str, bytes],
) -> tuple[dict[str, int], list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Count processes whose INITIAL environment carries a needle — and say what was read.

    On Linux this reads ``/proc/<pid>/environ`` — the environment image the kernel
    copied at ``exec``, which is the whole reason the entrypoint re-execs itself after
    taking the key out of its environment, and the read path ``shell_env.py``
    documents as ``cat /proc/$PPID/environ``. **Procfs is the only source this probe
    will use**: the Darwin substitute was written, measured and removed
    (``_environ_blobs`` records why), and without procfs the watcher reports BLOCKED
    rather than guessing. Only counts, PIDs and comm names are returned; never a byte
    of the environment itself.

    The fourth element is the OBSERVED set: every process whose environment was read,
    matched or not. Coverage is a claim about a window — "this sampled while the child
    was alive" — and a claim about a window that cannot be checked against the artifact
    is the same defect as a green probe that never looked (agent review round 3).
    """
    counts: dict[str, int] = dict.fromkeys(needles, 0)
    hits: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    blobs = _environ_blobs()
    if not blobs:
        return counts, hits, 0, observed
    for label, blob, comm in blobs:
        pid: Any = int(label) if label.isdigit() else label
        observed.append({"pid": pid, "comm": comm})
        for needle_label, needle in needles.items():
            if needle and needle in blob:
                counts[needle_label] += 1
                hits.append({"pid": pid, "comm": comm, "needle": needle_label})
    return counts, hits, len(blobs), observed


def _agent_pid(pid_file: Path | None) -> int | None:
    """The agent process, from the launcher's own pid file.

    The launcher writes its pid because the entrypoint cannot know it (the agent runs
    in the foreground) and a cmdline scan is unreliable: ``lop`` renames itself with
    ``prctl``, so by the time the watcher samples, the cmdline it would have matched on
    is gone. Fallback: the cmdline scan, for a run whose launcher is a different build.
    """
    if pid_file is not None:
        try:
            return int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
    return _find_agent_pid()


def _find_agent_pid() -> int | None:
    """The launcher process, found by cmdline scan.

    The watcher is a sibling of the agent rather than its parent (both are children
    of the entrypoint), which is what makes the memory check below a same-uid
    NON-DESCENDANT read — the case ``yama/ptrace_scope`` exists to decide.

    ``None`` where there is no procfs, which is also where the memory question cannot
    be answered at all: the caller records that as ``mem_openable: null`` with the
    platform named, rather than as ``False``, because "no procfs" is not "protected".
    """
    if not _procfs_available():
        return None
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = (
                Path(f"/proc/{entry}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "replace")
            )
        except OSError:
            continue
        if "lop_launch" in cmdline or "lop-launch" in cmdline:
            return int(entry)
    return None


def _residual(agent_pid: int | None) -> dict[str, Any]:
    """The two residuals the environ claim cannot speak for.

    (1) The key lives in the agent's MEMORY, and every process here is uid 10001.
    (2) Whether a same-uid non-descendant may read that memory is the kernel's
    answer, not ours: ``yama/ptrace_scope`` decides, and the check is an actual
    open of ``/proc/<pid>/mem`` rather than a restatement of the policy.
    """
    residual: dict[str, Any] = {"platform": sys.platform}
    try:
        residual["yama_ptrace_scope"] = (
            Path("/proc/sys/kernel/yama/ptrace_scope").read_text(encoding="utf-8").strip()
        )
    except OSError as error:
        residual["yama_ptrace_scope"] = f"unreadable: {type(error).__name__}"
    if agent_pid is None:
        residual["mem_openable"] = None
        residual["mem_target"] = None
        return residual
    if not _procfs_available():
        # No procfs: nothing to open, and that is NOT the same as "protected".
        residual["mem_target"] = {"pid": agent_pid, "comm": "ps"}
        residual["mem_openable"] = None
        residual["mem_note"] = (
            f"no procfs on {sys.platform}; the memory question is measured on Linux only"
        )
        return residual
    residual["mem_target"] = {"pid": agent_pid, "comm": _comm_of(str(agent_pid))}
    try:
        with open(f"/proc/{agent_pid}/mem", "rb"):
            residual["mem_openable"] = True
            residual["mem_note"] = (
                "a same-uid, NON-descendant process opened the agent's memory: the key is "
                "recoverable from /proc/<pid>/mem (the in-memory residual, disclosed)"
            )
    except OSError as error:
        residual["mem_openable"] = False
        residual["mem_error"] = f"{type(error).__name__}: {error.strerror or ''}".strip()
    return residual


def watch_environ(
    out: Path,
    key: str,
    prefix_chars: int,
    stop_file: Path,
    interval_ms: int,
    max_seconds: float,
    max_samples: int | None = None,
    agent_pid_file: Path | None = None,
) -> int:
    """Probe 4e: sample every process's initial environment while the agent runs.

    WHY A WATCHER AND NOT A ONE-SHOT. Probe 4c-env spawns ``env`` from the probe
    process before the agent exists, so it can see "the entrypoint forgot to unset
    ``LOP_POC_MODEL_KEY``" and cannot see "the value was handed to the agent as its
    launch environment" — the pair of processes that matters only coexist while the
    agent is running. This samples that window, and the key it searches for is
    itself delivered over the key fd, never through the environment it is scanning.

    The report is rewritten after every sample, so a killed watcher still leaves its
    last state behind. Exit 0 iff no process's environ ever matched.
    """
    needles = _needles(key, prefix_chars)
    if not _procfs_available():
        # BLOCKED, not a pass: without procfs this probe can observe nothing, and a
        # green reading from an instrument that never looked is the failure mode
        # AGENTS.md names ("a dead instrument returns a reading, not an error").
        message = {
            "blocked": True,
            "reason": (
                f"no procfs on {sys.platform}: /proc/<pid>/environ is the only source "
                "this probe uses"
            ),
        }
        out.write_text(json.dumps(message, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(message, sort_keys=True), file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "samples": 0,
        "interval_ms": interval_ms,
        "key_length_bytes": len(key),
        "key_sha256_first8": hashlib.sha256(key.encode()).hexdigest()[:8] if key else None,
        "processes_scanned_max": 0,
        "matches_by_needle": dict.fromkeys(needles, 0),
        "matching_processes": [],
        "observed_processes": [],
        "residual": {},
    }
    deadline = time.monotonic() + max_seconds
    while True:
        counts, hits, scanned, observed = _proc_environ_scan(needles)
        report["samples"] += 1
        report["processes_scanned_max"] = max(report["processes_scanned_max"], scanned)
        # What was OBSERVED, not only what matched: PIDs and comms are names, and a byte
        # of no environment is recorded. This is what makes the coexistence row's
        # "while 4e sampled" checkable against the artifact instead of asserted.
        for entry in observed:
            if entry not in report["observed_processes"]:
                report["observed_processes"].append(entry)
        report["observed_processes"] = report["observed_processes"][:32]
        for label, count in counts.items():
            report["matches_by_needle"][label] = max(report["matches_by_needle"][label], count)
        for hit in hits:
            if hit not in report["matching_processes"]:
                report["matching_processes"].append(hit)
        # KEEP THE FIRST DEFINITIVE MEMORY READING, not the last one. The watcher's last
        # sample lands after the agent has exited, so probing then opens a pid that no
        # longer exists and reports FileNotFoundError — a dead-process artifact mistaken
        # for a measurement (measured, on the first five runs of this revision). The
        # first sample that HAS a target pid is the one taken while the agent was alive.
        candidate = _residual(_agent_pid(agent_pid_file))
        if not report["residual"] or (
            report["residual"].get("mem_target") is None and candidate.get("mem_target") is not None
        ):
            report["residual"] = candidate
        total = sum(report["matches_by_needle"].values())
        report["pass"] = total == 0
        report["note"] = (
            f"{total} process(es) whose INITIAL environment carried the key, over "
            f"{report['samples']} sample(s) and up to {report['processes_scanned_max']} "
            f"readable /proc/<pid>/environ"
        )
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if stop_file.exists() or time.monotonic() >= deadline:
            break
        if max_samples is not None and report["samples"] >= max_samples:
            break
        time.sleep(interval_ms / 1000)
    # The digest goes to stdout, which is CloudWatch: the artifact carries the full
    # record, and this is what makes the verdict readable in the task log without
    # downloading anything. Counts and NAMES only — never a byte of an environment.
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "samples": report["samples"],
                "processes_scanned_max": report["processes_scanned_max"],
                "observed_processes": report["observed_processes"],
                "matches_by_needle": report["matches_by_needle"],
                "matching_processes": report["matching_processes"][:10],
            },
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 1


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
    parser.add_argument(
        "--watch-environ",
        action="store_true",
        help="watcher mode: sample every process's initial environment until --stop-file appears",
    )
    parser.add_argument("--stop-file", type=Path, help="watcher mode: stop when this path exists")
    parser.add_argument("--interval-ms", type=int, default=1000, help="watcher sampling interval")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=7300.0,
        help="watcher hard bound (agent deadline + slack)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="watcher mode: stop after N samples (used by the red/green unit test)",
    )
    parser.add_argument(
        "--agent-pid-file",
        type=Path,
        help="watcher mode: the launcher writes its pid here for the memory probe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    key = _read_key(args.key_fd)
    if args.watch_environ:
        if args.out is None or args.stop_file is None:
            print("--watch-environ needs --out and --stop-file", file=sys.stderr)
            return 2
        return watch_environ(
            args.out,
            key,
            args.key_prefix_chars,
            args.stop_file,
            args.interval_ms,
            args.max_seconds,
            args.max_samples,
            args.agent_pid_file,
        )
    if args.scan_dir is not None:
        return _main_scan(args.scan_dir, key, args.key_prefix_chars)
    probes = [
        probe_credentials(args.model_secret_arn, args.region),
        probe_egress(),
        probe_no_secret_on_disk(key, args.key_prefix_chars),
        probe_key_absent_from_child_env(key),
        probe_platform(),
        probe_ps_absent(),
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

    EXIT 2 FOR AN EMPTY KEY, not 0. A caller that treats rc 0 as "clean" would
    otherwise report a clean scan of zero files when the key never arrived at all
    — the failure mode ``verify``'s local scan had, and exactly the shape AGENTS.md
    calls a dead instrument returning a reading.
    """
    if not key:
        print(
            json.dumps(
                {"error": "no key was delivered on the key fd; nothing was scanned"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
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
