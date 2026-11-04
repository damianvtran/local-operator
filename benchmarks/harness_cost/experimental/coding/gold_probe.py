"""In-container gold-isolation and quiescence probe (stdlib; run as root in ``main``).

WHY (review-rigor POPULATION-SCREENING-DRAFT "Minimal owner QA"): Harbor's
shared-container order is source-supported, not runtime-proven. Before the
first model request this probe proves the agent cannot see the task's hidden
inputs: ``/solution`` and ``/tests`` are absent, ``/logs/verifier`` is empty,
and no readable file anywhere in the container has the bytes of any host-side
``solution/`` or ``tests/`` file (the host passes only their SHA-256 digests,
never their content). After the agent phase and BEFORE Harbor uploads tests, it
proves quiescence: no process survives except the container's keep-alive init
and this probe, so nothing can read later tests/rewards.

The zero-paid container S4 plants sentinel-only files in a scratch copy of a
DEV task's ``solution/`` and ``tests/``; seeing them absent before inference and
the tests sentinel present in ``/tests`` after upload is the positive control
that the detector works.

``egress <gate_port>...`` is the per-trial fence check (``egress.py``): a public
IP, a public name, an unlisted host-gateway port and an unlisted gate port must
all fail, and every listed gate port must connect -- the gate is the one route.

Usage: ``python3 gold_probe.py isolation <digest>...`` |
``quiescence`` | ``egress <gate_port>...``. Prints one JSON line; exit 0 = clean.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path

# Kernel/virtual trees and the read-only frozen toolchain (hash-attested
# separately) are skipped; everything else, including /tmp and $HOME, is read.
SKIP = ("/proc", "/sys", "/dev", "/opt/bench", "/opt/bench-support", "/run/bench-ca")
MAX_BYTES = 4 * 1024 * 1024
MAX_FILES = 400_000


def isolation(digests: set[str], root: str = "/") -> dict[str, object]:
    found: list[str] = []
    scanned = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if any(dirpath == s or dirpath.startswith(s + "/") for s in SKIP):
            dirnames[:] = []
            continue
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
                if not (st.st_mode & 0o170000 == 0o100000) or st.st_size > MAX_BYTES:
                    continue
                scanned += 1
                if scanned > MAX_FILES:
                    truncated = True
                    break
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() in digests:
                    found.append(path)
            except OSError:
                continue
        if truncated:
            break
    verifier_logs = sorted(p.name for p in Path("/logs/verifier").glob("*")) if Path(
        "/logs/verifier"
    ).is_dir() else []
    ok = (
        not found
        and not truncated
        and not Path("/solution").exists()
        and not Path("/tests").exists()
        and not verifier_logs
    )
    return {
        "check": "gold-isolation",
        "ok": ok,
        "scanned_files": scanned,
        "truncated": truncated,
        "matched_paths": found,
        "solution_present": Path("/solution").exists(),
        "tests_present": Path("/tests").exists(),
        "verifier_log_entries": verifier_logs,
        "digests_checked": len(digests),
    }


def quiescence() -> dict[str, object]:
    me = os.getpid()
    ancestry = {me, os.getppid()}
    survivors = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        # PID 1 is Harbor's ``sleep infinity`` keep-alive; the probe's own
        # docker-exec shell is its parent.
        if pid == 1 or pid in ancestry:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            state = next(
                (ln.split()[1] for ln in (entry / "status").read_text().splitlines()
                 if ln.startswith("State:")),
                "?",
            )
        except OSError:
            continue  # exited while scanning
        if state == "Z":
            continue  # zombie: cannot run or read anything
        survivors.append({"pid": pid, "cmd": cmd.strip()[:200]})
    return {"check": "quiescence", "ok": not survivors, "survivors": survivors}


def _connects(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=5):
            return "connected"
    except OSError as exc:
        return type(exc).__name__


def _resolves(name: str) -> str:
    try:
        socket.getaddrinfo(name, 443)
        return "resolved"
    except OSError as exc:
        return type(exc).__name__


def egress(gate_ports: list[int], gate: str = "bench-gate") -> dict[str, object]:
    denied = {
        "public-ip-tcp-1.1.1.1:443": _connects("1.1.1.1", 443),
        "public-ip-tcp-8.8.8.8:53": _connects("8.8.8.8", 53),
        "public-dns-example.com": _resolves("example.com"),
        "host-gateway-unlisted-22": _connects("host.docker.internal", 22),
        "gate-unlisted-port-1": _connects(gate, 1),
    }
    allowed = {f"{gate}:{p}": _connects(gate, p) for p in gate_ports}
    ok = (
        bool(gate_ports)
        and all(v not in ("connected", "resolved") for v in denied.values())
        and all(v == "connected" for v in allowed.values())
    )
    return {"check": "egress", "ok": ok, "denied": denied, "allowed": allowed}


def main(argv: list[str]) -> int:
    if argv[:1] == ["isolation"]:
        result = isolation(set(argv[1:]))
    elif argv[:1] == ["quiescence"]:
        result = quiescence()
    elif argv[:1] == ["egress"] and all(a.isdigit() for a in argv[1:]):
        result = egress([int(a) for a in argv[1:]])
    else:
        print("usage: gold_probe.py isolation <sha256>... | quiescence | egress <port>...",
              file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
