"""In-container gold-isolation and quiescence probe (stdlib; run as root in ``main``).

WHY (review-rigor POPULATION-SCREENING-DRAFT "Minimal owner QA"): Harbor's
shared-container order is source-supported, not runtime-proven. Before the
first model request this probe proves the agent cannot see the task's hidden
inputs: ``/solution`` and ``/tests`` are absent, ``/logs/verifier`` is empty,
and no readable file anywhere in the container has the bytes of any host-side
``solution/`` or ``tests/`` file (the host passes only their SHA-256 digests,
never their content). After the agent phase and BEFORE Harbor uploads tests, it
records quiescence: every process still alive besides the container's keep-alive
init and this probe, as pid, SHA-256 of its command line and its listening TCP
ports. That is a disclosed check, never a remedy: services such as
nginx-request-logging's nginx or kv-store-grpc's server are what the tests
grade, so ``bench_verifier`` grades exactly as Harbor's stock verifier does and
reports survivors beside the score (R6 F15).

Out of scope, stated rather than implied: the isolation scan matches WHOLE-file
hashes only. Gold bytes inside an archive, a concatenation or a re-encoding are
not detected, and files planted to run at verify time are not processes.

The zero-paid container S4 plants sentinel-only files in a scratch copy of a
DEV task's ``solution/`` and ``tests/``; seeing them absent before inference and
the tests sentinel present in ``/tests`` after upload is the positive control
that the detector works.

``egress --canary <port> <gate_port>...`` is the per-trial fence check
(``egress.py``): public IPs and a known-OPEN run-owned host port (the canary,
which the gate never forwards) must have NO route -- ENETUNREACH, EHOSTUNREACH
or a timeout; a refusal means a packet got there and back, so it fails (R6
F16) -- public DNS must not resolve, unlisted gate ports must not connect, and
every listed gate port must connect: the gate is the one route.

Usage: ``python3 gold_probe.py isolation <digest>...`` |
``quiescence`` | ``egress --canary <port> <gate_port>...``. Prints one JSON line; exit 0 = clean.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

# Kernel/virtual trees and the read-only frozen toolchain (hash-attested
# separately) are skipped; everything else, including /tmp and $HOME, is read.
SKIP = ("/proc", "/sys", "/dev", "/opt/bench", "/opt/bench-support", "/run/bench-ca")
# No size cut-off (R6 F20): a skipped large copy of a gold file was a silent
# pass. Files are hashed streaming, so size costs time, never memory.
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
                if not st.st_mode & 0o170000 == 0o100000:
                    continue
                scanned += 1
                if scanned > MAX_FILES:
                    truncated = True
                    break
                with open(path, "rb") as fh:
                    if hashlib.file_digest(fh, "sha256").hexdigest() in digests:
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


def _listening_inodes(proc_root: Path) -> dict[str, int]:
    """Socket inode -> port for every LISTEN (state 0A) TCP socket in this netns."""
    ports: dict[str, int] = {}
    for table in ("tcp", "tcp6"):
        try:
            rows = (proc_root / "net" / table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A":
                ports[cols[9]] = int(cols[1].rsplit(":", 1)[1], 16)
    return ports


def quiescence(proc_root: Path = Path("/proc")) -> dict[str, object]:
    """Record (never signal) every surviving agent-started process.

    Command lines are reported as SHA-256 only: an agent's argv can carry a
    value that must not land in a published trial record.
    """
    me = os.getpid()
    ancestry = {me, os.getppid()}
    listening = _listening_inodes(proc_root)
    survivors = []
    for entry in sorted(proc_root.iterdir(), key=lambda e: (len(e.name), e.name)):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        # PID 1 is Harbor's ``sleep infinity`` keep-alive; the probe's own
        # docker-exec shell is its parent.
        if pid == 1 or pid in ancestry:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").strip()
            state = next(
                (ln.split()[1] for ln in (entry / "status").read_text().splitlines()
                 if ln.startswith("State:")),
                "?",
            )
        except OSError:
            continue  # exited while scanning
        if state == "Z":
            continue  # zombie: cannot run or read anything
        ports = set()
        try:
            for fd in (entry / "fd").iterdir():
                target = os.readlink(fd)
                if target.startswith("socket:[") and target[8:-1] in listening:
                    ports.add(listening[target[8:-1]])
        except OSError:
            pass  # fds unreadable or closing: record the process without ports
        survivors.append({
            "pid": pid,
            "cmd_sha256": hashlib.sha256(cmd).hexdigest(),
            "listening_ports": sorted(ports),
        })
    return {"check": "quiescence", "ok": not survivors, "survivors": survivors}


# Outcomes that prove no packet reached a listener. ECONNREFUSED is NOT one:
# a refusal is a reply, so a route exists (R6 F16).
NO_ROUTE = frozenset({"ENETUNREACH", "EHOSTUNREACH", "timeout", "gaierror"})


def _connects(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=5):
            return "connected"
    except socket.gaierror:
        return "gaierror"
    except TimeoutError:
        return "timeout"
    except OSError as exc:
        return errno.errorcode.get(exc.errno or 0, type(exc).__name__)


def _resolves(name: str) -> str:
    try:
        socket.getaddrinfo(name, 443)
        return "resolved"
    except OSError as exc:
        return type(exc).__name__


def egress(gate_ports: list[int], canary_port: int, gate: str = "bench-gate",
           host: str = "host.docker.internal") -> dict[str, object]:
    """``canary_port`` is a run-owned host listener the gate does NOT forward.

    It is known OPEN, so only "no route" can make it fail; a closed port (the
    old ``:22`` probe) fails the same way with or without a route. ``host``
    must resolve (``main`` carries a host-gateway hosts entry) or the canary
    tested name lookup, not routing.
    """
    no_route = {
        "public-ip-tcp-1.1.1.1:443": _connects("1.1.1.1", 443),
        "public-ip-tcp-8.8.8.8:53": _connects("8.8.8.8", 53),
        f"host-canary-{canary_port}": _connects(host, canary_port),
    }
    denied = {
        "public-dns-example.com": _resolves("example.com"),
        "gate-unlisted-port-1": _connects(gate, 1),
        f"gate-canary-{canary_port}": _connects(gate, canary_port),
    }
    host_name = _resolves(host)
    allowed = {f"{gate}:{p}": _connects(gate, p) for p in gate_ports}
    ok = (
        bool(gate_ports)
        and canary_port not in gate_ports
        and host_name == "resolved"
        and all(v in NO_ROUTE for v in no_route.values())
        and all(v not in ("connected", "resolved") for v in denied.values())
        and all(v == "connected" for v in allowed.values())
    )
    return {"check": "egress", "ok": ok, "no_route": no_route, "denied": denied,
            "host_gateway_name": host_name, "allowed": allowed}


def main(argv: list[str]) -> int:
    if argv[:1] == ["isolation"]:
        result = isolation(set(argv[1:]))
    elif argv[:1] == ["quiescence"]:
        result = quiescence()
    elif (argv[:2] == ["egress", "--canary"] and len(argv) > 3
          and all(a.isdigit() for a in argv[2:])):
        result = egress([int(a) for a in argv[3:]], canary_port=int(argv[2]))
    else:
        print("usage: gold_probe.py isolation <sha256>... | quiescence | "
              "egress --canary <port> <gate_port>...",
              file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
