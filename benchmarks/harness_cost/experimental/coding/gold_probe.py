"""In-container gold-isolation and quiescence probe (stdlib; run as root in ``main``).

WHY (review-rigor POPULATION-SCREENING-DRAFT "Minimal owner QA"): Harbor's
shared-container order is source-supported, not runtime-proven. Before the
first model request this probe proves the agent cannot see the task's hidden
inputs: ``/solution`` and ``/tests`` are absent, ``/logs/verifier`` is empty,
and no readable file has hidden solution/test bytes. Exact public COPY
path/hash pairs from the reviewed, task/image-bound manifest are classified
separately; raw strict matches are retained (hashes only, never content). After
the agent phase and BEFORE Harbor uploads tests, it records quiescence: every process still alive besides the container's keep-alive
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

``isolation ... --authorized-solution <tree-json> <sha256>`` exists ONLY for
zero-model oracle qualification (``oracle_qualification.py``, design step 5):
digests supplied by that exact uploaded ``/solution`` tree are classified
``authorized_solution_matches``; tests-only digests stay rejected. No paid
path passes it (``bench_verifier.BenchVerifier._isolation_extra_args``).

Usage: ``python3 gold_probe.py isolation <digest>...`` |
``quiescence`` | ``egress --canary <port> <gate_port>...``. Prints one JSON line; exit 0 = clean.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import socket
import sys
from pathlib import Path

# Kernel/virtual trees and the read-only frozen toolchain (hash-attested
# separately) are skipped; everything else, including /tmp and $HOME, is read.
SKIP = ("/proc", "/sys", "/dev", "/opt/bench", "/opt/bench-support", "/run/bench-ca")
# No size cut-off (R6 F20): a skipped large copy of a gold file was a silent
# pass. Files are hashed streaming, so size costs time, never memory.
MAX_FILES = 400_000


def provenance_args() -> list[str]:
    """Both host boundaries pass the identical frozen manifest, not container env."""
    names = ("BENCH_PUBLIC_PROVENANCE", "BENCH_PUBLIC_PROVENANCE_SHA256",
             "BENCH_TASK_PAYLOAD_SHA256", "BENCH_IMAGE_DIGEST")
    values = [os.environ.get(name, "") for name in names]
    if not all(values):
        raise ValueError("missing public provenance manifest/binding")
    return ["--public-provenance", *values]


def _public_pairs(manifest, task_payload_sha256, image_digest, digests):
    """Malformed/stale provenance cannot turn a strict match into public input."""
    def is_sha(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    if (not isinstance(manifest, dict) or manifest.get("version") != 1
            or not is_sha(task_payload_sha256)
            or manifest.get("task_payload_sha256") != task_payload_sha256
            or not isinstance(image_digest, str) or not image_digest.startswith("sha256:")
            or not is_sha(image_digest[7:]) or manifest.get("image_digest") != image_digest
            or not is_sha(manifest.get("policy_sha256"))
            or not isinstance(manifest.get("solution_digests"), list)
            or not isinstance(manifest.get("gold_digests"), list)
            or not all(is_sha(d) for d in manifest["gold_digests"])
            or set(manifest["gold_digests"]) != digests or not digests
            or not all(is_sha(d) and d in digests for d in manifest["solution_digests"])
            or not isinstance(manifest.get("public_files"), list)):
        raise ValueError("malformed or stale public provenance")
    pairs = set()
    paths = set()
    for entry in manifest["public_files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("malformed public pair")
        path, sha = entry["path"], entry["sha256"]
        if (not isinstance(path, str) or not path.startswith("/")
                or str(Path(path)) != path or ".." in Path(path).parts
                or path in paths or not is_sha(sha)
                or any(path == p or path.startswith(p + "/")
                       for p in (*SKIP, "/solution", "/tests", "/logs/verifier"))):
            raise ValueError("invalid public path/digest")
        paths.add(path)
        pairs.add((path, sha))
    return pairs


def _authorized_tree(authorized, manifest) -> dict[str, str]:
    """The exact uploaded ``/solution`` tree an oracle qualification may carry.

    ORACLE-ONLY (ORACLE-ISOLATION-DESIGN.md step 5): only
    ``oracle_qualification.QualificationVerifier`` passes this, from a host-side
    receipt. Every non-empty digest must be a manifest solution digest, so a
    verifier-only (tests) digest can never be authorized through it.
    """
    if not isinstance(authorized, dict) or not authorized:
        raise ValueError("malformed authorized solution tree")
    solutions = set(manifest.get("solution_digests", []))
    for rel, sha in authorized.items():
        if (not isinstance(rel, str) or not rel or rel.startswith("/")
                or ".." in Path(rel).parts or str(Path(rel)) != rel
                or not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                or (sha != EMPTY_SHA256 and sha not in solutions)):
            raise ValueError("authorized solution entry is not a manifest solution file")
    return authorized


def _solution_tree(base: Path) -> dict[str, str]:
    tree = {}
    for path in sorted(base.rglob("*")) if base.is_dir() else []:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            tree[str(path.relative_to(base))] = "non-regular"
        elif path.is_file():
            with open(path, "rb") as fh:
                tree[str(path.relative_to(base))] = hashlib.file_digest(fh, "sha256").hexdigest()
    return tree


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
DISCLOSABLE = frozenset({"solution", "tests-fixture"})
CLASSES = DISCLOSABLE | {"verifier-code"}


def _digest_classes(manifest, digests) -> dict[str, str] | None:
    """The manifest's frozen class map, or ``None`` (all matches refuse)."""
    classes = manifest.get("digest_classes") if isinstance(manifest, dict) else None
    if (not isinstance(classes, dict) or set(classes) != set(digests)
            or not all(v in CLASSES for v in classes.values())):
        return None
    return classes


# The oracle upload target; a module constant so offline tests can relocate it.
SOLUTION = Path("/solution")


def isolation(digests: set[str], root: str = "/", *, manifest=None,
              task_payload_sha256=None, image_digest=None,
              authorized_solution=None) -> dict[str, object]:
    found: list[str] = []
    raw_matches = []
    error = None
    try:
        public_pairs = _public_pairs(manifest, task_payload_sha256, image_digest, digests)
    except (ValueError, TypeError) as exc:
        public_pairs, error = set(), str(exc)
    authorized: dict[str, str] = {}
    if authorized_solution is not None and error is None:
        try:
            authorized = _authorized_tree(authorized_solution, manifest)
        except (ValueError, TypeError) as exc:
            error = str(exc)
    elif authorized_solution is not None:
        error = error or "authorized solution without valid provenance"
    # A public/solution collision is disclosed, never silently authorized.
    solutions = set(manifest.get("solution_digests", [])) if not error else set()
    overlap = sorted({sha for _, sha in public_pairs} & solutions)
    public_matches, rejected, authorized_matches = [], [], []
    authorized_digests = set(authorized.values()) - {EMPTY_SHA256}
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
                    sha = hashlib.file_digest(fh, "sha256").hexdigest()
                    if sha in digests:
                        found.append(path)
                        match = {"path": path, "sha256": sha}
                        raw_matches.append(match)
                        if (path, sha) in public_pairs and sha not in solutions:
                            public_matches.append(match)
                        elif sha in authorized_digests:
                            # Supplied by the authorized oracle source itself.
                            authorized_matches.append(match)
                        else:
                            rejected.append(match)
            except OSError:
                continue
        if truncated:
            break
    verifier_logs = sorted(p.name for p in Path("/logs/verifier").glob("*")) if Path(
        "/logs/verifier"
    ).is_dir() else []
    # Without an authorization /solution must be absent; with one, it must be
    # EXACTLY the recorded upload (no extra, missing or changed entry).
    present = SOLUTION.exists()
    tree = _solution_tree(SOLUTION) if present and authorized else {}
    solution_ok = (not present) if not authorized else (not present or tree == authorized)
    ok = (
        not rejected
        and error is None
        and not overlap
        and not truncated
        and solution_ok
        and not Path("/tests").exists()
        and not verifier_logs
    )
    # S9b pre-grading rule. ``ok`` above stays the strict decision (and the
    # exit code), which is all the pre-inference boundary reads. A hidden match
    # that reached grading was absent pre-inference (that scan refuses any), so
    # with /tests withheld and the fence intact it can only be the agent's own
    # work -- e.g. a correct answer equal to an expected-output fixture. Such
    # solution/tests-fixture matches are DISCLOSED and graded; verifier-code or
    # unclassified matches, and every structural failure, still refuse. The
    # fence half of the rule is checked host-side by BenchVerifier.
    classes = _digest_classes(manifest, digests) if error is None else None
    disclosed, refusing = [], []
    for match in rejected:
        kind = (classes or {}).get(match["sha256"], "unclassified")
        (disclosed if kind in DISCLOSABLE else refusing).append({**match, "class": kind})
    post_agent_ok = (
        error is None
        and not refusing
        and not overlap
        and not truncated
        and solution_ok
        and not Path("/tests").exists()
        and not verifier_logs
    )
    return {
        "check": "gold-isolation",
        "ok": ok,
        "post_agent_ok": post_agent_ok,
        "disclosed_matches": disclosed,
        "refusing_matches": refusing,
        "strict_ok": not found and not truncated and not Path("/solution").exists()
        and not Path("/tests").exists() and not verifier_logs,
        "scanned_files": scanned,
        "truncated": truncated,
        "matched_paths": found,
        "raw_matches": raw_matches,
        "public_matches": public_matches,
        "rejected_matches": rejected,
        "solution_public_overlap": overlap,
        "authorized_solution_matches": authorized_matches,
        "authorized_solution_tree_match": (tree == authorized) if authorized else None,
        "provenance_error": error,
        "manifest_sha256": hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
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
        args = argv[1:]
        manifest, payload, image = None, None, None
        authorized = None
        if "--authorized-solution" in args:
            # Oracle qualification only; see _authorized_tree. A tree whose
            # digest does not verify becomes a malformed value, so it fails.
            index = args.index("--authorized-solution")
            try:
                raw_tree, tree_sha = args[index + 1:index + 3]
                authorized = (json.loads(raw_tree) if hashlib.sha256(
                    raw_tree.encode()).hexdigest() == tree_sha else "digest mismatch")
            except ValueError:
                authorized = "malformed"
            args = args[:index] + args[index + 3:]
        if "--public-provenance" in args:
            index = args.index("--public-provenance")
            try:
                raw, expected, payload, image = args[index + 1:]
                if hashlib.sha256(raw.encode()).hexdigest() != expected:
                    raise ValueError("manifest hash mismatch")
                manifest = json.loads(raw)
            except (ValueError, TypeError):
                manifest = None
            args = args[:index]
        result = isolation(set(args), manifest=manifest,
                           task_payload_sha256=payload, image_digest=image,
                           authorized_solution=authorized)
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
