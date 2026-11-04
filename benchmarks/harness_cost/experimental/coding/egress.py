"""Paid-trial container egress: an internal network whose only exit is the meter gate.

WHY: ``--network none`` (the zero-paid probes) cannot reach the host meter, and
Harbor's public default lets any tool dial anything, unmetered. Two designs
were weighed:

* host-gateway address + host firewall rule: on Docker Desktop for macOS the
  containers run in a Linux VM, so a host ``pf`` rule never sees their traffic
  and the VM's iptables are not ours to own. The fence would be unenforceable
  from the host, and Harbor's own egress sidecar would mean pulling a second
  image (the one-image-at-a-time rule).
* **internal Docker network + explicit gate container (chosen)**: ``main``
  joins only a run-owned ``internal: true`` network, which Docker creates with
  no default route and no external DNS. A ``bench-gate`` container, on that
  network AND an uplink, runs ``gate_relay.py`` (stdlib, frozen toolchain
  Python, SAME already-pulled task image -- no extra pull) and forwards a fixed
  port map to fixed host ports: the meter, the LoP edge, the Codex bridge.
  Docker itself enforces the fence; the gate is the single, auditable exit.

The verifier keeps Harbor's normal shared-container run, which for Terminal-Bench
fetches pytest/uv from the internet. ``bench_verifier.BenchVerifier`` therefore
proves the agent is quiescent, THEN connects ``main`` to the uplink, THEN runs
Harbor's stock ``Verifier.verify()``. Inference never had a route; verification
has exactly Harbor's normal one.

CONSTRAINTS:
* ``main`` declares explicit ``networks`` so Harbor never splices its egress
  sidecar (docker.py ``_egress_controlled_service_names``); the task's
  network_mode stays public so Harbor does not require dynamic policy support.
* The compose override is generated per run; names are derived from the
  validated run id and never from task input.
* Every host port is a positive int; the gate listens on the SAME numbers so a
  container URL ``http://bench-gate:<port>`` names exactly one host service.
* Agent-phase package egress (``../package_egress.py``): the gate also serves
  an exact-host forward proxy on ``PACKAGE_PROXY_PORT`` (never in the relayed
  port map), allowlist frozen in ``package-egress-allowlist.json``, model hosts
  always denied, one row per connection into the run-owned ``log_dir`` -- the
  gate's ONLY writable mount, never mounted into ``main``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

GATE_SERVICE = "bench-gate"
INTERNAL_NETWORK = "bench_internal"
UPLINK_NETWORK = "bench_uplink"
GATE_READY = "/tmp/gate.ready"
TOOLCHAIN = "/opt/bench"
GATE_PYTHON = "/opt/bench/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14"
SUPPORT = "/opt/bench-support"
# Each container role gets these and nothing else from the host.
CA_TARGET = "/run/bench-ca/edge-ca.pem"
# The package proxy's listen port inside the gate. Not a host port: the gate
# itself dials allowlisted registries through its uplink.
PACKAGE_PROXY_PORT = 3128
PACKAGE_ALLOWLIST = "package-egress-allowlist.json"
EGRESS_LOG = "/var/log/bench-egress"
_RUN = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def _port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 65536:
        raise ValueError("gate ports must be integers in 1..65535")
    return value


def gate_url(port: int, path: str = "") -> str:
    """The container-side URL that reaches host port ``port`` through the gate."""
    return f"http://{GATE_SERVICE}:{_port(port)}{path}"


def compose_override(
    run_id: str, ports: list[int], image: str, support_dir: Path, toolchain: Path,
    log_dir: Path,
) -> dict[str, object]:
    """Harbor ``--extra-docker-compose`` body for one paid trial.

    ``support_dir`` holds gate_relay.py (mounted read-only); ``image`` is the
    task's own digest-pinned image, reused so the gate needs no extra pull.
    """
    if not _RUN.fullmatch(run_id):
        raise ValueError("invalid run identifier")
    ports = [_port(p) for p in ports]
    if not ports or len(set(ports)) != len(ports):
        raise ValueError("gate needs unique ports")
    if "@sha256:" not in image:
        raise ValueError("gate image must be the digest-pinned task image")
    if PACKAGE_PROXY_PORT in ports:
        raise ValueError("a host port collides with the package proxy port")
    relay_args = [
        "--package-proxy", str(PACKAGE_PROXY_PORT),
        "--allowlist", f"{SUPPORT}/{PACKAGE_ALLOWLIST}",
        "--log", f"{EGRESS_LOG}/connections.jsonl",
        *(f"{p}={p}" for p in ports),
    ]
    return {
        "services": {
            "main": {
                # Never pull implicitly: the driver pulls the digest-pinned
                # image once, behind the disk gate (2026-11-03 incident: an
                # implicit --platform pull landed below the 21 GiB gate).
                "pull_policy": "never",
                # Explicit networks: Harbor leaves main alone (no sidecar), and
                # the only attached network is internal (no route, no ext DNS).
                "networks": [INTERNAL_NETWORK],
                "depends_on": {GATE_SERVICE: {"condition": "service_healthy"}},
                # A hosts-file NAME only, no route (main has no gateway): the
                # egress canary must reach routing, not fail at DNS (R6 F16).
                "extra_hosts": ["host.docker.internal:host-gateway"],
            },
            GATE_SERVICE: {
                "image": image,
                "platform": "linux/amd64",
                "pull_policy": "never",
                "entrypoint": [GATE_PYTHON, f"{SUPPORT}/gate_relay.py", *relay_args],
                "networks": [INTERNAL_NETWORK, UPLINK_NETWORK],
                # Docker Desktop resolves this to the macOS host's loopback
                # services; Linux engines need the explicit host-gateway entry.
                "extra_hosts": ["host.docker.internal:host-gateway"],
                "read_only": True,
                "tmpfs": ["/tmp:size=1m"],
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "mem_limit": "128m",
                "pids_limit": 64,
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(Path(toolchain).resolve()),
                        "target": TOOLCHAIN,
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": str(Path(support_dir).resolve()),
                        "target": SUPPORT,
                        "read_only": True,
                    },
                    {
                        # Per-connection package log: the trial record's source.
                        "type": "bind",
                        "source": str(Path(log_dir).resolve()),
                        "target": EGRESS_LOG,
                        "read_only": False,
                    },
                ],
                "healthcheck": {
                    "test": ["CMD", GATE_PYTHON, "-c",
                             f"import os,sys; sys.exit(not os.path.exists({GATE_READY!r}))"],
                    "interval": "1s",
                    "timeout": "5s",
                    "retries": 30,
                },
                "labels": {"bench.run_id": run_id, "bench.role": "gate"},
            },
        },
        "networks": {
            INTERNAL_NETWORK: {"internal": True, "labels": {"bench.run_id": run_id}},
            UPLINK_NETWORK: {"labels": {"bench.run_id": run_id}},
        },
    }


def write_override(path: Path, override: dict[str, object]) -> Path:
    # JSON is valid YAML; Harbor's own generated overlays are JSON too.
    path.write_text(json.dumps(override, indent=2) + "\n")
    return path


def support_mounts(support_dir: Path, ca_file: Path | None) -> list[dict[str, object]]:
    """Read-only binds for ``main``: probe/launcher scripts and (LoP) the run CA.

    Read-only matters: a writable launcher or CA bundle would let a model tool
    replace the edge checks or trust its own CA. The CA is bound as a single
    FILE (R6 N6): the edge's run directory also holds its live request log,
    which the agent has no business reading.
    """
    mounts: list[dict[str, object]] = [
        {"type": "bind", "source": str(Path(support_dir).resolve()), "target": SUPPORT,
         "read_only": True}
    ]
    if ca_file is not None:
        if Path(ca_file).name != Path(CA_TARGET).name:
            raise ValueError("mount the edge CA certificate file, not a directory")
        mounts.append({"type": "bind", "source": str(Path(ca_file).resolve()),
                       "target": CA_TARGET, "read_only": True})
    return mounts


# Docker CLI global options that take a value (docker 2x/Desktop). An unknown
# leading option is refused: guessing its arity is how ``--context x run``
# once read ``x`` as the verb (R6 F17 / P17).
_GLOBAL_VALUED = {"--config", "-c", "--context", "-H", "--host", "-l", "--log-level",
                  "--tlscacert", "--tlscert", "--tlskey"}
_GLOBAL_FLAGS = {"-D", "--debug", "--tls", "--tlsverify"}
_COMPOSE_VALUED = {"-f", "--file", "-p", "--project-name", "--profile", "--env-file",
                   "--project-directory", "--ansi", "--progress", "--parallel"}
_COMPOSE_FLAGS = {"--compatibility", "--dry-run", "--all-resources"}
# Verbs that may fetch an image when it is absent locally.
_RUN_VERBS = {("run",), ("create",), ("container", "run"), ("container", "create")}
_COMPOSE_RUN_VERBS = {"up", "run", "create"}
_PULL_VERBS = {("pull",), ("image", "pull"), ("build",), ("image", "build"),
               ("buildx",), ("compose", "pull"), ("compose", "build")}


def _skip_options(argv: list[str], i: int, valued: set[str], flags: set[str]) -> int:
    while i < len(argv) and argv[i].startswith("-"):
        name = argv[i].split("=", 1)[0]
        if name in valued:
            i += 1 if "=" in argv[i] else 2
        elif name in flags:
            i += 1
        else:
            raise ValueError(f"unrecognised docker option before the verb: {name}")
    return i


def _leads_with_pull_never(opts: list[str]) -> bool:
    """True iff the command's options OPEN with ``--pull never``.

    Rig convention, chosen because it needs no knowledge of option arity:
    anywhere later, ``--pull`` might be the container's own argv (after the
    image) or the value of an unknown option, and guessing which is how a
    guard gets bypassed.
    """
    return opts[:2] == ["--pull", "never"] or opts[:1] == ["--pull=never"]


def require_no_implicit_pull(argv: list[str], *, explicit_pull: bool = False) -> list[str]:
    """Refuse a docker invocation that could pull an image implicitly.

    WHY (2026-11-03 incident): ``docker run --platform linux/amd64 <tag>``
    silently pulled a second platform variant of an image already on disk,
    below the disk gate. Every docker invocation on the rig must name
    ``--pull never`` when its verb can fetch an image (``run``/``create``,
    ``compose up/run/create``); ``--platform`` is accepted only alongside it.
    Pulls and builds happen only through the driver's explicit, disk-gated
    ``docker pull`` (``explicit_pull=True``).

    CONSTRAINT: ``--pull never`` must be the FIRST option after the verb
    (``docker run --pull never ...``, ``docker compose up --pull never ...``);
    see ``_leads_with_pull_never``. Global options before the verb are parsed
    against a fixed table, and an unknown one is refused.
    """
    if argv[:1] != ["docker"]:
        raise ValueError("not a docker invocation")
    i = _skip_options(argv, 1, _GLOBAL_VALUED, _GLOBAL_FLAGS)
    rest = argv[i:]
    if not rest:
        raise ValueError("docker invocation without a verb")
    if rest[0] == "compose":
        j = _skip_options(rest, 1, _COMPOSE_VALUED, _COMPOSE_FLAGS)
        verb = tuple(rest[j:j + 1])
        if ("compose", *verb) in _PULL_VERBS:
            raise ValueError("compose pull/build is not allowed on the rig")
        if verb and verb[0] in _COMPOSE_RUN_VERBS:
            if not _leads_with_pull_never(rest[j + 1:]):
                raise ValueError(f"docker compose {verb[0]} must pass --pull never")
        return argv
    verb = tuple(rest[:2]) if rest[0] in {"container", "image"} else tuple(rest[:1])
    opts = rest[len(verb):]
    if verb in _PULL_VERBS:
        if not explicit_pull or verb != ("pull",):
            raise ValueError("pulls happen only through the driver's disk-gated pull")
        return argv
    if verb in _RUN_VERBS:
        if not _leads_with_pull_never(opts):
            raise ValueError(f"docker {' '.join(verb)} must pass --pull never")
    elif any(a == "--platform" or a.startswith("--platform=") for a in opts):
        raise ValueError("--platform only together with --pull never")
    return argv


def run_docker(argv: list[str], *, explicit_pull: bool = False, **kwargs):
    """The ONE way rig code shells out to docker: guard, then ``subprocess.run``.

    Wired into every driver/verifier/oracle docker call (R6 F17); a test
    walks their ASTs so a bare ``subprocess.run(["docker", ...])`` fails CI.
    """
    return subprocess.run(require_no_implicit_pull(argv, explicit_pull=explicit_pull), **kwargs)


# Support files the gate and the trial need, copied into one per-run directory.
# Paths are relative to coding/; each is copied flat (by basename).
SUPPORT_FILES = ("gate_relay.py", "gold_probe.py", "lop_launch.py", "../package_egress.py",
                 PACKAGE_ALLOWLIST)
