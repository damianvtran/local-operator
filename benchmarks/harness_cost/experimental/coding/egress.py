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
"""

from __future__ import annotations

import json
import re
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
_RUN = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def _port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 65536:
        raise ValueError("gate ports must be integers in 1..65535")
    return value


def gate_url(port: int, path: str = "") -> str:
    """The container-side URL that reaches host port ``port`` through the gate."""
    return f"http://{GATE_SERVICE}:{_port(port)}{path}"


def compose_override(
    run_id: str, ports: list[int], image: str, support_dir: Path, toolchain: Path
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
    relay_args = [f"{p}={p}" for p in ports]
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


def support_mounts(support_dir: Path, ca_dir: Path | None) -> list[dict[str, object]]:
    """Read-only binds for ``main``: probe/launcher scripts and (LoP) the run CA.

    Read-only matters: a writable launcher or CA bundle would let a model tool
    replace the edge checks or trust its own CA.
    """
    mounts: list[dict[str, object]] = [
        {"type": "bind", "source": str(Path(support_dir).resolve()), "target": SUPPORT,
         "read_only": True}
    ]
    if ca_dir is not None:
        mounts.append({"type": "bind", "source": str(Path(ca_dir).resolve()),
                       "target": str(Path(CA_TARGET).parent), "read_only": True})
    return mounts


def require_no_implicit_pull(argv: list[str]) -> list[str]:
    """Refuse a ``docker run``/``create`` that could pull an image implicitly.

    WHY (2026-11-03 incident): ``docker run --platform linux/amd64 <tag>``
    silently pulled a second platform variant of an image already on disk,
    below the disk gate. Every rig/probe docker invocation that names an image
    must carry ``--pull never``; ``--platform`` is accepted only alongside it.
    Pulls happen only through the driver's explicit, disk-gated ``docker pull``.
    """
    if argv[:1] != ["docker"]:
        raise ValueError("not a docker invocation")
    verb = next((a for a in argv[1:] if not a.startswith("-")), "")
    if verb in {"run", "create"} and "--pull" not in argv and "--pull=never" not in argv:
        raise ValueError(f"docker {verb} must pass --pull never")
    if "--pull" in argv and argv[argv.index("--pull") + 1] != "never":
        raise ValueError("--pull must be never")
    if any(a == "--platform" or a.startswith("--platform=") for a in argv) and not (
        "--pull=never" in argv or ("--pull" in argv and argv[argv.index("--pull") + 1] == "never")
    ):
        raise ValueError("--platform only together with --pull never")
    return argv


# Support files the gate and the trial need, copied into one per-run directory.
SUPPORT_FILES = ("gate_relay.py", "gold_probe.py", "lop_launch.py")
