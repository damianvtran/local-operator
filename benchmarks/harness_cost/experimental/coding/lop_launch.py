"""Launch released Local Operator inside a Harbor task container, edge-metered only.

WHY: the benchmark measures released, unmodified Local Operator v0.94.22
(``500883ef56ca504325ffd80c5b90bf2435d00e3f``). That release talks only to its
first-party URLs, so it is metered solely through the per-run TLS edge
(``../lop_edge.py``) that runs on the HOST and holds the real Radient bearer.
Inside the container the only route out is the bench gate (``egress.py``), which
relays the edge's port; the child gets ``HTTPS_PROXY`` naming
``bench-gate:<edge port>`` with this run's identity, a read-only run CA bundle,
and the edge's dummy ``RADIENT_API_KEY``. This is the container counterpart of
``../lop_launch.py`` (the desktop launcher); the two differ only where the
topology does:

* the edge host is the gate service, not loopback (the internal Docker network
  is the egress fence here, replacing the desktop ``sandbox-exec`` profile);
* source identity is the frozen overlay ``/opt/bench/lop-baseline-500883ef``
  (``uv pip install --target`` of the tag), checked by its ``direct_url.json``
  commit, its RECORD hashes for every ``local_operator`` file, and that every
  imported ``local_operator`` module resolves inside it -- there is no git
  checkout in the container to ask.

CONSTRAINT: every check runs before ``local_operator.cli`` is imported, so an
unmetered or wrong-source launch cannot happen by omission. Nothing here reads
or prints a credential; the dummy key is the only one the container ever holds.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from urllib.parse import urlparse

RELEASED_REVISION = "500883ef56ca504325ffd80c5b90bf2435d00e3f"
OVERLAY = Path("/opt/bench/lop-baseline-500883ef")
GATE_HOST = "bench-gate"
CA_BUNDLE = "/run/bench-ca/edge-ca.pem"
# Loopback only: the in-container MCP replay (public-full arm) is the sole
# thing that may bypass the edge, exactly as on the desktop lane.
NO_PROXY = "127.0.0.1,localhost,::1"
# Must equal ``lop_edge.DUMMY_RADIENT_KEY`` (asserted by test_lop_container):
# the edge strips it and injects the host-held bearer on the decisions route.
DUMMY_RADIENT_KEY = 'bench-dummy'


def require_edge_environment(environ: dict[str, str] | os._Environ[str] = os.environ) -> None:
    """Refuse unless every paid client will route through this run's edge.

    Same proxy-spelling rules as the desktop launcher: urllib's
    ``getproxies()`` (httpx/requests) gives LOWERCASE names precedence even
    when empty, so each lowercase spelling must be absent or identical to the
    uppercase one; ``ALL_PROXY`` (a SOCKS escape hatch) is refused.
    """
    for upper in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        lower = upper.lower()
        if lower in environ and environ[lower] != environ.get(upper):
            raise SystemExit(f"{lower} conflicts with {upper}; the edge would be bypassed")
    if "ALL_PROXY" in environ or "all_proxy" in environ:
        raise SystemExit("ALL_PROXY must not be set for a metered run")
    proxy = urlparse(environ.get("HTTPS_PROXY", ""))
    if (
        proxy.scheme != "http"
        or proxy.hostname != GATE_HOST
        or not proxy.port
        or not proxy.username
        or not proxy.password
        or proxy.path not in ("", "/")
    ):
        raise SystemExit("HTTPS_PROXY must name this run's metering edge via the bench gate")
    if "HTTP_PROXY" in environ and environ["HTTP_PROXY"] != environ["HTTPS_PROXY"]:
        raise SystemExit("HTTP_PROXY must name the same edge as HTTPS_PROXY")
    if environ.get("SSL_CERT_FILE") != CA_BUNDLE or environ.get("REQUESTS_CA_BUNDLE") != CA_BUNDLE:
        raise SystemExit("SSL_CERT_FILE and REQUESTS_CA_BUNDLE must name the mounted run CA")
    if not Path(CA_BUNDLE).is_file() or os.access(CA_BUNDLE, os.W_OK):
        # Writable would let a tool swap in its own CA and MITM-bypass the meter.
        raise SystemExit("run CA bundle must be present and read-only")
    if environ.get("NO_PROXY") != NO_PROXY:
        raise SystemExit("NO_PROXY must exempt loopback only")
    if environ.get("RADIENT_API_KEY") != DUMMY_RADIENT_KEY:
        raise SystemExit("RADIENT_API_KEY must be the edge's dummy value")


def _record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def verify_overlay(overlay: Path = OVERLAY) -> int:
    """Prove the overlay is the released tag's bytes; return files verified.

    ``direct_url.json`` names the commit uv installed from; RECORD hashes bind
    every ``local_operator/`` file to that install, so an edited or added module
    is refused (an unlisted ``.py`` under the package is a mismatch too).
    """
    dist = overlay / "local_operator-0.94.22.dist-info"
    direct = json.loads((dist / "direct_url.json").read_text())
    if (direct.get("vcs_info") or {}).get("commit_id") != RELEASED_REVISION:
        raise SystemExit("overlay is not released revision " + RELEASED_REVISION)
    listed: dict[str, str] = {}
    with (dist / "RECORD").open(newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("local_operator/") and row[1]:
                listed[row[0]] = row[1]
    package = overlay / "local_operator"
    present = {
        str(p.relative_to(overlay))
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if not listed or present != set(listed):
        raise SystemExit("overlay local_operator files differ from its RECORD")
    for rel, expected in listed.items():
        if _record_hash(overlay / rel) != expected:
            raise SystemExit("overlay file hash differs from RECORD: " + rel)
    return len(listed)


def verify_imports(overlay: Path = OVERLAY) -> None:
    """Every imported ``local_operator`` module must come from the overlay."""
    import local_operator

    package = (overlay / "local_operator").resolve()
    if [Path(p).resolve() for p in local_operator.__path__] != [package]:
        raise SystemExit("local_operator does not import from the frozen overlay")
    for name, module in tuple(sys.modules.items()):
        if name == "local_operator" or name.startswith("local_operator."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(package):
                raise SystemExit("module imported outside the overlay: " + name)


def export_overlay_path() -> None:
    """PREPEND the overlay to ``PYTHONPATH`` for LoP's tool children (R6 F18).

    WHY PYTHONPATH (not only sys.path): LoP's own re-launches of
    ``/opt/bench/lop/bin/lop`` (forks, subagents) must also import the
    overlay, never the unreleased candidate in the venv's site-packages; this
    is the method the overlay was qualified with (native-first-wire v2).

    SCOPE: this process's environment only, so it reaches LoP and the tool
    processes LoP spawns -- never Harbor, the verifier or other harnesses.
    Prepending keeps an image's own entries (multi-source-data-merger sets
    ``PYTHONPATH=/app``). Recorded side effect: a task Python started by a
    LoP tool also sees the overlay (``local_operator`` + its dist-info).
    """
    overlay = str(OVERLAY)
    rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != overlay]
    os.environ["PYTHONPATH"] = os.pathsep.join([overlay, *rest])


def main() -> None:
    require_edge_environment()
    verify_overlay()
    # The overlay holds only local_operator + its dist-info; third-party deps
    # come from the frozen /opt/bench/lop venv this interpreter belongs to.
    if sys.path[:1] != [str(OVERLAY)]:
        sys.path.insert(0, str(OVERLAY))
    export_overlay_path()
    verify_imports()
    sys.argv = ["lop", *sys.argv[1:]]
    runpy.run_module("local_operator.cli", run_name="__main__")


if __name__ == "__main__":
    main()
