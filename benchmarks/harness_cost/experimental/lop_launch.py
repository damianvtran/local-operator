"""Launch only the released, unmodified Local Operator v0.94.22 source install.

WHY: the benchmark measures the product as shipped. It pins the released
v0.94.22 commit (``500883ef…``) by FULL SHA — never the unreleased
``07c324e9`` thinking-replay branch the DEV qualification used — and refuses a
tree with any tracked change or untracked file (an untracked
``sitecustomize.py`` or module is importable, so it is a modification too).

No product code is patched. The former ``lop_shim`` rebinding of
``clients.ANTHROPIC_API_URL`` is gone: Local Operator talks to its real
first-party URLs, and the per-run edge (``lop_edge.py``) meters them at the
network boundary through the inherited ``HTTPS_PROXY`` + run-CA environment.
This launcher therefore refuses to start unless that environment names a
loopback edge with a run identity and a CA bundle, so an unmetered launch
cannot happen by omission (the kernel sandbox is the second, independent
backstop).

Interpreter and root are supplied by the caller; source identity is checked
before any CLI import.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

REVIEWED_REVISION = "500883ef56ca504325ffd80c5b90bf2435d00e3f"
NO_PROXY = "127.0.0.1,localhost,::1"


def _require_edge_environment() -> None:
    """Refuse unless every paid client will route through this run's edge.

    urllib's ``getproxies()`` (which httpx and requests use) gives LOWERCASE
    names precedence, even when empty, so each lowercase spelling must be
    absent or identical to its uppercase one; ``ALL_PROXY`` (a SOCKS escape
    hatch) is refused in either spelling.
    """
    for upper in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        lower = upper.lower()
        if lower in os.environ and os.environ[lower] != os.environ.get(upper):
            raise SystemExit(f"{lower} conflicts with {upper}; the edge would be bypassed")
    if "ALL_PROXY" in os.environ or "all_proxy" in os.environ:
        raise SystemExit("ALL_PROXY must not be set for a metered run")
    proxy = urlparse(os.environ.get("HTTPS_PROXY", ""))
    if (
        proxy.scheme != "http"
        or proxy.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not proxy.port
        or not proxy.username
        or not proxy.password
    ):
        raise SystemExit("HTTPS_PROXY must name this run's loopback metering edge")
    if "HTTP_PROXY" in os.environ and os.environ["HTTP_PROXY"] != os.environ["HTTPS_PROXY"]:
        raise SystemExit("HTTP_PROXY must name the same edge as HTTPS_PROXY")
    bundle = os.environ.get("SSL_CERT_FILE", "")
    if not bundle or os.environ.get("REQUESTS_CA_BUNDLE") != bundle or not Path(bundle).is_file():
        raise SystemExit("SSL_CERT_FILE and REQUESTS_CA_BUNDLE must name the run CA bundle")
    if os.environ.get("NO_PROXY") != NO_PROXY:
        # Only loopback (the bench MCP server) may bypass the edge.
        raise SystemExit("NO_PROXY must exempt loopback only")


def main() -> None:
    _require_edge_environment()
    python = os.environ.get("LOP_BENCH_PYTHON")
    if not python:
        raise SystemExit("LOP_BENCH_PYTHON must select the reviewed source interpreter")
    if Path(python).absolute() != Path(sys.executable).absolute():
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    root = Path(os.environ["LOP_BENCH_SOURCE_ROOT"]).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != REVIEWED_REVISION:
        raise SystemExit("benchmark source is not reviewed revision " + REVIEWED_REVISION)
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip():
        raise SystemExit("benchmark source must have no modifications or untracked files")
    spec = importlib.util.find_spec("local_operator")
    if not spec or not spec.origin or Path(spec.origin).resolve().parent != root / "local_operator":
        raise SystemExit("selected interpreter does not import the reviewed source install")
    sys.argv = ["lop", *sys.argv[1:]]
    runpy.run_module("local_operator.cli", run_name="__main__")


if __name__ == "__main__":
    main()
