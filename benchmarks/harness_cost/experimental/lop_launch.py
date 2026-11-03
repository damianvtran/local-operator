"""Launch only the explicitly selected, reviewed Local Operator source install.

The benchmark must not silently execute the operator's installed 0.94.22.
Interpreter/root are supplied by the caller; source identity is checked before
any CLI import. The proxy shim stays in this process, not inherited PYTHONPATH,
so task subprocesses keep their ordinary Python environment.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys

REVIEWED_REVISION = "07c324e9"


def main() -> None:
    from urllib.parse import urlparse
    proxy = urlparse(os.environ.get("BENCH_ANTHROPIC_URL", ""))
    if proxy.scheme != "http" or proxy.hostname not in {"127.0.0.1", "localhost", "::1"} or not proxy.port:
        raise SystemExit("BENCH_ANTHROPIC_URL must name the local metering proxy")
    python = os.environ.get("LOP_BENCH_PYTHON")
    if not python:
        raise SystemExit("LOP_BENCH_PYTHON must select the reviewed source interpreter")
    if Path(python).absolute() != Path(sys.executable).absolute():
        os.execv(python, [python, str(Path(__file__).resolve()), *sys.argv[1:]])
    root = Path(os.environ["LOP_BENCH_SOURCE_ROOT"]).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not head.startswith(REVIEWED_REVISION):
        raise SystemExit("benchmark source is not reviewed revision " + REVIEWED_REVISION)
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("benchmark source must have no tracked modifications")
    spec = importlib.util.find_spec("local_operator")
    if not spec or not spec.origin or Path(spec.origin).resolve().parent != root / "local_operator":
        raise SystemExit("selected interpreter does not import the reviewed source install")
    # run_path avoids globally changing import search paths for spawned tooling.
    runpy.run_path(str(Path(__file__).parent / "lop_shim" / "sitecustomize.py"))
    sys.argv = ["lop", *sys.argv[1:]]
    runpy.run_module("local_operator.cli", run_name="__main__")


if __name__ == "__main__":
    main()
