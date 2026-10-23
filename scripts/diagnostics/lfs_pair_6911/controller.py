"""One bounded relay QA allocation. Stdlib controller, real probe OFFHOST only.

run_bounded owns this process group; the pytest child inherits it. We never
print pytest bytes, even on errors, and delete synthetic roots only after wait.
Other three flake-ticket falsifiers are NOT implemented by this partial lane.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for relative, expected in manifest["sources"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit("source identity STOP")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    child = None
    reaped = False
    receipts = None
    result = {"kind": "pair-diagnostic", "executed": False, "reaped": False,
              "timeout": False, "output_overflow": False,
              "python": list(sys.version_info[:3]),
              "executable_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
              "sources": manifest["sources"]}
    # Do not use TemporaryDirectory: its implicit exit deletes live files if wait
    # fails. Any unsettled child leaves the root intact for runner disposal.
    work = Path(tempfile.mkdtemp(prefix="lfs-pair-"))
    try:
        env = {"PATH": os.environ["PATH"], "HOME": str(work),
               "LOCAL_OPERATOR_CONFIG_DIR": str(work / "config"),
               "PYTHONDONTWRITEBYTECODE": "1", "TERM": "xterm-256color",
               "LFS_PAIR_RECEIPTS": str(work / "receipts")}
        args = [sys.executable, "-B", "-m", "pytest", str(HERE / "probe.py"),
                "-n0", "-q", "--tb=no", "-o", "addopts=", "--maxfail=1"]
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)  # No setsid: owned PG.
        try:
            result["executed"] = True
            total = 0
            deadline = time.monotonic() + 60
            if os.getpgid(child.pid) != os.getpgrp():
                raise RuntimeError("unowned child process group")
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        result["timeout"] = True
                        raise RuntimeError("child timeout")
                    for key, _ in selector.select(0.1):
                        chunk = os.read(key.fileobj.fileno(), 16384)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        total += len(chunk)
                        if total > 262144:
                            result["output_overflow"] = True
                            raise RuntimeError("output cap")
        finally:
            # Healthy wait first, then exact child kill and unconditional final wait.
            try:
                rc = child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                rc = child.wait(timeout=4)
            reaped = True
            result.update(rc=rc, reaped=True, output_bytes=locals().get("total", 0))
        if rc != 0 or result["timeout"] or result["output_overflow"]:
            raise RuntimeError("probe failure")
        names = ["old-held", "new-release", "new-omission"]
        paths = sorted((work / "receipts").glob("*.json"))
        if sorted(p.name for p in paths) != sorted(n + ".json" for n in names):
            raise RuntimeError("missing/extra variants")
        receipts = []
        allowed = {"variant", "observation_reached", "target_pass", "failure",
                   "append_done", "release_thread_joined", "wall_s", "wait_s"}
        for name in names:
            path = work / "receipts" / (name + ".json")
            if path.stat().st_size > 4096:
                raise RuntimeError("receipt bound")
            row = json.loads(path.read_text())
            if set(row) != allowed or row["variant"] != name:
                raise RuntimeError("receipt schema")
            if row["observation_reached"] is not True or row["release_thread_joined"] is not True:
                raise RuntimeError("incomplete proof")
            if row["target_pass"] is not (name == "new-release"):
                raise RuntimeError("wrong oracle")
            if row["failure"] != (None if name == "new-release" else "original_member_admitted_assertion"):
                raise RuntimeError("wrong failure")
            if name == "new-omission" and (not isinstance(row["wait_s"], (int, float)) or row["wait_s"] < 5):
                raise RuntimeError("missing timeout witness")
            receipts.append(row)
        result["variants"] = receipts
        result["qualified"] = True
    except Exception:
        # No exception text, repr, locals or raw pytest output crosses the boundary.
        result["qualified"] = False
    finally:
        # A failed final wait remains unsettled. No second reap allowance and no
        # root deletion; run_bounded still owns the whole group and its backstop.
        result["reaped"] = reaped
        if reaped:
            import shutil
            shutil.rmtree(work)
        result["synthetic_root_removed"] = reaped and not work.exists()
        (out / "pair.json").write_text(json.dumps(result) + "\n")
    raise SystemExit(0 if result.get("qualified") and result["synthetic_root_removed"] else 1)


if __name__ == "__main__":
    main()
