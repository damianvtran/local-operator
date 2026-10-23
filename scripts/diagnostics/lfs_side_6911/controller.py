"""Two ORIGINAL selectors in one serial child, 30+5 TOTAL, offhost only.
Reuse reviewed PAIR lifecycle/owned-set/identity functions unchanged by import.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table, validate_identity

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NODE = "tests/unit/tui/test_sidebar_repaint.py::test_identical_catalog_does_not_invalidate_rendered_frame"


def qualify(proof, expected):
    rows, reports = proof["rows"], proof["reports"]
    if [r["variant"] for r in rows] != ["baseline", "render"] or [r["token"] for r in rows] != [1, 2]:
        raise ValueError("missing/duplicate variant")
    if len(reports) != 6 or [r["phase"] for r in reports] != ["setup", "call", "teardown"] * 2:
        raise ValueError("missing phases")
    if any(not r["node"] or (r["phase"] != "call" and r["outcome"] != "passed") for r in reports):
        raise ValueError("setup/teardown STOP")
    for index, row in enumerate(rows):
        validate_identity(row["identity"], expected)
        if row["prepare_calls"] != 1 or row["renders"] != index or row["restored"] is not True:
            raise ValueError("instrumentation STOP")
        if type(row["count"]) is not int or not 0 <= row["count"] <= 50:
            raise ValueError("count STOP")
        for stage in ("before", "after"):
            state = row[stage]
            if set(state) != {"baseline_missing", "changed_positions"} or type(state["baseline_missing"]) is not bool:
                raise ValueError("paint schema")
            if any(type(i) is not int or not 0 <= i < 64 for i in state["changed_positions"]):
                raise ValueError("paint positions")
    baseline, forced = rows
    if not forced["call_pass"] or forced["count"] != 0 or reports[4]["outcome"] != "passed":
        raise ValueError("counterfactual failure")
    if baseline["call_pass"] and baseline["count"] == 0 and proof["exitstatus"] == 0:
        return "INCONCLUSIVE_NOT_EXONERATED"
    if baseline["assertion"] and baseline["count"] == 50 and reports[1]["outcome"] == "failed" and proof["exitstatus"] == 1:
        return "STALE_RENDER_BASELINE_CANDIDATE"
    raise ValueError("unclassified baseline")


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for label, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / label).read_bytes()).hexdigest() != sha:
            raise SystemExit("source STOP")
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lfs-side-"))
    known, pgid = {}, os.getpgrp()
    expected = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        ppid=os.getpid(), pgid=pgid, sources={p:manifest["sources"][p] for p in manifest["loaded_sources"]})
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work),
        XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
        XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
        LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", TERM="xterm-256color",
        LFS_SIDE_ROOT=str(work), LFS_SIDE_PROOF=str(work / "proof.json"),
        LFS_SIDE_EXPECTED=json.dumps(expected))
    result, lifecycle = dict(qualified=False), {}
    def observe():return owned(known, pgid)
    def kill_owned():
        for pid, facts in observe().items():
            fresh = table().get(pid)
            if fresh and fresh["start"] == facts["start"]:os.kill(pid, 9)
    def remove():
        path = work / "proof.json"
        if path.stat().st_size > 32768:raise ValueError("proof cap")
        proof = json.loads(path.read_text())
        result["verdict"] = qualify(proof, expected)
        # Strict structural projection, not wholesale plugin output forwarding.
        result["rows"] = [{k:r[k] for k in ("variant", "token", "count", "prepare_calls", "renders", "assertion", "call_pass", "restored", "before", "after", "identity")} for r in proof["rows"]]
        shutil.rmtree(work)
    try:
        child = subprocess.Popen([sys.executable,"-B","-m","pytest",NODE,
            "-n0","-q","--tb=no","-o","addopts=",
            "-p","tests.unit.tui.lfs_side_plugin_6911"],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            expected["pid"] = child.pid
            if os.getpgid(child.pid) != pgid:raise RuntimeError("group STOP")
            total, deadline = 0, time.monotonic()+30
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout,selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() >= deadline:raise RuntimeError("time STOP")
                    observe()
                    for key,_ in selector.select(0.1):
                        chunk=os.read(key.fileobj.fileno(),16384)
                        if not chunk:selector.unregister(key.fileobj)
                        total+=len(chunk)
                        if total>262144:raise RuntimeError("output STOP")
        finally:
            lifecycle=finish_owned(child,observe,kill_owned,remove)
        result["qualified"]=not lifecycle["errors"] and lifecycle["synthetic_root_removed"] and lifecycle["rc"] in (0,1)
    except Exception:
        result["qualified"]=False
    finally:
        result["lifecycle"]=lifecycle
        (out/"side.json").write_text(json.dumps(result)+"\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__=="__main__":main()
