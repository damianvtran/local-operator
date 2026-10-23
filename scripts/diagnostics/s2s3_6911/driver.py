"""One off-host allocation, four serial children in run_bounded's owned PG.

No local use authorized. All cleanup is settlement-gated; no TemporaryDirectory
finalizer can delete a sandbox while a child or tracked descendant still lives.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from scripts.diagnostics.s2s3_6911.oracle import CHILD_SECONDS, MODES, final, judge, finish_child, project

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table():
    rows = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            text = (entry / "stat").read_text()
            fields = text[text.rindex(")")+2:].split()
            rows[int(entry.name)] = dict(state=fields[0], ppid=int(fields[1]),
                                         pgid=int(fields[2]), start=fields[19])
        except (OSError, ValueError):
            if entry.exists():
                raise RuntimeError("process observation unknown")
    return rows


def owned(known, pgid):
    rows = table()
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if pid != os.getpid() and pid not in known and (row["pgid"] == pgid or row["ppid"] in known):
                known[pid] = row["start"]
                changed = True
    return {pid: row for pid, row in rows.items()
            if known.get(pid) == row["start"] and row["state"] != "Z"}


def case(mode, out):
    home = Path(tempfile.mkdtemp(prefix=mode+"-", dir=out))
    proof = home / "proof.json"
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "TMPDIR": str(home),
           "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
           "LOCAL_OPERATOR_CONFIG_DIR": str(home / "config"),
           "XDG_CONFIG_HOME": str(home / "config"), "XDG_CACHE_HOME": str(home / "cache"),
           "S2S3_MODE": mode, "S2S3_OUT": str(proof), "PYTEST_ADDOPTS": ""}
    command = [sys.executable, "-B", "-m"]
    if mode == "synthetic":
        command += ["scripts.diagnostics.s2s3_6911.plugin"]
    else:
        node = "test_s2_send_to_agent_end_keeps_the_loop_under_the_bar" if mode == "s2-original" else "test_s3_harvest_with_unlisted_model_and_cold_memo_keeps_the_loop_free"
        command += ["pytest", "tests/unit/test_tui_responsiveness.py::"+node,
                    "-q", "-n0", "-p", "scripts.diagnostics.s2s3_6911.plugin", "--tb=short"]
    row = dict(mode=mode, rc=None, signal=None, reaped=False, timed_out=False,
               cleanup_errors=[], identity_ok=False)
    child = None
    known = {}
    pgid = os.getpgrp()
    data = bytearray()
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)  # INHERITS owned PG, no setsid.
        row.update(pid=child.pid, pgid=pgid)
        deadline = time.monotonic()+CHILD_SECONDS
        with selectors.DefaultSelector() as select:
            select.register(child.stdout, selectors.EVENT_READ)
            while select.get_map():
                owned(known, pgid)
                if time.monotonic() >= deadline:
                    row["timed_out"] = True
                    raise RuntimeError("child bound")
                for key, _ in select.select(0.1):
                    chunk = os.read(key.fileobj.fileno(), 16384)
                    if not chunk:
                        select.unregister(key.fileobj)
                    else:
                        data.extend(chunk)
                        if len(data) > 262144:
                            raise RuntimeError("child output bound")
    finally:
        if child is not None:
            def kill_owned():
                for pid, facts in owned(known, pgid).items():
                    fresh = table().get(pid)
                    if fresh and fresh["start"] == facts["start"]:
                        os.kill(pid, signal.SIGKILL)
            def parse():
                if proof.stat().st_size > 200000:
                    raise ValueError("proof bound")
                return json.loads(proof.read_text())
            row.update(finish_child(child, lambda: owned(known, pgid), kill_owned,
                                    parse, lambda: shutil.rmtree(home),
                                    fault=row["timed_out"]))
        row["signal"] = -row["rc"] if type(row["rc"]) is int and row["rc"] < 0 else None
        row["identity_ok"] = row.get("data", {}).get("identity") == project(identity()) if row.get("data") else False
        # NO raw pytest output file, forwarding, traceback or arbitrary error text.
        (out / (mode+".json")).write_text(json.dumps(project(row)))
    row["verdict"] = judge(row)
    return row


def identity(require_loaded=False, original_module=None):
    import importlib.metadata
    manifest = json.loads((HERE / "manifest.json").read_text())
    actual = {path: sha(ROOT/path) for path in manifest["sources"]}
    if actual != manifest["sources"] or sys.version_info[:3] != (3, 13, 16):
        raise RuntimeError("source/Python identity mismatch")
    textual = importlib.metadata.version("textual")
    if not manifest.get("textual_pin") or textual != manifest["textual_pin"]:
        raise RuntimeError("Textual exact canonical pin unresolved or changed")
    modules = {"tests/unit/test_tui_responsiveness.py": "tests.unit.test_tui_responsiveness",
               "tests/loop_callback_spans.py": "tests.loop_callback_spans",
               "scripts/diagnostics/s2s3_6911/plugin.py": "scripts.diagnostics.s2s3_6911.plugin"}
    loaded = {}
    for path, name in modules.items():
        expected = (ROOT/path).resolve()
        if require_loaded:
            module = original_module if path == "tests/unit/test_tui_responsiveness.py" and original_module is not None else sys.modules.get(name)
            if path.endswith("/plugin.py") and module is None:
                module = sys.modules.get("__main__")
            observed = Path(module.__file__).resolve() if module is not None else None
            if observed != expected:
                raise RuntimeError("actual loaded module path mismatch")
        loaded[path] = dict(path=path, sha256=sha(expected))
    return dict(python=list(sys.version_info[:3]), textual=textual, sources=actual,
                loaded=loaded, executable=str(Path(sys.executable).resolve()),
                executable_sha256=sha(Path(sys.executable).resolve()),
                versions={n: importlib.metadata.version(n) for n in ("textual", "pytest", "pytest-asyncio", "pydantic")})


def main():
    identity()  # No app/case dispatch until exact closure identity is qualified.
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        rows.append(case(mode, out))
    result = final(rows)
    (out / "result.json").write_text(json.dumps(result))
    raise SystemExit(1 if result["original_failure"] else 0)


if __name__ == "__main__":
    main()
