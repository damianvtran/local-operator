"""PAIR-only offhost controller; run_bounded owns the inherited process group.

Lifecycle pattern follows s2s3_6911.finish_child: healthy wait, unconditional
final wait, observed owned-set settlement, then (and ONLY then) root removal.
Pure injected seams below never inspect the kernel unless main supplies them.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = "tests/unit/network/test_lfs_pair_diag_6911.py"
VARIANTS = ("old-held", "new-release", "new-omission")


def finish_owned(child, observe, kill_owned, remove):
    """Never let a kill/observation/wait error skip final wait or permit deletion."""
    errors, rc = [], None
    try:
        try:
            rc = child.wait(timeout=1)
        except (TimeoutError, subprocess.TimeoutExpired):
            pass
        except Exception:
            errors.append("healthy-wait")
    finally:
        try:
            if child.poll() is None or observe():
                kill_owned()
        except Exception:
            errors.append("owned-kill-or-observe")
            try:
                child.kill()
            except Exception:
                errors.append("direct-kill")
        finally:
            try:
                rc = child.wait(timeout=4)
            except Exception:
                errors.append("final-wait")
    settled = False
    try:
        settled = type(rc) is int and not observe()
    except Exception:
        errors.append("final-observe")
    if not settled:
        errors.append("owned-unsettled")
    removed = False
    if settled and not errors:
        try:
            remove()
            removed = True
        except Exception:
            errors.append("remove")
    return dict(rc=rc, reaped=type(rc) is int, owned_settled=settled,
                errors=errors, synthetic_root_removed=removed)


def cleanup_steps(actions):
    """Release/join/wrapper-exit/restoration/receipt all get their attempt."""
    errors = []
    for label, action in actions:
        try:
            if action() is False:
                errors.append(label)
        except Exception:
            errors.append(label)
    return errors


def table():
    rows = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            text = (entry / "stat").read_text()
            fields = text[text.rindex(")") + 2:].split()
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
    # Zombies are not settled: require disappearance, not merely non-running.
    return {pid: row for pid, row in rows.items() if known.get(pid) == row["start"]}


def validate_identity(identity, expected):
    if not isinstance(identity, dict) or identity != expected:
        raise ValueError("loaded identity STOP")


def read_receipts(work, expected):
    """Keep bounded AVAILABLE proof even when pytest rc is unexpectedly nonzero."""
    result = []
    for name in VARIANTS:
        path = work / "receipts" / (name + ".json")
        if not path.exists():
            continue
        if path.stat().st_size > 8192:
            raise ValueError("receipt bound")
        row = json.loads(path.read_text())
        keys = {"variant", "observation_reached", "target_pass", "failure", "append_done",
                "wrapper_exited", "release_thread_joined", "wall_s", "wait_s", "errors", "identity"}
        if set(row) != keys or row["variant"] != name:
            raise ValueError("receipt schema")
        validate_identity(row["identity"], expected)
        for key in ("observation_reached", "target_pass", "append_done", "wrapper_exited", "release_thread_joined"):
            if type(row[key]) is not bool:
                raise ValueError("receipt type")
        labels = {"release", "waiter", "producer-exit", "keys", "restore"}
        if not isinstance(row["errors"], list) or any(e not in labels for e in row["errors"]):
            raise ValueError("cleanup labels")
        if row["failure"] not in (None, "original_member_admitted_assertion", "unexpected"):
            raise ValueError("failure label")
        for key in ("wall_s", "wait_s"):
            if row[key] is not None and (type(row[key]) not in (float, int) or not 0 <= row[key] <= 65):
                raise ValueError("clock bound")
        result.append(row)
    if any(p.name not in {n + ".json" for n in VARIANTS} for p in (work / "receipts").glob("*.json")):
        raise ValueError("extra receipt")
    return result


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for path, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha:
            raise SystemExit("source identity STOP")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lfs-pair-"))
    child, known, lifecycle = None, {}, {}
    pgid = os.getpgrp()
    result = dict(qualified=False, stage="setup", variants=[])
    expected = dict(python=list(sys.version_info[:3]),
                    executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                    ppid=os.getpid(), pgid=pgid,
                    sources={p: manifest["sources"][p] for p in manifest["loaded_sources"]}, isolated=True)
    for name in ("tmp", "config", "cache", "data", "runtime"):
        (work / name).mkdir(mode=0o700)
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work / "tmp"),
               XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
               XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
               LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), PYTHONPATH=str(ROOT),
               PYTHONDONTWRITEBYTECODE="1", TERM="xterm-256color",
               LFS_PAIR_ROOT=str(work), LFS_PAIR_RECEIPTS=str(work / "receipts"),
               LFS_PAIR_EXPECTED=json.dumps(expected), PYTEST_ADDOPTS="")
    def observe():
        return owned(known, pgid)
    def kill_owned():
        for pid, row in observe().items():
            current = table().get(pid)
            if current and current["start"] == row["start"]:
                os.kill(pid, 9)
    try:
        args = [sys.executable, "-B", "-m", "pytest", PROBE, "-n0", "-q", "--tb=no",
                "-o", "addopts=", "--maxfail=1"]
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            result["stage"] = "observe"
            expected["pid"] = child.pid
            # Child learns its own PID; comparison is against the Popen PID, not PGID.
            if os.getpgid(child.pid) != pgid:
                raise RuntimeError("unowned group")
            total, deadline = 0, time.monotonic() + 60
            result["stage"] = "probe"
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("timeout")
                    observe()
                    for key, _ in selector.select(0.1):
                        chunk = os.read(key.fileobj.fileno(), 16384)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        total += len(chunk)
                        if total > 262144:
                            raise RuntimeError("output cap")
            result["output_bytes"] = total
        finally:
            # Parse before deletion but AFTER settlement. finish_owned's remove
            # callback performs bounded receipt validation then removes the root.
            def remove():
                result["variants"] = read_receipts(work, expected)
                shutil.rmtree(work)
            lifecycle = finish_owned(child, observe, kill_owned, remove)
        result["stage"] = "qualification"
        rows = result["variants"]
        result["qualified"] = (lifecycle.get("rc") == 0 and not lifecycle["errors"]
            and len(rows) == 3 and all(not row["errors"] and row["observation_reached"]
            and row["wrapper_exited"] and row["release_thread_joined"]
            and row["target_pass"] == (row["variant"] == "new-release")
            and row["failure"] == (None if row["variant"] == "new-release" else "original_member_admitted_assertion")
            and (row["variant"] != "new-omission" or row["wait_s"] >= 5) for row in rows))
    except Exception:
        result["qualified"] = False
    finally:
        # If lifecycle errors prohibited removal, still read safe available records.
        if work.exists() and child is not None:
            try:
                result["variants"] = read_receipts(work, expected)
            except Exception:
                result["receipt_error"] = True
        result["lifecycle"] = lifecycle
        (out / "pair.json").write_text(json.dumps(result) + "\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__ == "__main__":
    main()
