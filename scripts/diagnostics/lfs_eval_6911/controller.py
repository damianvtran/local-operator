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
NODE = "tests/unit/tools/test_eval_tool.py::test_render_does_not_block_the_event_loop"


def qualify_rows(rows, reports):
    """Controller calls AFTER full module fixtures and owned-process settlement.
    Identity/isolation/source-binding validation is MANDATORY before this helper;
    this helper is not a replacement controller or a lifecycle proof.
    """
    if len(rows) != 2 or [r['mode'] for r in rows] != ['baseline', 'noinject'] or [r['token'] for r in rows] != [1, 2]:
        raise ValueError('EVAL missing/order/duplicate STOP')
    if len(reports) != 6 or [r['phase'] for r in reports] != ['setup', 'call', 'teardown'] * 2:
        raise ValueError('EVAL phases STOP')
    if any(r['outcome'] != 'passed' for r in reports if r['phase'] != 'call'):
        raise ValueError('EVAL fixture STOP')
    for i, r in enumerate(rows):
        if r['injected_calls'] != 1 or r['active'] != 0 or r['restored'] is not True:
            raise ValueError('EVAL injection cardinality/cleanup STOP')
        if (r['delegated'], r['skipped']) != ((1, 0) if i == 0 else (0, 1)):
            raise ValueError('EVAL mutation STOP')
        if len(r['spill_threads']) != 1 or r['spill_threads'][0] == r['loop_thread']:
            raise ValueError('EVAL render ran inline STOP')
        if not r['call_pass'] and not r['heartbeat_assertion']:
            raise ValueError('EVAL non-heartbeat failure STOP')
        if reports[3 * i + 1]['outcome'] != ('passed' if r['call_pass'] else 'failed'):
            raise ValueError('EVAL call receipt mismatch STOP')
    if all(r['call_pass'] for r in rows):
        return 'INCONCLUSIVE_NOT_EXONERATED'
    if all(not r['call_pass'] for r in rows):
        return 'INJECTION_NOT_NECESSARY_IN_THIS_COHORT_ROOT_UNKNOWN'
    return 'N1_VARIANT_DIFFERENCE_NOT_CAUSAL_PROOF'

def qualify(proof, expected):
    if set(proof) != {'rows', 'reports', 'exitstatus'}:
        raise ValueError('EVAL protocol keys')
    allowed = {'mode','token','injected_calls','delegated','skipped','active','loop_thread',
               'spill_threads','restored','call_pass','assertion','heartbeat_assertion','identity'}
    for row in proof['rows']:
        if set(row) != allowed:
            raise ValueError('EVAL row schema')
        validate_identity(row['identity'], expected)
        for key in ('token','injected_calls','delegated','skipped','active','loop_thread'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('EVAL numeric schema')
        if not isinstance(row['spill_threads'], list) or len(row['spill_threads']) > 1 or any(type(x) is not int for x in row['spill_threads']):
            raise ValueError('EVAL thread schema')
        for key in ('restored','call_pass','assertion','heartbeat_assertion'):
            if type(row[key]) is not bool:
                raise ValueError('EVAL boolean schema')
    for report in proof['reports']:
        if set(report) != {'phase','outcome'} or report['phase'] not in ('setup','call','teardown') or report['outcome'] not in ('passed','failed','skipped'):
            raise ValueError('EVAL report schema')
    result = qualify_rows(proof['rows'], proof['reports'])
    expected_rc = int(any(not row['call_pass'] for row in proof['rows']))
    if proof['exitstatus'] != expected_rc:
        raise ValueError('EVAL pytest status')
    return result


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for label, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / label).read_bytes()).hexdigest() != sha:
            raise SystemExit("source STOP")
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lfs-eval-"))
    known, pgid = {}, os.getpgrp()
    expected = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        ppid=os.getpid(), pgid=pgid, sources={p:manifest["sources"][p] for p in manifest["loaded_sources"]})
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work),
        XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
        XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
        LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", TERM="xterm-256color",
        LFS_EVAL_ROOT=str(work), LFS_EVAL_PROOF=str(work / "proof.json"),
        LFS_EVAL_EXPECTED=json.dumps(expected))
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
        result["rows"] = [{k:r[k] for k in ("mode", "token", "injected_calls", "delegated", "skipped", "active", "loop_thread", "spill_threads", "restored", "call_pass", "assertion", "heartbeat_assertion", "identity")} for r in proof["rows"]]
        shutil.rmtree(work)
    try:
        child = subprocess.Popen([sys.executable,"-B","-m","pytest",NODE,
            "-n0","-q","--tb=no","-o","addopts=",
            "-p","tests.unit.tools.lfs_eval_plugin_6911"],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            expected["pid"] = child.pid
            if os.getpgid(child.pid) != pgid:raise RuntimeError("group STOP")
            total, deadline = 0, time.monotonic()+45
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
        (out/"eval.json").write_text(json.dumps(result)+"\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__=="__main__":main()
