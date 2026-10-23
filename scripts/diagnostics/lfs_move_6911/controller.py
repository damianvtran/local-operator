"""Two original MOVE variants in one serial child, 45+5 TOTAL, offhost only.
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
NODE = "tests/unit/network/test_mobility_promotion_identity.py::test_a_peer_that_answers_the_old_way_cannot_make_this_device_promote"


def validate_row(row, mode, token, outcome):
    if row['mode'] != mode or row['token'] != token or row['pair_calls'] != 1 or row['restored'] is not True:
        raise ValueError('MOVE order/helper/restoration STOP')
    if outcome != ('passed' if row['call_pass'] else 'failed'):
        raise ValueError('MOVE outcome STOP')
    if row['call_pass']:
        if row['move_calls'] != 2 or not all(row[k] for k in ('first_ok', 'request_seen', 'reply_seen')):
            raise ValueError('MOVE vacuous original path STOP')
    elif mode != 'baseline' or row['move_calls'] != 1 or not row['first_unreachable'] or not row['first_assertion']:
        raise ValueError('MOVE unexpected original failure STOP')
    if mode == 'fresh' and (not all(row[k] for k in ('dial_attempted','dial_ok','new_generation','selected_new','correlated','selected_alive','request_seen','reply_seen','call_pass')) or row['selected_closing']):
        raise ValueError('MOVE fresh actual selected transport STOP')


def qualify_rows(rows, reports):
    if len(rows) != 2 or [r['mode'] for r in rows] != ['baseline', 'fresh'] or [r['token'] for r in rows] != [1, 2]:
        raise ValueError('MOVE cardinality/order STOP')
    if len(reports) != 6 or [r['phase'] for r in reports] != ['setup', 'call', 'teardown'] * 2:
        raise ValueError('MOVE phases STOP')
    if any(r['outcome'] != 'passed' for r in reports if r['phase'] != 'call'):
        raise ValueError('MOVE fixtures STOP')
    for i, row in enumerate(rows):
        validate_row(row, ('baseline', 'fresh')[i], i+1, reports[3*i+1]['outcome'])
    baseline, fresh = rows
    if not all(fresh[k] for k in ('dial_attempted', 'dial_ok', 'new_generation', 'selected_new', 'correlated', 'selected_alive', 'request_seen', 'reply_seen', 'call_pass')) or fresh['selected_closing']:
        raise ValueError('MOVE fresh selected transport STOP')
    if baseline['call_pass']:
        return 'INCONCLUSIVE_NOT_EXONERATED'
    # The original staging helper immediately asserts first move ok; the
    # expected negative is THAT failure, not an unrelated later authority guard.
    if baseline['move_calls'] == 1 and baseline['first_unreachable']:
        return 'N1_TRANSPORT_CANDIDATE_NOT_ROOT_CURE'
    raise ValueError('MOVE non-transport baseline failure STOP')

def qualify(proof, expected):
    if set(proof) != {'rows', 'reports', 'exitstatus'}:
        raise ValueError('MOVE protocol keys')
    allowed = {'reply_seen', 'request_seen', 'first_assertion', 'first_ok', 'move_calls', 'call_pass', 'first_unreachable', 'dial_ok', 'correlated', 'restored', 'identity', 'pair_calls', 'selected_alive', 'token', 'dial_attempted', 'selected_closing', 'mode', 'new_generation', 'selected_new'}
    for row in proof['rows']:
        if set(row) != allowed:
            raise ValueError('MOVE row schema STOP')
        validate_identity(row['identity'], expected)
        for key in ('token','pair_calls','move_calls'):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError('MOVE integer schema STOP')
        for key in allowed - {'mode','token','pair_calls','move_calls','identity'}:
            if type(row[key]) is not bool:
                raise ValueError('MOVE boolean schema STOP')
    for report in proof['reports']:
        if set(report) != {'phase','outcome'} or report['phase'] not in ('setup','call','teardown') or report['outcome'] not in ('passed','failed','skipped'):
            raise ValueError('MOVE report schema')
    result = qualify_rows(proof['rows'], proof['reports'])
    expected_rc = int(any(not row['call_pass'] for row in proof['rows']))
    if proof['exitstatus'] != expected_rc:
        raise ValueError('MOVE pytest status')
    return result


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for label, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / label).read_bytes()).hexdigest() != sha:
            raise SystemExit("source STOP")
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lfs-move-"))
    known, pgid = {}, os.getpgrp()
    expected = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        ppid=os.getpid(), pgid=pgid, sources={p:manifest["sources"][p] for p in manifest["loaded_sources"]})
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work),
        XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
        XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
        LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", TERM="xterm-256color",
        LFS_MOVE_ROOT=str(work), LFS_MOVE_PROOF=str(work / "proof.json"),
        LFS_MOVE_EXPECTED=json.dumps(expected))
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
        result["rows"] = [{k:r[k] for k in ('mode', 'token', 'pair_calls', 'move_calls', 'dial_attempted', 'dial_ok', 'new_generation', 'selected_new', 'correlated', 'selected_alive', 'selected_closing', 'request_seen', 'reply_seen', 'first_ok', 'first_unreachable', 'call_pass', 'restored', 'first_assertion', 'identity')} for r in proof['rows']]
        shutil.rmtree(work)
    try:
        child = subprocess.Popen([sys.executable,"-B","-m","pytest",NODE,
            "-n0","-q","--tb=no","-o","addopts=",
            "-p","tests.unit.network.lfs_move_plugin_6911"],cwd=ROOT,env=env,
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
        (out/"move.json").write_text(json.dumps(result)+"\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__=="__main__":main()
