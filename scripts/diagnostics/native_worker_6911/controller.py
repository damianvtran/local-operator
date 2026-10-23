"""Generic native Worker producer, one child 30+5 TOTAL, offhost only.
Reuse reviewed PAIR lifecycle/owned-set/identity functions unchanged by import.
"""
import hashlib
import math
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



def qualify(proof, expected):
    if set(proof) != {'identity','native_sources','compiled_producer','loaded','data','executor_settled','threads_settled'}:
        raise ValueError('native proof schema STOP')
    validate_identity(proof['identity'], expected['identity'])
    if proof['native_sources'] != expected['native_sources'] or proof['compiled_producer'] != expected['compiled_producer'] or proof['loaded'] != expected['identity']['sources']:
        raise ValueError('native identity STOP')
    if proof['executor_settled'] is not True or proof['threads_settled'] is not True:
        raise ValueError('native settlement STOP')
    data = proof['data']
    if set(data) != {'quality','rows','windows','cpu_results','wait_wall_ms','offload','cancel_seen','error_seen','ready','trailing','restored','overhead_observations'}:
        raise ValueError('native result keys STOP')
    if any(data[k] is not True for k in ('cancel_seen','error_seen','ready','trailing','restored')):
        raise ValueError('native control STOP')
    q = data['quality']
    if set(q) != {'qualified','unknown','overflow','unsupported','eager'} or q['qualified'] is not True or any(type(q[k]) is not int or q[k] != 0 for k in q if k != 'qualified'):
        raise ValueError('native quality STOP')
    labels = set(expected['labels']); phases = {'one-before','one-after','two-before','two-after','wait-before','wait-after','offload-before','offload-after','cancel-enter','error-enter'}
    if len(data['rows']) != 10 or {r['phase'] for r in data['rows']} != phases:
        raise ValueError('native witness STOP')
    def number(v):
        if type(v) not in (int,float) or not math.isfinite(v) or v < 0:
            raise ValueError('native metric STOP')
    for row in data['rows']:
        if set(row) != {'phase','label','thread'} or row['label'] not in labels:
            raise ValueError('native private/unknown label STOP')
        number(row['thread'])
    if not 1 <= len(data['windows']) <= 257:
        raise ValueError('native window cap STOP')
    observed = set()
    for window in data['windows']:
        if set(window) != {'callbacks','cpu_ms','wall_ms','worker_spans'} or len(window['worker_spans']) > 5:
            raise ValueError('native window schema STOP')
        for k in ('callbacks','cpu_ms','wall_ms'): number(window[k])
        for span in window['worker_spans']:
            if set(span) != {'label','cpu_ms','wall_ms'} or span['label'] not in labels:
                raise ValueError('native span privacy STOP')
            observed.add(span['label']); number(span['cpu_ms']); number(span['wall_ms'])
    by = {r['phase']:r['label'] for r in data['rows']}; roots = {by[p+'-before'] for p in ('one','two','wait','offload')}
    if len(roots) != 4 or not roots <= observed:
        raise ValueError('empty/four native IDs STOP')
    def burn(v):
        if set(v) != {'thread','cpu_ms'}: raise ValueError('CPU schema STOP')
        number(v['thread']); number(v['cpu_ms'])
    if len(data['cpu_results']) != 2 or any(len(g) != 2 for g in data['cpu_results']):
        raise ValueError('CPU count STOP')
    for group in data['cpu_results']:
        for v in group: burn(v)
    burn(data['offload']); number(data['wait_wall_ms'])
    overhead = data['overhead_observations']
    if len(overhead) != 2 or [v['enabled'] for v in overhead] != [False,True]:
        raise ValueError('annotation cohort STOP')
    for v in overhead:
        if set(v) != {'enabled','requested_callbacks','observed_callbacks','outer_cpu_ms','outer_wall_ms'} or v['requested_callbacks'] != 128 or v['observed_callbacks'] < 128:
            raise ValueError('annotation observation STOP')
        for k in ('requested_callbacks','observed_callbacks','outer_cpu_ms','outer_wall_ms'): number(v[k])
    return 'GENERIC_NATIVE_PRODUCER_QUALIFIED_NOT_S2_OR_PRODUCT_EXONERATION'


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for label, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / label).read_bytes()).hexdigest() != sha:
            raise SystemExit("source STOP")
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="native-worker-"))
    known, pgid = {}, os.getpgrp()
    expected = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        ppid=os.getpid(), pgid=pgid, sources={p:manifest["sources"][p] for p in manifest["loaded_sources"]})
    from scripts.diagnostics.native_worker_6911.carrier import native_sources, compiled_hash
    from tests.worker_code_identity import source_allowlist
    expected = dict(identity=expected, sources=manifest['sources'], native_sources=native_sources(),
        compiled_producer=compiled_hash(), labels=list(source_allowlist(ROOT,
        ('scripts/diagnostics/native_worker_6911/producer.py',)).values()))
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work),
        XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
        XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
        LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", TERM="xterm-256color",
        NATIVE_WORKER_ROOT=str(work), NATIVE_WORKER_PROOF=str(work / "proof.json"),
        NATIVE_WORKER_EXPECTED=json.dumps(expected))
    result, lifecycle = dict(qualified=False), {}
    def observe():return owned(known, pgid)
    def kill_owned():
        for pid, facts in observe().items():
            fresh = table().get(pid)
            if fresh and fresh["start"] == facts["start"]:os.kill(pid, 9)
    def remove():
        path = work / "proof.json"
        if path.stat().st_size > 262144:raise ValueError("proof cap")
        proof = json.loads(path.read_text())
        result["verdict"] = qualify(proof, expected)
        # Strict structural projection, not wholesale plugin output forwarding.
        result['producer'] = proof['data']
        result['identity'] = proof['identity']
        shutil.rmtree(work)
    try:
        child = subprocess.Popen([sys.executable,"-B","-m","scripts.diagnostics.native_worker_6911.carrier"],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            expected["identity"]["pid"] = child.pid
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
        result["qualified"]=not lifecycle["errors"] and lifecycle["synthetic_root_removed"] and lifecycle["rc"] == 0
    except Exception:
        result["qualified"]=False
    finally:
        result["lifecycle"]=lifecycle
        (out/"native.json").write_text(json.dumps(result)+"\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__=="__main__":main()
