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
    if set(proof) != {'identity','native_sources','producer_source_sha256','compiled_equal','loaded','data','executor_settled','threads_settled'}:
        raise ValueError('native proof schema STOP')
    validate_identity(proof['identity'], expected['identity'])
    if proof['native_sources'] != expected['native_sources'] or (proof['compiled_equal'] is not True or proof['producer_source_sha256'] != expected['producer_source_sha256']) or proof['loaded'] != expected['identity']['sources']:
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


def partial_proof(proof):
    out = {'stage':'UNKNOWN','rows':[], 'reports':[]}
    if type(proof) is not dict:
        return out
    stages = {'SOURCE','IMPORT_PRODUCER','IMPORT_HELPERS','IMPORT_TEXTUAL','REPO_PATHS','NATIVE_PATHS','COMPILED_COMPARE','IDENTITY','PRODUCER','EXECUTOR','COMPLETE'}
    if type(proof.get('stage')) is str and proof['stage'] in stages:
        out['stage'] = proof['stage']
    if type(proof.get('observed')) is dict:
        out['observed'] = {k:v for k,v in proof['observed'].items()
            if k in ('producer_imported','helpers_imported','textual_imported','repo_paths','native_paths','compiled_equal') and type(v) is bool}
    rows, reports = proof.get('rows'), proof.get('reports')
    if type(rows) is list:
        out['row_count_capped'] = min(len(rows), 3)
        for row in rows[:2]:
            if type(row) is not dict: continue
            safe = {}
            for k, v in row.items():
                if k in ('mode',) and v in ('baseline','fresh'): safe[k] = v
                elif k in ('token','pair_calls','move_calls') and type(v) is int and 0 <= v <= 2: safe[k] = v
                elif k in ('dial_attempted','dial_ok','new_generation','selected_new','correlated','selected_alive','selected_closing','request_seen','reply_seen','first_ok','first_unreachable','call_pass','restored','first_assertion') and type(v) is bool: safe[k] = v
            out['rows'].append(safe)
    if type(reports) is list:
        out['report_count_capped'] = min(len(reports), 7)
        for report in reports[:6]:
            if type(report) is dict and report.get('phase') in ('setup','call','teardown') and report.get('outcome') in ('passed','failed','skipped'):
                out['reports'].append({'phase':report['phase'],'outcome':report['outcome']})
    guards = proof.get('guards')
    if type(guards) is list:
        out['guards'] = [g for g in guards[:8] if type(g) is str and g in ('DIAL','SELECTION','CORRELATION','REQUEST','FIXTURE','ROW')]
    if type(proof.get('data')) is dict:
        data = proof['data']
        out['native_counts'] = {k:min(len(data[k]),258) for k in ('rows','windows') if type(data.get(k)) is list}
        out['native_flags'] = {k:data[k] for k in ('ready','trailing','restored','cancel_seen','error_seen') if type(data.get(k)) is bool}
    return out


def finalize_proof(work, result, expected, settled=True, reader=None, deleter=None):
    # ONLY the reviewed common lifecycle may invoke this with settled=True.
    # Proof rejection and actual filesystem deletion are independent outcomes.
    result['full_proof_qualified'] = False
    result['proof_stage'] = 'UNKNOWN'
    result['root_deletion'] = 'NOT_ATTEMPTED'
    if not settled:
        result['proof_stage'] = 'UNSETTLED'
        return
    try:
        try:
            if reader is None:
                path = work / 'proof.json'
                if not path.exists():
                    result['proof_stage'] = 'MISSING_PROOF'
                    return
                if path.stat().st_size > 262144:
                    result['proof_stage'] = 'OVERCAP'
                    return
                proof = json.loads(path.read_text())
            else:
                proof = reader()
            result['partial'] = partial_proof(proof)
            result['proof_stage'] = 'QUALIFICATION'
            verdict = qualify(proof, expected)
            result['verdict'] = verdict
            result['producer'] = proof['data']
            result['identity'] = proof['identity']
            result['full_proof_qualified'] = True
            result['proof_stage'] = 'QUALIFIED'
        except (ValueError, KeyError, TypeError, IndexError, OSError):
            if result['proof_stage'] == 'UNKNOWN': result['proof_stage'] = 'MALFORMED_PROOF'
    finally:
        result['root_deletion'] = 'FAILED'
        (deleter or (lambda: shutil.rmtree(work)))()
        result['root_deletion'] = 'REMOVED'


def build_env(work, expected):
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work),
        XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
        XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
        LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), LOCAL_OPERATOR_RUNTIME_ROOT=str(work / "runtime"), PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", TERM="xterm-256color",
        NATIVE_WORKER_ROOT=str(work), NATIVE_WORKER_PROOF=str(work / "proof.json"),
        NATIVE_WORKER_EXPECTED=json.dumps(expected))
    (work / 'runtime').mkdir(exist_ok=True)
    return env


STYLE_FILES = ('tests/unit/test_tui_responsiveness.py','tests/unit/test_worker_code_identity.py',
               'tests/worker_code_identity.py','tests/loop_callback_spans.py')


def style_summary(tool, inner_rc, launched, timeout, overcap, settled, output, monitor_completed, monitor_failed):
    files = []
    if type(output) is bytes and len(output) <= 262144:
        for line in output.decode('utf-8', errors='replace').splitlines():
            path = None
            if tool == 'black' and line.startswith('would reformat '):
                path = line[len('would reformat '):]
            if tool == 'isort' and line.startswith('ERROR: ') and line.endswith(' Imports are incorrectly sorted and/or formatted.'):
                path = line[7:-len(' Imports are incorrectly sorted and/or formatted.')]
            if path is not None:
                path = path.removeprefix(str(ROOT)+'/')
                if path in STYLE_FILES and path not in files: files.append(path)
    physical = settled.get('reaped') is True and settled.get('owned_settled') is True and not settled.get('errors')
    safe = monitor_completed and not monitor_failed and launched and not timeout and not overcap and settled.get('reaped') is True and settled.get('owned_settled') is True and not settled.get('errors')
    reason = 'PASS' if safe and inner_rc == 0 else 'TOOL_FAILURE'
    if safe and inner_rc == 1 and files and tool in ('black','isort'): reason = 'FORMAT_REQUIRED'
    if not launched: reason = 'LAUNCH_FAILED'
    elif timeout: reason = 'TIMEOUT'
    elif overcap: reason = 'OUTPUT_CAP'
    elif not physical: reason = 'UNSETTLED'
    elif monitor_failed or not monitor_completed: reason = 'MONITOR_FAILURE'
    return dict(inner_rc=inner_rc, launched=launched, timeout=timeout, output_cap=overcap,
        monitor_completed=monitor_completed, monitor_failed=monitor_failed,
        reaped=settled.get('reaped') is True, owned_settled=settled.get('owned_settled') is True,
        reason=reason, files=files if reason == 'FORMAT_REQUIRED' else [])


def style_main(tool, destination):
    if tool not in ('black','isort','flake8'): raise SystemExit(2)
    options = {'black':['--check'],'isort':['--check-only'],'flake8':['--jobs','1']}[tool]
    known, pgid = {}, os.getpgrp()
    def observe(): return owned(known, pgid)
    def kill_owned():
        for pid, facts in observe().items():
            fresh = table().get(pid)
            if fresh and fresh['start'] == facts['start']: os.kill(pid, 9)
    child, settled, output = None, {}, bytearray()
    timed, capped = False, False
    monitor_completed, monitor_failed = False, False
    try:
        child = subprocess.Popen([sys.executable,'-B','-m',tool]+options+list(STYLE_FILES),
            cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+20
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout,selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() >= deadline:
                        timed=True
                        raise ValueError('style bound STOP')
                    observe()
                    for key,_ in selector.select(0.05):
                        chunk=os.read(key.fileobj.fileno(),16384)
                        if not chunk: selector.unregister(key.fileobj)
                        if len(output)+len(chunk)>262144:
                            capped=True
                            raise ValueError('style output STOP')
                        output.extend(chunk)
            monitor_completed = True  # ONLY healthy EOF completes the monitor.
        finally:
            settled=finish_owned(child,observe,kill_owned,lambda:None)
    except Exception:
        monitor_failed = True  # Latch unexpected faults even if final wait succeeds.
    summary=style_summary(tool,settled.get('rc'),child is not None,timed,capped,settled,bytes(output),monitor_completed,monitor_failed)
    Path(destination).write_text(json.dumps(summary)+'\n')
    raise SystemExit(0 if summary['reason']=='PASS' else 1)


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for label, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / label).read_bytes()).hexdigest() != sha:
            raise SystemExit("source STOP")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="native-worker-"))
    known, pgid = {}, os.getpgrp()
    expected = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        ppid=os.getpid(), pgid=pgid, sources={p:manifest["sources"][p] for p in manifest["loaded_sources"]})
    from scripts.diagnostics.native_worker_6911.carrier import native_sources
    from tests.worker_code_identity import source_allowlist
    expected = dict(identity=expected, sources=manifest['sources'], native_sources=native_sources(),
        producer_source_sha256=manifest['sources']['scripts/diagnostics/native_worker_6911/producer.py'], labels=list(source_allowlist(ROOT,
        ('scripts/diagnostics/native_worker_6911/producer.py',)).values()))
    env = build_env(work, expected)
    result, lifecycle = dict(qualified=False), {}
    def observe():return owned(known, pgid)
    def kill_owned():
        for pid, facts in observe().items():
            fresh = table().get(pid)
            if fresh and fresh["start"] == facts["start"]:os.kill(pid, 9)
    def remove():
        finalize_proof(work, result, expected)

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
        result["qualified"]=result.get("full_proof_qualified") is True and not lifecycle["errors"] and lifecycle["synthetic_root_removed"] and lifecycle["rc"] == 0
    except Exception:
        result["qualified"]=False
    finally:
        result["lifecycle"]=lifecycle
        (out/"native.json").write_text(json.dumps(result)+"\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--style':
        style_main(sys.argv[2],sys.argv[3])
    else:
        main()
