"""Two original MOVE variants in one serial child, 45+5 TOTAL, offhost only.
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
NODE = "tests/unit/test_tui_responsiveness.py::test_s2_send_to_agent_end_keeps_the_loop_under_the_bar"


def project_measurement(raw, labels):
    out = {k:raw[k] for k in ('worst_cpu_ms','worst_wall_ms','noise_floor_ms','wall_cpu_descheduling_ratio')}
    callbacks = raw['callbacks']
    out['worker_identity'] = callbacks['worker_identity']
    out['loop_thread_id'] = callbacks['loop_thread_id']
    for key in ('worst_cpu_windows','worst_wall_windows'):
        windows = []
        for window in callbacks[key][:3]:
            cb = window['callbacks']
            value = dict(cpu_gap_ms=window['cpu_gap_ms'],wall_gap_ms=window['wall_gap_ms'],
                count=cb['count'],cpu_ms=cb['cpu_ms'],wall_ms=cb['wall_ms'],gc_count=len(cb['gc']))
            for group in ('top_cpu','top_wall','gc'):
                value[group] = [dict(cpu_ms=s['cpu_ms'],wall_ms=s['wall_ms'],
                    code_id=s['detail'] if s['detail'] in labels else '',
                    attribution='WORKER' if s['detail'] in labels else 'UNKNOWN',
                    gc_generation={'gc.gen0':0,'gc.gen1':1,'gc.gen2':2}.get(s['name']) if group=='gc' else None) for s in cb[group][:5]]
            windows.append(value)
        out[key] = windows
    return out


def qualify(proof, expected):
    if set(proof) != {'rows','reports','guards','exitstatus'} or len(proof['rows']) != 1 or proof['guards']:
        raise ValueError('S2 cardinality/schema STOP')
    row=proof['rows'][0]
    if set(row) != {'token','identity','native_sources','phases','summary','effective','call_pass','performance_oracle','restored'} or row['token'] != 1 or row['restored'] is not True:
        raise ValueError('S2 row STOP')
    validate_identity(row['identity'],expected)
    if row['native_sources'] != expected_native: raise ValueError('Textual native source STOP')
    if type(row['call_pass']) is not bool or type(row['performance_oracle']) is not bool:
        raise ValueError('S2 outcome type STOP')
    phases=proof['reports']
    if any(set(p) != {'phase','outcome'} or p['phase'] not in ('setup','call','teardown') or p['outcome'] not in ('passed','failed','skipped') for p in phases):
        raise ValueError('S2 report schema STOP')
    if len(phases)!=3 or [p['phase'] for p in phases]!=['setup','call','teardown'] or any(p['outcome']!='passed' for p in phases if p['phase']!='call'):
        raise ValueError('S2 fixtures STOP')
    rc=0 if row['call_pass'] else 1
    if proof['exitstatus']!=rc or phases[1]['outcome']!=('passed' if rc==0 else 'failed') or (rc and row['performance_oracle'] is not True):
        raise ValueError('S2 unexpected assertion STOP')
    summary=row['summary']
    if set(summary) != {'worst_cpu_ms','worst_wall_ms','noise_floor_ms','wall_cpu_descheduling_ratio','worker_identity','loop_thread_id','worst_cpu_windows','worst_wall_windows'}:
        raise ValueError('S2 summary schema STOP')
    quality=summary['worker_identity']
    if set(quality) != {'qualified','unknown','overflow','unsupported','eager'} or any(type(quality[k]) is not int or quality[k]<0 for k in quality if k!='qualified') or type(quality['qualified']) is not bool:
        raise ValueError('S2 worker quality missing/unarmed STOP')
    if quality['qualified'] != (not any(quality[k] for k in ('unknown','overflow','unsupported','eager'))):
        raise ValueError('S2 contradictory quality STOP')
    def numeric(v):
        if type(v) not in (int,float) or not math.isfinite(v) or v<0: raise ValueError('S2 numeric STOP')
    for k in ('worst_cpu_ms','worst_wall_ms','wall_cpu_descheduling_ratio','loop_thread_id'):numeric(summary[k])
    if summary['noise_floor_ms'] is not None:numeric(summary['noise_floor_ms'])
    if set(row['effective']) != {'cpu_ms','wall_ms'}:raise ValueError('S2 bounds STOP')
    for v in row['effective'].values():numeric(v)
    if [p['phase'] for p in row['phases']] != ['IDLE_BEGIN','TURN_BEGIN','TURN_END']:raise ValueError('S2 phases STOP')
    for p in row['phases']:
        if set(p) != {'phase','cpu_s','wall_s'}:raise ValueError('S2 phase privacy STOP')
        numeric(p['cpu_s']);numeric(p['wall_s'])
    for key in ('worst_cpu_windows','worst_wall_windows'):
        if not 1 <= len(summary[key]) <= 3:raise ValueError('S2 window cap STOP')
        for window in summary[key]:
            if set(window) != {'cpu_gap_ms','wall_gap_ms','count','cpu_ms','wall_ms','gc_count','top_cpu','top_wall','gc'}:raise ValueError('S2 callback schema STOP')
            for k in ('cpu_gap_ms','wall_gap_ms','count','cpu_ms','wall_ms','gc_count'):numeric(window[k])
            for group in ('top_cpu','top_wall','gc'):
                if len(window[group])>5:raise ValueError('S2 span cap STOP')
                for span in window[group]:
                    if set(span) != {'cpu_ms','wall_ms','code_id','attribution','gc_generation'} or span['attribution'] not in ('WORKER','UNKNOWN') or (span['code_id'] and span['code_id'] not in trusted_labels):raise ValueError('S2 code privacy STOP')
                    if span['gc_generation'] not in (None,0,1,2):raise ValueError('GC generation STOP')
                    numeric(span['cpu_ms']);numeric(span['wall_ms'])
    return 'ORIGINAL_S2_OBSERVED_ATTRIBUTION_PARTIAL_UNKNOWN' if not quality['qualified'] else 'ORIGINAL_S2_OBSERVED_NOT_EXONERATION'


def partial_proof(proof):
    out = {'stage':'UNKNOWN','rows':[], 'reports':[]}
    if type(proof) is not dict:
        return out
    stages = {'SOURCE','NATIVE_IMPORT','IDENTITY','PRODUCER','EXECUTOR','COMPLETE'}
    if type(proof.get('stage')) is str and proof['stage'] in stages:
        out['stage'] = proof['stage']
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
            if type(row.get('phases')) is list:
                safe['phases']=[v['phase'] for v in row['phases'][:4] if type(v) is dict and v.get('phase') in ('IDLE_BEGIN','TURN_BEGIN','TURN_END')]
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
            global trusted_labels
            from tests.worker_code_identity import source_allowlist
            trusted_labels=set(source_allowlist(ROOT).values())
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
            result['rows'] = result['partial']['rows']
            result['identities'] = [row['identity'] for row in proof['rows']]
            result['measurement'] = proof['rows'][0]
            result['full_proof_qualified'] = True
            result['proof_stage'] = 'QUALIFIED'
        except (ValueError, KeyError, TypeError, IndexError, OSError):
            if result['proof_stage'] == 'UNKNOWN': result['proof_stage'] = 'MALFORMED_PROOF'
    finally:
        result['root_deletion'] = 'FAILED'
        (deleter or (lambda: shutil.rmtree(work)))()
        result['root_deletion'] = 'REMOVED'


def main():
    global expected_native
    from scripts.diagnostics.native_worker_6911.carrier import native_sources
    expected_native=native_sources()
    manifest = json.loads((HERE / "manifest.json").read_text())
    for label, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / label).read_bytes()).hexdigest() != sha:
            raise SystemExit("source STOP")
    if sys.version_info[:2] != (3, 13):
        raise SystemExit("Python family STOP")
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="s2-static-"))
    known, pgid = {}, os.getpgrp()
    expected = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        ppid=os.getpid(), pgid=pgid, sources={p:manifest["sources"][p] for p in manifest["loaded_sources"]})
    env = dict(PATH=os.environ["PATH"], HOME=str(work), TMPDIR=str(work),
        XDG_CONFIG_HOME=str(work / "config"), XDG_CACHE_HOME=str(work / "cache"),
        XDG_DATA_HOME=str(work / "data"), XDG_RUNTIME_DIR=str(work / "runtime"),
        LOCAL_OPERATOR_CONFIG_DIR=str(work / "config"), LOCAL_OPERATOR_RUNTIME_ROOT=str(work / "runtime"), PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="", TERM="xterm-256color",
        S2_PROFILE_ROOT=str(work), S2_PROFILE_PROOF=str(work / "proof.json"),
        S2_PROFILE_EXPECTED=json.dumps(expected))
    (work/"runtime").mkdir(exist_ok=True)
    result, lifecycle = dict(qualified=False), {}
    def observe():return owned(known, pgid)
    def kill_owned():
        for pid, facts in observe().items():
            fresh = table().get(pid)
            if fresh and fresh["start"] == facts["start"]:os.kill(pid, 9)
    def remove():
        finalize_proof(work, result, expected)

    try:
        child = subprocess.Popen([sys.executable,"-B","-m","pytest",NODE,
            "-n0","-q","--tb=no","-o","addopts=",
            "-p","tests.unit.lfs_s2_static_plugin_6911"],cwd=ROOT,env=env,
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
        result["qualified"]=result.get("full_proof_qualified") is True and not lifecycle["errors"] and lifecycle["synthetic_root_removed"] and lifecycle["rc"] in (0,1)
    except Exception:
        result["qualified"]=False
    finally:
        result["lifecycle"]=lifecycle
        (out/"s2.json").write_text(json.dumps(result)+"\n")
    raise SystemExit(0 if result["qualified"] else 1)


if __name__=="__main__":main()
