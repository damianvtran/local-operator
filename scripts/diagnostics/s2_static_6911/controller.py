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


def serialize_profile(entries,catalog,restored):
    from types import CodeType
    import math
    if type(restored) is not bool or not restored:raise ValueError('profile hook STOP')
    if len(entries)>20000:raise ValueError('profile entry cap STOP')
    def identity(code):
        if type(code) is not CodeType:return ''
        return catalog.get((code.co_filename,code.co_firstlineno,code.co_qualname),'')
    def measure(entry):
        values=dict(calls=entry.callcount,recursive=entry.reccallcount,inline_ms=entry.inlinetime*1000,cumulative_ms=entry.totaltime*1000)
        if any(type(values[k]) is not int or values[k]<0 for k in ('calls','recursive')):raise ValueError('profile count STOP')
        if any(type(values[k]) not in (int,float) or not math.isfinite(values[k]) or values[k]<0 for k in ('inline_ms','cumulative_ms')):raise ValueError('profile clock STOP')
        return values
    known=[];unknown=0
    for entry in entries:
        code_id=identity(entry.code)
        if code_id:known.append((entry,dict(code_id=code_id,**measure(entry))))
        else:unknown+=1
    known.sort(key=lambda pair:pair[1]['cumulative_ms'],reverse=True)
    roots=[p for p in known if p[1]['code_id'].endswith(':OperatorApp._boot_session')]
    selected=(roots+[p for p in known if p not in roots])[:20]
    ids={p[1]['code_id'] for p in selected};edges=[]
    for entry,row in selected:
        for child in entry.calls or ():
            target=identity(child.code)
            if target in ids:edges.append(dict(parent=row['code_id'],child=target,**measure(child)))
    edges.sort(key=lambda row:row['cumulative_ms'],reverse=True)
    return dict(scope='ORIGINAL_S2_CALL_APP_ONLY_THREAD_CPU',perturbed=True,hook_restored=restored,unknown_entries=unknown,dropped_rows=max(0,len(known)-len(selected)),rows=[p[1] for p in selected],edges=edges[:20])


def validate_profile(profile,labels):
    if type(profile) is not dict or set(profile)!={'scope','perturbed','hook_restored','unknown_entries','dropped_rows','rows','edges'}:raise ValueError('profile schema STOP')
    if profile['scope']!='ORIGINAL_S2_CALL_APP_ONLY_THREAD_CPU' or profile['perturbed'] is not True or profile['hook_restored'] is not True:raise ValueError('profile scope STOP')
    for key in ('unknown_entries','dropped_rows'):
        if type(profile[key]) is not int or profile[key]<0:raise ValueError('profile quality STOP')
    if not 1<=len(profile['rows'])<=20 or len(profile['edges'])>20:raise ValueError('profile output cap STOP')
    ids=set()
    for group in ('rows','edges'):
        for row in profile[group]:
            keys={'calls','recursive','inline_ms','cumulative_ms'}|({'code_id'} if group=='rows' else {'parent','child'})
            if set(row)!=keys:raise ValueError('profile privacy STOP')
            for key in ('code_id',) if group=='rows' else ('parent','child'):
                if row[key] not in labels:raise ValueError('profile static ID STOP')
            for key in ('calls','recursive'):
                if type(row[key]) is not int or row[key]<0:raise ValueError('profile count STOP')
            for key in ('inline_ms','cumulative_ms'):
                if type(row[key]) not in (int,float) or not math.isfinite(row[key]) or row[key]<0:raise ValueError('profile clock STOP')
            if group=='rows':
                if row['code_id'] in ids:raise ValueError('profile duplicate STOP')
                ids.add(row['code_id'])
    if not any(v.endswith(':OperatorApp._boot_session') for v in ids):raise ValueError('profile boot root absent STOP')
    if any(edge[key] not in ids for edge in profile['edges'] for key in ('parent','child')):raise ValueError('profile edge STOP')


def qualify(proof, expected):
    if set(proof) != {'rows','reports','guards','exitstatus'} or len(proof['rows']) != 1 or proof['guards']:
        raise ValueError('S2 cardinality/schema STOP')
    row=proof['rows'][0]
    if set(row) != {'token','identity','native_sources','phases','summary','effective','call_pass','performance_oracle','restored','cpu_profile','postcall'} or row['token'] != 1 or row['restored'] is not True:
        raise ValueError('S2 row STOP')
    validate_identity(row['identity'],expected)
    post=safe_postcall(row['postcall'])
    if post is None or post['stage']!='PROFILE_VALIDATE' or any(post[k] is not True for k in ('disable_ok','getstats_ok','catalog_ok','counts_ok','serialize_ok','profile_validate_ok','hookRestored')):raise ValueError('postcall incomplete STOP')
    if any(post[k] is None for k in ('entry_count','nativeCodeCount','static_matchCount','negativeCPUcount','invalidCPUcount')) or post['bootSeen'] is not True or not post['static_matchCount']<=post['nativeCodeCount']<=post['entry_count']<=20000:raise ValueError('postcall counts incomplete STOP')
    validate_profile(row['cpu_profile'],trusted_labels)
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


def safe_postcall(value):
    if type(value) is not dict:return None
    counts=('entry_count','nativeCodeCount','static_matchCount','negativeCPUcount','invalidCPUcount')
    flags=('bootSeen','hookRestored','disable_ok','getstats_ok','catalog_ok','counts_ok','serialize_ok','profile_validate_ok')
    if set(value)!=set(counts)|set(flags)|{'stage'}:return None
    if value['stage'] not in (None,'POSTCALL_DISABLE','GETSTATS','CATALOG','COUNTS','SERIALIZE','PROFILE_VALIDATE'):return None
    if any(value[k] is not None and (type(value[k]) is not int or not 0<=value[k]<=1000000) for k in counts):return None
    if any(value[k] is not None and type(value[k]) is not bool for k in flags):return None
    return {k:value[k] for k in ('stage',*counts,*flags)}


def safe_available(proof,expected):
    result={'rows':[]}
    if type(proof) is not dict or type(proof.get('rows')) is not list:return result
    for row in proof['rows'][:1]:
        if type(row) is not dict:continue
        safe={'postcall':safe_postcall(row.get('postcall')),'recorder':None,'effective':None,'identity':None}
        for group,keys in (('summary',('worst_cpu_ms','worst_wall_ms','noise_floor_ms','wall_cpu_descheduling_ratio')),('effective',('cpu_ms','wall_ms'))):
            raw=row.get(group)
            if type(raw) is dict:
                values={k:raw.get(k) for k in keys}
                if all(v is None or (type(v) in (int,float) and math.isfinite(v) and v>=0) for v in values.values()):safe['recorder' if group=='summary' else 'effective']=values
        try:
            validate_identity(row.get('identity'),expected)
            safe['identity']=row['identity']
        except (ValueError,TypeError):pass
        result['rows'].append(safe)
    return result


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
            result['available'] = safe_available(proof,expected)
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
    if sys.version_info[:2] != (3, 12):
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
