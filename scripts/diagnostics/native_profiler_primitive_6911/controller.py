"""ONE no-App primitive child. Mini PASS is never large-App authorization.

Pinned reference lines833–850 describe profiler_disable, NOT deallocation.
Own object weakref release plus tool/events are not a global callback snapshot.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table
from scripts.diagnostics.native_profiler_primitive_6911.producer import PHASES,CAPS,TOLERANCE,runtime_identity,source_catalog,validate_phase,judge
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
COMPONENT='PRIMITIVE_PRODUCER_ONLY_NOT_APP_AUTHORIZATION'


def phase_pass(row,phase,labels):
    validate_phase(row,labels)
    if row['phase']!=phase or row['mode']!='INTEGER_NS' or row['status']!='PASS' or row['first_guard'] is not None:raise ValueError('phase outcome STOP')
    if len({r['code_id'] for r in row['rows']})!=len(row['rows']) or len({(e['parent'],e['child']) for e in row['own_edges']})!=len(row['own_edges']):raise ValueError('duplicate source ID STOP')
    if row['entries']!=row['unknown_entries']+len(row['rows']) or row['edges']<row['unknown_edges']+len(row['own_edges']):raise ValueError('quality cardinality STOP')
    # The shared schema intentionally retains signed/nullable failure evidence.
    # A successful graph must independently reject contradictions in EVERY
    # published own row/edge, not trust global counters or only the chosen root.
    for item in (*row['rows'],*row['own_edges']):
        for key in ('callcount','reccallcount'):
            if type(item[key]) is not int or item[key]<0:raise ValueError('full count quality STOP')
        for key in ('inlinetime','totaltime'):
            if type(item[key]) not in (int,float) or not math.isfinite(item[key]) or item[key]<0:raise ValueError('full time quality STOP')
    reference=row['reference']
    if type(reference) is not dict or reference.get('ready') is not True or reference.get('trailing') is not True:raise ValueError('readiness STOP')
    if phase in ('DIRECT','NESTED'):
        root='PRIMITIVE:'+('direct' if phase=='DIRECT' else 'nested')
        okay=reference['markers']==1 and type(reference['markers']) is int and judge(row,root,reference['reference_ms'],reference['markers'])
    elif phase=='ASYNC':
        okay=all(type(reference[k]['markers']) is int and reference[k]['markers']==3 for k in ('work','wait')) and judge(row,'PRIMITIVE:async_work_target',reference['work']['reference_ms'],3) and judge(row,'PRIMITIVE:async_wait_target',reference['wait']['reference_ms'],3,reference['unrelated_ms'])
    else:okay=reference.get('worker_distinct') is True and type(reference['markers']) is int and reference['markers']==3 and judge(row,'PRIMITIVE:offload_target',reference['reference_ms'],3,reference['worker_ms'])
    edges={'DIRECT':('PRIMITIVE:direct','PRIMITIVE:burn'),'NESTED':('PRIMITIVE:nested','PRIMITIVE:recurse'),'ASYNC':('PRIMITIVE:async_wait_target','PRIMITIVE:resume_target')}
    if phase in edges:okay=okay and any((e['parent'],e['child'])==edges[phase] and type(e['callcount']) is int and e['callcount']>0 for e in row['own_edges'])
    if phase=='NESTED':okay=okay and any(e['code_id']=='PRIMITIVE:recurse' and e['callcount']==4 and e['reccallcount']==3 for e in row['rows'])
    if not okay:raise ValueError('independent accounting STOP')


def qualify(proof,expected,labels):
    keys={'component','identity','caps','tolerance','integer','float_comparison','remaining_unrun','qualified','application_authorized'}
    if type(proof) is not dict or set(proof)!=keys or proof['component']!=COMPONENT or proof['identity']!=expected:raise ValueError('identity/schema STOP')
    if proof['caps']!=CAPS or proof['tolerance']!=TOLERANCE or proof['application_authorized'] is not False or proof['qualified'] is not True:raise ValueError('policy STOP')
    if proof['float_comparison'] is not None or proof['remaining_unrun']!=[] or type(proof['integer']) is not list or len(proof['integer'])!=4:raise ValueError('integer-only proof STOP')
    for row,phase in zip(proof['integer'],PHASES):phase_pass(row,phase,labels)
    return 'NATIVE_PRIMITIVES_QUALIFIED_NOT_APP_OR_CPROFILE_QUALITY_POLICY_WAIVER'


def safe_partial(proof,expected,labels):
    out=dict(stage='UNKNOWN',identity=None,integer=None,float_comparison=None,remaining_unrun=None)
    if type(proof) is not dict:return out
    if proof.get('identity')==expected:out['identity']=expected
    def phase(row):
        try:validate_phase(row,labels);return dict(row)
        except (ValueError,TypeError,KeyError):
            result=dict(schema_valid=False)
            if type(row) is dict:
                if row.get('phase') in PHASES:result['phase']=row['phase']
                if row.get('mode') in ('INTEGER_NS','FLOAT_SECONDS'):result['mode']=row['mode']
                for k in ('negative_entries','negative_edges','invalid_entries','invalid_edges'):
                    v=row.get(k);result[k]=v if type(v) is int and 0<=v<=40000 else None
            return result
    if type(proof.get('integer')) is list:out.update(stage='OBSERVED',integer=[phase(v) for v in proof['integer'][:4]])
    if proof.get('float_comparison') is not None:out['float_comparison']=phase(proof['float_comparison'])
    remaining=proof.get('remaining_unrun')
    if type(remaining) is list and len(remaining)<=4 and all(v in PHASES for v in remaining):out['remaining_unrun']=list(remaining)
    if proof.get('status')=='SETUP_FAILURE':out['stage']='SETUP_FAILURE'
    return out


def finalize(work,result,expected,labels,settled=True,reader=None,deleter=None):
    result.update(full_proof_qualified=False,proof_stage='UNKNOWN',root_deletion='NOT_ATTEMPTED')
    if not settled:result['proof_stage']='UNSETTLED';return
    try:
        try:
            if reader is None:
                path=work/'proof.json'
                if path.is_symlink() or not path.is_file() or path.stat().st_size>CAPS['output_bytes']:raise ValueError('proof unavailable STOP')
                proof=json.loads(path.read_bytes())
            else:proof=reader()
            result['partial']=safe_partial(proof,expected,labels);result['proof_stage']='QUALIFICATION'
            result['verdict']=qualify(proof,expected,labels);result['full_proof_qualified']=True;result['proof_stage']='QUALIFIED'
        except (ValueError,TypeError,KeyError,IndexError,OSError):pass
    finally:
        result['root_deletion']='FAILED';(deleter or (lambda:shutil.rmtree(work)))();result['root_deletion']='REMOVED'


def finish_result(result,lifecycle,completed):
    okay=result.get('full_proof_qualified') is True and completed is True and result.get('root_deletion')=='REMOVED' and type(lifecycle.get('rc')) is int and lifecycle['rc']==0 and lifecycle.get('reaped') is True and lifecycle.get('owned_settled') is True and lifecycle.get('synthetic_root_removed') is True and lifecycle.get('errors')==[]
    result.update(qualified=okay,application_authorized=False,monitor_completed=completed,lifecycle=lifecycle)
    return result


def main():
    if sys.version_info[:3]!=(3,12,15):raise SystemExit('runtime version STOP')
    manifest=json.loads((HERE/'manifest.json').read_text())
    for label,sha in manifest['sources'].items():
        path=ROOT/label
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise SystemExit('source STOP')
    labels=set(source_catalog().values());identity=runtime_identity();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='native-profiler-'));pgid=os.getpgrp();known={}
    for name in ('home','config','tmp','cache','data','runtime'):(work/name).mkdir()
    env=dict(PATH=os.environ['PATH'],HOME=str(work/'home'),LOCAL_OPERATOR_CONFIG_DIR=str(work/'config'),LOCAL_OPERATOR_RUNTIME_ROOT=str(work/'runtime'),TMPDIR=str(work/'tmp'),XDG_CONFIG_HOME=str(work/'config'),XDG_CACHE_HOME=str(work/'cache'),XDG_DATA_HOME=str(work/'data'),PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),NATIVE_PRIMITIVE_EXPECTED_FILE=str(work/'expected.json'),NATIVE_PRIMITIVE_PROOF=str(work/'proof.json'))
    result=dict(qualified=False,application_authorized=False,full_proof_qualified=False,partial=dict(stage='UNKNOWN'));lifecycle={};completed=False;expected={}
    def observe():return owned(known,pgid)
    def kill():
        for pid,row in observe().items():
            current=table().get(pid)
            if current and current['start']==row['start']:os.kill(pid,9)
    def remove():finalize(work,result,expected,labels)
    try:
        child=subprocess.Popen([sys.executable,'-B','-m','scripts.diagnostics.native_profiler_primitive_6911.producer'],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            expected=dict(identity,pid=child.pid,ppid=os.getpid(),pgid=pgid)
            if os.getpgid(child.pid)!=pgid:raise ValueError('group STOP')
            temporary=work/'expected.tmp';temporary.write_text(json.dumps(expected));temporary.replace(work/'expected.json');child.stdin.write(b'R');child.stdin.flush();child.stdin.close()
            deadline=time.monotonic()+30;total=0
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout,selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic()>=deadline:raise ValueError('timeout STOP')
                    observe()
                    for key,_ in selector.select(.05):
                        data=os.read(key.fileobj.fileno(),16384)
                        if not data:selector.unregister(key.fileobj)
                        total+=len(data)
                        if total>CAPS['output_bytes']:raise ValueError('output cap STOP')
            completed=True
        finally:
            try:
                if child.stdin is not None and not child.stdin.closed:child.stdin.close()
            finally:
                try:lifecycle=finish_owned(child,observe,kill,remove)
                finally:
                    if child.stdout is not None:child.stdout.close()
        finish_result(result,lifecycle,completed)
    except Exception:result['qualified']=False
    finally:
        result['lifecycle']=lifecycle;result['monitor_completed']=completed;(out/'primitive.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    raise SystemExit(0 if result['qualified'] else 1)


if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except Exception:raise SystemExit(1) from None
