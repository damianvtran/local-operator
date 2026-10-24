"""Six original S3 controlled observations, two serial children; not natural stalls."""
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
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table, validate_identity
from scripts.diagnostics.native_worker_6911.carrier import native_sources
from tests.unit.lfs_s3_positive_plugin_6911 import NODE, MODES, NEGATIVE_MODES, build_row, validate_row, validate_negative_row
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def positive_row(row, mode, index, expected, native):
    validate_row(row)
    if row['mode'] != mode or row['token'] != index:
        raise ValueError('order STOP')
    validate_identity(row['identity'], expected)
    if row['native_sources'] != native or row['stage'] != 'RESTORED':
        raise ValueError('source STOP')
    for key in ('compiled', 'target_seen', 'held_before_manual', 'worker_offloop', 'sampler_stop_returned', 'gate_released', 'call_pass', 'entry_restored', 'discovery_restored', 'patches_restored'):
        if row[key] is not True:
            raise ValueError('witness STOP')
    if row['held_ticks'] is None or row['held_ticks'] < 3 or row['entry_calls'] < 1 or (row['owned_submissions'] < 1) or (row['manual_calls'] != 3) or (row['manual_cpu_ms'] >= 500) or (row['loop_gate_waits'] != 0) or (row['pending_before_invalidate'] != 0):
        raise ValueError('observation STOP')
    if row['shadow_wall_ms'] is None or row['async_injections'] != (0 if index == 1 else 2) or (index == 2 and row['shadow_wall_ms'] <= 500):
        raise ValueError('async shadow STOP')

ALL_MODES=MODES+NEGATIVE_MODES
BATCHES=(ALL_MODES[:3],ALL_MODES[3:])


def case_row(row,mode,expected,native):
    token=ALL_MODES.index(mode)+1
    if mode in MODES:
        positive_row(row,mode,token,expected,native);return
    validate_negative_row(row);validate_identity(row['identity'],expected)
    if row['mode']!=mode or row['token']!=token or row['native_sources']!=native or row['stage']!='RESTORED' or row['call_pass'] is not False:raise ValueError('negative source/outcome STOP')
    for key in ('compiled','target_seen','sampler_stop_returned','gate_released','entry_restored','discovery_restored','patches_restored'):
        if row[key] is not True:raise ValueError('negative cleanup STOP')
    if row['entry_calls']<1 or row['owned_submissions']<1 or row['pending_before_invalidate']!=0 or row['loop_gate_waits']!=0 or row['async_injections']!=0 or row['manual_cpu_ms'] is None:raise ValueError('negative ownership STOP')
    state=row['control'];oracle={'manual-cpu':'ORIGINAL_MANUAL_ASSERT','inline-offload':'ORIGINAL_HELD_ASSERT','body-fault':'BODY_FAULT','body-stop-fault':'BODY_STOP_GROUP'}[mode]
    if state['oracle']!=oracle:raise ValueError('negative oracle STOP')
    body=mode in ('body-fault','body-stop-fault');count=1 if body else 3
    if row['manual_calls']!=count or state['manual_started']!=count or state['manual_completed']!=(0 if body else 3) or len(state['manual_samples_ms'])!=count:raise ValueError('manual cardinality STOP')
    if state['body_fault_raised'] is not body or state['stop_fault_raised'] is not (mode=='body-stop-fault'):raise ValueError('fault observation STOP')
    if mode=='inline-offload':
        if state['inline_calls']<1 or state['inline_refused'] is not True:raise ValueError('inline refusal STOP')
    else:
        if row['held_ticks'] is None or row['held_ticks']<3 or row['held_before_manual'] is not True or row['worker_offloop'] is not True:raise ValueError('negative held witness STOP')
        if state['inline_calls']!=0 or state['inline_refused'] is not None:raise ValueError('unexpected inline STOP')
    if mode=='manual-cpu' and (row['manual_cpu_ms'] is None or row['manual_cpu_ms']<=500 or sum(state['manual_samples_ms'])<=500):raise ValueError('CPU control STOP')


def qualify(proof,expected,native,modes):
    if type(modes) is not tuple or modes not in BATCHES:raise ValueError('batch selection STOP')
    if type(proof) is not dict or set(proof)!={'component','rows','reports','exitstatus'} or proof['component']!='S3_POSITIVE_PRODUCER_ONLY' or type(proof['exitstatus']) is not int or proof['exitstatus']!=1:raise ValueError('proof/raw expected failure RC STOP')
    if type(proof['rows']) is not list or len(proof['rows'])!=3 or type(proof['reports']) is not list or len(proof['reports'])!=9:raise ValueError('batch cardinality STOP')
    for offset,(mode,row) in enumerate(zip(modes,proof['rows'])):
        case_row(row,mode,expected,native)
        for report,phase in zip(proof['reports'][3*offset:3*offset+3],('setup','call','teardown')):
            outcome='failed' if phase=='call' and mode in NEGATIVE_MODES else 'passed'
            if type(report) is not dict or type(report.get('token')) is not int or report!=dict(mode=mode,token=ALL_MODES.index(mode)+1,phase=phase,outcome=outcome):raise ValueError('actual report STOP')
    return 'THREE_CONTROLLED_ORACLES_QUALIFIED_NOT_NATURAL_STALLS'


def batch_complete(result):
    lifecycle=result.get('lifecycle',{})
    return result.get('monitor_completed') is True and result.get('full_proof_qualified') is True and type(lifecycle.get('rc')) is int and lifecycle.get('rc')==1 and lifecycle.get('reaped') is True and lifecycle.get('owned_settled') is True and lifecycle.get('synthetic_root_removed') is True and lifecycle.get('errors')==[] and result.get('root_deletion')=='REMOVED'


def aggregate(batches):
    full=len(batches)==2 and all(batch_complete(row) for row in batches)
    return dict(qualified=full,full_proof_qualified=full,batches=batches,verdict='SIX_CONTROLLED_ORACLES_NOT_NATIVE_PERFORMANCE_CURE' if full else 'UNQUALIFIED')

def partial(proof,expected):
    out=dict(stage='UNKNOWN',rows=None,reports=None)
    if type(proof) is not dict:return out
    if type(proof.get('rows')) is list:
        out['rows']=[];out['row_count_capped']=min(len(proof['rows']),4)
        for row in proof['rows'][:3]:
            if type(row) is not dict:continue
            safe={}
            for key in ('mode','stage'):
                if row.get(key) in (ALL_MODES if key=='mode' else ('SOURCE','ARMED','CALL','RESTORED')):safe[key]=row[key]
            for key in ('token','entry_calls','held_ticks','manual_calls','async_injections','loop_gate_waits','owned_submissions','pending_before_invalidate'):
                value=row.get(key);safe[key]=value if type(value) is int and 0<=value<=10000 else None
            for key in ('manual_cpu_ms','shadow_wall_ms'):
                value=row.get(key);safe[key]=value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None
            for key in ('compiled','target_seen','held_before_manual','worker_offloop','sampler_stop_returned','gate_released','call_pass','entry_restored','discovery_restored','patches_restored'):
                safe[key]=row.get(key) if type(row.get(key)) is bool else None
            try:validate_identity(row.get('identity'),expected);safe['identity']=row['identity']
            except (ValueError,TypeError):pass
            if type(row.get('control')) is dict:
                control=row['control'];clean={}
                for k in ('manual_started','manual_completed','inline_calls'):
                    v=control.get(k);clean[k]=v if type(v) is int and 0<=v<=10000 else None
                for k in ('inline_refused','body_fault_raised','stop_fault_raised'):
                    v=control.get(k);clean[k]=v if type(v) is bool else None
                v=control.get('oracle');clean['oracle']=v if v in ('UNAVAILABLE','ORIGINAL_PASS','ORIGINAL_MANUAL_ASSERT','ORIGINAL_HELD_ASSERT','BODY_FAULT','BODY_STOP_GROUP','UNEXPECTED') else 'UNAVAILABLE'
                values=control.get('manual_samples_ms');clean['manual_samples_ms']=list(values) if type(values) is list and len(values)<=3 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values) else None
                safe['control']=clean
            out['rows'].append(safe)
    if type(proof.get('reports')) is list:
        out['report_count_capped']=min(len(proof['reports']),10);out['reports']=[]
        for row in proof['reports'][:9]:
            if type(row) is dict and row.get('mode') in ALL_MODES and row.get('phase') in ('setup','call','teardown') and row.get('outcome') in ('passed','failed','skipped'):
                out['reports'].append({k:row[k] for k in ('mode','phase','outcome')})
    return out


def finalize(work,result,expected,native,modes,reader=None,deleter=None):
    result.update(full_proof_qualified=False,proof_stage='UNKNOWN',root_deletion='NOT_ATTEMPTED')
    try:
        try:
            if reader is None:
                path=work/'proof.json'
                if not path.is_file() or path.is_symlink() or path.stat().st_size>262144:raise ValueError('proof missing STOP')
                proof=json.loads(path.read_bytes())
            else:proof=reader()
            result['partial']=partial(proof,expected);result['proof_stage']='QUALIFICATION'
            result['verdict']=qualify(proof,expected,native,modes);result['rows']=proof['rows'];result['reports']=proof['reports'];result['raw_pytest_rc']=proof['exitstatus'];result['full_proof_qualified']=True;result['proof_stage']='QUALIFIED'
        except (ValueError,TypeError,KeyError,IndexError,OSError):pass
    finally:
        result['root_deletion']='FAILED';(deleter or (lambda:shutil.rmtree(work)))();result['root_deletion']='REMOVED'


def run_batch(manifest,modes):
    work=Path(tempfile.mkdtemp(prefix='s3-controls-'))
    for name in ('home','config','tmp','cache','data','runtime','pytest'):(work/name).mkdir()
    native=native_sources();pgid=os.getpgrp();known={}
    expected=dict(python=list(sys.version_info[:3]),executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),ppid=os.getpid(),pgid=pgid,sources={p:manifest['sources'][p] for p in manifest['loaded_sources']})
    env=dict(PATH=os.environ['PATH'],HOME=str(work/'home'),LOCAL_OPERATOR_CONFIG_DIR=str(work/'config'),TMPDIR=str(work/'tmp'),XDG_CONFIG_HOME=str(work/'config'),XDG_CACHE_HOME=str(work/'cache'),XDG_DATA_HOME=str(work/'data'),LOCAL_OPERATOR_RUNTIME_ROOT=str(work/'runtime'),PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TERM='xterm-256color',S3_NEGATIVE_MODES=json.dumps(modes),S3_PRODUCER_ROOT=str(work),S3_PRODUCER_PROOF=str(work/'proof.json'),S3_PRODUCER_EXPECTED=json.dumps(expected),S3_PRODUCER_NATIVE=json.dumps(native))
    result=dict(qualified=False,full_proof_qualified=False,partial=dict(stage='UNKNOWN',rows=None,reports=None));lifecycle={};monitor=False
    def observe():return owned(known,pgid)
    def kill():
        for pid,row in observe().items():
            fresh=table().get(pid)
            if fresh and fresh['start']==row['start']:os.kill(pid,9)
    def remove():finalize(work,result,expected,native,modes)
    try:
        child=subprocess.Popen([sys.executable,'-B','-m','pytest','-p','pytest_asyncio.plugin','-p','xdist.plugin','-p','tests.unit.lfs_s3_positive_plugin_6911','-n','0','-o','addopts=','--basetemp',str(work/'pytest'),'-q',NODE],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        try:
            expected['pid']=child.pid
            if os.getpgid(child.pid)!=pgid:raise ValueError('group STOP')
            total=0;deadline=time.monotonic()+30
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout,selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic()>=deadline:raise ValueError('timeout STOP')
                    observe()
                    for key,_ in selector.select(.05):
                        data=os.read(key.fileobj.fileno(),16384)
                        if not data:selector.unregister(key.fileobj)
                        total+=len(data)
                        if total>262144:raise ValueError('output STOP')
            monitor=True
        finally:lifecycle=finish_owned(child,observe,kill,remove)
        result.update(monitor_completed=monitor,lifecycle=lifecycle);result['qualified']=batch_complete(result)
    except Exception:result['qualified']=False
    finally:
        result['monitor_completed']=monitor;result['lifecycle']=lifecycle
    return result


def main():
    if sys.version_info[:2]!=(3,12):raise SystemExit('version STOP')
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,sha in manifest['sources'].items():
        path=ROOT/name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise SystemExit('source STOP')
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    batches=[]
    try:
        for modes in BATCHES:
            result=run_batch(manifest,modes);batches.append(result)
            if not batch_complete(result):break
    finally:
        receipt=aggregate(batches);(out/'s3.json').write_text(json.dumps(receipt)+'\n')
    raise SystemExit(0 if receipt['qualified'] else 1)


if __name__=='__main__':main()
