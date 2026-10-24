"""Canonical top-root lifecycle, strict new-v1 primitive consumer; no App."""
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
from scripts.diagnostics.owner_thread_graph_6911.producer import runtime_identity, phase_pass, overhead_pass, PHASES, CAPS
from scripts.diagnostics.owner_thread_graph_6911.collector import finalize as graph_final
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def qualify(proof,expected,labels=None):
    if type(proof) is not dict or set(proof)!={'identity','status','phases','overhead','application_authorized'} or proof['identity']!=expected or proof['status']!='COMPLETE' or proof['application_authorized'] is not False:raise ValueError('proof STOP')
    overhead_pass(proof['overhead'],expected)
    if type(proof['phases']) is not list or len(proof['phases'])!=4:raise ValueError('phase count STOP')
    for phase,row in zip(PHASES,proof['phases']):
        if row['phase']!=phase:raise ValueError('phase order STOP')
        phase_pass(row,expected)
    return 'OWNER_THREAD_V1_FOUR_PHASES_NO_APP'


def safe_partial(proof,expected,labels=None):
    safe=dict(status='UNKNOWN',identity=None,phases=None)
    if type(proof) is not dict:return safe
    if proof.get('identity')==expected:safe['identity']=expected
    if proof.get('status') in ('SETUP_FAILURE','OVERHEAD','COMPLETE')+PHASES:safe['status']=proof['status']
    if type(proof.get('phases')) is list:
        safe['phases']=[]
        for row in proof['phases'][:4]:
            if type(row) is not dict or row.get('phase') not in PHASES:continue
            graph=row.get('graph');identity=graph.get('identity') if type(graph) is dict else None
            safe['phases'].append(dict(phase=row['phase'],graph=graph_final(graph,identity)))
    return safe


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
    labels=None;identity=runtime_identity();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='native-profiler-'));pgid=os.getpgrp();known={}
    for name in ('home','config','tmp','cache','data','runtime'):(work/name).mkdir()
    env=dict(PATH=os.environ['PATH'],HOME=str(work/'home'),LOCAL_OPERATOR_CONFIG_DIR=str(work/'config'),LOCAL_OPERATOR_RUNTIME_ROOT=str(work/'runtime'),TMPDIR=str(work/'tmp'),XDG_CONFIG_HOME=str(work/'config'),XDG_CACHE_HOME=str(work/'cache'),XDG_DATA_HOME=str(work/'data'),PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT),OWNER_GRAPH_EXPECTED_FILE=str(work/'expected.json'),OWNER_GRAPH_PROOF=str(work/'proof.json'))
    result=dict(qualified=False,application_authorized=False,full_proof_qualified=False,partial=dict(stage='UNKNOWN'));lifecycle={};completed=False;expected={}
    def observe():return owned(known,pgid)
    def kill():
        for pid,row in observe().items():
            current=table().get(pid)
            if current and current['start']==row['start']:os.kill(pid,9)
    def remove():finalize(work,result,expected,labels)
    try:
        child=subprocess.Popen([sys.executable,'-B','-m','scripts.diagnostics.owner_thread_graph_6911.producer'],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
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
        result['lifecycle']=lifecycle;result['monitor_completed']=completed;(out/'owner-graph.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    raise SystemExit(0 if result['qualified'] else 1)


if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except Exception:raise SystemExit(1) from None
