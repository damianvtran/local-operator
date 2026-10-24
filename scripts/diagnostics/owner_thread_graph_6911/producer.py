"""Standalone instrumented owner-thread quality; NEVER authorizes App execution.

Predeclared bounds: same-scope error <= max(5ms,25% reference); suspended
wait/offload <= direct resume CPU+5ms and < half unrelated/worker >=15ms.
Fixed 1000 no-op operations: profile CPU <=250ms and signed extra <=100ms.
No bias subtraction, float retry, fitted thresholds or product-policy changes.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import CodeType
from scripts.diagnostics.owner_thread_graph_6911.collector import Collector, validate, finalize as graph_final

ROOT=Path(__file__).resolve().parents[3]
REL='scripts/diagnostics/owner_thread_graph_6911/producer.py'
COLLECTOR='scripts/diagnostics/owner_thread_graph_6911/collector.py'
PHASES=('DIRECT','NESTED','ASYNC','OFFLOAD')
CAPS=dict(output_bytes=262144,timeout_s=30,grace_s=5,memory_mb=512,operations=1000)
NAMES=('root','direct','nested','recurse','burn','segment','async_work','async_wait','async_phase','offload','offload_worker','noop','fixed_ops')


def burn(ns=20000000):
    start=time.thread_time_ns();value=0
    while time.thread_time_ns()-start<ns:
        for _ in range(1000):value=(value+1)%997


def direct(state):
    before=time.thread_time_ns();burn();state['reference_ns']=time.thread_time_ns()-before


def recurse(depth):
    if depth:return recurse(depth-1)
    burn()


def nested(state):
    before=time.thread_time_ns();recurse(3);state['reference_ns']=time.thread_time_ns()-before


async def segment(part,gates,ready,work):
    for i in range(3):
        before=time.thread_time_ns()
        if work:burn(6000000)
        part['reference_ns']+=time.thread_time_ns()-before;part['segments']+=1
        if i<2:ready[i].set();await gates[i].wait()


async def async_work(part,gates,ready):await segment(part,gates,ready,True)
async def async_wait(part,gates,ready):await segment(part,gates,ready,False)


async def async_phase(state):
    for label,fn in (('work',async_work),('wait',async_wait)):
        gates=[asyncio.Event(),asyncio.Event()];ready=[asyncio.Event(),asyncio.Event()];part=dict(reference_ns=0,segments=0);state[label]=part;task=asyncio.create_task(fn(part,gates,ready))
        try:
            for i in range(2):
                await asyncio.wait_for(ready[i].wait(),2.)
                if label=='wait' and i==0:
                    before=time.thread_time_ns();burn();state['unrelated_ns']=time.thread_time_ns()-before
                gates[i].set()
            await task
        finally:
            for gate in gates:gate.set()
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)


def offload_worker(state,loop,ready,release):
    state['worker_distinct']=threading.get_ident()!=state['owner'];state['worker_hook_none']=sys.getprofile() is None;loop.call_soon_threadsafe(ready.set)
    if not release.wait(2.):raise RuntimeError('barrier STOP')
    before=time.thread_time_ns();burn();state['worker_ns']=time.thread_time_ns()-before


async def offload(state):
    loop=asyncio.get_running_loop();ready=asyncio.Event();release=threading.Event();state['reference_ns']=0;state['segments']=0
    before=time.thread_time_ns();future=loop.run_in_executor(None,offload_worker,state,loop,ready,release);state['reference_ns']+=time.thread_time_ns()-before;state['segments']+=1
    try:
        await ready.wait()
        before=time.thread_time_ns();release.set();state['reference_ns']+=time.thread_time_ns()-before;state['segments']+=1
        await future
        before=time.thread_time_ns();state['segments']+=1;state['reference_ns']+=time.thread_time_ns()-before
    finally:
        release.set();await asyncio.gather(future,return_exceptions=True)


def noop():return None

def fixed_ops(state):
    for _ in range(CAPS['operations']):noop()


def root(phase,state):
    state['ready']=True
    if phase=='DIRECT':direct(state)
    elif phase=='NESTED':nested(state)
    elif phase=='ASYNC':asyncio.run(async_phase(state));state['loop_closed']=True
    elif phase=='OFFLOAD':asyncio.run(offload(state));state['loop_closed']=True
    elif phase=='OVERHEAD':fixed_ops(state)
    else:raise ValueError('phase STOP')
    state['trailing']=True


def catalog():
    path=ROOT/REL;compiled=compile(path.read_bytes(),str(path),'exec',flags=0,dont_inherit=True,optimize=0)
    codes={c.co_name:c for c in compiled.co_consts if type(c) is CodeType};pairs=[]
    for name in NAMES:
        code=globals()[name].__code__
        if code!=codes[name]:raise ValueError('native code source STOP')
        pairs.append((code,'PRIMITIVE:'+name.upper()))
    return tuple(pairs)


def runtime_identity():
    return dict(python=list(sys.version_info[:3]),executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),producer_sha256=hashlib.sha256((ROOT/REL).read_bytes()).hexdigest(),collector_sha256=hashlib.sha256((ROOT/COLLECTOR).read_bytes()).hexdigest())


def graph_identity(expected,owner,pairs):
    return {k:expected[k] for k in ('python','executable_sha256','producer_sha256','pid','ppid','pgid')}|dict(owner_thread=owner,static_ids=[label for _,label in pairs])


def observe(phase,expected,pairs):
    owner=threading.get_ident();identity=graph_identity(expected,owner,pairs);engine=Collector(root.__code__,pairs,identity,identity,time.thread_time_ns,threading.get_ident,sys.getprofile,sys.setprofile);state=dict(owner=owner,ready=False,trailing=False)
    if engine.failure is not None:raise ValueError('hook conflict STOP')
    try:
        # No Python helper may intervene between installation and reviewed root.
        sys.setprofile(engine.callback)
        root(phase,state)
    except Exception:
        # Keep the partial graph and original trailing-state failure; never
        # replace a failed primitive with a fabricated successful observation.
        pass
    finally:engine.cleanup()
    graph=engine.export()
    return build_phase(phase,graph,state,identity)


def build_phase(phase,graph,state,identity):
    if phase not in PHASES+('OVERHEAD',):raise ValueError('phase STOP')
    return dict(phase=phase,graph=graph,state=state,identity=identity)


def phase_pass(row,expected):
    if type(row) is not dict or set(row)!={'phase','graph','state','identity'} or row['phase'] not in PHASES+('OVERHEAD',):raise ValueError('row STOP')
    identity=row['identity'];owner=identity.get('owner_thread') if type(identity) is dict else None
    if type(owner) is not int or owner<=0:raise ValueError('owner STOP')
    labels=['PRIMITIVE:'+v.upper() for v in NAMES];wanted={k:expected[k] for k in ('python','executable_sha256','producer_sha256','pid','ppid','pgid')}|dict(owner_thread=owner,static_ids=labels)
    if identity!=wanted:raise ValueError('identity STOP')
    validate(row['graph'],wanted)
    state=row['state'];phase=row['phase'];keys={'owner','ready','trailing'}|({'reference_ns'} if phase in ('DIRECT','NESTED') else {'work','wait','unrelated_ns','loop_closed'} if phase=='ASYNC' else {'reference_ns','segments','worker_ns','worker_distinct','worker_hook_none','loop_closed'} if phase=='OFFLOAD' else set())
    if type(state) is not dict or set(state)!=keys or state['owner']!=owner or type(state['owner']) is not int or state['ready'] is not True or state['trailing'] is not True:raise ValueError('state STOP')
    nodes=row['graph']['nodes']
    def own(label):
        values=[n for n in nodes if n['static_id']=='PRIMITIVE:'+label]
        if len(values)!=1:raise ValueError('node STOP')
        return values[0]
    def number(v):
        if type(v) is not int or v<0:raise ValueError('reference STOP')
        return v
    def near(label,reference):
        n=own(label);ref=number(reference)
        if abs(n['inclusive_ns']-ref)>max(5000000,ref//4):raise ValueError('tolerance STOP')
        return n
    if phase in ('DIRECT','NESTED'):
        near(phase,state['reference_ns'])
        if phase=='NESTED' and own('RECURSE')['segments']!=4:raise ValueError('recursion STOP')
    elif phase=='ASYNC':
        if state['loop_closed'] is not True:raise ValueError('loop STOP')
        for part in ('work','wait'):
            v=state[part]
            if type(v) is not dict or set(v)!={'reference_ns','segments'} or type(v['segments']) is not int or v['segments']!=3:raise ValueError('resume STOP')
            n=near('ASYNC_'+part.upper(),v['reference_ns'])
            if n['segments']!=3:raise ValueError('observed resume STOP')
        reference=number(state['wait']['reference_ns']);unrelated=number(state['unrelated_ns']);measured=own('ASYNC_WAIT')['inclusive_ns']
        if measured>reference+5000000 or (unrelated>=15000000 and measured*2>=unrelated):raise ValueError('wait separation STOP')
    elif phase=='OFFLOAD':
        if any(state[k] is not True for k in ('worker_distinct','worker_hook_none','loop_closed')) or type(state['segments']) is not int or state['segments']!=3:raise ValueError('worker STOP')
        n=near('OFFLOAD',state['reference_ns']);worker=number(state['worker_ns'])
        if n['segments']!=3 or n['inclusive_ns']>state['reference_ns']+5000000 or (worker>=15000000 and n['inclusive_ns']*2>=worker):raise ValueError('worker separation STOP')
    else:
        if own('NOOP')['segments']!=CAPS['operations']:raise ValueError('operations STOP')
    return True


def overhead_pass(value,expected):
    if type(value) is not dict or set(value)!={'unprofiled_ns','profiled_ns','difference_ns','row'}:raise ValueError('overhead schema STOP')
    if any(type(value[k]) is not int for k in ('unprofiled_ns','profiled_ns','difference_ns')) or min(value['unprofiled_ns'],value['profiled_ns'])<0 or value['difference_ns']!=value['profiled_ns']-value['unprofiled_ns']:raise ValueError('overhead type STOP')
    phase_pass(value['row'],expected)
    if value['row']['phase']!='OVERHEAD' or value['profiled_ns']>250000000 or value['difference_ns']>100000000:raise ValueError('overhead bound STOP')
    return True


def main():
    output=Path(os.environ['OWNER_GRAPH_PROOF']);receipt=dict(identity=None,status='SETUP_FAILURE',phases=[],overhead=None,application_authorized=False)
    try:
        if sys.version_info[:3]!=(3,12,15) or sys.flags.optimize!=0 or sys.stdin.buffer.read(1)!=b'R':raise ValueError('setup STOP')
        expected=json.loads(Path(os.environ['OWNER_GRAPH_EXPECTED_FILE']).read_bytes());identity=dict(runtime_identity(),pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp())
        if identity!=expected or sys.getprofile() is not None or threading.getprofile() is not None:raise ValueError('header/hook STOP')
        receipt['identity']=identity;pairs=catalog();receipt['status']='OVERHEAD'
        before=time.thread_time_ns();root('OVERHEAD',{});unprofiled=time.thread_time_ns()-before
        before=time.thread_time_ns();row=observe('OVERHEAD',expected,pairs);profiled=time.thread_time_ns()-before
        receipt['overhead']=dict(unprofiled_ns=unprofiled,profiled_ns=profiled,difference_ns=profiled-unprofiled,row=row);overhead_pass(receipt['overhead'],expected)
        for phase in PHASES:
            receipt['status']=phase;row=observe(phase,expected,pairs);receipt['phases'].append(row);phase_pass(row,expected)
        receipt['status']='COMPLETE'
    except Exception:pass
    raw=(json.dumps(receipt,allow_nan=False)+'\n').encode()
    if len(raw)>CAPS['output_bytes']:raise SystemExit(1)
    output.write_bytes(raw);raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)


if __name__=='__main__':main()
