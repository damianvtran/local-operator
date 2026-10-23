"""COMPONENT ONLY. Primitive accounting preflight, never authority to run App.

Tolerances are fixed before data: 5ms or 25% direct scope CPU. They are NOT
application budgets/calibration. Any entry/edge negative invalidates the phase.
"""
import asyncio
import cProfile
import _lsprof
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType
import weakref

PHASES=('DIRECT','NESTED','ASYNC','OFFLOAD')
CAPS=dict(wall_s=30,grace_s=5,memory_mib=512,output_bytes=262144,entries=20000,edges=40000)
TOLERANCE=dict(absolute_ms=5.0,relative=0.25,separation_min_ms=15.0,separation_fraction=0.5)


def burn(seconds=.02):
    start=time.thread_time();value=0
    while time.thread_time()-start<seconds:value+=1


def direct(state):
    state['ready']=True
    begin=time.thread_time();burn();state['reference_ms']=(time.thread_time()-begin)*1000;state['markers']=1;state['trailing']=True


def recurse(depth):
    if depth:return recurse(depth-1)
    burn()


def nested(state):
    state['ready']=True
    begin=time.thread_time();recurse(3);state['reference_ms']=(time.thread_time()-begin)*1000;state['markers']=1;state['trailing']=True


async def resume_target(state,gates,ready,work):
    for i in range(3):
        begin=time.thread_time()
        if work:burn(.006)
        state['reference_ms']+=(time.thread_time()-begin)*1000;state['markers']+=1
        if i<2:ready[i].set();await gates[i].wait()


async def async_work_target(state,gates,ready):
    await resume_target(state,gates,ready,True)


async def async_wait_target(state,gates,ready):
    await resume_target(state,gates,ready,False)


async def async_phase(state):
    for label,work in (('work',True),('wait',False)):
        gates=[asyncio.Event(),asyncio.Event()];ready=[asyncio.Event(),asyncio.Event()]
        part=dict(reference_ms=0.,markers=0);state[label]=part
        task=asyncio.create_task((async_work_target if work else async_wait_target)(part,gates,ready))
        try:
            for i in range(2):
                await asyncio.wait_for(ready[i].wait(),2.);state['ready']=True
                if not work and i==0:
                    start=time.thread_time();burn();state['unrelated_ms']=(time.thread_time()-start)*1000
                gates[i].set()
            await task
        finally:
            for gate in gates:gate.set()
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    state['trailing']=True


def offload_worker(state,loop,ready,release):
    state['worker_distinct']=threading.get_ident()!=state['loop_ident'];loop.call_soon_threadsafe(ready.set)
    if not release.wait(2.):raise RuntimeError('owned worker barrier STOP')
    start=time.thread_time();burn();state['worker_ms']=(time.thread_time()-start)*1000


async def offload_target(state,ready,release):
    begin=time.thread_time();loop=asyncio.get_running_loop();future=loop.run_in_executor(None,offload_worker,state,loop,ready,release)
    state['reference_ms']+=(time.thread_time()-begin)*1000;state['markers']+=1
    try:
        await asyncio.wait_for(ready.wait(),2.)
        begin=time.thread_time();state['ready']=True;state['markers']+=1;release.set();state['reference_ms']+=(time.thread_time()-begin)*1000
        await future
    finally:release.set();await asyncio.gather(future,return_exceptions=True)
    begin=time.thread_time();state['markers']+=1;state['reference_ms']+=(time.thread_time()-begin)*1000


async def offload_phase(state):
    state.update(reference_ms=0.,markers=0,loop_ident=threading.get_ident());ready=asyncio.Event();release=threading.Event()
    try:await offload_target(state,ready,release)
    finally:release.set()
    state.pop('loop_ident');state['trailing']=True


def template(phase,mode):
    if phase not in PHASES or mode not in ('INTEGER_NS','FLOAT_SECONDS'):raise ValueError('selection STOP')
    return dict(phase=phase,mode=mode,status='UNKNOWN',accounting_qualified=False,ownership_qualified=False,entries=None,edges=None,unknown_entries=None,unknown_edges=None,own_edges=[],negative_entries=None,negative_edges=None,invalid_entries=None,invalid_edges=None,first_guard=None,rows=[],reference=None,hook_entry=None,hook_serialize=None,hook_cleanup=None,tool_entry_free=None,tool_cleanup_free=None,events_cleanup_zero=None,own_callbacks_released=None,callback_scope='OWN_PROFILE_WEAKREF_NOT_GLOBAL_CALLBACK_SNAPSHOT',executor_settled=None)


def measurements(entries,catalog,row):
    if type(entries) not in (list,tuple) or len(entries)>CAPS['entries']:raise ValueError('entry cap STOP')
    row.update(entries=len(entries),edges=0,unknown_entries=0,unknown_edges=0,negative_entries=0,negative_edges=0,invalid_entries=0,invalid_edges=0)
    pending_rows=[];pending_edges=[]
    def measure(entry,edge,parent=None):
        label=catalog.get(entry.code) if type(entry.code) is CodeType else None
        if label is None:row['unknown_edges' if edge else 'unknown_entries']+=1
        bad=negative=False;values={}
        for field in ('callcount','reccallcount','inlinetime','totaltime'):
            value=getattr(entry,field);count=field in ('callcount','reccallcount')
            invalid=type(value) is not int if count else type(value) not in (int,float) or not math.isfinite(value)
            bad|=invalid;negative|=not invalid and value<0
            if (invalid or value<0) and row['first_guard'] is None:
                row['first_guard']=dict(scope='EDGE' if edge else 'ENTRY',field=field,kind='TYPE_OR_NONFINITE' if invalid else 'NEGATIVE',code_id=label,value=None if invalid or label is None else value,unit='COUNT' if count else 'SECONDS')
            values[field]=value if not invalid else None
        row['invalid_edges' if edge else 'invalid_entries']+=int(bad)
        row['negative_edges' if edge else 'negative_entries']+=int(negative)
        if label is not None and not edge:pending_rows.append(dict(code_id=label,**values))
        if edge and label is not None and parent is not None:
            if len(pending_edges)>=20:raise ValueError('known edge cap STOP')
            pending_edges.append(dict(parent=parent,child=label,**values))
    for entry in entries:
        measure(entry,False)
        if entry.calls is not None and type(entry.calls) not in (list,tuple):raise ValueError('edge shape STOP')
        for edge in entry.calls or ():
            row['edges']+=1
            if row['edges']>CAPS['edges']:raise ValueError('edge cap STOP')
            measure(edge,True,catalog.get(entry.code) if type(entry.code) is CodeType else None)
    if len(pending_rows)>20:raise ValueError('own rows cap STOP')
    if not any(row[k] for k in ('negative_entries','negative_edges','invalid_entries','invalid_edges')):
        row['rows']=pending_rows;row['own_edges']=pending_edges


def judge(row,scope_id,reference,markers,comparison=None):
    """Same-scope inclusive time only; cumulative entries are never summed."""
    if any(row.get(k) is None or row[k]!=0 for k in ('negative_entries','negative_edges','invalid_entries','invalid_edges')):return False
    matches=[x for x in row['rows'] if x['code_id']==scope_id]
    if len(matches)!=1 or type(markers) is not int or matches[0]['callcount']!=markers:return False
    measured=matches[0]['totaltime']*1000
    if type(reference) not in (int,float) or not math.isfinite(reference) or reference<0:return False
    if abs(measured-reference)>max(TOLERANCE['absolute_ms'],TOLERANCE['relative']*reference):return False
    if comparison is not None and (comparison<15. or measured>reference+5. or not measured<comparison/2):return False
    return True


def source_catalog():
    path=Path(__file__).resolve();source=path.read_text();frozen=compile(source,str(path),'exec',flags=0,dont_inherit=True,optimize=0);known={c.co_qualname:c for c in frozen.co_consts if type(c) is CodeType};catalog={}
    for name in ('burn','direct','recurse','nested','resume_target','async_work_target','async_wait_target','async_phase','offload_worker','offload_target','offload_phase'):
        fn=globals()[name]
        if type(fn) is not FunctionType or type(fn.__code__) is not CodeType or fn.__code__.co_filename!=str(path) or known.get(name)!=fn.__code__:raise ValueError('compiled source STOP')
        catalog[fn.__code__]='PRIMITIVE:'+name
    return catalog


def run_phase(phase,mode,catalog):
    row=template(phase,mode);tool=sys.monitoring.PROFILER_ID
    row['hook_entry']=sys.getprofile() is None;row['tool_entry_free']=sys.monitoring.get_tool(tool) is None and sys.monitoring.get_events(tool)==0
    if not row['hook_entry'] or not row['tool_entry_free']:row['status']='SETUP_FAILURE';return row
    state={};profiler=None;reference=None
    try:
        profiler=cProfile.Profile(timer=time.thread_time_ns,timeunit=1e-9) if mode=='INTEGER_NS' else cProfile.Profile(timer=time.thread_time)
        reference=weakref.ref(profiler);profiler.enable(subcalls=True,builtins=False)
        if phase=='DIRECT':direct(state)
        elif phase=='NESTED':nested(state)
        elif phase=='ASYNC':asyncio.run(async_phase(state))
        else:asyncio.run(offload_phase(state))
        row['executor_settled']=True
        profiler.disable();row['hook_serialize']=sys.getprofile() is None
        measurements(profiler.getstats(),catalog,row);row['reference']=state
        if phase in ('DIRECT','NESTED'):passed=judge(row,'PRIMITIVE:'+('direct' if phase=='DIRECT' else 'nested'),state['reference_ms'],state['markers'])
        elif phase=='ASYNC':
            passed=judge(row,'PRIMITIVE:async_work_target',state['work']['reference_ms'],state['work']['markers']) and judge(row,'PRIMITIVE:async_wait_target',state['wait']['reference_ms'],state['wait']['markers'],state['unrelated_ms'])
        else:passed=state.get('ready') is True and state.get('trailing') is True and state.get('worker_distinct') is True and judge(row,'PRIMITIVE:offload_target',state['reference_ms'],state['markers'],state['worker_ms'])
        required={'DIRECT':('PRIMITIVE:direct','PRIMITIVE:burn'),'NESTED':('PRIMITIVE:nested','PRIMITIVE:recurse'),'ASYNC':('PRIMITIVE:async_wait_target','PRIMITIVE:resume_target')}
        if phase in required:passed=passed and any((x['parent'],x['child'])==required[phase] and type(x['callcount']) is int and x['callcount']>0 for x in row['own_edges'])
        if phase=='NESTED':passed=passed and any(x['code_id']=='PRIMITIVE:recurse' and x['callcount']==4 and x['reccallcount']==3 for x in row['rows'])
        row['accounting_qualified']=passed;row['status']='PASS' if passed else 'ACCOUNTING_FAILURE'
    except Exception:row['status']='SETUP_OR_PROTOCOL_FAILURE'
    finally:
        if profiler is not None:
            try:profiler.disable()
            except Exception:row['status']='OWNERSHIP_FAILURE'
        profiler=None
        row['hook_cleanup']=sys.getprofile() is None;row['tool_cleanup_free']=sys.monitoring.get_tool(tool) is None;row['events_cleanup_zero']=sys.monitoring.get_events(tool)==0;row['own_callbacks_released']=reference() is None if reference is not None else None
        row['ownership_qualified']=all(row[k] is True for k in ('hook_entry','hook_serialize','hook_cleanup','tool_entry_free','tool_cleanup_free','events_cleanup_zero','own_callbacks_released'))
        if not row['ownership_qualified']:row['accounting_qualified']=False;row['status']='OWNERSHIP_FAILURE'
    return row


def runtime_identity():
    paths=dict(producer=Path(__file__).resolve(),cprofile=Path(cProfile.__file__).resolve(),lsprof=Path(_lsprof.__file__).resolve(),executable=Path(sys.executable).resolve())
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):raise ValueError('runtime source STOP')
    return dict(python=list(sys.version_info[:3]),pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()})


def validate_phase(row,labels):
    if type(row) is not dict or set(row)!=set(template('DIRECT','INTEGER_NS')):raise ValueError('phase schema STOP')
    for key,cap in (('entries',CAPS['entries']),('edges',CAPS['edges'])):
        if row[key] is not None and (type(row[key]) is not int or not 0<=row[key]<=cap):raise ValueError('cardinality STOP')
    if row['phase'] not in PHASES or row['mode'] not in ('INTEGER_NS','FLOAT_SECONDS'):raise ValueError('mode STOP')
    if row['status'] not in ('UNKNOWN','PASS','ACCOUNTING_FAILURE','SETUP_FAILURE','SETUP_OR_PROTOCOL_FAILURE','OWNERSHIP_FAILURE'):raise ValueError('status STOP')
    for k in ('negative_entries','negative_edges','invalid_entries','invalid_edges','unknown_entries','unknown_edges'):
        if row[k] is not None and (type(row[k]) is not int or not 0<=row[k]<=40000):raise ValueError('counter STOP')
    for k in ('hook_entry','hook_serialize','hook_cleanup','tool_entry_free','tool_cleanup_free','events_cleanup_zero','own_callbacks_released','executor_settled'):
        if row[k] is not None and type(row[k]) is not bool:raise ValueError('observed bool STOP')
    for k in ('accounting_qualified','ownership_qualified'):
        if type(row[k]) is not bool:raise ValueError('qualification bool STOP')
    if type(row['rows']) is not list or len(row['rows'])>20:raise ValueError('row cap STOP')
    for item in row['rows']:
        if set(item)!={'code_id','callcount','reccallcount','inlinetime','totaltime'} or item['code_id'] not in labels:raise ValueError('private row STOP')
        if any(type(item[k]) is not int for k in ('callcount','reccallcount') if item[k] is not None) or any(type(item[k]) not in (int,float) or not math.isfinite(item[k]) for k in ('inlinetime','totaltime') if item[k] is not None):raise ValueError('row numeric STOP')
    # References contain only explicit scalar observations / the two async parts.
    def refs(value):
        if value is None:return
        if type(value) is not dict or set(value)-{'reference_ms','markers','work','wait','unrelated_ms','worker_ms','worker_distinct','ready','trailing'}:raise ValueError('reference privacy STOP')
        for k,v in value.items():
            if k in ('work','wait'):refs(v)
            elif k in ('worker_distinct','ready','trailing'):
                if type(v) is not bool:raise ValueError('reference bool STOP')
            elif type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError('reference numeric STOP')
    refs(row['reference'])
    if row['callback_scope']!='OWN_PROFILE_WEAKREF_NOT_GLOBAL_CALLBACK_SNAPSHOT':raise ValueError('callback scope STOP')
    if type(row['own_edges']) is not list or len(row['own_edges'])>20:raise ValueError('edge cap STOP')
    for edge in row['own_edges']:
        if type(edge) is not dict or set(edge)!={'parent','child','callcount','reccallcount','inlinetime','totaltime'} or edge['parent'] not in labels or edge['child'] not in labels:raise ValueError('edge privacy STOP')
        for key in ('callcount','reccallcount','inlinetime','totaltime'):
            value=edge[key]
            if value is not None and (type(value) not in ((int,) if key in ('callcount','reccallcount') else (int,float)) or not math.isfinite(value)):raise ValueError('edge numeric STOP')
    failure=row['first_guard']
    if failure is not None:
        if type(failure) is not dict or set(failure)!={'scope','kind','code_id','field','value','unit'} or failure['scope'] not in ('ENTRY','EDGE') or failure['field'] not in ('callcount','reccallcount','inlinetime','totaltime'):raise ValueError('failure schema STOP')
        if failure['kind'] not in ('TYPE_OR_NONFINITE','NEGATIVE') or failure['unit'] not in ('COUNT','SECONDS'):raise ValueError('failure kind STOP')
        if failure['code_id'] is None and failure['value'] is not None:raise ValueError('unknown value STOP')
        if failure['code_id'] is not None and failure['code_id'] not in labels:raise ValueError('failure ID STOP')
        if failure['value'] is not None and (type(failure['value']) not in (int,float) or not math.isfinite(failure['value'])):raise ValueError('failure number STOP')
    if row['status']=='PASS' and (row['accounting_qualified'] is not True or row['ownership_qualified'] is not True or any(row[k] is not True for k in ('hook_entry','hook_serialize','hook_cleanup','tool_entry_free','tool_cleanup_free','events_cleanup_zero','own_callbacks_released','executor_settled'))):raise ValueError('incomplete pass STOP')
    if row['accounting_qualified'] and (not row['ownership_qualified'] or row['status']!='PASS' or any(row[k]!=0 for k in ('negative_entries','negative_edges','invalid_entries','invalid_edges'))):raise ValueError('quality STOP')
    return row


def produce(expected):
    identity=runtime_identity()
    if sys.version_info[:2]!=(3,12) or type(expected) is not dict or identity!=expected:raise ValueError('parent identity STOP')
    catalog=source_catalog();rows=[];comparison=None
    for phase in PHASES:
        row=run_phase(phase,'INTEGER_NS',catalog);validate_phase(row,set(catalog.values()));rows.append(row)
        if row['status']!='PASS':
            if row['status']=='ACCOUNTING_FAILURE' and row['ownership_qualified']:
                comparison=run_phase(phase,'FLOAT_SECONDS',catalog);validate_phase(comparison,set(catalog.values()))
            break
    result=dict(component='PRIMITIVE_PRODUCER_ONLY_NOT_APP_AUTHORIZATION',identity=identity,caps=CAPS,tolerance=TOLERANCE,integer=rows,float_comparison=comparison,remaining_unrun=list(PHASES[len(rows):]),qualified=len(rows)==4 and all(r['status']=='PASS' for r in rows),application_authorized=False)
    return result


if __name__=='__main__':
    try:
        # Parent writes its independently Popen-bound identity file, then releases
        # this one-byte pipe barrier. No profiler exists before the readback.
        if sys.stdin.buffer.read(1)!=b'R':raise ValueError('parent readiness STOP')
        expected_path=Path(os.environ['NATIVE_PRIMITIVE_EXPECTED_FILE'])
        if expected_path.is_symlink() or not expected_path.is_file() or expected_path.stat().st_size>65536:raise ValueError('parent header STOP')
        result=produce(json.loads(expected_path.read_bytes()))
    except Exception:result=dict(component='PRIMITIVE_PRODUCER_ONLY_NOT_APP_AUTHORIZATION',qualified=False,application_authorized=False,status='SETUP_FAILURE',integer=None,float_comparison=None,remaining_unrun=None)
    data=json.dumps(result,allow_nan=False).encode()
    if len(data)>CAPS['output_bytes']:raise SystemExit('artifact cap STOP')
    Path(os.environ['NATIVE_PRIMITIVE_PROOF']).write_bytes(data+b'\n')
    raise SystemExit(0 if result['qualified'] else 1)
