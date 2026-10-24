"""COMPONENT ONLY: two original S3 calls; consumer/lifecycle allocation is separate."""
import asyncio
import ast
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import CodeType, FunctionType
from unittest.mock import patch
import pytest
from scripts.diagnostics.native_worker_6911.carrier import native_sources

NODE='tests/unit/test_tui_responsiveness.py::test_s3_harvest_with_unlisted_model_and_cold_memo_keeps_the_loop_free'
MODES=('healthy','async-delay')
ROWS=[]
REPORTS=[]
CALLS=0
SOURCES=('tests/unit/test_tui_responsiveness.py','tests/unit/lfs_s3_positive_plugin_6911.py','tests/conftest.py','tests/unit/tui/test_app_pilot.py','local_operator/tui/app.py','local_operator/model/configure.py','local_operator/model/discovery.py','local_operator/model/costs.py')


def build_row(mode,token,identity):
    if mode not in MODES or type(token) is not int or token not in (1,2):raise ValueError('case STOP')
    return dict(mode=mode,token=token,identity=identity,native_sources=None,stage='SOURCE',compiled=False,
        entry_calls=0,target_seen=False,held_ticks=None,held_before_manual=False,
        manual_calls=0,manual_cpu_ms=0.0,shadow_wall_ms=None,async_injections=0,
        loop_gate_waits=0,worker_offloop=None,owned_submissions=0,
        pending_before_invalidate=None,sampler_stop_returned=False,gate_released=None,
        call_pass=False,entry_restored=False,discovery_restored=False,patches_restored=False)


def validate_row(row):
    keys=set(build_row('healthy',1,None))
    if type(row) is not dict or set(row)!=keys:raise ValueError('row privacy STOP')
    identity=row['identity']
    if type(identity) is not dict or set(identity)!={'python','executable_sha256','pid','ppid','pgid','sources'}:raise ValueError('identity schema STOP')
    def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
    if type(identity['python']) is not list or len(identity['python'])!=3 or identity['python'][:2]!=[3,12] or any(type(v) is not int for v in identity['python']):raise ValueError('Python STOP')
    if not sha(identity['executable_sha256']) or any(type(identity[k]) is not int or identity[k]<=0 for k in ('pid','ppid','pgid')):raise ValueError('process binding STOP')
    if type(identity['sources']) is not dict or set(identity['sources'])!=set(SOURCES) or not all(sha(v) for v in identity['sources'].values()):raise ValueError('source binding STOP')
    if row['mode'] not in MODES or type(row['token']) is not int or row['token'] not in (1,2):raise ValueError('case STOP')
    if row['stage'] not in ('SOURCE','ARMED','CALL','RESTORED'):raise ValueError('stage STOP')
    for key in ('entry_calls','manual_calls','async_injections','loop_gate_waits','owned_submissions'):
        if type(row[key]) is not int or not 0<=row[key]<=10000:raise ValueError('count STOP')
    for key in ('held_ticks','pending_before_invalidate'):
        if row[key] is not None and (type(row[key]) is not int or not 0<=row[key]<=10000):raise ValueError('optional count STOP')
    for key in ('manual_cpu_ms','shadow_wall_ms'):
        if row[key] is not None and (type(row[key]) not in (int,float) or not math.isfinite(row[key]) or row[key]<0):raise ValueError('clock STOP')
    for key in ('compiled','target_seen','held_before_manual','sampler_stop_returned','call_pass','entry_restored','discovery_restored','patches_restored'):
        if type(row[key]) is not bool:raise ValueError('bool STOP')
    if any(row[k] is not None and type(row[k]) is not bool for k in ('worker_offloop','gate_released')):raise ValueError('thread STOP')
    return row


def code_matches(function,source,filename):
    if type(function) is not FunctionType or function.__code__.co_filename!=filename:return False
    frozen=compile(source,filename,'exec',flags=0,dont_inherit=True)
    found=[]
    def walk(code):
        if type(code) is CodeType:
            if code.co_qualname==function.__code__.co_qualname:found.append(code)
            for value in code.co_consts:
                if type(value) is CodeType:walk(value)
    walk(frozen)
    return len(found)==1 and found[0]==function.__code__



# Opt-in producer controls only. The positive controller intentionally does not
# accept these records; extending its policy is a separately reviewed component.
NEGATIVE_MODES=('manual-cpu','inline-offload','body-fault','body-stop-fault')

class BodyFault(RuntimeError):pass
class StopFault(RuntimeError):pass


def selected_modes():
    raw=os.environ.get('S3_NEGATIVE_MODES')
    if raw is None:return MODES
    value=json.loads(raw)
    allowed=(NEGATIVE_MODES,('healthy','async-delay','manual-cpu'),('inline-offload','body-fault','body-stop-fault'))
    if type(value) is not list or tuple(value) not in allowed:raise ValueError('mode selection STOP')
    return tuple(value)


def negative_state(mode):
    if mode not in NEGATIVE_MODES:raise ValueError('negative mode STOP')
    return dict(manual_started=0,manual_completed=0,manual_samples_ms=[],inline_calls=0,
        inline_refused=None,body_fault_raised=False,stop_fault_raised=False,oracle='UNAVAILABLE')


def build_negative_row(mode,token,identity):
    if mode not in NEGATIVE_MODES or type(token) is not int or not 1<=token<=6:raise ValueError('negative case STOP')
    row=build_row('healthy',1,identity);row.update(mode=mode,token=token,control=negative_state(mode));return row


def validate_negative_row(row):
    if type(row) is not dict or 'control' not in row:raise ValueError('negative schema STOP')
    control=row['control'];base={k:v for k,v in row.items() if k!='control'}
    if base.get('mode') not in NEGATIVE_MODES or type(base.get('token')) is not int or not 1<=base['token']<=6:raise ValueError('negative case STOP')
    base.update(mode='healthy',token=1);validate_row(base)
    if type(control) is not dict or set(control)!=set(negative_state('body-fault')):raise ValueError('control privacy STOP')
    for key in ('manual_started','manual_completed','inline_calls'):
        if type(control[key]) is not int or not 0<=control[key]<=10000:raise ValueError('control count STOP')
    if control['manual_completed']>control['manual_started']:raise ValueError('completion STOP')
    samples=control['manual_samples_ms']
    if type(samples) is not list or len(samples)>3 or len(samples)>control['manual_started'] or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in samples):raise ValueError('sample STOP')
    if control['inline_refused'] is not None and type(control['inline_refused']) is not bool:raise ValueError('refusal STOP')
    if any(type(control[k]) is not bool for k in ('body_fault_raised','stop_fault_raised')):raise ValueError('fault bool STOP')
    if control['oracle'] not in ('UNAVAILABLE','ORIGINAL_PASS','ORIGINAL_MANUAL_ASSERT','ORIGINAL_HELD_ASSERT','BODY_FAULT','BODY_STOP_GROUP','UNEXPECTED'):raise ValueError('oracle STOP')
    return row


def controlled_manual(mode,state,invoke,clock):
    state['manual_started']+=1;start=clock()
    try:
        if mode in ('body-fault','body-stop-fault') and state['manual_started']==1:
            state['body_fault_raised']=True;raise BodyFault()
        if mode=='manual-cpu':
            while clock()-start<.2:pass
        result=invoke();state['manual_completed']+=1;return result
    finally:state['manual_samples_ms'].append((clock()-start)*1000)


def failure_oracle(error,mode,state,manual_code,manual_line,held_code):
    if error is None:return 'ORIGINAL_PASS'
    if mode=='body-fault' and type(error) is BodyFault:return 'BODY_FAULT'
    if mode=='body-stop-fault' and type(error) is ExceptionGroup and len(error.exceptions)==2 and tuple(type(e) for e in error.exceptions)==(BodyFault,StopFault):return 'BODY_STOP_GROUP'
    if type(error) is AssertionError:
        trace=error.__traceback__
        while trace is not None:
            if mode=='manual-cpu' and trace.tb_frame.f_code is manual_code and trace.tb_lineno==manual_line:return 'ORIGINAL_MANUAL_ASSERT'
            if mode=='inline-offload' and trace.tb_frame.f_code is held_code and state['inline_refused'] is True:return 'ORIGINAL_HELD_ASSERT'
            trace=trace.tb_next
    return 'UNEXPECTED'


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__==NODE.split('::')[1]:
        metafunc.parametrize('_s3_mode',selected_modes(),indirect=True)


@pytest.fixture(autouse=True)
def _s3_mode(request):return request.param


@pytest.fixture(autouse=True)
def _owned_config(isolate_environment,monkeypatch):
    root=Path(os.environ['S3_PRODUCER_ROOT']).resolve()
    for key in ('HOME','TMPDIR','LOCAL_OPERATOR_RUNTIME_ROOT'):
        if not Path(os.environ[key]).resolve().is_relative_to(root):raise ValueError('isolation STOP')
    monkeypatch.setenv('LOCAL_OPERATOR_CONFIG_DIR',str(root/'config'))


def pytest_collection_modifyitems(items):
    modes=selected_modes()
    if len(items)!=len(modes) or len({id(x) for x in items})!=len(modes) or [x.callspec.params['_s3_mode'] for x in items]!=list(modes) or any(x.nodeid.split('[')[0]!=NODE for x in items):raise ValueError('two distinct original calls STOP')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global CALLS
    CALLS+=1
    mode=item.callspec.params['_s3_mode']
    modes=selected_modes()
    if not 1<=CALLS<=len(modes) or mode!=modes[CALLS-1]:raise ValueError('call order STOP')
    from local_operator.tui import app as app_module
    from local_operator.model import configure,discovery,costs
    from textual.pilot import Pilot
    import tests.conftest as policy
    from tests.unit.tui import test_app_pilot as pilot_policy
    from scripts.diagnostics.native_worker_6911.carrier import native_sources
    from scripts.diagnostics.lfs_pair_6911.controller import validate_identity
    root=Path(__file__).resolve().parents[2]
    module=item.module
    paths={'tests/unit/test_tui_responsiveness.py':module.__file__,
        'tests/unit/lfs_s3_positive_plugin_6911.py':__file__,
        'tests/conftest.py':policy.__file__,'tests/unit/tui/test_app_pilot.py':pilot_policy.__file__,
        'local_operator/tui/app.py':app_module.__file__,'local_operator/model/configure.py':configure.__file__,
        'local_operator/model/discovery.py':discovery.__file__,'local_operator/model/costs.py':costs.__file__}
    hashes={}
    for label,path in paths.items():
        actual=Path(path).resolve()
        if actual!=root/label:raise ValueError('source path STOP')
        hashes[label]=hashlib.sha256(actual.read_bytes()).hexdigest()
    identity=dict(python=list(sys.version_info[:3]),executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),sources=hashes)
    expected=json.loads(os.environ['S3_PRODUCER_EXPECTED'])
    validate_identity(identity,dict(expected,pid=os.getpid()))
    if native_sources()!=json.loads(os.environ['S3_PRODUCER_NATIVE']):raise ValueError('native source STOP')
    row=build_negative_row(mode,(MODES+NEGATIVE_MODES).index(mode)+1,identity) if mode in NEGATIVE_MODES else build_row(mode,CALLS,identity)
    control=row.get('control')
    row['native_sources']=native_sources()
    if row['native_sources']!=json.loads(os.environ['S3_PRODUCER_NATIVE']):raise ValueError('native API binding STOP')
    source=(root/'tests/unit/test_tui_responsiveness.py').read_text()
    # item.obj may be pytest-asyncio's sync runner wrapper. The original
    # module function is assertion-rewritten, so bind source/native identity;
    # only the unrewritten helper methods use structural compiled equality.
    target=module.__dict__[NODE.split('::')[1]]
    filename=str(root/'tests/unit/test_tui_responsiveness.py')
    definition=next(n for n in ast.parse(source).body if isinstance(n,ast.AsyncFunctionDef) and n.name==NODE.split('::')[1])
    first=min([definition.lineno]+[d.lineno for d in definition.decorator_list])
    if type(target) is not FunctionType or target.__code__.co_filename!=filename or target.__code__.co_qualname!=definition.name or target.__code__.co_firstlineno!=first:raise ValueError('target source identity STOP')
    row['compiled']=all(code_matches(fn,source,filename) for fn in (module._HeldDiscovery.__call__,module._HeldDiscovery.await_held))
    if not row['compiled']:raise ValueError('compiled helper STOP')
    original_entry=configure.refresh_model_info_background;original_discovery=discovery.available_models
    original_init=module._HeldDiscovery.__init__;original_harvest=app_module.OperatorApp._harvest_subagent_costs
    original_held_call=module._HeldDiscovery.__call__
    original_pause=Pilot.pause;original_stop=module.StallRecorder.stop;original_invalidate=configure.invalidate_model_info_cache
    owned=set();held=[];started=None;loop_thread=threading.get_ident()
    target_entry_active=False
    manual_line=next(n.lineno for n in ast.walk(definition) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare) and ast.dump(n.test)==ast.dump(ast.parse('harvest_cpu_ms < 500.0',mode='eval').body))
    with ExitStack() as stack:
        def held_call(self,*args,**kwargs):
            before=self.inline_violations
            try:return original_held_call(self,*args,**kwargs)
            finally:
                if self.inline_violations>before and threading.get_ident()==loop_thread:control['inline_refused']=True
        def init(self,*args,**kwargs):
            original_init(self,*args,**kwargs);held.append(self)
            if len(held)!=1:raise ValueError('held cardinality STOP')
            if mode=='inline-offload':
                loop=asyncio.get_running_loop();original_executor=loop.run_in_executor
                configure_source=(root/'local_operator/model/configure.py').read_text()
                if not code_matches(original_entry,configure_source,str(root/'local_operator/model/configure.py')):raise ValueError('offload source STOP')
                if not code_matches(original_entry,(root/'local_operator/model/configure.py').read_text(),str(root/'local_operator/model/configure.py')):raise ValueError('entry compiled source STOP')
                roots=[c for c in original_entry.__code__.co_consts if type(c) is CodeType and c.co_qualname=='refresh_model_info_background.<locals>._refresh']
                if len(roots)!=1:raise ValueError('offload root STOP')
                def inline(executor,function,*values):
                    if target_entry_active and executor is None and not values and type(function) is FunctionType and function.__code__==roots[0] and function.__code__.co_filename==str(root/'local_operator/model/configure.py'):
                        control['inline_calls']+=1;function()
                        future=loop.create_future();future.set_result(None);return future
                    return original_executor(executor,function,*values)
                stack.enter_context(patch.object(loop,'run_in_executor',inline))
                stack.enter_context(patch.object(module._HeldDiscovery,'__call__',held_call))
            wait=self._gate.wait
            def observed_wait(*a,**kw):
                if threading.get_ident()==loop_thread:row['loop_gate_waits']+=1
                return wait(*a,**kw)
            stack.enter_context(patch.object(self._gate,'wait',observed_wait))
        def entry(provider,model):
            nonlocal target_entry_active
            row['entry_calls']+=1;owned.add((provider,model));row['owned_submissions']=len(owned)
            if (provider,model)==('kimi','never-heard-of-it'):row['target_seen']=True
            if mode=='inline-offload' and (provider,model)==('kimi','never-heard-of-it'):
                target_entry_active=True
                try:return original_entry(provider,model)
                finally:target_entry_active=False
            return original_entry(provider,model)
        def invalidate():
            if owned:row['pending_before_invalidate']=len(owned.intersection(configure._paint_refreshing))
            return original_invalidate()
        def harvest(self,*args,**kwargs):
            nonlocal started
            manual=sys._getframe(1).f_code is target.__code__
            if not manual:return original_harvest(self,*args,**kwargs)
            if row['manual_calls']==0:
                if mode!='inline-offload' and not row['held_before_manual']:raise ValueError('pre-manual witness STOP')
                started=time.perf_counter()
            before=time.thread_time()
            try:
                if control is not None:return controlled_manual(mode,control,lambda:original_harvest(self,*args,**kwargs),time.thread_time)
                return original_harvest(self,*args,**kwargs)
            finally:row['manual_cpu_ms']+=(time.thread_time()-before)*1000;row['manual_calls']+=1
        async def pause(self,*args,**kwargs):
            manual=sys._getframe(1).f_code is target.__code__
            if manual and mode=='async-delay' and row['manual_calls'] in (1,2):
                await asyncio.sleep(.32);row['async_injections']+=1
            result=await original_pause(self,*args,**kwargs)
            if manual and row['manual_calls']==0 and mode!='inline-offload':
                if len(held)!=1:raise ValueError('held setup STOP')
                witness=await held[0].await_held(minimum_ticks=3,timeout_s=2.0)
                row['held_ticks']=witness['ticks'];row['held_before_manual']=witness['parked']>=1 and witness['ticks']>=3
                row['worker_offloop']=held[0].worker_thread!=loop_thread
            if manual and row['manual_calls']==3 and started is not None:row['shadow_wall_ms']=(time.perf_counter()-started)*1000
            return result
        async def stop(self,*args,**kwargs):
            result=await original_stop(self,*args,**kwargs);row['sampler_stop_returned']=True
            if mode=='body-stop-fault' and control['body_fault_raised'] and not control['stop_fault_raised']:
                control['stop_fault_raised']=True;raise StopFault()
            return result
        try:
            for obj,name,value in ((module._HeldDiscovery,'__init__',init),(configure,'refresh_model_info_background',entry),(configure,'invalidate_model_info_cache',invalidate),(app_module.OperatorApp,'_harvest_subagent_costs',harvest),(Pilot,'pause',pause),(module.StallRecorder,'stop',stop)):
                stack.enter_context(patch.object(obj,name,value))
            row['stage']='ARMED'
            outcome=yield
            row['stage']='CALL';row['call_pass']=outcome.excinfo is None
            if control is not None:
                control['oracle']=failure_oracle(None if outcome.excinfo is None else outcome.excinfo[1],mode,control,target.__code__,manual_line,module._HeldDiscovery.await_held.__code__)
        finally:
            row['gate_released']=held[0]._gate.is_set() if len(held)==1 else None
            row['entry_restored']=configure.refresh_model_info_background is entry
            row['discovery_restored']=discovery.available_models is original_discovery
            stack.close()
            row['patches_restored']=all((configure.refresh_model_info_background is original_entry,configure.invalidate_model_info_cache is original_invalidate,module._HeldDiscovery.__init__ is original_init,app_module.OperatorApp._harvest_subagent_costs is original_harvest,Pilot.pause is original_pause,module.StallRecorder.stop is original_stop,module._HeldDiscovery.__call__ is original_held_call))
            row['stage']='RESTORED';ROWS.append(validate_negative_row(row) if control is not None else validate_row(row))


def pytest_runtest_logreport(report):
    modes=selected_modes()
    mode=report.nodeid.split('[')[-1].rstrip(']')
    REPORTS.append(dict(mode=mode if mode in modes else 'UNKNOWN',token=(MODES+NEGATIVE_MODES).index(mode)+1 if mode in modes else None,phase=report.when if report.when in ('setup','call','teardown') else 'UNKNOWN',outcome=report.outcome if report.outcome in ('passed','failed','skipped') else 'UNKNOWN'))
    if len(REPORTS)>3*len(modes):raise ValueError('report cap STOP')


def pytest_sessionfinish(session,exitstatus):
    Path(os.environ['S3_PRODUCER_PROOF']).write_text(json.dumps(dict(component='S3_POSITIVE_PRODUCER_ONLY',rows=ROWS,reports=REPORTS,exitstatus=int(exitstatus)))+'\n')
