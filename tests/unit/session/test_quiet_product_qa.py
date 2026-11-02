"""SOURCE ONLY until exact coordinator admission; ordinary repo pytest fixtures.

Real Session ingress/dispatch, counted scripted provider and actual eval worker.
Only custom effect tool/provider are synthetic. No native notifier/peer transport.
Goal/serving supplement and certificate coverage is explicitly NOT claimed here.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import pytest

from local_operator.harness.types import (
    AgentEndEvent, AgentTool, Message, StreamEndEvent, StreamTextDelta,
    StreamToolCallDelta, TextContent, ToolExecutionEndEvent,
    ToolExecutionStartEvent, ToolResult,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.session.transcript import Transcript
from local_operator.tools import eval as eval_tool
from tests.unit.session.test_session import make_session
import local_operator.session.session as session_module

WT = Path(__file__).resolve().parents[3]
REPORTS = Path(os.environ.get('RUNNER_TEMP', '/tmp')) / 'quiet-qa-temporary'


def call(name, args=None, cid='synthetic-call', index=0):
    return StreamToolCallDelta(index=index, id=cid, name=name,
                               argument_delta=json.dumps(args or {}))


def batch(*calls):
    return [*calls, StreamEndEvent(stop_reason='toolUse')]


def final(text='QA_FINAL_SENTINEL'):
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason='stop')]


class RecordingProvider:
    def __init__(self, scripts, timeline):
        self.scripts, self.timeline, self.requests = scripts, timeline, []

    def __call__(self, request, signal):
        self.requests.append(request)
        self.timeline.append({'kind': 'provider', 'ordinal': len(self.requests),
                              'purpose': getattr(request, 'purpose', None),
                              'request': request.model_dump(mode='json')})
        if len(self.requests) > len(self.scripts):
            raise AssertionError('Unscripted provider inference')
        script = self.scripts[len(self.requests)-1]
        async def stream():
            for event in script:
                yield event
        return stream()


async def settled(session, predicate=lambda: True):
    async def wait():
        while not (predicate() and not session.is_streaming
                   and not session._turn_lock.locked()):
            await asyncio.sleep(.005)
    await asyncio.wait_for(wait(), 8)


def serialized(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    return str(value)


# Proof uploads are opt-in to CI only and contain projections, never full prompts,
# request dumps, raw history, environment or arbitrary output. Raw fixture state
# may remain temporary for assertions; only this allowlisted projection uploads.
PROOF_LIMIT = 96 * 1024
_PROOF_SERIAL = {}
_SENTINELS = ('QA_FINAL_SENTINEL', 'PROBE_UNWANTED_FINAL',
              'QA_NEW_USER_ANSWER', 'QA_AFTER_REQUEST_ERROR',
              'QA_REQUIRED_AFTER_BARRIER', 'QA typed job handled', 'SUMMARY')

def _origin(path):
    p=Path(path).resolve()
    try: relative=str(p.relative_to(WT))
    except ValueError: return {'outside_checkout':True}
    return {'path':relative,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def _event_proof(event):
    if not isinstance(event,dict): return {}
    out={k:event[k] for k in ('type','generation','quiet_requested','quiet_accepted',
        'aborted','notify','tool_name','is_error') if k in event}
    out['error_present']=bool(event.get('error'))
    result=event.get('result')
    if isinstance(result,dict):
        out['is_error']=bool(result.get('is_error'))
        details=result.get('details') or {}
        if isinstance(details,dict):
            out['markers']={k:details[k] for k in ('__quiet_turn','quiet_request_pending',
                'quiet_accepted') if isinstance(details.get(k),bool)}
        parts=result.get('content') or []
        content='\n'.join(p.get('text','') for p in parts if isinstance(p,dict))
        out['sentinel_tags']=[tag for tag in _SENTINELS if tag in content]
        for line in content.splitlines():
            if line.startswith('QA_WORKER_IDENTITY='):
                try:
                    worker=json.loads(line.split('=',1)[1])
                    out['worker_origin']=_origin(worker['path'])
                    out['worker_ids']={k:worker[k] for k in ('pid','pgid') if isinstance(worker.get(k),int)}
                except (ValueError,KeyError,TypeError): out['worker_origin_parse_error']=True
    return out

def worker_identity_code():
    expected=hashlib.sha256(Path(eval_tool.__file__).with_name('eval_worker.py').read_bytes()).hexdigest()
    return ('import os,json,hashlib,pathlib,sys\n'
        'p=pathlib.Path(sys.modules["__main__"].__file__).resolve()\n'
        f'assert p.is_relative_to(pathlib.Path({str(WT)!r}))\n'
        f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected!r}\n'
        "print('QA_WORKER_IDENTITY='+json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),"
        "'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))\n")


def emit_proof(label,report):
    if os.environ.get('CI','').strip().lower() not in {'1','true'}: return
    # Hash node identity instead of interpolating arbitrary test ids into paths.
    node=os.environ.get('PYTEST_CURRENT_TEST',label).split(' (',1)[0]
    key=hashlib.sha256((node+'|'+label).encode()).hexdigest()[:24]
    serial=_PROOF_SERIAL.get(key,0);_PROOF_SERIAL[key]=serial+1
    proof={'node_digest':key,'label':label,'source':{
        'session':_origin(session_module.__file__),'eval':_origin(eval_tool.__file__)},
        'counts':{},'events':[],'timeline':[]}
    for name,value in report.items():
        if isinstance(value,(bool,int,float)) and name in {
            'provider_requests','requests','request_count','summary_requests',
            'ordinary_before','ordinary_after','injected'}: proof['counts'][name]=value
        if name.endswith('requests') and isinstance(value,list):
            proof['counts'][name]=len(value)
            proof.setdefault('purposes',{})[name]=[v.get('purpose') for v in value if isinstance(v,dict)]
    for e in report.get('events',report.get('ends',[])):
        if isinstance(e,dict): proof['events'].append(_event_proof(e))
    for row in report.get('timeline',[]):
        if not isinstance(row,dict):continue
        if row.get('kind')=='event': proof['timeline'].append({'event':_event_proof(row.get('event',{}))})
        else: proof['timeline'].append({k:row[k] for k in ('kind','ordinal','purpose','judge',
            'held_requested','held_accepted') if k in row})
    if isinstance(report.get('effects'),list):
        proof['side_effect_count']=len(report['effects'])
        proof['side_effect_tags']=['synthetic_effect_written']*min(len(report['effects']),16)
    if isinstance(report.get('deliveries'),list): proof['delivery_count']=len(report['deliveries'])
    for keyname in ('records','cold','raw_replay','entries','continuations','owned_kernels',
                    'before_context','after_context','calls'):
        value=report.get(keyname)
        if isinstance(value,list):
            proof['counts'][keyname]=len(value)
            if keyname=='cold':proof['counts']['cold_valid']=sum(v is not None for v in value)
            if keyname=='owned_kernels':proof['kernel_exit']=[{k:r[k] for k in ('pid','before','after') if k in r} for r in value]
    if isinstance(report.get('calls'),list):
        proof['admissions']=[{'event':_event_proof(v.get('event',{})),
            'goal_loop_running':bool(v.get('goal_loop_running'))} for v in report['calls']]
    for phase in ('start','finish'):
        value=report.get(phase)
        if isinstance(value,dict):
            proof['goal_'+phase]={'status':value.get('status'),
                'goal_present':bool(value.get('goal')),'token_present':bool(value.get('token'))}
    attention=report.get('attention')
    if isinstance(attention,dict):
        proof['attention']={k:attention[k] for k in ('kind','notify','unseen','qa_ends','qa_accepted') if k in attention}
        proof['attention']['completion_present']=bool(attention.get('completion_token'))
        reopened=attention.get('qa_reopened_attention')
        if isinstance(reopened,dict):proof['restart']={k:reopened[k] for k in ('kind','notify','unseen') if k in reopened}
    # Raw tool/message text is never retained; tags only recognize fixed synthetic
    # sentinels and cannot turn arbitrary provider output into uploaded prose.
    packed=json.dumps(report,default=str)
    proof['sentinel_tags']=[tag for tag in _SENTINELS if tag in packed]
    encoded=json.dumps(proof,sort_keys=True).encode()
    if len(encoded)>PROOF_LIMIT:
        # Preserve counts+an explicit truncation fact; do not cut JSON mid-byte.
        proof['timeline']=proof['timeline'][:80];proof['events']=proof['events'][:80]
        proof['truncated']=True
        encoded=json.dumps(proof,sort_keys=True).encode()
    if len(encoded)>PROOF_LIMIT:
        proof.pop('timeline',None);proof.pop('events',None)
        proof['truncated']=True
        encoded=json.dumps(proof,sort_keys=True).encode()
    if len(encoded)>PROOF_LIMIT:
        encoded=json.dumps({'node_digest':key,'label':label,'counts':proof['counts'],
            'truncated':True,'omitted_oversize_details':True},sort_keys=True).encode()
    assert len(encoded)<=PROOF_LIMIT, 'bounded proof projection exceeded its byte limit'
    directory=WT/'test-artifacts'/'quiet-proof';directory.mkdir(parents=True,exist_ok=True)
    with (directory/(key+'-'+str(serial)+'.json')).open('xb') as output:output.write(encoded)


CASES = [
    'direct', 'wrapped', 'wrapped-post-tool', 'wrapped-exception',
    'forged-marker', 'user-direct', 'user-wrapped',
    'wake-required', 'wake-quiet', 'notify-then-quiet',
    'queued-user', 'after-accepted-user',
    'batch-quiet-early', 'batch-quiet-final', 'batch-failed-sibling',
    'parallel-two-effects', 'ordinary-error', 'cancel-and-fresh', 'restart-attention',
]


@pytest.mark.parametrize('case', CASES)
def test_real_session_product(case, tmp_path, monkeypatch):
    asyncio.run(run_case(case, tmp_path, monkeypatch))


async def run_case(case, tmp_path, monkeypatch):
    # Repo autouse isolate_environment runs before this body. Bind config/cwd
    # explicitly to that per-test HOME, never to an inherited fallback path.
    home = Path.home().resolve()
    assert home.is_absolute() and home != WT
    assert not any(k.startswith('CMUX_') for k in os.environ)
    # env -i removes inherited LOP_* before imports; ordinary repo fixtures
    # legitimately restore their documented safety gates afterward.
    monkeypatch.setenv('LOCAL_OPERATOR_CONFIG_DIR', str(home / 'config'))
    monkeypatch.chdir(home)
    assert Path(session_module.__file__).resolve().is_relative_to(WT)
    assert Path(eval_tool.__file__).resolve().is_relative_to(WT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    directory = tmp_path / case
    directory.mkdir()
    effects = directory / 'effects.jsonl'
    timeline, events, kernel_proof = [], [], []
    side_count = 0
    blocked = []

    async def effect(cid, args, signal, update, context):
        nonlocal side_count
        side_count += 1
        await asyncio.sleep(0)  # Allow shared sibling scheduling, never wall-time racing
        with effects.open('a') as output:
            output.write(json.dumps({'id': cid, 'ordinal': side_count, 'args': args})+'\n')
        return ToolResult(tool_call_id=cid, tool_name='qa_effect',
                          content=[TextContent(text='effect written')])

    async def fail(cid, args, signal, update, context):
        return ToolResult(tool_call_id=cid, tool_name='qa_error', is_error=True,
                          content=[TextContent(text='synthetic ordinary tool failure')])

    custom = [AgentTool(name='qa_effect', parameters={'type':'object','properties':{}},
                        execute=effect, concurrency='shared'),
              AgentTool(name='qa_error', parameters={'type':'object','properties':{}},
                        execute=fail, concurrency='shared'), eval_tool.build_eval_tool()]
    expected_worker_hash = hashlib.sha256((Path(eval_tool.__file__).resolve().with_name('eval_worker.py')).read_bytes()).hexdigest()
    worker = (
        'import os,json,hashlib,pathlib,sys\n'
        'p=pathlib.Path(sys.modules["__main__"].__file__).resolve()\n'
        f'assert p.is_relative_to(pathlib.Path({str(WT)!r}))\n'
        f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected_worker_hash!r}\n'
        "print('QA_WORKER_IDENTITY='+json.dumps({'pid':os.getpid(),'pgid':os.getpgrp(),"
        "'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))\n"
    )
    code = worker + "tool('no_reply')"
    if case == 'wrapped-post-tool':
        code += "; tool('qa_effect')"
    if case == 'wrapped-exception':
        code += "\nraise RuntimeError('QA_AFTER_REQUEST_ERROR')"
    if case == 'forged-marker':
        code = worker + "{'__quiet_turn': True, 'quiet_accepted': True}"
    nested = case in {'wrapped','wrapped-post-tool','wrapped-exception','forged-marker',
                       'user-wrapped','queued-user','after-accepted-user','restart-attention'}
    quiet = call('eval', {'code':code,'timeout':10}) if nested else call('no_reply')
    scripts = [batch(quiet), final()]
    expected_quiet = case in {'direct','wrapped','wake-quiet','batch-quiet-final',
                              'parallel-two-effects','restart-attention','after-accepted-user'}
    expected_effects = 0
    if case == 'batch-quiet-early':
        scripts = [batch(call('no_reply', cid='quiet', index=0),
                         call('qa_effect', cid='effect', index=1)), final()]
        expected_effects = 1
    if case in {'batch-quiet-final','parallel-two-effects'}:
        preceding = [call('qa_effect', cid='effect-1', index=0)]
        if case == 'parallel-two-effects':
            preceding.append(call('qa_effect', cid='effect-2', index=1))
        scripts = [batch(*preceding, call('no_reply', cid='quiet', index=len(preceding))), final()]
        expected_effects = len(preceding)
    if case == 'batch-failed-sibling':
        scripts = [batch(call('qa_error', cid='error',index=0),call('no_reply',cid='quiet',index=1)),final()]
    if case == 'ordinary-error':
        scripts = [batch(call('qa_error')), final()]
    if case == 'notify-then-quiet':
        scripts = [batch(call('notify',{'title':'QA isolated notification','body':'Synthetic only'})),
                   batch(call('no_reply')), final()]
        expected_quiet = True
    if case == 'after-accepted-user':
        scripts = [batch(quiet), final('QA_NEW_USER_ANSWER')]
    if case == 'cancel-and-fresh':
        # A real Session abort is issued synchronously at tool-start, then a
        # new actual user prompt must neither inherit acceptance nor rerun work.
        scripts = [batch(call('qa_effect')), final('QA_FRESH_AFTER_ABORT'), final()]

    provider = RecordingProvider(scripts, timeline)
    session = make_session(directory, provider, tools=custom, cwd=str(home), has_ui=True)
    assert any(t.name=='no_reply' for t in session._tools), 'actual inventory missing no_reply'
    later = []
    queued = False
    def observe(event):
        nonlocal queued
        events.append(event)
        timeline.append({'kind':'event','event':serialized(event)})
        if case == 'queued-user' and isinstance(event, ToolExecutionStartEvent) and not queued:
            queued = True
            session.steer('QA queued actual user', message_id='qa-queued-user')
        if case == 'cancel-and-fresh' and isinstance(event, ToolExecutionStartEvent) and not queued:
            queued = True
            session.abort()
        if (case == 'after-accepted-user' and isinstance(event, AgentEndEvent)
                and event.quiet_accepted and not queued):
            queued = True
            # Public ingress scheduled during terminal subscriber fan-out;
            # assertions below use emitted boundary order, not internal calls.
            later.append(asyncio.create_task(session.prompt('QA later actual user',
                message_id='qa-after-accepted-user')))
    unsubscribe = session.subscribe(observe)
    async def approve(name, description):
        return name in {'eval','qa_effect','qa_error','no_reply','notify'}
    session.set_approval_handler(approve)
    snapshot = {}
    reopened = None
    try:
        if case in {'user-direct','user-wrapped'}:
            await asyncio.wait_for(session.prompt('QA actual user requires words'),20)
        elif case in {'wake-required','wake-quiet','notify-then-quiet'}:
            wake = WakeSchedule(id='qa-wake',message='QA synthetic wake',next_due_at=0,
                                created_at=0,notify=case!='wake-quiet')
            await session._deliver_wake(DueWake(schedule=wake,occurrence=1,planned_total=1,final=True))
        else:
            await session.receive_peer_message('QA synthetic peer result',wake=True,
                sender={'pid':700003,'conversation_name':'qa-synthetic'},
                message_id='peer-'+'c'*32)
        await settled(session, lambda: bool(provider.requests))
        if later:
            await asyncio.wait_for(asyncio.gather(*later),20)
            await settled(session)
        if case == 'cancel-and-fresh':
            await asyncio.wait_for(session.prompt('QA fresh real user after cancellation'),20)
            await settled(session)
        snapshot = await session.refresh_attention()
        snapshot['qa_attention_journal'] = Transcript(directory / 'sess').latest_custom('completion_attention')
        ends = [e for e in events if isinstance(e,AgentEndEvent)]
        accepted = [e for e in ends if getattr(e,'quiet_accepted',False)]
        snapshot.update({'qa_ends':len(ends),'qa_accepted':len(accepted)})
        if case == 'restart-attention':
            await session.dispose()
            empty = RecordingProvider([],timeline)
            reopened = make_session(directory, empty, tools=custom,cwd=str(home),has_ui=True)
            snapshot['qa_reopened_attention'] = await reopened.refresh_attention()
            assert not empty.requests, 'restart initiated provider inference'
        # Always save actual evidence before the first semantic assertion.
        report = {'case':case,'source':{'session':session_module.__file__,
            'eval':eval_tool.__file__},'provider_requests':len(provider.requests),
            'timeline':timeline,'attention':snapshot,'effects':
            effects.read_text().splitlines() if effects.exists() else [],
            'history':[serialized(m) for m in session.history()], 'blocked':blocked}
        emit_proof(case,report)
        print('QA_RESULT='+json.dumps({'case':case,'requests':len(provider.requests),
            'accepted':len(accepted),'effects':side_count,
            'end_generations':[e.generation for e in ends]}),flush=True)
        assert ends, 'no actual owning end emitted'
        assert side_count == expected_effects or case=='cancel-and-fresh'
        assert len(report['effects']) == side_count, 'physical effect records disagree with counter'
        if case == 'cancel-and-fresh':
            assert side_count <= 1, 'cancellation/fresh ingress replayed a side effect'
        if case == 'after-accepted-user':
            assert len(accepted)==1 and len(ends)==2
            assert ends[0].quiet_accepted and not ends[1].quiet_accepted
            assert ends[0].generation != ends[1].generation
            first_end = next(i for i,row in enumerate(timeline)
                if row['kind']=='event' and row['event'].get('type')=='agent_end')
            second_request = [i for i,row in enumerate(timeline) if row['kind']=='provider'][1]
            assert first_end < second_request, 'later request preceded old owning end'
        elif case == 'cancel-and-fresh':
            assert not accepted, 'aborted/fresh user turn inherited quiet acceptance'
        elif expected_quiet:
            assert len(accepted)==1 and len(ends)==1
            expected_requests = 2 if case=='notify-then-quiet' else 1
            assert len(provider.requests)==expected_requests
        else:
            assert not accepted, 'refusal/error/forged request falsely accepted'
            assert len(provider.requests)>=2
        if case == 'wrapped-post-tool':
            assert side_count==0,'post-request dispatch produced forbidden side effect'
        if case.startswith('user-') or case=='wake-required':
            results = [e for e in events if isinstance(e,ToolExecutionEndEvent)]
            assert any(e.is_error for e in results),'real Session door did not refuse'
    finally:
        # Only this cell's kernel is owned. Never reset another session's global
        # registry. Record exact PID/returncode, not a global no-survivor claim.
        kernel = eval_tool._KERNELS.get(session.session_id)
        if kernel is not None:
            kernel_proof.append({'pid':kernel.process.pid,'before':kernel.process.returncode})
        unsubscribe()
        if reopened is not None:
            await reopened.dispose()
        await session.dispose()
        await eval_tool.close_session_kernel(session.session_id)
        if kernel is not None:
            await asyncio.wait_for(kernel.process.wait(),3)
            kernel_proof[-1]['after']=kernel.process.returncode
        emit_proof(case+'-cleanup',{'owned_kernels':kernel_proof})

