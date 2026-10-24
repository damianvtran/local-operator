from pathlib import Path
import os,json,copy,sys
b=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])/'owner-thread-v1';ns={};exec(compile((b/'collector.py').read_text(),'<actual-owner-collector>','exec'),ns)
f={'sys':sys};exec(compile('def root():\n return sys._getframe()\ndef child():\n yield\ndef other():\n yield\n','<owned-static-fixture>','exec'),f)
held=[]
def frame(name):
 if name=='root':return f[name]()
 g=f[name]();held.append(g);return g.gi_frame
root=frame('root');child=frame('child');other=frame('other');catalog=tuple((f[n].__code__,'PRIMITIVE:'+n.upper()) for n in ('root','child'));identity=dict(python=[3,12,15],executable_sha256='a'*64,producer_sha256='b'*64,pid=1,ppid=2,pgid=3,owner_thread=4,static_ids=[label for _,label in catalog]);results=[]
def make(times):
 state=dict(hook=None,thread=4,clock_reads=0,writes=[]);values=iter(times)
 def clock():state['clock_reads']+=1;return next(values)
 def setter(v):state['writes'].append(v);state['hook']=v
 engine=ns['Collector'](root.f_code,catalog,identity,identity,clock,lambda:state['thread'],lambda:state['hook'],setter);state['hook']=engine.callback;return engine,state

def finish(engine):
 engine.cleanup();record=engine.export();return record,ns['finalize'](record,identity)
# Actual engine uses exact native FrameType/CodeType. Events/clocks/hooks/thread
# observations below are SIMULATED; no sys.setprofile or actual thread delivery.
e,s=make([0,10,20,30]);e.callback(root,'call',None);e.callback(child,'call',None);e.callback(child,'return',None);e.callback(root,'return',None);good,full=finish(e);assert full['qualified'];assert good['nodes'][0]['own_ns']==20 and good['nodes'][0]['inclusive_ns']==30 and good['edges'][0]['inclusive_ns']==10;results.append('DIRECT_NESTED_ACTUAL_ENGINE_VALIDATOR_FINALIZER')
e,s=make([0,10,20,30,40,50]);recursive=frame('child');e.callback(root,'call',None);e.callback(child,'call',None);e.callback(recursive,'call',None);e.callback(recursive,'return',None);e.callback(child,'return',None);e.callback(root,'return',None);record,full=finish(e);assert full['qualified'] and record['nodes'][1]['segments']==2 and len(record['edges'])==2;results.append('RECURSIVE_DISTINCT_ACTIVATIONS')
e,s=make([0,10,20,30,130,140,150,160]);e.callback(root,'call',None);e.callback(child,'call',None);e.callback(child,'return',None);e.callback(other,'call',None);e.callback(other,'return',None);e.callback(child,'call',None);e.callback(child,'return',None);e.callback(root,'return',None);record,full=finish(e);assert full['qualified'] and record['nodes'][1]['inclusive_ns']==20 and record['nodes'][2]['static_id'] is None and record['nodes'][2]['own_ns']==100;results.append('YIELD_POP_RESUME_UNRELATED_UNKNOWN_CPU_NOT_CHARGED')
e,s=make([0]);s['thread']=5;e.callback(root,'call',None);assert not e.stack and s['clock_reads']==0;assert not finish(e)[1]['qualified'];results.append('FOREIGN_DELIVERY_BEFORE_CLOCK_STACK')
unknown_good=copy.deepcopy(record)
for bad in (-1,None,True,float('inf')):
 for group in ('nodes','edges'):
  for field in ('own_ns','inclusive_ns','segments'):
   record=copy.deepcopy(good);record[group][0][field]=bad;assert not ns['finalize'](record,identity)['qualified']
 record=copy.deepcopy(unknown_good);record['nodes'][2]['own_ns']=bad;assert not ns['finalize'](record,identity)['qualified']
results.append('ALL_NODE_EDGE_SIGNED_MISSING_BOOL_NONFINITE_REJECT')
e,s=make([100,97]);e.callback(root,'call',None);e.callback(other,'call',None);record,full=finish(e);assert not full['qualified'] and full['signed_value']==-3 and full['failure']=='CLOCK_NEGATIVE';results.append('SIGNED_NEGATIVE_DIAGNOSTIC_PRESERVED')
for kind in ('unmatched','unfinished','cap','foreign_hook','serialization','header','schema'):
 e,s=make([0,1,2]);e.callback(root,'call',None)
 if kind=='unmatched':e.callback(child,'return',None)
 elif kind=='cap':e.events=ns['CAPS']['events'];e.callback(child,'call',None)
 elif kind=='foreign_hook':s['hook']=object()
 elif kind=='serialization':e.nodes[0]['own_ns']='PRIVATE_PAYLOAD'
 elif kind in ('header','schema'):
  record=copy.deepcopy(good)
  if kind=='header':record['identity']['pid']=True
  else:record['PRIVATE_PAYLOAD']='NEVER_ECHO'
  full=ns['finalize'](record,identity);assert not full['qualified'] and 'PRIVATE' not in json.dumps(full);continue
 record,full=finish(e);assert not full['qualified'] and 'PRIVATE' not in json.dumps(full)
 if kind=='foreign_hook':assert s['hook'] is not None and not s['writes'] and record['hook_argument_used'] is True and record['hook_after_cleanup'] is False
results.append('UNMATCHED_UNFINISHED_CAP_FOREIGN_HOOK_SERIALIZATION_HEADER_EXTRA_REJECT')
record=copy.deepcopy(good);record['completed']=False;assert not ns['finalize'](record,identity)['qualified'];record['signed_value']='PRIVATE';assert ns['finalize'](record,identity)['signed_value'] is None
hook=object();e=ns['Collector'](root.f_code,catalog,identity,identity,lambda:0,lambda:4,lambda:hook,lambda v:(_ for _ in ()).throw(AssertionError()));e.callback(root,'call',None);assert e.failure=='HOOK_CONFLICT' and not e.stack
results.append('INCOMPLETE_BAD_SIGNED_PRIVATE_AND_ENTRY_HOOK_CONFLICT_REJECT')
for g in held:g.close()
(b/'pure-results.json').write_text(json.dumps(dict(groups=results,passed=True,simulated_events_clocks_hooks_threads=True,native_hook_verified=False),indent=2)+'\n');print('\n'.join(results));print('ALL_PURE_CONTROLS_PASS_NO_NATIVE_HOOK_EXECUTION')
