from pathlib import Path
import os,ast,copy,json
b=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])/'owner-thread-integrated';c={};exec(compile((b.parent/'owner-thread-v1/collector.py').read_text(),'<actual-collector>','exec'),c)
p=dict(CodeType=c['CodeType'],Collector=c['Collector'],validate=c['validate'],graph_final=c['finalize']);tree=ast.parse((b/'producer.py').read_text());selected={'graph_identity','build_phase','phase_pass','overhead_pass'}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef)) and (isinstance(n,ast.FunctionDef) and n.name in selected or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('PHASES','CAPS','NAMES') for t in n.targets))],type_ignores=[]),'<actual-producer-contract>','exec'),p)
f={'sys':__import__('sys')};text='def root():\n return sys._getframe()\n'+''.join('def '+name+'():\n yield\n' for name in p['NAMES'] if name!='root');exec(compile(text,'<native-code-fixtures>','exec'),f);pairs=tuple((f[name].__code__,'PRIMITIVE:'+name.upper()) for name in p['NAMES']);expected=dict(python=[3,12,15],executable_sha256='a'*64,producer_sha256='b'*64,collector_sha256='c'*64,pid=1,ppid=2,pgid=3);identity=p['graph_identity'](expected,4,pairs);held=[]
def graph(phase):
 state={'hook':None,'time':0}
 def clock():state['time']+=100;return state['time']
 e=c['Collector'](f['root'].__code__,pairs,identity,identity,clock,lambda:4,lambda:state['hook'],lambda v:state.update(hook=v));state['hook']=e.callback;rf=f['root']();e.callback(rf,'call',None)
 names={'DIRECT':['direct'],'NESTED':['nested','recurse','recurse','recurse','recurse'],'ASYNC':['async_work']*3+['async_wait']*3,'OFFLOAD':['offload']*3,'OVERHEAD':['noop']*1000}[phase]
 if phase=='NESTED':
  frames=[]
  for name in names:
   g=f[name]();held.append(g);frames.append(g.gi_frame);e.callback(g.gi_frame,'call',None)
  for fr in reversed(frames):e.callback(fr,'return',None)
 else:
  for name in names:
   g=f[name]();held.append(g);e.callback(g.gi_frame,'call',None);e.callback(g.gi_frame,'return',None)
 e.callback(rf,'return',None);return e.export()
def row(phase):
 state=dict(owner=4,ready=True,trailing=True)
 if phase in ('DIRECT','NESTED'):state['reference_ns']=100
 elif phase=='ASYNC':state.update(work=dict(reference_ns=300,segments=3),wait=dict(reference_ns=300,segments=3),unrelated_ns=20000000,loop_closed=True)
 elif phase=='OFFLOAD':state.update(reference_ns=300,segments=3,worker_distinct=True,worker_hook_none=True,worker_ns=20000000,loop_closed=True)
 result=p['build_phase'](phase,graph(phase),state,identity);assert p['phase_pass'](result,expected);return result
rows=[row(v) for v in p['PHASES']];overhead=dict(unprofiled_ns=100,profiled_ns=200,difference_ns=100,row=row('OVERHEAD'));assert p['overhead_pass'](overhead,expected)
q=dict(p,Path=Path,json=json,shutil=None);ct=ast.parse((b/'controller.py').read_text());exec(compile(ast.Module(body=[n for n in ct.body if isinstance(n,ast.FunctionDef) and n.name in ('qualify','safe_partial','finalize','finish_result')],type_ignores=[]),'<actual-consumer-finalizer>','exec'),q)
proof=dict(identity=expected,status='COMPLETE',phases=rows,overhead=overhead,application_authorized=False);assert q['qualify'](proof,expected);healthy=dict(rc=0,reaped=True,owned_settled=True,synthetic_root_removed=True,errors=[]);deleted=[];out={};q['finalize'](Path('.'),out,expected,None,reader=lambda:proof,deleter=lambda:deleted.append(1));assert q['finish_result'](out,healthy,True)['qualified'] and deleted==[1]
mutations=[('pid',lambda x:x['identity'].update(pid=5)),('binding',lambda x:x['phases'][0]['identity'].update(executable_sha256='d'*64)),('unknown_negative',lambda x:x['phases'][0]['graph']['nodes'].append(dict(static_id=None,segments=1,own_ns=-3,inclusive_ns=1))),('bool_count',lambda x:x['phases'][0]['graph']['nodes'][0].update(segments=True)),('nonfinite',lambda x:x['phases'][0]['graph']['nodes'][0].update(own_ns=float('inf'))),('hook',lambda x:x['phases'][0]['graph'].update(hook_seen_at_export=False)),('phase',lambda x:x['phases'].reverse()),('incomplete',lambda x:x['phases'][0]['graph'].update(completed=False)),('private',lambda x:x['phases'][0]['state'].update(PRIVATE='hidden')),('tolerance',lambda x:x['phases'][0]['state'].update(reference_ns=100000000)),('overhead',lambda x:x['overhead'].update(difference_ns=101))]
for label,change in mutations:
 bad=copy.deepcopy(proof);change(bad);out={};deleted=[];q['finalize'](Path('.'),out,expected,None,reader=lambda:bad,deleter=lambda:deleted.append(1));assert not q['finish_result'](out,healthy,True)['qualified'] and deleted==[1] and 'hidden' not in json.dumps(out),label
for rc in (None,True,1):assert not q['finish_result'](dict(full_proof_qualified=True,root_deletion='REMOVED'),dict(healthy,rc=rc),True)['qualified']
out={};deleted=[];q['finalize'](Path('.'),out,expected,None,settled=False,reader=lambda:proof,deleter=lambda:deleted.append(1));assert not deleted and not out['full_proof_qualified']
for g in held:g.close()
(b/'pure-results.json').write_text(json.dumps(dict(healthy_four=True,overhead=True,mutations_rejected=[v[0] for v in mutations],raw_none_bool_nonzero_rejected=True,unsettled_not_removed=True,observations='SIMULATED native events/clocks/hooks/phase references/lifecycle; actual engine/builders/validators/finalizers',main_popen_barrier_not_exercised=True),indent=2)+'\n');print('ACTUAL_ENGINE_4_PHASE_BUILDERS_V1_CONSUMER_FINALIZERS_HEALTHY_AND_11_REJECTIONS; NO_NATIVE_HOOKS; MAIN_FAKES_PENDING')
