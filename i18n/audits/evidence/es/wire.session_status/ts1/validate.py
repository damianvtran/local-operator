import os,sys,json,hashlib,shutil
from pathlib import Path
repo=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status'); out=Path(__file__).parent
sys.path.insert(0,str(repo))
from local_operator.i18n import catalogues,runtime
from scripts.i18n import check,ledger
ns='wire.session_status'; source=repo/f'local_operator/i18n/catalogues/en/{ns}.json'; context=source.with_suffix('.context.json'); target=Path('/Users/damian/.local-operator/sessions/b39246e0d986/scratchpad/terminal-status-es/wire.session_status.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(context)=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert sha(target)=='b13626dbcbc240488738922d287663f0ede62b191e461ed1767de9a56a0ee128'
def strict_pairs(pairs):
 d={}
 for k,v in pairs:
  assert k not in d,('duplicate',k)
  d[k]=v
 return d
en=json.loads(source.read_text(),object_pairs_hook=strict_pairs); es=json.loads(target.read_text(),object_pairs_hook=strict_pairs)
fixture=out/'fixture'
for loc in ['en','es']:(fixture/loc).mkdir(parents=True,exist_ok=True)
shutil.copyfile(source,fixture/'en'/source.name); shutil.copyfile(context,fixture/'en'/context.name)
catalogues._CATALOGUES=fixture
p=fixture/'es'/source.name
results={}
def validate(name,data,want):
 p.write_text(json.dumps(data,ensure_ascii=False))
 problems=check.check_catalogues(); assert bool(problems)==want,(name,problems)
 results[name]=problems
 print(name+': '+('REJECTED as expected' if want else 'PASS (0 problems)')+' '+json.dumps(problems,ensure_ascii=False))
validate('normal_admission',es,False)
bad=dict(es);bad[ns+'.retired_label']=bad[ns+'.retired_label'].replace('{base}','{wrong}')
validate('negative_placeholder',bad,True)
bad=dict(es);bad[ns+'.dormant']=bad[ns+'.dormant'].replace('many {# de despertares inactivos} ','')
validate('negative_missing_many_REPO_VALIDATOR_GAP',bad,False)
assert set(runtime.plural_selectors(bad[ns+'.dormant'])[0]) != {'one','many','other'}
print('Independent CLDR completeness control REJECTED missing many; repo validator only checks legal categories plus other for plural-to-plural (check.py:712-738)')
bad=dict(es);del bad[ns+'.complete'];validate('negative_dropped_key',bad,True)
bad=dict(es);bad[ns+'.signal_stop']=bad[ns+'.signal_stop'][:-1];validate('negative_broken_icu',bad,True)
validate('restored_normal_admission',es,False)
assert list(en)==list(es) and len(es)==17
old=json.loads((repo/f'local_operator/i18n/catalogues/es/{ns}.json').read_text())
assert es[ns+'.retired_label']==old[ns+'.retired_label']
def arms(nodes):
 r=[]
 for n in nodes:
  if isinstance(n,runtime._Arg):
   if n.kind=='select':r.append((n.name,tuple(a for a,_ in n.parts)))
   for _,body in n.parts:r.extend(arms(body))
 return r
for k in en:
 assert runtime.message_arguments(en[k])==runtime.message_arguments(es[k]),k
 assert arms(runtime.parse_message(en[k]))==arms(runtime.parse_message(es[k])),k
 assert not any(c in es[k] for c in '\n\r\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069'),k
assert runtime.plural_selectors(es[ns+'.dormant'])==( ('one','many','other'), )
print('Independent checks PASS: strict JSON, 17 keys, exact frozen retired_label, placeholder kinds/names, all machine select arms, CLDR one/many/other, single line, no bidi controls')
samples=[]
for name in ['signal_stop','unseen_signal_stop','forced_kill','unseen_forced_kill']:
 for cmd in ['none','/stop --all','lop sessions stop --session synthetic-qa']:
  s=runtime.render_message(es[ns+'.'+name],{'command':cmd},'es')
  assert (s.endswith(' mediante '+cmd) if cmd!='none' else ('none' not in s and 'mediante' not in s))
  samples.append({'key':name,'params':{'command':cmd},'text':s})
for count in [0,1,2,1000000,1500000]:samples.append({'key':'dormant','params':{'count':count},'text':runtime.render_message(es[ns+'.dormant'],{'count':count},'es')})
for name in ['error_cause','unseen_error_cause']:
 for cause in list(arms(runtime.parse_message(es[ns+'.'+name]))[0][1])+['synthetic_unknown']:
  s=runtime.render_message(es[ns+'.'+name],{'cause':cause},'es');assert s
  samples.append({'key':name,'params':{'cause':cause},'text':s})
(out/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
(out/'mechanical.json').write_text(json.dumps({'validator_controls':results,'sample_count':len(samples),'input_hashes':{n:{'path':str(p),'sha256':sha(p)} for n,p in [('en',source),('context',context),('es',target)]}},ensure_ascii=False,indent=2)+'\n')
print('PASS:',len(samples),'isolated ICU format samples (not integrated runtime/visual proof)')
