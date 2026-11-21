import os, sys, json, hashlib, runpy, copy, contextlib, io, re
from pathlib import Path
R=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
S=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0,str(R))
from local_operator.i18n import catalogues, runtime
ns='wire.session_status'
enp=R/f'local_operator/i18n/catalogues/en/{ns}.json'
cp=enp.with_name(ns+'.context.json')
tp=Path('/Users/damian/.local-operator/sessions/a14cc322d1da/scratchpad/wire.session_status.fr.json')
def strict(p):
 def pairs(xs):
  d={}
  for k,v in xs:
   assert k not in d, ('duplicate',k)
   d[k]=v
  return d
 return json.loads(p.read_text(),object_pairs_hook=pairs)
en=strict(enp);fr=strict(tp);ctx=strict(cp)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(enp)=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(cp)=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert sha(tp)=='1b48011bda6364f5091c78e236020980018fa15934ff9424feca4280f5a9429d'
fixture=S/'auditor-catalogue-fixture'
for locale in ['en','fr']:(fixture/locale).mkdir(parents=True,exist_ok=True)
(fixture/'en'/enp.name).write_bytes(enp.read_bytes());(fixture/'en'/cp.name).write_bytes(cp.read_bytes())
f=fixture/'fr'/enp.name
catalogues._CATALOGUES=fixture
checker=runpy.run_path(str(R/'scripts/i18n/check.py'),run_name='independent_fr_validator')
def check(label,data,expect):
 f.write_text(json.dumps(data,ensure_ascii=False))
 out=io.StringIO()
 with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):rc=checker['main'](['--catalogues-only'])
 print(f'CASE {label} actual_rc={rc} expected_rc={expect}\n{out.getvalue()}',flush=True)
 assert rc==expect,(label,rc)
check('normal admission',fr,0)
for label,mutate in [('missing key',lambda d:d.pop(ns+'.recent')),('wrong placeholder',lambda d:d.update({ns+'.retired_label':d[ns+'.retired_label'].replace('{base}','{wrong}')})),('invalid few CLDR',lambda d:d.update({ns+'.dormant':d[ns+'.dormant'].replace('many {','few {')}))]:
 d=copy.deepcopy(fr);mutate(d);check(label,d,1)
check('restored passing control',fr,0)
def shapes(nodes,kind):
 out=[]
 for n in nodes:
  if isinstance(n,runtime._Arg):
   if n.kind==kind:out.append((n.name,tuple(x for x,_ in n.parts)))
   for _,body in n.parts:out+=shapes(body,kind)
 return out
def own_guard(d):
 assert set(d)==set(en)
 for k in en:
  assert runtime.message_arguments(en[k])==runtime.message_arguments(d[k]),k
  assert shapes(runtime.parse_message(en[k]),'select')==shapes(runtime.parse_message(d[k]),'select'),k
  assert re.findall(r'`[^`]*`|%[sd]|<[^>]+>',en[k])==re.findall(r'`[^`]*`|%[sd]|<[^>]+>',d[k]),k
  assert '\n' not in d[k]
 assert shapes(runtime.parse_message(d[ns+'.dormant']),'plural')==[('count',('one','many','other'))]
own_guard(fr)
d=copy.deepcopy(fr);d[ns+'.dormant']=d[ns+'.dormant'].replace(' many {# de réveils en sommeil}','')
check('missing many admitted by repo validator (documented limit)',d,0)
try:own_guard(d)
except AssertionError:print('PASS independent all-CLDR guard rejects missing many; repo validator checks legal categories and other, not exhaustive categories')
else:raise AssertionError('all-CLDR guard failed')
check('final restored passing control',fr,0)
d=copy.deepcopy(fr);d[ns+'.error_cause']=d[ns+'.error_cause'].replace('user_stop {','arret_utilisateur {')
try:own_guard(d)
except AssertionError:print('PASS independent select-arm guard rejects translated machine identifier')
else:raise AssertionError('select-arm guard failed')
assert len(fr)==17
assert fr[ns+'.retired_label']==strict(R/f'local_operator/i18n/catalogues/fr/{ns}.json')[ns+'.retired_label']
print('PASS 17/17 strict JSON, keys, argument types, machine select arms, literals, one-line; unchanged legacy retired_label')
for k in ['signal_stop','unseen_signal_stop','forced_kill','unseen_forced_kill']:
 for cmd in ['none','/stop --all']:
  text=runtime.render_message(fr[ns+'.'+k],{'command':cmd},'fr')
  assert ('none' not in text) if cmd=='none' else (cmd in text)
  print('FORMAT',k,repr(cmd),'=>',text)
for count in [0,1,2,1000000,1500000]:print('FORMAT dormant',count,'=>',runtime.render_message(fr[ns+'.dormant'],{'count':count},'fr'))
for key in ['error_cause','unseen_error_cause']:
 ast=runtime.parse_message(fr[ns+'.'+key]);arg=ast[0]
 for cause,_ in arg.parts:
  print('FORMAT',key,cause,'=>',runtime.render_message(fr[ns+'.'+key],{'cause':cause},'fr'))
from rich.cells import cell_len
for k in fr:
 if k.endswith('error_cause'):continue
 params={'base':'Interrompue','command':'/stop --all','count':1000000}
 text=runtime.render_message(fr[k],params,'fr')
 print('WIDTH',k,cell_len(text),'cells',len(text),'chars; no fixed source-side cap')
print('PASS string-level formatting only; NO live app, runtime lifecycle, visual or release clearance')
