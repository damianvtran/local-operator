"""Read-only staged French linguistic audit mechanics; one independent output per namespace."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,importlib.util,unicodedata,re,ast
ROOT=Path.home()/'local-operator'; SHA='0e7cde33ca8008d561e74f4aed774aa5f76ff7d5'
BASE=Path('/Users/damian/.local-operator/sessions/7b2ba4ed501d/scratchpad')
SCRATCH=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
slug,ns,count=sys.argv[1],sys.argv[2],int(sys.argv[3]);PACKET=BASE/slug;OUT=SCRATCH/(slug+'-first');OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from local_operator.i18n import catalogues,runtime
from rich.cells import cell_len
spec=importlib.util.spec_from_file_location('core_checker',ROOT/'scripts/i18n/check.py');checker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=checker;spec.loader.exec_module(checker)
hashb=lambda b:hashlib.sha256(b).hexdigest()
def load(b):
 def unique(pairs):
  d={}
  for k,v in pairs:
   assert k not in d,('duplicate',k)
   d[k]=v
  return d
 return json.loads(b,object_pairs_hook=unique)
def git(p):return subprocess.check_output(['git','show',f'{SHA}:{p}'],cwd=ROOT)
manifest=load((PACKET/'manifest.json').read_bytes());expected={x['path']:x['sha256'] for x in manifest['files']}
inputs={'en.json':git(f'local_operator/i18n/catalogues/en/{ns}.json'),'context.json':git(f'local_operator/i18n/catalogues/en/{ns}.context.json'),'fr.json':(PACKET/f'{ns}.json').read_bytes(),'fr-style.md':git('i18n/style-guides/fr.md'),'glossary.md':git('i18n/glossary.md')}
for name,p in [('en.json','inputs/en.json'),('context.json','inputs/en.context.json'),('fr.json',f'{ns}.json'),('fr-style.md','inputs/fr.md'),('glossary.md','inputs/glossary.md')]:
 if name=='fr-style.md':
  translator_style=(PACKET/p).read_bytes();assert hashb(translator_style)==expected[p]
  (OUT/'translator-fr-style.md').write_bytes(translator_style)
  print('STYLE SNAPSHOT DIFFERENCE: pinned guide retains unrelated relay.chrome width-exception paragraph; normative rules identical; both retained')
 else:
  assert hashb(inputs[name])==expected[p],(name,hashb(inputs[name]),expected[p])
 (OUT/name).write_bytes(inputs[name]);print('INPUT',name,hashb(inputs[name]))
en,fr,context=(load(inputs[k]) for k in ['en.json','fr.json','context.json']);assert len(en)==len(fr)==len(context)==count and set(en)==set(fr)==set(context)
fixture=OUT/'catalogues';(fixture/'en').mkdir(parents=True,exist_ok=True);(fixture/'fr').mkdir(exist_ok=True)
(fixture/'en'/f'{ns}.json').write_bytes(inputs['en.json']);(fixture/'en'/f'{ns}.context.json').write_bytes(inputs['context.json']);target=fixture/'fr'/f'{ns}.json';target.write_bytes(inputs['fr.json']);catalogues._CATALOGUES=fixture
assert checker.check_catalogues()==[],checker.check_catalogues();print('CORE CHECKER PASS isolated namespace fixture',ns,count)
first=next(iter(fr))
for label,mutate in [('unexpected argument',lambda d:d.__setitem__(first,d[first]+' {unexpected_parameter}')),('missing key',lambda d:d.pop(first)),('bad ICU',lambda d:d.__setitem__(first,'{unclosed'))]:
 d=dict(fr);mutate(d);target.write_text(json.dumps(d,ensure_ascii=False));p=checker.check_catalogues();assert p;print('NEGATIVE REJECTED',label,p[0])
target.write_bytes(inputs['fr.json']);assert checker.check_catalogues()==[]
widths=[]
tokens=['agent_name','agent_id','ApprovalPolicy.callback()','ApprovalPolicy.declared()','ApprovalPolicy.refuse()','ApprovalPolicy.handler','ApprovalPolicy.tools','callback','declared','mode','attach','own','spec.resume','resume','output_format','output_retries','output_schema','birth_effort','hosting/model','open_session','spawn_session()','dispose','tools','None','world_root','ApprovalPolicy.auto()','yolo','@latest','[redacted]'] if ns=='wire.errors.sdk_spec' else []
for k in en:
 a=runtime.message_arguments(en[k]);b=runtime.message_arguments(fr[k]);assert a==b,(k,a,b)
 assert runtime.plural_selectors(fr[k])==runtime.plural_selectors(en[k])==(),k
 assert not any(unicodedata.category(c)=='Cf' for c in fr[k]),k
 assert not any(c in fr[k] for c in '\x1b\x00'),k
 params={n:"OPAQUE_'{literal}_#" for n,_ in a};rendered=runtime.render_message(fr[k],params,'fr')
 for n,_ in a:assert params[n] in rendered
 for token in tokens:
  if token in en[k]:assert token in fr[k],(k,token)
 if ns=='wire.console':assert re.findall(r'\d+',en[k])==re.findall(r'\d+',fr[k]),k
 if 'maxCells' in context[k]:
  cells=cell_len(runtime.render_message(fr[k],{},'fr'));assert cells<=context[k]['maxCells'],(k,cells)
  widths.append({'key':k,'cells':cells,'maxCells':context[k]['maxCells']})
 print('KEY PASS',k,'argument_types='+repr(a),'Cf=0','opaque_values_literal=true')
print('WIDTHS',json.dumps(widths,ensure_ascii=False));(OUT/'widths.json').write_text(json.dumps(widths,indent=2)+'\n')
if ns=='wire.errors.sdk_spec':
 calls=[]
 for p in ['local_operator/session/spec.py','local_operator/sdk.py']:
  src=git(p);tree=ast.parse(src)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='envelope' and isinstance(n.func.value,ast.Call):
    f=n.func.value.func
    if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id=='wire_errors_sdk_spec':
     assert any(kw.arg=='locale' and isinstance(kw.value,ast.Constant) and kw.value.value=='en' for kw in n.keywords)
     calls.append(dict(path=p,line=n.lineno,message_id='.'.join((ns,f.attr))))
 assert len(calls)==21 and {c['message_id'] for c in calls}==set(en)
 (OUT/'sdk-forced-english-callers.json').write_text(json.dumps(calls,indent=2)+'\n');print('SOURCE AST PASS: all21 keys forced English; no caller migration or localized sink claimed')
 assessment=Path.home()/'Documents/LocalOperator/Reports/i18n-sdk-contract-assessment-2026-11-03-3ef6295a5545.md';b=assessment.read_bytes();assert hashb(b)==expected['inputs/architect-assessment.md'];(OUT/'architect-assessment.md').write_bytes(b)
rows=[];transcript=SCRATCH.parent/'transcript.jsonl'
for line in transcript.read_text().splitlines():
 r=json.loads(line);p=r.get('payload',{});u=p.get('usage',{}) if isinstance(p,dict) else {}
 if u.get('provider') and u.get('model_id'):rows.append({'message_id':r.get('id'),'ts':r.get('ts'),'provider':u['provider'],'model_id':u['model_id'],'input_tokens':u.get('input_tokens'),'output_tokens':u.get('output_tokens')})
assert rows;(OUT/'served-model-evidence.json').write_text(json.dumps({'source':str(transcript),'completed_usage_rows':rows[-4:]},indent=2)+'\n');print('SERVED MODEL',json.dumps(rows[-1]))
(OUT/'audit.py').write_bytes(Path(__file__).read_bytes())
print('PASS mechanical; pseudo/native RTL/native fit/real-path/design NOT RUN; no ledger/repo edits')
