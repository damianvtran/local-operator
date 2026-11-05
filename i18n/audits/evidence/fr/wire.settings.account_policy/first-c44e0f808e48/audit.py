"""Independent, bounded fourteen-key CORE audit. Writes only this session's scratch."""
from pathlib import Path
import sys, json, hashlib, subprocess, importlib.util, unicodedata, re, shutil
ROOT=Path.home()/'local-operator'
OUT=Path(__file__).parent
PACKET=Path('/Users/damian/.local-operator/sessions/7b2ba4ed501d/scratchpad/fr-account-policy-0e7cde33')
NS='wire.settings.account_policy'
SHA='0e7cde33ca8008d561e74f4aed774aa5f76ff7d5'
CONTEXT=ROOT/'.worktrees/account-policy-context-backfill/local_operator/i18n/catalogues/en'/f'{NS}.context.json'
sys.path.insert(0,str(ROOT))
from local_operator.i18n import catalogues, runtime
from rich.cells import cell_len
spec=importlib.util.spec_from_file_location('core_i18n_checker',ROOT/'scripts/i18n/check.py')
checker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=checker;spec.loader.exec_module(checker)

def sha(b):return hashlib.sha256(b).hexdigest()
def load(b):
 def unique(pairs):
  d={}
  for k,v in pairs:
   assert k not in d,('duplicate JSON key',k)
   d[k]=v
  return d
 return json.loads(b,object_pairs_hook=unique)
inputs={
 'en':subprocess.check_output(['git','show',f'{SHA}:local_operator/i18n/catalogues/en/{NS}.json'],cwd=ROOT),
 'fr':(PACKET/f'{NS}.json').read_bytes(),
 'translator_context':(PACKET/'inputs/en.context.json').read_bytes(),
 'audited_context':CONTEXT.read_bytes(),
}
expected={'en':'1ad13f8198e28cb19fe0ecb2a6ef9b2a6990bc0d047566b71f5dd5322f594b25','fr':'f998c7a608a0023275a39f4c75ff7db65d107924c78db3907c7d0a28b2d30744','translator_context':'ccee657198d84a0bef6d20ae0343c9a864567e5a622a1b9736b1974ca296a254','audited_context':'2c23cc38b8eb9fb354eda0e1dd476d53b1064ffe88c2e821b9e1b723b8508c02'}
for role,b in inputs.items():
 assert sha(b)==expected[role],(role,sha(b))
 (OUT/f'{role}.json').write_bytes(b)
 print('INPUT',role,sha(b))
en,fr,old,new=(load(inputs[k]) for k in ['en','fr','translator_context','audited_context'])
assert len(en)==len(fr)==14 and set(en)==set(fr)==set(new)==set(old)
assert len([k for k in old if old[k]!=new[k]])==14
for k in old:
 assert {x:v for x,v in old[k].items() if x!='note'}=={x:v for x,v in new[k].items() if x!='note'},k
print('CONTEXT PASS: exactly fourteen note-only changes; all other fields identical')
fixture=OUT/'catalogues';(fixture/'en').mkdir(parents=True,exist_ok=True);(fixture/'fr').mkdir(exist_ok=True)
(fixture/'en'/f'{NS}.json').write_bytes(inputs['en']);(fixture/'en'/f'{NS}.context.json').write_bytes(inputs['audited_context'])
target=fixture/'fr'/f'{NS}.json';target.write_bytes(inputs['fr']);catalogues._CATALOGUES=fixture
problems=checker.check_catalogues();assert problems==[],problems
print('CORE VALIDATOR PASS: check_catalogues(), isolated one-namespace en/fr fixture')
for name,mutation in [('added argument',lambda d:d.__setitem__(NS+'.label',d[NS+'.label']+' {unexpected}')),('missing key',lambda d:d.pop(NS+'.error.invalid_json')),('invalid ICU',lambda d:d.__setitem__(NS+'.label','{bad'))]:
 changed=dict(fr);mutation(changed);target.write_text(json.dumps(changed,ensure_ascii=False))
 failures=checker.check_catalogues();assert failures,name
 print('NEGATIVE CONTROL REJECTED',name,failures[0])
target.write_bytes(inputs['fr']);assert checker.check_catalogues()==[]
widths=[]
for k in en:
 assert runtime.message_arguments(en[k])==runtime.message_arguments(fr[k])==(),k
 assert runtime.plural_selectors(fr[k])==(),k
 assert runtime.render_message(fr[k],{},'fr')==fr[k],k
 assert not any(unicodedata.category(c)=='Cf' for c in fr[k]),k
 for token in ('true','false','value','JSON'):
  pattern=r'(?<!\w)'+re.escape(token)+r'(?!\w)'
  assert len(re.findall(pattern,en[k]))==len(re.findall(pattern,fr[k])),(k,token)
 if 'maxCells' in new[k]:
  n=cell_len(fr[k]);assert n<=new[k]['maxCells'],(k,n)
  widths.append({'key':k,'cells':n,'maxCells':new[k]['maxCells']})
 print('KEY PASS',k,'args=0 plurals=0 Cf=0 render-exact=true')
print('WIDTHS',json.dumps(widths,ensure_ascii=False))
(OUT/'widths.json').write_text(json.dumps(widths,ensure_ascii=False,indent=2)+'\n')
# Capture only structured model usage from this session's completed assistant turns.
transcript=OUT.parent.parent/'transcript.jsonl'
usage=[]
for line in transcript.read_text().splitlines():
 row=json.loads(line);p=row.get('payload',{});u=p.get('usage',{}) if isinstance(p,dict) else {}
 if u.get('provider') and u.get('model_id'):
  usage.append({'message_id':row.get('id'),'ts':row.get('ts'),'provider':u['provider'],'model_id':u['model_id'],'input_tokens':u.get('input_tokens'),'output_tokens':u.get('output_tokens')})
assert usage
(OUT/'served-model-evidence.json').write_text(json.dumps({'source':str(transcript),'completed_usage_rows':usage[-4:]},indent=2)+'\n')
print('ACTUAL COMPLETED USAGE',json.dumps(usage[-1]))
print('NOT RUN: pseudo, RTL rendering, native/app fit, real-path integration, design, builds, installs, scanner, full suite, ledger writes')
print('PASS: fourteen-key isolated mechanical audit')
