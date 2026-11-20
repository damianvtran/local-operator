"""Independent scoped candidate admission; no repo writes or catalogue activation."""
import sys,json,hashlib,re
from pathlib import Path
ROOT=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from local_operator.i18n import runtime as r
from scripts.i18n.check import arguments_match,_check_plural_categories
NS='wire.session_status'
paths={'en':ROOT/f'local_operator/i18n/catalogues/en/{NS}.json','context':ROOT/f'local_operator/i18n/catalogues/en/{NS}.context.json','target':Path('/Users/damian/.local-operator/sessions/8c6e64f5562b/scratchpad/terminal-status-hi/wire.session_status.json'),'style':ROOT/'i18n/style-guides/hi.md','glossary':ROOT/'i18n/glossary.md'}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
assert hashes['en']['sha256']=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert hashes['context']['sha256']=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert hashes['target']['sha256']=='bff9faa6c1eaf819f0aa8943d53add199288e4611c736a24bdb7ebb8bd340b81'
def unique(pairs):
 d={}
 for k,v in pairs:
  assert k not in d,('duplicate key',k)
  d[k]=v
 return d
en,ctx,hi=[json.loads(paths[k].read_text(),object_pairs_hook=unique) for k in ['en','context','target']]
plural=json.loads((ROOT/'local_operator/i18n/data/plural_rules.json').read_text())
assert len(hi)==17 and set(en)==set(ctx)==set(hi)
def structure(nodes):
 result=[]
 for n in nodes:
  if isinstance(n,r._Arg):
   if n.parts: result.append((n.name,n.kind,tuple(k for k,_ in n.parts)))
   for _,children in n.parts: result.extend(structure(children))
 return result
def admission(key,text):
 assert arguments_match(r.message_arguments(en[key]),r.message_arguments(text),translated_source=text,locale='hi',plural=plural),'arguments'
 assert structure(r.parse_message(en[key]))==structure(r.parse_message(text)),'machine arms'
 problems=[]
 _check_plural_categories(problems,'hi',NS,key,text,plural,'hi')
 assert not problems,problems
 for occurrence in r.plural_arguments(text):
  assert set(plural['locales']['hi']['categories']) <= {b.selector for b in occurrence.branches},'missing CLDR'
  assert all(b.renders_value and not b.empty for b in occurrence.branches),'lost count'
 assert re.findall(r'`[^`]*`|</?[^>]+>|%[sdif]',en[key])==re.findall(r'`[^`]*`|</?[^>]+>|%[sdif]',text),'literal tokens'
 assert '\n' not in text
 return True
for key,text in hi.items(): assert admission(key,text)
base=json.loads((ROOT/f'local_operator/i18n/catalogues/hi/{NS}.json').read_text())
assert hi[NS+'.retired_label']==base[NS+'.retired_label']
samples=[]
for key,text in hi.items():
 if key.endswith('error_cause'):
  arms=[k for k,_ in r.parse_message(text)[0].parts]
  for cause in arms+['unrecognized_cause','']:
   output=r.render_message(text,{'cause':cause},'hi')
   assert output.startswith('त्रुटि')
   samples.append({'key':key,'params':{'cause':cause},'output':output})
 elif key.endswith(('signal_stop','forced_kill')):
  for command in ['none','/stop --all','lop sessions stop --session abc-123','opaque-{id}/路径']:
   output=r.render_message(text,{'command':command},'hi')
   assert (command not in output) if command=='none' else (output.count(command)==1)
   samples.append({'key':key,'params':{'command':command},'output':output})
 elif key.endswith('.dormant'):
  for count in [0,1,2,5,21,55,1234567,0.5,1.5]:
   output=r.render_message(text,{'count':count},'hi')
   # Hindi CLDR one also covers 0 <= n < 1 (integer part zero).
   assert ('जागृतियाँ' in output)==(count>=1 and count!=1)
   samples.append({'key':key,'params':{'count':count},'output':output})
 else:
  params={'base':'बाधित'} if key.endswith('.retired_label') else {}
  samples.append({'key':key,'params':params,'output':r.render_message(text,params,'hi')})
controls=[]
def reject(name,key,text):
 try: admission(key,text)
 except (AssertionError,r.MessageSyntaxError) as e: controls.append({'name':name,'result':'rejected','diagnostic':str(e)}); return
 raise AssertionError('negative admitted: '+name)
reject('renamed placeholder',NS+'.retired_label','{renamed} — अपडेट')
reject('wrong placeholder type',NS+'.retired_label','{base, number} — अपडेट')
reject('malformed ICU',NS+'.dormant','{count, plural, one {#}')
reject('renamed machine arm',NS+'.signal_stop',hi[NS+'.signal_stop'].replace('none {','absent {'))
reject('missing Hindi one arm',NS+'.dormant','रुका हुआ ({count, plural, other {# जागृतियाँ निष्क्रिय}})')
reject('dropped plural number',NS+'.dormant',hi[NS+'.dormant'].replace('# जागृति','जागृति'))
reject('invalid Hindi category',NS+'.dormant',hi[NS+'.dormant'].replace('one {','few {'))
gap_text='{count, plural, other {# जागृतियाँ निष्क्रिय}}'
gap_problems=[]
_check_plural_categories(gap_problems,'hi',NS,NS+'.dormant',gap_text,plural,'hi')
gap_match=arguments_match(r.message_arguments(en[NS+'.dormant']),r.message_arguments(gap_text),translated_source=gap_text,locale='hi',plural=plural)
result={'status':'PASS','key_count':len(hi),'input_hashes':hashes,'samples':samples,'negative_controls':controls,'validator_gap':{'missing_one_problems':gap_problems,'same_kind_arguments_match':gap_match,'description':'Repo same-kind plural checks do not require all CLDR categories. Independent admission checks full category set; actual target contains one/other and is correct.'},'scope':'Repository ICU parser and scoped check.py helpers, plus independent admission; not full-corpus CLI, integrated runtime, visual, shaping or pseudo proof.'}
(OUT/'checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('PASS 17/17 keys; frozen hashes; valid duplicate-free JSON; placeholder names/types; exact machine arms; full Hindi CLDR categories; count preservation; legacy retired_label unchanged')
print(f'PASS {len(samples)} formatter-only samples (all cause arms, fallback, command none/opaque, counts 0/1/2/5/21/55/1234567/0.5/1.5)')
print(f'PASS {len(controls)} meaningful negative controls rejected by scoped admission')
print('GAP CONFIRMED: repo same-kind plural checker admits missing Hindi one; actual target has one/other. Not a target finding.')
print('NOT PERFORMED: full suite, integrated runtime, visual, pseudo, shaping, native-client proof, repository writes, history apply')
