import sys, os, json, hashlib, copy
from pathlib import Path
from unittest.mock import patch
R=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
sys.path.insert(0,str(R))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
fmt = runtime.fmt
S=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD']); NS='wire.session_status'
paths={'en':R/f'local_operator/i18n/catalogues/en/{NS}.json','context':R/f'local_operator/i18n/catalogues/en/{NS}.context.json','fr':Path('/Users/damian/.local-operator/sessions/a14cc322d1da/scratchpad/wire.session_status.fr.json'),'zh-CN':Path('/Users/damian/.local-operator/sessions/e9c45b5505eb/scratchpad/wire.session_status.zh-CN.json')}
data={k:json.loads(p.read_text()) for k,p in paths.items()}
expected={'en':'52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326','context':'5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525','fr':'1b48011bda6364f5091c78e236020980018fa15934ff9424feca4280f5a9429d','zh-CN':'6461115924a504b57c11d077a0e735cf08af4bdae59b17997e81b4fc5bf4a649'}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
assert all(hashes[k]['sha256']==h for k,h in expected.items()),hashes
F=S/'scoped-catalogues'; F.mkdir(exist_ok=True)
for loc in ('en','fr','zh-CN'):
 (F/loc).mkdir(exist_ok=True)
 (F/loc/f'{NS}.json').write_bytes(paths[loc].read_bytes())
(F/'en'/f'{NS}.context.json').write_bytes(paths['context'].read_bytes())
def admit(loc, value):
 (F/loc/f'{NS}.json').write_text(json.dumps(value,ensure_ascii=False))
 with patch.object(catalogues,'catalogue_root',return_value=F): return check.check_catalogues()
results={'hashes':hashes,'normal_admission':{},'negative_controls':{},'per_locale':{}}
for loc in ('fr','zh-CN'):
 target=data[loc]; problems=admit(loc,target); assert not problems,problems
 results['normal_admission'][loc]=problems
 controls={}
 mutations={}
 bad=copy.deepcopy(target); bad.pop(NS+'.complete');mutations['missing_key']=bad
 bad=copy.deepcopy(target);bad[NS+'.retired_label']=bad[NS+'.retired_label'].replace('{base}','{wrong}');mutations['wrong_placeholder']=bad
 bad=copy.deepcopy(target);bad[NS+'.dormant']=bad[NS+'.dormant'].rsplit('}',1)[0];mutations['broken_icu']=bad
 bad=copy.deepcopy(target);bad[NS+'.dormant']='{count, plural, few {# X} other {# X}}';mutations['invalid_locale_category']=bad
 for name,bad in mutations.items():
  p=admit(loc,bad);assert p,(loc,name);controls[name]={'rejected':True,'actual_output':p}
 if loc=='fr':
  bad=copy.deepcopy(target);bad[NS+'.dormant']=bad[NS+'.dormant'].replace(' many {# de réveils en sommeil}','')
  p=admit(loc,bad);controls['missing_required_many']={'rejected':bool(p),'actual_output':p,'interpretation':'Known validator gap: legal category check does not require all categories for ordinary plurals; actual target has one/many/other.'}
 assert not admit(loc,target)
 results['negative_controls'][loc]=controls
 per=[]
 for k,en in data['en'].items():
  text=target[k];ast=runtime.parse_message(text);sourceast=runtime.parse_message(en)
  assert runtime.message_arguments(en)==runtime.message_arguments(text),(k,loc)
  row={'key':k,'source':en,'current_text':text,'branches':[]}
  args=[n for n in ast if isinstance(n,runtime._Arg)]
  sargs=[n for n in sourceast if isinstance(n,runtime._Arg)]
  if args and args[0].kind=='select':
   a=args[0]; sa=sargs[0];assert set(dict(a.parts))==set(dict(sa.parts))
   vals=([x for x,_ in a.parts]+['unknown_future_cause']) if a.name=='cause' else ['none','/stop --all','lop sessions stop abc --force']
   for v in vals:
    out=runtime.render_message(text,{a.name:v},loc)
    if a.name=='command':
     assert ('none' not in out) if v=='none' else v in out
    row['branches'].append({'params':{a.name:v},'output':out})
  elif args and args[0].kind=='plural':
   required={'one','many','other'} if loc=='fr' else {'other'}
   assert set(dict(args[0].parts))==required
   for n in [0,1,2,5,21,55,1000000,1500000,1000000000000,0.5,1.5,2.5]:
    row['branches'].append({'params':{'count':n},'category':fmt.plural_category(loc,n),'output':runtime.render_message(text,{'count':n},loc)})
  else:
   params={'base':'BASE_STATUS'} if k.endswith('.retired_label') else {}
   row['branches'].append({'params':params,'output':runtime.render_message(text,params,loc)})
  per.append(row)
 results['per_locale'][loc]=per
(S/'execution-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'hashes':hashes,'normal_admission':results['normal_admission'],'negative_controls':results['negative_controls'],'keys_per_locale':{k:len(v) for k,v in results['per_locale'].items()}},ensure_ascii=False,indent=2))
