from pathlib import Path
import sys,json,hashlib,re,shutil
root=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
sys.path.insert(0,str(root))
from scripts.i18n import check
from local_operator.i18n import catalogues,runtime
out=Path(__file__).parent
ns='wire.teams.effective'
inputs={'en':root/f'local_operator/i18n/catalogues/en/{ns}.json','context':root/f'local_operator/i18n/catalogues/en/{ns}.context.json','hi':Path('/Users/damian/.local-operator/sessions/47e37aebb87c/scratchpad/effective-reasons-hi/catalogues/hi/wire.teams.effective.json')}
expected={'en':'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','context':'01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','hi':'e2a1d7d25325c08db1f414105c75609e7c3a4f65d8a877169064974a9d5f1d92'}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in inputs.items()}
assert all(hashes[k]['sha256']==v for k,v in expected.items())
def members(pairs):
 d={}
 for k,v in pairs:
  assert k not in d, ('duplicate',k)
  d[k]=v
 return d
data={k:json.loads(p.read_text(),object_pairs_hook=members) for k,p in inputs.items()}
assert set(data['en'])==set(data['hi'])==set(data['context']) and len(data['en'])==9
fixture=out/'catalogues'
for k,p in inputs.items():
 dest=fixture/('en' if k=='context' else k)/p.name
 dest.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(p,dest)
catalogues.catalogue_root=lambda:fixture
problems=check.check_catalogues()
assert not problems,problems
rows=[]
for k,en in data['en'].items():
 hi=data['hi'][k]; args=runtime.message_arguments(en)
 assert args==runtime.message_arguments(hi)
 assert not re.search(r',\s*(?:plural|select|selectordinal)\s*,',en+hi)
 literals=[t for t in ['subagents.models.','provider/model'] if t in en]
 if k.endswith('tier_invalid_mapping'): literals+=['model','effort']
 assert all(t in hi for t in literals)
 assert not re.search('[\u200e\u200f\u202a-\u202e\u2066-\u2069\u200c\u200d\u0966-\u096f]',hi)
 rows.append({'key':k,'arguments':[list(a) for a in args],'technical_literals':literals,'latin_runs':re.findall('[A-Za-z][A-Za-z./]*',hi),'source':en,'current_text':hi,'mechanical':'passed','plurals':'none; CLDR categories not applicable','width_hint':{a:b for a,b in data['context'][k].items() if a in ['maxCells','maxChars']}})
result={'input_hashes':hashes,'key_count':9,'normal_catalogue_validator':'scripts.i18n.check.check_catalogues()','problems':problems,'per_key':rows}
(out/'mechanical.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
