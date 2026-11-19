import sys, json, hashlib, unicodedata, re, shutil
from pathlib import Path
root=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
sys.path.insert(0,str(root))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
out=Path(__file__).parent
ns='wire.teams.effective'
paths={'en':root/f'local_operator/i18n/catalogues/en/{ns}.json','context':root/f'local_operator/i18n/catalogues/en/{ns}.context.json','vi':Path('/Users/damian/.local-operator/sessions/bdce0d00e792/scratchpad/effective-reasons-vi/wire.teams.effective.json')}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
expected={'en':'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','context':'01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','vi':'4d510633cda89708fa842a0812dc9fff2836383e93e1322c6a6d532cf52cc057'}
assert all(hashes[k]['sha256']==v for k,v in expected.items()),hashes
fixture=out/'fixture'
for locale in ('en','vi'):(fixture/locale).mkdir(parents=True,exist_ok=True)
for kind,loc in [('en','en'),('context','en'),('vi','vi')]:shutil.copyfile(paths[kind],fixture/loc/paths[kind].name)
catalogues._CATALOGUES=fixture
base=json.loads(paths['vi'].read_text()); en=json.loads(paths['en'].read_text()); ctx=json.loads(paths['context'].read_text())
assert len(base)==len(en)==len(ctx)==9

def extra(values):
    problems=[]
    for key,value in values.items():
        if value!=unicodedata.normalize('NFC',value):problems.append(key+': non-NFC')
        for token in ('subagents.models.','provider/model'):
            if token in en[key] and en[key].count(token)!=value.count(token):problems.append(key+': changed token '+token)
        if key.endswith('tier_invalid_mapping'):
            for token in ('model','effort'):
                if not re.search(r'(?<![\w.])'+token+r'(?![\w.])',value):problems.append(key+': missing config token '+token)
        if re.findall(r'`[^`]*`',en[key])!=re.findall(r'`[^`]*`',value):problems.append(key+': backtick mismatch')
    return problems

def run(label,values,negative=False):
    (fixture/'vi'/f'{ns}.json').write_text(json.dumps(values,ensure_ascii=False),encoding='utf-8')
    issues=check.check_catalogues()+extra(values)
    assert bool(issues)==negative,(label,issues)
    print(json.dumps({'case':label,'expected_rejection':negative,'issues':issues,'result':'PASS'},ensure_ascii=False))
run('candidate',base)
for label,key,replacement in [('missing_key',ns+'.team_missing',None),('placeholder_rename',ns+'.team_missing',base[ns+'.team_missing'].replace('{role}','{name}')),('placeholder_type',ns+'.team_missing',base[ns+'.team_missing'].replace('{role}','{role, number}')),('technical_token',ns+'.tier_invalid_selector',base[ns+'.tier_invalid_selector'].replace('provider/model','provider:modèle')),('nfc',ns+'.team_missing',unicodedata.normalize('NFD',base[ns+'.team_missing']))]:
    bad=base.copy()
    if replacement is None:bad.pop(key)
    else:bad[key]=replacement
    run(label,bad,True)
run('restored_candidate',base)
rows=[]
for key,value in base.items():
    want=runtime.message_arguments(en[key]); got=runtime.message_arguments(value)
    assert want==got
    assert not runtime.plural_selectors(value)
    rows.append({'key':key,'source':en[key],'target':value,'arguments':got,'nfc':True,'plural_selectors':[],'context_maxCells':ctx[key].get('maxCells'),'characters':len(value)})
(out/'mechanical.json').write_text(json.dumps({'input_hashes':hashes,'key_count':9,'repo_validator':'scripts.i18n.check.check_catalogues with scratch-only en/vi namespace fixture','per_key':rows,'result':'passed'},ensure_ascii=False,indent=2)+'\n')
print('ALL PASS: 9 keys; exact hashes; repository validator; 5 rejecting negative controls; restored positive control; NFC and literal checks. No build/install/test suite or ledger apply.')
