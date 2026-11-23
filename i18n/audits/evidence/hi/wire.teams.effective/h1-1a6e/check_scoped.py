"""Scratch-only scoped admission; invokes repository validator on isolated bytes."""
import hashlib, json, re, shutil, sys
from pathlib import Path
repo=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
sys.path.insert(0,str(repo))
from scripts.i18n import check, ledger
from local_operator.i18n import catalogues, runtime
out=Path(__file__).parent
root=out/'fixture'/'catalogues'
ns='wire.teams.effective'
paths={'en':repo/f'local_operator/i18n/catalogues/en/{ns}.json','context':repo/f'local_operator/i18n/catalogues/en/{ns}.context.json','target':Path('/Users/damian/.local-operator/sessions/47e37aebb87c/scratchpad/effective-reasons-hi/catalogues/hi/wire.teams.effective.json'),'translator_receipt':Path('/Users/damian/.local-operator/sessions/47e37aebb87c/scratchpad/effective-reasons-hi/translator-receipt.json')}
expected={'en':'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','context':'01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','target':'e2a1d7d25325c08db1f414105c75609e7c3a4f65d8a877169064974a9d5f1d92'}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
for k,v in expected.items(): assert hashes[k]['sha256']==v,(k,hashes[k])
for k,loc,name in [('en','en',ns+'.json'),('context','en',ns+'.context.json'),('target','hi',ns+'.json')]:
    dest=root/loc/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(paths[k],dest)
catalogues._CATALOGUES=root
problems=check.check_catalogues()
print('Repository scripts.i18n.check.check_catalogues isolated root:',root)
print('Validator problems:',json.dumps(problems,ensure_ascii=False));assert not problems
# Independent strict parse rejects duplicate members as well as JSON errors.
def strict(p):
    def pairs(items):
        d={}
        for k,v in items:
            assert k not in d,('duplicate',k)
            d[k]=v
        return d
    return json.loads(p.read_text(),object_pairs_hook=pairs)
en,hi,context=(strict(paths[k]) for k in ('en','target','context'))
assert set(en)==set(hi)==set(context)
rows=[]
for k,v in hi.items():
    assert runtime.message_arguments(en[k])==runtime.message_arguments(v)
    assert not runtime.plural_selectors(v)
    assert re.findall(r'\{[^{}]+\}',en[k])==re.findall(r'\{[^{}]+\}',v)
    assert re.findall(r'`[^`]*`|</?[^>]+>',en[k])==re.findall(r'`[^`]*`|</?[^>]+>',v)
    assert not any(c in v for c in '\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
    for token in ('subagents.models.','provider/model'):
        assert en[k].count(token)==v.count(token),(k,token)
    latin=re.findall(r'[A-Za-z]+(?:[./][A-Za-z]+)*',re.sub(r'\{[^{}]*\}','',v))
    assert set(latin)<= {'subagents.models.','subagents.models','provider/model','model','effort'},(k,latin)
    rows.append({'key':k,'arguments':list(runtime.message_arguments(v)),'codepoints':len(v),'ascii_tokens':latin,'width_hint':context[k].get('maxCells'),'mechanical':'passed'})
print('Independent strict JSON/key/argument/literal/control checks: PASS (9/9); plural/select arms: none')
# Negative control proves the scoped validator examines candidate arguments.
bad=dict(hi);bad[ns+'.team_missing']=bad[ns+'.team_missing'].replace('{role}','{wrong}')
(root/'hi'/f'{ns}.json').write_text(json.dumps(bad,ensure_ascii=False))
negative=check.check_catalogues();assert any('arguments' in p for p in negative),negative
print('Negative control renamed role placeholder: REJECTED',json.dumps(negative,ensure_ascii=False))
shutil.copyfile(paths['target'],root/'hi'/f'{ns}.json')
assert not check.check_catalogues()
print('Restored candidate control: PASS')
(out/'mechanical.json').write_text(json.dumps({'input_hashes':hashes,'keys':rows,'validator_problems':problems,'negative_control':negative},ensure_ascii=False,indent=2)+'\n')
print('Input hashes:',json.dumps(hashes,ensure_ascii=False))
