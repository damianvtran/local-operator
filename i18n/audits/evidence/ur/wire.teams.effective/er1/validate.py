import hashlib,json,os,shutil,sys,unicodedata
from pathlib import Path
root=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
sys.path.insert(0,str(root))
from local_operator.i18n import catalogues,runtime
from scripts.i18n import check
out=Path(__file__).parent
fixture=out/'catalogue-fixture'
ns='wire.teams.effective'
en=root/f'local_operator/i18n/catalogues/en/{ns}.json'
context=en.with_name(ns+'.context.json')
target=Path('/Users/damian/.local-operator/sessions/8dce6c4d11c0/scratchpad/ur/wire.teams.effective.json')
translator=target.with_name('translator-receipt.json')
expected=['fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','c43b87b8f9cc396da817e7c20da28c3b2a3b00668e6003ba4a4f0872c7ef345a']
for p,want in zip([en,context,target],expected):
    got=hashlib.sha256(p.read_bytes()).hexdigest(); assert got==want; print('INPUT',p,got)
print('TRANSLATOR_RECEIPT_SHA256',hashlib.sha256(translator.read_bytes()).hexdigest())
def strict(p):
    def pairs(xs):
        d={}
        for k,v in xs:
            assert k not in d,('duplicate',k)
            d[k]=v
        return d
    return json.loads(p.read_text(),object_pairs_hook=pairs)
e,t,c=strict(en),strict(target),strict(context)
assert len(e)==len(t)==len(c)==9 and e.keys()==t.keys()==c.keys()
for loc,src in [('en',en),('en',context),('ur',target)]:
    dest=fixture/loc/src.name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dest)
catalogues._CATALOGUES=fixture
print('SCOPED_NORMAL_VALIDATOR scripts.i18n.check.main([--catalogues-only]); catalogue root:',fixture)
assert check.main(['--catalogues-only'])==0
negative=dict(t); key=ns+'.team_missing'; negative[key]=negative[key].replace('{role}','{role, number}')
(fixture/'ur'/target.name).write_text(json.dumps(negative,ensure_ascii=False))
problems=check.check_catalogues(); assert any('arguments' in x and key in x for x in problems)
print('NEGATIVE_TYPE_CONTROL_REJECTED',problems)
shutil.copyfile(target,fixture/'ur'/target.name)
assert check.main(['--catalogues-only'])==0
controls=set(range(0x202a,0x202f))|set(range(0x2066,0x206a))|{0x200e,0x200f,0x061c}
for k in e:
    assert runtime.message_arguments(e[k])==runtime.message_arguments(t[k])
    assert not runtime.plural_selectors(e[k]) and not runtime.plural_selectors(t[k])
    assert not any(ord(ch) in controls for ch in t[k])
    args=runtime.message_arguments(t[k]); params={name:{'role':'QA-ٹیم-7','tier':'med','selector':'openai/gpt-6.1-sol'}[name] for name,kind in args}
    formatted=runtime.render_message(t[k],params,'ur')
    assert all(value in formatted for value in params.values())
    print('KEY_PASS',k,'ARGS',args,'CHARS',len(t[k]),'FORMATTED_LOGICAL_ORDER',json.dumps(formatted,ensure_ascii=False))
for suffix in ['tier_invalid_value','tier_invalid_mapping','tier_unconfigured','tier_invalid_selector']:
    assert 'subagents.models.{tier}' in t[ns+'.'+suffix]
for suffix in ['tier_invalid_value','tier_invalid_mapping','tier_invalid_selector']:
    assert 'provider/model' in t[ns+'.'+suffix]
assert all(x in t[ns+'.tier_invalid_mapping'] for x in ['model','effort'])
assert all('maxCells' not in value and 'maxChars' not in value for value in c.values())
print('PASS 9/9 keys; exact name/type/order parity; no plural/select messages; no manual direction controls; config literals retained; logical mixed-direction formatting only, no visual/bidi/fallback/pseudo evidence')
