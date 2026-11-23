import sys, json, hashlib, unicodedata, re, shutil
from pathlib import Path
from unittest.mock import patch
REPO=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
OUT=Path(__file__).parent
sys.path.insert(0,str(REPO))
from scripts.i18n import check
from local_operator.i18n import runtime
source=REPO/'local_operator/i18n/catalogues/en/wire.teams.effective.json'
context=source.with_name('wire.teams.effective.context.json')
target=Path('/Users/damian/.local-operator/sessions/bdce0d00e792/scratchpad/effective-reasons-vi/wire.teams.effective.json')
receipt=target.with_name('translator-receipt.json')
expected=['fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','4d510633cda89708fa842a0812dc9fff2836383e93e1322c6a6d532cf52cc057','ff505f4a6c875eba2bd2a75eeab0e1f64001790bdbddaae62866329e7beca6fd']
for p,sha in zip([source,context,target,receipt],expected):
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==sha,(str(p),actual)
    print('HASH PASS',p,actual)
fixture=OUT/'fixture'
for locale,p in [('en',source),('en',context),('vi',target)]:
    dest=fixture/locale/p.name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(p,dest)
with patch.object(check.catalogues,'catalogue_root',return_value=fixture):
    result=check.main(['--catalogues-only'])
assert result==0
# Independently check duplicate JSON keys, ordered placeholder names/types,
# technical tokens, normalization, controls, and source width hints.
def unique(pairs):
    d={}
    for k,v in pairs:
        assert k not in d,k
        d[k]=v
    return d
en=json.loads(source.read_text(),object_pairs_hook=unique)
vi=json.loads(target.read_text(),object_pairs_hook=unique)
ctx=json.loads(context.read_text(),object_pairs_hook=unique)
assert list(en)==list(vi)==list(ctx)
rows=[]
for k,s in en.items():
    t=vi[k]
    args=runtime.message_arguments(s)
    assert args==runtime.message_arguments(t)
    assert re.findall(r'\{[^{}]+\}',s)==re.findall(r'\{[^{}]+\}',t)
    assert runtime.plural_selectors(s)==runtime.plural_selectors(t)==()
    assert unicodedata.normalize('NFC',t)==t
    assert all(unicodedata.category(c)!='Cf' for c in t)
    for token in ['subagents.models.','provider/model']:
        assert s.count(token)==t.count(token),(k,token)
    if k.endswith('tier_invalid_mapping'):
        assert re.search(r'\bmodel\b',t) and re.search(r'\beffort\b',t)
    assert re.findall(r'`[^`]*`|</?[^>]+>|--[a-zA-Z][\w-]*',s)==re.findall(r'`[^`]*`|</?[^>]+>|--[a-zA-Z][\w-]*',t)
    assert 'maxCells' not in ctx[k] and 'maxChars' not in ctx[k]
    row={'key':k,'arguments':[list(x) for x in args], 'placeholder_sequence':re.findall(r'\{[^{}]+\}',t),'characters':len(t),'NFC':True,'mechanical':'passed'}
    rows.append(row)
    print('KEY PASS',json.dumps(row,ensure_ascii=False))
(OUT/'mechanical.json').write_text(json.dumps({'key_count':len(rows),'rows':rows,'validator_exit_code':result,'plural_select':'absent in all 9 source and target strings; vi other category not invoked'},ensure_ascii=False,indent=2)+'\n')
print('PASS 9/9; no fixed width hints. Runtime/pseudo/fallback/bidi/visual NOT RUN.')
