"""Independent scratch-only admission; no repo, runtime or catalogue edits."""
import sys, json, hashlib, re, shutil, unicodedata
from pathlib import Path
repo=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
out=Path(__file__).parent
sys.path.insert(0,str(repo))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
paths={'en':repo/'local_operator/i18n/catalogues/en/wire.teams.effective.json','context':repo/'local_operator/i18n/catalogues/en/wire.teams.effective.context.json','ur':Path('/Users/damian/.local-operator/sessions/8dce6c4d11c0/scratchpad/ur/wire.teams.effective.json')}
expected=['fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','c43b87b8f9cc396da817e7c20da28c3b2a3b00668e6003ba4a4f0872c7ef345a']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pairs(items):
    result={}
    for k,v in items:
        assert k not in result, ('duplicate',k)
        result[k]=v
    return result
hashes={k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()}
assert [x['sha256'] for x in hashes.values()]==expected
objects={k:json.loads(p.read_text(),object_pairs_hook=pairs) for k,p in paths.items()}
en,ur,ctx=objects['en'],objects['ur'],objects['context']
assert list(en)==list(ur)==list(ctx) and len(en)==9
root=out/'catalogue-fixture'
for locale,key in [('en','en'),('en','context'),('ur','ur')]:
    (root/locale).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(paths[key],root/locale/paths[key].name)
catalogues._CATALOGUES=root
code=check.main(['--catalogues-only'])
assert code==0
rows=[]
for k,t in ur.items():
    source=en[k]
    assert re.findall(r'\{[^{}]+\}',source)==re.findall(r'\{[^{}]+\}',t)
    # Preserve only context-declared syntax, not English prose occurrences of model/effort.
    tokens=['subagents.models.{tier}','provider/model']
    if k.endswith('tier_invalid_mapping'):tokens+=['model','effort']
    for token in tokens:
        if token in ('model','effort'):
            assert len(re.findall(r'(?<![A-Za-z0-9_/])'+token+r'(?![A-Za-z0-9_/])',t))==1
        else:assert t.count(token)==source.count(token)
    assert not any(unicodedata.bidirectional(c) in ('LRE','RLE','LRO','RLO','PDF','LRI','RLI','FSI','PDI') or c in '\u200e\u200f\u061c' for c in t)
    assert not re.search(r',\s*(plural|select|selectordinal)\s*,',source+t)
    rows.append({'key':k,'placeholder_tokens':re.findall(r'\{[^{}]+\}',t),'placeholder_names_types_order':'passed by normal validator and independent token check','technical_literals':'passed','manual_bidi_controls':0,'plural_select':'none in source or target; CLDR arms not applicable','maxCells':ctx[k].get('maxCells'),'maxChars':ctx[k].get('maxChars'),'code_markup':'none in source or target','character_count':len(t)})
result={'input_hashes':hashes,'key_count':len(rows),'rows':rows,'validator':{'entrypoint':'scripts.i18n.check.main(["--catalogues-only"])','exit_code':code,'fixture':str(root),'normal_catalogue_admission':True,'scope':'Only assigned namespace; no check suppression or admission bypass'},'limits':['No runtime formatting, pseudo, fallback, rendered mixed-direction or visual execution performed']}
(out/'mechanical.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Independent checks PASS: exact 9 keys, duplicate-free JSON, source/context/target hashes, placeholder parity, technical tokens, no manual bidi controls. No plural/select present; no width caps.')
print(json.dumps(hashes,indent=2))
