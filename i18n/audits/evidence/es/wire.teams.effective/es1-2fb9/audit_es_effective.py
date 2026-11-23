import os, sys, json, hashlib, re, unicodedata
from pathlib import Path
from unittest.mock import patch
ROOT = Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
OUT = Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0, str(ROOT))
from scripts.i18n import check
NS='wire.teams.effective'
files={
'en': ROOT/f'local_operator/i18n/catalogues/en/{NS}.json',
'context': ROOT/f'local_operator/i18n/catalogues/en/{NS}.context.json',
'es': Path('/Users/damian/.local-operator/sessions/8fba0458db34/scratchpad/es.wire.teams.effective.json'),
'provenance': Path('/Users/damian/.local-operator/sessions/8fba0458db34/scratchpad/es.wire.teams.effective.translator-provenance.json'),
'style': ROOT/'i18n/style-guides/es.md', 'glossary': ROOT/'i18n/glossary.md'}
expected={'en':'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','context':'01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','es':'aad6efce35fa6fccc25573506b0ec3d3ee042f6e23f6e52d5440d2289a519e25','provenance':'ce95251a453698bb6421eff617e1711c529222a091d11945a6b39751e666be75'}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in files.items()}
for k,v in expected.items(): assert hashes[k]['sha256']==v,(k,hashes[k])
def unique(pairs):
    d={}
    for k,v in pairs:
        assert k not in d, f'duplicate key: {k}'
        d[k]=v
    return d
en=json.loads(files['en'].read_text(),object_pairs_hook=unique)
es=json.loads(files['es'].read_text(),object_pairs_hook=unique)
context=json.loads(files['context'].read_text(),object_pairs_hook=unique)
assert len(en)==len(es)==len(context)==9 and list(en)==list(es)==list(context)
def scoped_validator(target):
    # Only input enumeration/loading is adapted to the assigned scratch candidate.
    # Real repository catalogue validation logic and ICU parser remain unchanged.
    with patch.object(check.catalogues,'namespaces',return_value=[NS]), patch.object(check.catalogues,'locales',return_value=['en','es']), patch.object(check.catalogues,'load_catalogue',side_effect=lambda loc,ns: en if loc=='en' else target), patch.object(check.catalogues,'context',return_value=context):
        return check.check_catalogues()
problems=scoped_validator(es)
assert problems==[], problems
bad=dict(es); bad[NS+'.team_missing']=bad[NS+'.team_missing'].replace('{role}','')
negative=scoped_validator(bad)
assert negative and any('team_missing' in p for p in negative),negative
plural=json.loads((ROOT/'local_operator/i18n/data/plural_rules.json').read_text())['locales']['es']['categories']
assert set(plural)=={'one','many','other'}
rows=[]
for key,source in en.items():
    target=es[key]
    assert re.findall(r'\{[^{}]+\}',source)==re.findall(r'\{[^{}]+\}',target)
    want=check.msg_runtime.message_arguments(source); got=check.msg_runtime.message_arguments(target)
    assert want==got
    assert not check.msg_runtime.plural_selectors(target)
    for token in ['subagents.models.','provider/model']:
        assert source.count(token)==target.count(token),(key,token)
    if key.endswith('tier_invalid_mapping'):
        assert re.search(r'\ba model\b',target) and re.search(r'\ba effort\b',target)
    assert not re.search(r'[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]',target)
    assert not re.search(r'[`<>]',source+target)
    assert not any(k in context[key] for k in ['maxCells','maxChars'])
    rows.append({'key':key,'arguments':repr(got),'characters':len(target),'combining_codepoints':[f'U+{ord(c):04X}' for c in target if unicodedata.combining(c)],'mechanical':'PASS'})
result={'input_hashes':hashes,'key_count':9,'json_duplicate_keys':'none','key_order_parity':True,'validator':{'executed':True,'entrypoint':'scripts.i18n.check.check_catalogues','scope':'Only assigned namespace; in-memory enumeration/load adapters for scratch candidate, no repository catalogue mutation; not the full check.py CLI','problems':problems,'negative_control':{'mutation':'In-memory removal of {role}, candidate file untouched','actual_problems':negative,'result':'detected'}},'cldr_categories':plural,'plural_select_count':0,'technical_tokens':'PASS; config model/effort identities checked contextually (English effort prose legitimately becomes esfuerzo)','bidi_controls':'none','rows':rows}
(OUT/'es.effective.mechanical.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
