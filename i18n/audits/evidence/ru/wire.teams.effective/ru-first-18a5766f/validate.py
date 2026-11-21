"""Independent Russian first-audit mechanical probe; scratch files only."""
import contextlib, hashlib, io, json, os, re, sys
from pathlib import Path
ROOT = Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
OUT = Path(__file__).resolve().parent
for name in list(os.environ):
    if name.startswith('CMUX_'):
        del os.environ[name]
sys.path.insert(0, str(ROOT))
from scripts.i18n import check
NS = 'wire.teams.effective'
paths = {
 'en': ROOT / f'local_operator/i18n/catalogues/en/{NS}.json',
 'context': ROOT / f'local_operator/i18n/catalogues/en/{NS}.context.json',
 'ru': Path('/Users/damian/.local-operator/sessions/9588c120c894/scratchpad/ru/wire.teams.effective.json'),
 'translator_receipt': Path('/Users/damian/.local-operator/sessions/9588c120c894/scratchpad/ru/translator-receipt.json'),
}
expected = ['fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','82486a592ce66624c8d7b1c0b45e05ba5f1b49d811d652f65f1ac6d784dc02e1','71261e4589b0bbacce7d1605a841f5a7fe5f1fed54c64f5ca413b89eb5cfb35b']
def strict_pairs(pairs):
    result = {}
    for k,v in pairs:
        assert k not in result, f'duplicate JSON key: {k}'
        result[k] = v
    return result
parsed = {}
for (name,path), digest in zip(paths.items(),expected):
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == digest, (name,actual)
    parsed[name] = json.loads(path.read_text(),object_pairs_hook=strict_pairs)
    print(f'HASH PASS {name} {actual}')
fixture = OUT/'catalogues'
for name,locale in [('en','en'),('context','en'),('ru','ru')]:
    dest=fixture/locale/paths[name].name
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(paths[name].read_bytes())
check.catalogues.catalogue_root=lambda:fixture
assert check.main(['--catalogues-only']) == 0
source,target,meta=parsed['en'],parsed['ru'],parsed['context']
assert len(source)==len(target)==len(meta)==9
assert list(source)==list(target)==list(meta)
for key, english in source.items():
    russian=target[key]
    args=check.msg_runtime.message_arguments(english)
    assert args==check.msg_runtime.message_arguments(russian),(key,args)
    assert not re.search(r'\{\w+,\s*(plural|select|selectordinal)',english+russian)
    for token in ['subagents.models.','provider/model']:
        assert english.count(token)==russian.count(token),(key,token)
    if key.endswith('tier_invalid_mapping'):
        for token in ['model','effort']:
            assert len(re.findall(r'(?<![A-Za-z_/])'+token+r'(?![A-Za-z_/])',russian))==1
    assert re.findall(r'`[^`]*`|</?[A-Za-z][^>]*>',english)==re.findall(r'`[^`]*`|</?[A-Za-z][^>]*>',russian)
    assert not any(c in russian for c in '\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
    print('KEY PASS',key,'arguments='+repr(args),'chars='+str(len(russian)))
# Ensure the stock validator actually reads this isolated target and rejects a broken placeholder.
probe_path=fixture/'ru'/paths['ru'].name
broken=dict(target)
broken[NS+'.team_missing']=broken[NS+'.team_missing'].replace('{role}','{wrong}')
probe_path.write_text(json.dumps(broken,ensure_ascii=False))
check.catalogues.load_catalogue.cache_clear() if hasattr(check.catalogues.load_catalogue,'cache_clear') else None
negative=check.main(['--catalogues-only'])
assert negative==1,'negative control not rejected'
print('NEGATIVE CONTROL PASS: renamed role rejected')
probe_path.write_bytes(paths['ru'].read_bytes())
check.catalogues.load_catalogue.cache_clear() if hasattr(check.catalogues.load_catalogue,'cache_clear') else None
assert check.main(['--catalogues-only'])==0
print('PASS: 9/9 strict JSON/key/context/argument/token checks; no plural/select arms; no bidi controls; fixture restored')
