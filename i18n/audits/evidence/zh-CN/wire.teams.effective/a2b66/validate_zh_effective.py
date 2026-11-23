"""Narrow audit: invoke frozen repository catalogue validator on copied inputs only.
Not an app/rendered-path or full repository check. No repository files are written.
"""
import hashlib, json, os, re, sys
from pathlib import Path
root = Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
sys.path.insert(0, str(root))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
scratch = Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
ns = 'wire.teams.effective'
paths = {'en': root/f'local_operator/i18n/catalogues/en/{ns}.json', 'context': root/f'local_operator/i18n/catalogues/en/{ns}.context.json', 'zh-CN': Path('/Users/damian/.local-operator/sessions/530f154495de/scratchpad/zh-CN.wire.teams.effective.json'), 'translator_receipt': Path('/Users/damian/.local-operator/sessions/530f154495de/scratchpad/zh-CN.wire.teams.effective.translator-receipt.json')}
expected = dict(zip(paths, ['fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','6d8d1fc410def690b03ad02ba6c759a5d7b3e82391d9c12dc441fbff7d73f454','cd670459e27140ac60a009c72ef40ddb90485c5e9a1deb82a7af241974432f4a']))
def strict(p):
    def pairs(items):
        result = {}
        for k,v in items:
            assert k not in result, f'duplicate key {k}'
            result[k]=v
        return result
    return json.loads(p.read_text(), object_pairs_hook=pairs)
for label,p in paths.items():
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    print('INPUT', label, str(p), digest, 'MATCH' if digest==expected[label] else 'MISMATCH')
    assert digest==expected[label]
en, target, context = strict(paths['en']),strict(paths['zh-CN']),strict(paths['context'])
fixture=scratch/'validator-fixture'
for locale,label in [('en','en'),('zh-CN','zh-CN')]:
    dest=fixture/locale/f'{ns}.json'; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(paths[label].read_bytes())
(fixture/'en'/f'{ns}.context.json').write_bytes(paths['context'].read_bytes())
catalogues._CATALOGUES=fixture
problems=check.check_catalogues()
print('REPOSITORY check.check_catalogues(), isolated namespace fixture:', json.dumps(problems))
assert not problems
assert len(en)==len(target)==len(context)==9 and set(en)==set(target)==set(context)
print('SELF CHECK: strict JSON, 9/9 key parity, 9/9 context coverage PASS')
bidi=set('\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f')
for k,s in en.items():
    t=target[k]
    a,b=runtime.message_arguments(s),runtime.message_arguments(t)
    assert a==b
    assert re.findall(r'\{[^{}]+\}',s)==re.findall(r'\{[^{}]+\}',t)
    assert not runtime.plural_selectors(s) and not runtime.plural_selectors(t)
    assert not bidi.intersection(t)
    for token in ['subagents.models.','provider/model']:
        assert s.count(token)==t.count(token),(k,token)
    if k.endswith('tier_invalid_mapping'):
        for token in ['model','effort']: assert token in t
    for token in re.findall(r'`[^`]*`|</?[^>]+>|--[a-z][a-z-]*',s): assert token in t
    params={name:{'tier':'hi','role':'demo-team','selector':'provider/model'}[name] for name,kind in a}
    result=runtime.render_message(t,params,'zh-CN')
    assert '{' not in result and '}' not in result
    print('KEY PASS',k,'ARGS',a,'FORMATTED',result)
print('CLDR zh-CN: no plural/select expressions, no arms required; other-only rule is not exercised')
print('No markup, backtick code, commands, flags, bidi controls or fixed maxCells budgets in these inputs')
print('NOT EXECUTED: full check.py/generate.py, ledger append, pseudo-localization, GUI/fallback/RTL rendering')
print('RESULT PASS: scoped mechanical + formatting only; formatting strings is NOT rendered visual evidence')
