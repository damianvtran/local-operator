import sys, os, json, hashlib, shutil, re, importlib.util
from pathlib import Path
repo=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
scratch=Path('/Users/damian/.local-operator/sessions/289d6604993c/scratchpad')
ns='wire.teams.effective'
source=repo/f'local_operator/i18n/catalogues/en/{ns}.json'
context=source.with_name(f'{ns}.context.json')
target=Path('/Users/damian/.local-operator/sessions/8fba0458db34/scratchpad/es.wire.teams.effective.json')
fixture=scratch/'es-second-fixture'
for locale in ['en','es']:
    (fixture/locale).mkdir(parents=True,exist_ok=True)
shutil.copyfile(source,fixture/'en'/source.name)
shutil.copyfile(context,fixture/'en'/context.name)
shutil.copyfile(target,fixture/'es'/source.name)
sys.path.insert(0,str(repo))
from scripts.i18n import check, ledger
from local_operator.i18n import catalogues
catalogues._CATALOGUES=fixture
problems=check.check_catalogues()
print('Repository validator: scripts.i18n.check.check_catalogues()')
print('Isolated catalogue root:',fixture)
print('Validator problems:',json.dumps(problems,ensure_ascii=False))
assert not problems

def strict(path):
    def pairs(xs):
        out={}
        for k,v in xs:
            assert k not in out, f'duplicate key {k}'
            out[k]=v
        return out
    return json.loads(path.read_text(),object_pairs_hook=pairs)
en,es,ctx=map(strict,[source,target,context])
assert len(en)==len(es)==len(ctx)==9
assert set(en)==set(es)==set(ctx)
rows=[]
for k,v in en.items():
    t=es[k]
    want=check.msg_runtime.message_arguments(v)
    got=check.msg_runtime.message_arguments(t)
    assert want==got
    assert re.findall(r'\{[^{}]*\}',v)==re.findall(r'\{[^{}]*\}',t)
    assert not re.search(r',\s*(plural|select|selectordinal)\b',v+' '+t)
    for token in ['subagents.models.','provider/model']:
        assert v.count(token)==t.count(token)
    assert re.findall(r'`[^`]*`',v)==re.findall(r'`[^`]*`',t)
    assert not any(ord(c) in [0x061c,0x200e,0x200f,*range(0x202a,0x202f),*range(0x2066,0x206a)] for c in t)
    rows.append({'key':k,'source':v,'current_text':t,'status':'passed','placeholder_types':repr(got),'chars':len(t),'maxCells':ctx[k].get('maxCells'),'mechanical':'Exact keys, ICU argument names/kinds and literal configuration syntax; no plural/select expressions, markup or code spans.'})
assert 'Asigna a model un selector provider/model y, opcionalmente, a effort' in es[ns+'.tier_invalid_mapping']
print('Independent strict JSON/key/placeholder/type/token/control checks: PASS 9/9')
print('Plural/select expressions: 0; Spanish CLDR arms not applicable')
print('Sidecar fixed cell-width limits: 0; rendered wrapping NOT RUN')
(scratch/'es.second.mechanical.json').write_text(json.dumps({'status':'passed','key_count':9,'validator_problems':problems,'per_key':rows},ensure_ascii=False,indent=2)+'\n')
