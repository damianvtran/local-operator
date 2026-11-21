"""Independent, namespace-scoped admission checks; fixtures and logs stay in scratch.
Not application-runtime, visual, shaping or pseudo evidence.
"""
import copy, hashlib, importlib.util, json, os, shutil, sys
from pathlib import Path
REPO=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
OUT=Path(__file__).parent
TARGET=Path('/Users/damian/.local-operator/sessions/3f1cf5b56ba0/scratchpad/terminal-status-ru/wire.session_status.json')
sys.path.insert(0,str(REPO))
from local_operator.i18n import catalogues, runtime
spec=importlib.util.spec_from_file_location('independent_checker',REPO/'scripts/i18n/check.py')
checker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=checker;spec.loader.exec_module(checker)
paths={'en':REPO/'local_operator/i18n/catalogues/en/wire.session_status.json','context':REPO/'local_operator/i18n/catalogues/en/wire.session_status.context.json','ru':TARGET}
hashes={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
assert hashes['en']['sha256']=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert hashes['context']['sha256']=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert hashes['ru']['sha256']=='b15cc559652777694f2e252404862095bbdc902f90fe72fcee930a03b698f47c'
en,ctx,ru=[json.loads(paths[k].read_text()) for k in ('en','context','ru')]
assert len(en)==len(ctx)==len(ru)==17 and set(en)==set(ru)==set(ctx)
fixture=OUT/'fixtures';(fixture/'en').mkdir(parents=True,exist_ok=True);(fixture/'ru').mkdir(exist_ok=True)
shutil.copyfile(paths['en'],fixture/'en/wire.session_status.json');shutil.copyfile(paths['context'],fixture/'en/wire.session_status.context.json')
catalogues._CATALOGUES=fixture
rp=fixture/'ru/wire.session_status.json'
def admit(data):
    rp.write_text(json.dumps(data,ensure_ascii=False));return checker.main(['--catalogues-only'])
print('=== NORMAL REPOSITORY CATALOGUE ADMISSION (scoped scratch root) ===')
assert admit(ru)==0
print('PASS normal candidate: 17 keys')
for label,mutate in [
 ('renamed placeholder',lambda d:d.__setitem__('wire.session_status.signal_stop',d['wire.session_status.signal_stop'].replace('{command','{cmd'))),
 ('missing key',lambda d:d.pop('wire.session_status.complete')),
 ('added key',lambda d:d.__setitem__('wire.session_status.extra','Лишнее')),
 ('invalid ICU',lambda d:d.__setitem__('wire.session_status.dormant',d['wire.session_status.dormant'][:-2]+')')),
 ('invalid plural category',lambda d:d.__setitem__('wire.session_status.dormant',d['wire.session_status.dormant'].replace('few {','two {'))),
 ('missing other',lambda d:d.__setitem__('wire.session_status.dormant',d['wire.session_status.dormant'].replace(' other {# пробуждения приостановлено}',''))),
]:
    data=copy.deepcopy(ru);mutate(data);result=admit(data);assert result!=0,label;print('PASS rejected negative:',label,'exit',result)
# Deliberately measure, rather than assume, the completeness gate for locale categories.
data=copy.deepcopy(ru);data['wire.session_status.dormant']=data['wire.session_status.dormant'].replace(' few {# пробуждения приостановлены}','');gap_result=admit(data)
print('CLDR missing-few control exit:',gap_result,'(0 means known validator completeness gap; actual target has few)')
assert admit(ru)==0
print('=== INDEPENDENT STRUCTURAL CHECKS AND MECHANICAL ICU EXPANSIONS ===')
def args(text):
    found=[]
    def visit(nodes):
        for n in nodes:
            if isinstance(n,runtime._Arg):
                found.append({'name':n.name,'kind':n.kind,'selectors':[s for s,_ in n.parts]})
                for _,branch in n.parts:visit(branch)
    visit(runtime.parse_message(text));return found
rows=[]
for k in en:
    assert runtime.message_arguments(en[k])==runtime.message_arguments(ru[k]),k
    a,b=args(en[k]),args(ru[k]);
    assert [(x['name'],x['selectors']) for x in a if x['kind']=='select']==[(x['name'],x['selectors']) for x in b if x['kind']=='select'],k
    assert '\n' not in ru[k] and '\r' not in ru[k]
    expansions=[]
    if any(x['name']=='command' for x in b):
        for command in ['none','/stop --all','lop sessions stop audit-synthetic-id']:
            text=runtime.render_message(ru[k],{'command':command},'ru')
            if command=='none':assert 'none' not in text and 'по команде' not in text
            else:assert text.endswith(' по команде '+command)
            expansions.append({'params':{'command':command},'text':text})
    elif any(x['name']=='cause' for x in b):
        for cause in next(x['selectors'] for x in b if x['name']=='cause')+['unrecognized_synthetic_cause']:
            text=runtime.render_message(ru[k],{'cause':cause},'ru');expansions.append({'params':{'cause':cause},'text':text})
        assert expansions[-1]['text']==('Ошибка (не просмотрена)' if 'unseen' in k else 'Ошибка')
    elif k.endswith('.dormant'):
        assert runtime.plural_selectors(ru[k])==(('one','few','many','other'),)
        for count in [0,1,2,5,11,21,22,55,101,1000001,1.5,2.5,0.5]:
            text=runtime.render_message(ru[k],{'count':count},'ru');expansions.append({'params':{'count':count},'text':text})
    elif k.endswith('.retired_label'):
        assert ru[k]==json.loads((REPO/'local_operator/i18n/catalogues/ru/wire.session_status.json').read_text())[k]
        expansions.append({'params':{'base':'Прервано'},'text':runtime.render_message(ru[k],{'base':'Прервано'},'ru')})
    else:expansions.append({'params':{},'text':ru[k]})
    row={'key':k,'en':en[k],'ru':ru[k],'context':ctx[k],'structure':b,'expansions':expansions}
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False))
(OUT/'mechanical.json').write_text(json.dumps({'hashes':hashes,'key_count':len(rows),'rows':rows,'cldr_missing_few_admission_exit':gap_result},ensure_ascii=False,indent=2)+'\n')
print('PASS exact placeholder kinds/names, all select arms, Russian four-category plural, command bytes and none omission, unknown-cause fallback, unchanged retired_label')
print('Mechanical ICU expansion only: no application, visual, pseudo or shaping proof claimed.')
