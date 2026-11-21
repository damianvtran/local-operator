import sys, json, hashlib, shutil, copy, re
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.i18n import check
from local_operator.i18n import catalogues, runtime
from local_operator.i18n import format as fmt
from decimal import Decimal
NS='wire.session_status'
TARGET=Path('/Users/damian/.local-operator/sessions/7b97856c53df/scratchpad/terminal-status-ur.json')
EN=ROOT/'local_operator/i18n/catalogues/en'/f'{NS}.json'
CONTEXT=EN.with_name(f'{NS}.context.json')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(EN)=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(CONTEXT)=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert sha(TARGET)=='4bd2a495aa050c93e2f380dd9e2b26dfd0fc0f8177138218f1ed3e528e05050b'
inputs={label:{'path':str(p),'sha256':sha(p)} for label,p in [('en',EN),('context',CONTEXT),('ur',TARGET),('style',ROOT/'i18n/style-guides/ur.md'),('glossary',ROOT/'i18n/glossary.md')]}
print('INPUT_HASHES',json.dumps(inputs))
def strict_pairs(pairs):
    d={}
    for k,v in pairs:
        assert k not in d, ('duplicate',k)
        d[k]=v
    return d
en=json.loads(EN.read_text(),object_pairs_hook=strict_pairs)
ur=json.loads(TARGET.read_text(),object_pairs_hook=strict_pairs)
assert len(en)==len(ur)==17 and en.keys()==ur.keys()
fixture=OUT/'fixture/catalogues'
for loc in ['en','ur']: (fixture/loc).mkdir(parents=True,exist_ok=True)
shutil.copyfile(EN,fixture/'en'/EN.name)
shutil.copyfile(CONTEXT,fixture/'en'/CONTEXT.name)
target=fixture/'ur'/EN.name
shutil.copyfile(TARGET,target)
def validate(data=None):
    if data is None: target.write_bytes(TARGET.read_bytes())
    else: target.write_text(json.dumps(data,ensure_ascii=False))
    with patch.object(catalogues,'catalogue_root',return_value=fixture):
        return check.check_catalogues()
assert validate()==[]
print('NORMAL_ADMISSION PASS scripts.i18n.check.check_catalogues scoped fixture en+ur/wire.session_status')
controls={}
def negative(name,change):
    data=copy.deepcopy(ur); change(data)
    problems=validate(data)
    controls[name]={'rejected':bool(problems),'actual_problems':problems}
    print('NEGATIVE',name,json.dumps(controls[name],ensure_ascii=False))
    assert problems,name
negative('dropped_key',lambda d:d.pop('wire.session_status.complete'))
negative('added_key',lambda d:d.update({'wire.session_status.extra':'اضافی'}))
negative('renamed_placeholder',lambda d:d.update({'wire.session_status.signal_stop':d['wire.session_status.signal_stop'].replace('{command','{cmd')}))
negative('invalid_icu',lambda d:d.update({'wire.session_status.dormant':d['wire.session_status.dormant'][:-2]}))
negative('missing_other',lambda d:d.update({'wire.session_status.dormant':'رکا ہوا ({count, plural, one {# بیداری غیر فعال}})'}))
# The normal validator allows missing locale categories for plural->plural, so
# independently check completeness and pin the discriminating counterfactual.
gap=copy.deepcopy(ur);gap['wire.session_status.dormant']='رکا ہوا ({count, plural, other {# بیداریاں غیر فعال}})'
problems=validate(gap)
controls['missing_one']={'rejected':bool(problems),'actual_problems':problems,'supplemental_audit_rejected':set(runtime.plural_selectors(gap['wire.session_status.dormant'])[0])!={'one','other'}}
print('CLDR_GAP_PROBE',json.dumps(controls['missing_one']))
assert controls['missing_one']['supplemental_audit_rejected']
assert validate()==[]
for k,v in ur.items():
    assert runtime.message_arguments(en[k])==runtime.message_arguments(v),(k,'arguments')
    assert not re.search('[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]',v),(k,'manual bidi')
    for selectors in runtime.plural_selectors(v): assert set(selectors)=={'one','other'}
    def structure(nodes):
        out=[]
        for n in nodes:
            if isinstance(n,runtime._Arg) and n.kind in ('select','plural'):
                out.append((n.name,n.kind,[s for s,_ in n.parts]))
                for _,body in n.parts:out.extend(structure(body))
        return out
    assert structure(runtime.parse_message(en[k]))==structure(runtime.parse_message(v)),(k,'selectors')
print('SUPPLEMENTAL PASS exact arguments/kinds/select branches/CLDR completeness/no bidi controls/duplicate JSON keys')
rows=[]
for k,v in ur.items():
    name=k.rsplit('.',1)[1]
    if name=='retired_label': params=[{'base':'Interrupted'}]
    elif name in ('signal_stop','unseen_signal_stop','forced_kill','unseen_forced_kill'):
        params=[{'command':x} for x in ['none','/stop --all','lop sessions stop abc123']]
    elif name in ('error_cause','unseen_error_cause'):
        node=runtime.parse_message(v)[0]
        params=[{'cause':arm} for arm,_ in node.parts]+[{'cause':'external_unknown'}]
    elif name=='dormant':params=[{'count':n} for n in [0,1,2,5,21,55,1000000,Decimal('1.5'),Decimal('1.0')]]
    else:params=[{}]
    for p in params:
        output=runtime.render_message(v,p,'ur')
        source=runtime.render_message(en[k],p,'en')
        row={'key':k,'params':{a:str(b) if isinstance(b,Decimal) else b for a,b in p.items()},'en_output':source,'ur_output':output}
        if 'count' in p: row['category']=fmt.plural_category('ur',p['count'])
        if 'command' in p:
            assert ('none' not in output) if p['command']=='none' else p['command'] in output
        if name=='dormant':assert ('بیداری غیر فعال' in output) if row['category']=='one' else ('بیداریاں غیر فعال' in output)
        rows.append(row)
        print('BRANCH',json.dumps(row,ensure_ascii=False))
print('NORMAL_ADMISSION_RESTORED',validate())
result={'input_hashes':inputs,'key_count':17,'rendered_branch_cases':len(rows),'branches':rows,'normal_admission_problems':validate(),'negative_controls':controls,'mechanical_verdict':'passed','limitations':['ICU string formatting only: not visual/runtime app/pseudo/shaping/RTL renderer certification','Missing-one plural->plural gap observed in repo validator; actual target includes both categories.']}
(OUT/'mechanical.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('SUCCESS',len(rows),'branch cases')
