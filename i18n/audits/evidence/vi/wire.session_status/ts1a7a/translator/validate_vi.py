import os, sys, json, hashlib, shutil, runpy, unicodedata, re
from pathlib import Path
from datetime import datetime, timezone
repo=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
scratch=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0,str(repo))
from local_operator.i18n import catalogues, runtime
ns='wire.session_status'
target=scratch/'wire.session_status.vi.json'
en=repo/f'local_operator/i18n/catalogues/en/{ns}.json'
context=en.with_name(ns+'.context.json')
old=repo/f'local_operator/i18n/catalogues/vi/{ns}.json'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(en)=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(context)=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
source=json.loads(en.read_text()); vi=json.loads(target.read_text()); previous=json.loads(old.read_text())
lines=[]
def log(s):
    lines.append(s); print(s)
assert list(source)==list(vi) and len(vi)==17
log('PASS: frozen en/context SHA-256; exact 17-key order, 16 new translations + 1 preserved label')
assert vi[ns+'.retired_label']==previous[ns+'.retired_label']
log('PASS: retired_label preserves exact frozen existing value')
assert all(unicodedata.normalize('NFC',s)==s and '\n' not in s for s in vi.values())
assert not any(any(unicodedata.bidirectional(c) in {'LRE','RLE','LRO','RLO','PDF','LRI','RLI','FSI','PDI'} for c in s) for s in vi.values())
log('PASS: all Vietnamese values NFC, one logical line, no manual bidi controls')
fixture=scratch/'catalogue-fixture'
for locale in ('en','vi'):
    (fixture/locale).mkdir(parents=True,exist_ok=True)
shutil.copyfile(en,fixture/'en'/en.name); shutil.copyfile(context,fixture/'en'/context.name); shutil.copyfile(target,fixture/'vi'/en.name)
catalogues._CATALOGUES=fixture
checker=runpy.run_path(str(repo/'scripts/i18n/check.py'),run_name='scoped_catalogue_validator')
problems=checker['check_catalogues']()
assert not problems,problems
log('PASS: repo scripts/i18n/check.py::check_catalogues() on isolated en/vi namespace: 0 problems')
def args(nodes):
    return [n for n in nodes if isinstance(n,runtime._Arg)]
select_count=0
for key in source:
    a=runtime.parse_message(source[key]); b=runtime.parse_message(vi[key])
    assert runtime.message_arguments(source[key])==runtime.message_arguments(vi[key]),key
    aa=args(a); bb=args(b)
    for x,y in zip(aa,bb):
        assert (x.name,x.kind,x.skeleton)==(y.name,y.kind,y.skeleton)
        if x.kind=='select':
            assert [p[0] for p in x.parts]==[p[0] for p in y.parts],key
            select_count+=1
        if y.kind=='plural':
            assert [p[0] for p in y.parts]==['other'],key
    assert re.findall(r'`[^`]*`|%[sd]|https?://\S+',source[key])==re.findall(r'`[^`]*`|%[sd]|https?://\S+',vi[key])
log(f'PASS: ICU argument names/kinds, {select_count} select argument/arm sequences, code/URL/printf tokens; vi plural other only')
command_cases=0
for suffix in ('signal_stop','unseen_signal_stop','forced_kill','unseen_forced_kill'):
    msg=vi[ns+'.'+suffix]
    none=runtime.render_message(msg,{'command':'none'},'vi')
    assert 'none' not in none
    for command in ('/stop --all','/kill --all','/stop abc-123'):
        rendered=runtime.render_message(msg,{'command':command},'vi')
        assert rendered.endswith(' qua '+command) and rendered.count(command)==1
        command_cases+=1
log(f'PASS: isolated ICU formatting: {command_cases} exact command-token cases + 4 sentinel none cases')
for count in (0,1,2,5,21,100):
    result=runtime.render_message(vi[ns+'.dormant'],{'count':count},'vi')
    assert str(count) in result and '{' not in result
log('PASS: isolated ICU count formatting for 0,1,2,5,21,100; no scheduling/cancellation assertion inferred')
cause_cases=0
for suffix in ('error_cause','unseen_error_cause'):
    msg=vi[ns+'.'+suffix]
    for arm,body in args(runtime.parse_message(msg))[0].parts:
        result=runtime.render_message(msg,{'cause':arm},'vi')
        assert result.startswith('Lỗi') and '{' not in result
        if suffix.startswith('unseen'): assert 'chưa đọc' in result
        cause_cases+=1
log(f'PASS: isolated ICU formats all {cause_cases} cause branches; unseen error outputs include chưa đọc')
# Counterfactuals prove the scoped catalogue and structural checks are discriminating.
bad=dict(vi); bad.pop(ns+'.recent'); (fixture/'vi'/en.name).write_text(json.dumps(bad,ensure_ascii=False))
assert any('missing key' in p for p in checker['check_catalogues']())
shutil.copyfile(target,fixture/'vi'/en.name)
assert not checker['check_catalogues']()
log('PASS: missing-key negative control detected; restored candidate returns 0 catalogue problems')
log('LIMIT: mechanical checks only; NOT an independent audit, integrated runtime, pseudo, visual or native proof')
log('GAP: unknown external diagnostics retain legacy English outside namespace; unresolved programme gap, not an accepted exception')
logpath=scratch/'vi-mechanical.log'; logpath.write_text('\n'.join(lines)+'\n',encoding='utf-8')
receipt={
 'locale':'vi','namespace':ns,'status':'translated','key_count':17,'new_keys_translated':16,'existing_keys_preserved':1,
 'date':datetime.now(timezone.utc).isoformat(), 'frozen_commit':'2930492e05a6968a2e1adfe34a091b94802bacd0',
 'translator':{'agent':'i18n-translator','task':'terminal-status-vi','session':'56c3029571f6','model':'openai/gpt-6.1-sol','actual_model_source':'session host Model header','fallback_used':False},
 'input_hashes':{name:{'path':str(p),'sha256':sha(p)} for name,p in [('en',en),('context',context),('existing_target',old),('style_guide',repo/'i18n/style-guides/vi.md'),('glossary',repo/'i18n/glossary.md')]},
 'artifacts':{name:{'path':str(p),'sha256':sha(p)} for name,p in [('target',target),('mechanical_log',logpath),('validation_script',Path(__file__))]},
 'mechanical_validation':{'repository_validator':'scripts/i18n/check.py::check_catalogues()','scope':'scratch fixture containing only frozen en, context, candidate vi namespace','catalogue_problems':0,'execution_log':str(logpath)},
 'audits':{'status':'not_audited','required_independent_safety_passes':2,'completed':0},
 'decisions':{'unseen':'chưa đọc is read status, separated from outcome by em dash','retirement':'đã ngừng để cập nhật explicitly names update and matches frozen retired_label','signal_vs_kill':'dừng bằng tín hiệu versus bị buộc kết thúc','runtime':'kept English per vi style guide','dormant':'lượt thức dậy tạm ngưng describes dormant wakes without cancellation or future firing','bounds':'ngưỡng đình trệ versus ngưỡng tiến triển keeps distinct source semantics without inventing a timeout duration'},
 'glossary_proposals':{'forced kill':'buộc kết thúc','dormant wake':'lượt thức dậy tạm ngưng','stall bound':'ngưỡng đình trệ','progress bound':'ngưỡng tiến triển','disposed session':'phiên bị giải phóng'},
 'limitations':['Two independent safety audits pending; translator has not self-audited','No integrated runtime, pseudo-localization, rendered visual or native proof','Unknown external diagnostic legacy-English path remains an unresolved programme gap','Scratch-only handoff; coordinator must retain final authoritative evidence in tracked canonical evidence root; no repository/history/generated writes performed']}
(scratch/'vi-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Receipt:',scratch/'vi-receipt.json')
print('Candidate SHA-256:',sha(target))
print('Log SHA-256:',sha(logpath))
