import os, sys, json, hashlib, re, contextlib, io
from pathlib import Path
from datetime import datetime, timezone
repo=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
scratch=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0,str(repo))
sys.dont_write_bytecode=True
from local_operator.i18n import runtime, catalogues
from scripts.i18n import check
ns='wire.session_status'
en_path=repo/f'local_operator/i18n/catalogues/en/{ns}.json'
context_path=repo/f'local_operator/i18n/catalogues/en/{ns}.context.json'
old_path=repo/f'local_operator/i18n/catalogues/zh-CN/{ns}.json'
target=scratch/f'{ns}.zh-CN.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(en_path)=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(context_path)=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
en=json.loads(en_path.read_text()); zh=json.loads(target.read_text()); old=json.loads(old_path.read_text())
assert len(zh)==17 and list(zh)==list(en)
assert zh[ns+'.retired_label']==old[ns+'.retired_label']
logs=['Pinned source/context SHA-256: PASS','JSON/key count/order: PASS (17 keys)','Historical retired_label value exact preservation: PASS']
plural=json.loads((repo/'local_operator/i18n/data/plural_rules.json').read_text())
for k in en:
    assert runtime.message_arguments(en[k])==runtime.message_arguments(zh[k]), k
    for selectors in runtime.plural_selectors(zh[k]): assert selectors==('other',), (k,selectors)
    assert '\n' not in zh[k]
    assert not any(c in zh[k] for c in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
    for pattern in (r'`[^`]*`',r'%[sdif]',r'</?[A-Za-z][^>]*>'):
        assert re.findall(pattern,en[k])==re.findall(pattern,zh[k]),k
    def selects(nodes):
        result=[]
        for n in nodes:
            if isinstance(n,runtime._Arg):
                if n.kind=='select': result.append((n.name,tuple(a for a,_ in n.parts)))
                for _, body in n.parts: result.extend(selects(body))
        return result
    assert selects(runtime.parse_message(en[k]))==selects(runtime.parse_message(zh[k])),k
logs.extend(['ICU parser and placeholder name/kind/order parity: PASS (17/17)','CLDR zh-CN: PASS (dormant other only)','All stable select arms and cause identifiers: PASS','Markup/backtick/printf-token parity; single-line and no bidi controls: PASS'])
# Run the repo catalogue checker on ONLY an isolated source/context/candidate fixture.
fixture=scratch/'zh-status-validation-catalogues'
for locale in ('en','zh-CN'): (fixture/locale).mkdir(parents=True,exist_ok=True)
(fixture/'en'/f'{ns}.json').write_bytes(en_path.read_bytes())
(fixture/'en'/f'{ns}.context.json').write_bytes(context_path.read_bytes())
(fixture/'zh-CN'/f'{ns}.json').write_bytes(target.read_bytes())
catalogues._CATALOGUES=fixture
out=io.StringIO()
with contextlib.redirect_stdout(out): code=check.main(['--catalogues-only'])
logs.append('Repository checker: scripts.i18n.check.main(["--catalogues-only"]), isolated candidate fixture')
logs.append(out.getvalue().rstrip()); assert code==0
samples=[]
for k,s in zh.items():
    if k.endswith('retired_label'): params_list=[{'base':'已中断'}]
    elif k.endswith('error_cause'):
        arms=runtime.parse_message(s)[0].parts
        params_list=[{'cause':a} for a,_ in arms]+[{'cause':'unknown_external_code'}]
    elif 'command' in s: params_list=[{'command':c} for c in ('none','/stop --all','lop sessions stop synthetic-id')]
    elif 'count' in s: params_list=[{'count':c} for c in (0,1,2,5,100)]
    else: params_list=[{}]
    for p in params_list:
        text=runtime.render_message(s,p,'zh-CN')
        if 'command' in p:
            if p['command']=='none': assert 'none' not in text and '通过' not in text
            else: assert p['command'] in text
        if 'cause' in p:
            assert p['cause'] not in text
            if p['cause']=='unknown_external_code': assert text==('未读：出错' if 'unseen' in k else '出错')
        samples.append(dict(message_id=k, params=p, text=text))
logs.append(f'Library-only ICU expansion: PASS ({len(samples)} cases, including all cause arms, unknown select fallback, command none/exact bytes, counts 0/1/2/5/100)')
logs.append('No UI/runtime/visual validation or independent language audit claimed; no repository writes')
(scratch/'zh-status-samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
(scratch/'zh-status-checks.log').write_text('\n'.join(logs)+'\n')
receipt={'locale':'zh-CN','namespace':ns,'status':'translated','key_count':17,'new_keys_translated':16,'preserved_existing_keys':['wire.session_status.retired_label'],'date':datetime.now(timezone.utc).isoformat(),'translator':{'agent':'i18n-translator','session':'e9c45b5505eb','model':'openai/gpt-6.1-sol','actual_model_source':'host session context Model: openai/gpt-6.1-sol','fallback_used':False},'frozen_head':'2930492e05a6968a2e1adfe34a091b94802bacd0','input_hashes':[{'path':str(p.relative_to(repo)),'sha256':sha(p)} for p in (en_path,context_path,old_path,repo/'i18n/style-guides/zh-CN.md',repo/'i18n/glossary.md')],'artifacts':[{'path':str(p),'sha256':sha(p)} for p in (target,scratch/'zh-status-checks.log',scratch/'zh-status-samples.json',Path(__file__))],'validation':{'checker':'scripts.i18n.check.main(["--catalogues-only"])','scope':'isolated scratch catalogue fixture containing only en source/context and zh-CN candidate','exit_code':code,'actual_stdout':out.getvalue(),'icu_expansion_cases':len(samples),'independent_audit':'NOT PERFORMED; two independent safety-copy audits follow','runtime_visual_proof':False},'choices':{'unseen':'未读： preserves read/unread state, not a new event','retired':'为更新而退出 explicitly attributes exit to update, not user action','signal_stop':'收到信号后停止, distinct from 强制终止; no invented signal escalation','runtime_killed':'未正常退出便消失；未收到停止请求 preserves absence of request without inventing cause','dormant':'唤醒已休眠; not cancelled or scheduled to fire','recent':'近期活动 clarifies recency of activity'},'glossary_proposals':{'runtime':'运行时','continuation':'续接','stall bound':'停滞时限（本次措辞：停滞超时）','progress bound':'进度时限','dormant wake':'已休眠的唤醒','disposed':'已释放'},'ambiguous_strings':['Retained retired_label uses historical 退役 unchanged by explicit instruction; new whole messages use clearer update attribution.','runtime_overdue uses 超过进度时限后退出以更新; progress-bound meaning follows English source/context without diagnosing an update failure.'],'known_programme_gap':'Unknown nonempty external diagnostics still retain legacy English outside this namespace. Stored prose untouched; no all-status-copy-complete claim.','retention':'Scratch handoff only per task; manager must retain canonical tracked evidence if adopted.'}
(scratch/'zh-status-translator-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('\n'.join(logs)); print('Target SHA-256:',sha(target)); print('Receipt:',scratch/'zh-status-translator-receipt.json')
