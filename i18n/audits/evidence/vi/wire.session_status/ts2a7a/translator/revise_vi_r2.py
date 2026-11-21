import os,sys,json,hashlib,runpy,unicodedata,difflib,shutil
from pathlib import Path
from datetime import datetime,timezone
repo=Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
scratch=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD']); sys.path.insert(0,str(repo))
from local_operator.i18n import runtime,catalogues
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=scratch/'wire.session_status.vi.json'
assert sha(original)=='963b8bf337981a328bb80b265cb49d033db10c7afb2838c4ec8d5f05df20c1d1'
old='runtime rời đi để cập nhật sau ngưỡng tiến triển'
new='runtime đã dừng để cập nhật sau khi hết thời hạn chờ tiến triển'
raw=original.read_bytes(); assert raw.count(old.encode())==2
candidate=scratch/'wire.session_status.vi.r2.json'; assert not candidate.exists()
candidate.write_bytes(raw.replace(old.encode(),new.encode()))
before=json.loads(raw); after=json.loads(candidate.read_bytes())
ns='wire.session_status'; keys=[ns+'.error_cause',ns+'.unseen_error_cause']
en=repo/f'local_operator/i18n/catalogues/en/{ns}.json'; context=en.with_name(ns+'.context.json')
frozen=repo/f'local_operator/i18n/catalogues/vi/{ns}.json'
assert sha(en)=='52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(context)=='5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert list(before)==list(after)==list(json.loads(en.read_bytes())) and len(after)==17
assert [k for k in before if before[k]!=after[k]]==keys
assert candidate.read_bytes().replace(new.encode(),old.encode())==raw
assert after[ns+'.retired_label']==json.loads(frozen.read_bytes())[ns+'.retired_label']
lines=[]
def log(s): lines.append(s);print(s)
log('PASS: original SHA-256 unchanged; frozen en/context verified; exact 17-key order; exactly 2 byte replacements and reversible byte-identical remainder; retired_label preserved')
assert all(unicodedata.normalize('NFC',s)==s and '\n' not in s for s in after.values())
log('PASS: NFC and single logical line for all 17 values')
def arg(nodes):return [n for n in nodes if isinstance(n,runtime._Arg)]
for key in keys:
    a=arg(runtime.parse_message(before[key]))[0]; b=arg(runtime.parse_message(after[key]))[0]
    assert (a.name,a.kind,a.skeleton)==(b.name,b.kind,b.skeleton)
    assert [x[0] for x in a.parts]==[x[0] for x in b.parts]
    for (arm,x),(arm2,y) in zip(a.parts,b.parts):
        if arm=='runtime_overdue':
            assert runtime.render_message(after[key],{'cause':arm},'vi')==('Lỗi — chưa đọc — ' if 'unseen_' in key else 'Lỗi — ')+new
        else:assert x==y
    assert runtime.message_arguments(before[key])==runtime.message_arguments(after[key])
log('PASS: both runtime_overdue bodies formatted with new copy; all other 24 cause bodies, selector identity/order and ICU arguments unchanged')
fixture=scratch/'catalogue-fixture-r2'
for loc in ('en','vi'):(fixture/loc).mkdir(parents=True,exist_ok=True)
shutil.copyfile(en,fixture/'en'/en.name);shutil.copyfile(context,fixture/'en'/context.name);shutil.copyfile(candidate,fixture/'vi'/en.name)
catalogues._CATALOGUES=fixture
check=runpy.run_path(str(repo/'scripts/i18n/check.py'),run_name='scoped_revision_check')['check_catalogues']
assert check()==[]
log('PASS: repository check.py::check_catalogues isolated en/vi namespace: 0 problems')
# Negative controls stay in scratch and never mutate candidate or frozen inputs.
bad=dict(after);bad.pop(ns+'.recent');(fixture/'vi'/en.name).write_text(json.dumps(bad,ensure_ascii=False))
assert any('missing key' in p for p in check())
log('PASS: missing-key negative control rejected')
bad=dict(after);bad[keys[0]]=bad[keys[0]].replace('{cause, select,','{wrong_cause, select,',1)
(fixture/'vi'/en.name).write_text(json.dumps(bad,ensure_ascii=False)); assert any('arguments' in p for p in check())
log('PASS: renamed-placeholder negative control rejected')
shutil.copyfile(candidate,fixture/'vi'/en.name);assert check()==[]
log('PASS: restored candidate: 0 problems')
log('LIMIT: translated revision only, NOT self-audited; both previous audit results inapplicable to new target SHA pending two independent delta audits; no runtime/visual/pseudo/native clearance')
log('GAP: external diagnostic legacy English remains unresolved, not an accepted exception')
logpath=scratch/'vi-r2-mechanical.log';logpath.write_text('\n'.join(lines)+'\n')
diffpath=scratch/'vi-r2.diff';diffpath.write_text(''.join(difflib.unified_diff(original.read_text().splitlines(True),candidate.read_text().splitlines(True),fromfile=original.name,tofile=candidate.name)))
report=Path('/Users/damian/.local-operator/sessions/5f0a2918d2f2/scratchpad/second-audits/vi/report.json')
secondreceipt=report.with_name('receipt.json')
inputs={'en':en,'context':context,'original_target':original,'frozen_existing_target':frozen,'second_audit_report':report,'second_audit_receipt':secondreceipt,'incidents_source':repo/'local_operator/incidents.py','update_cause_types':repo/'local_operator/session/runtime/types.py','update_cause_process':repo/'local_operator/session/runtime/process.py'}
receipt={'locale':'vi','namespace':ns,'status':'translated','revision':2,'finding_addressed':'VI-S2-01','key_count':17,'changed_keys':keys,'changed_select_arm':'runtime_overdue','date':datetime.now(timezone.utc).isoformat(),'frozen_commit':'2930492e05a6968a2e1adfe34a091b94802bacd0','translator':{'agent':'i18n-translator','session':'56c3029571f6','model':'openai/gpt-6.1-sol','actual_model_source':'host session Model header','fallback_used':False},'previous_translation_receipt':{'path':str(scratch/'vi-receipt.json'),'sha256':sha(scratch/'vi-receipt.json')},'input_hashes':{k:{'path':str(p),'sha256':sha(p)} for k,p in inputs.items()},'artifacts':{k:{'path':str(p),'sha256':sha(p)} for k,p in {'target':candidate,'mechanical_log':logpath,'two_key_diff':diffpath,'revision_script':Path(__file__)}.items()},'meaning_argument':{'new_body':new,'evidence':['local_operator/incidents.py:469-475: historical cause says newer build departure after BUILD_DRAIN_PROGRESS_S of no movement reported from in-flight work','local_operator/session/runtime/types.py:1164-1184: bounded handover was involuntary, distinct from ordinary retirement and deliberate user stop','local_operator/session/runtime/process.py:4098-4145: historical force-cut replaced; current runtime retains loaded build until turn completes'],'reasoning':'đã dừng removes anthropomorphic rời đi; hết thời hạn chờ tiến triển names expiry of a progress-wait window rather than reaching a quantity of progress. Update purpose explicit, no user stop, successful new-build load, numeric duration or general stall-watchdog diagnosis invented. Source shows progress-bound historical cause is its own update mechanism, not runtime_stall_bound.','glossary_proposal':{'progress bound (update waiting context)':'thời hạn chờ tiến triển'}},'mechanical_validation':{'repository_validator':'scripts/i18n/check.py::check_catalogues()','scope':'scratch en/vi namespace only','problems':0,'negative_controls':['missing key','renamed cause placeholder']},'audit_state':{'status':'not_audited','required_independent_delta_audits':2,'completed_for_new_hash':0,'second_original_result':'changes_requested','previous_audit_results_do_not_approve_new_target':True},'limitations':['No repo/generated/history writes, build, install, full suite or server','No integrated runtime, visual, pseudo or native proof','Unknown external diagnostic legacy-English path remains unresolved programme gap','Original candidate and all original receipts retained unchanged; canonical evidence retention delegated to coordinator']}
receiptpath=scratch/'vi-r2-receipt.json';assert not receiptpath.exists();receiptpath.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
assert sha(original)==receipt['input_hashes']['original_target']['sha256']
for a in json.loads(receiptpath.read_text())['artifacts'].values(): assert sha(Path(a['path']))==a['sha256']
print('Candidate SHA-256:',sha(candidate));print('Receipt SHA-256:',sha(receiptpath));print('Log SHA-256:',sha(logpath));print('Diff:',diffpath)
