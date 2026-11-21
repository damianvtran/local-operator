import sys, os, json, hashlib, re, importlib.util
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
OUT=Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0,str(ROOT))
from scripts.i18n import check
from rich.cells import cell_len
NS='wire.teams.effective'
EN=ROOT/'local_operator/i18n/catalogues/en'/f'{NS}.json'
CTX=EN.with_name(f'{NS}.context.json')
TARGETS={'fr':Path('/Users/damian/.local-operator/sessions/34a8703ccbff/scratchpad/wire.teams.effective.fr.json'),'zh-CN':Path('/Users/damian/.local-operator/sessions/530f154495de/scratchpad/zh-CN.wire.teams.effective.json')}
EXPECTED={'en':'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501','context':'01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417','fr':'9c7ffd5f52a5ad10267d739074e19c56c90f532b6c79b15314e2256655038226','zh-CN':'6d8d1fc410def690b03ad02ba6c759a5d7b3e82391d9c12dc441fbff7d73f454'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 def unique(pairs):
  d={}
  for k,v in pairs:
   assert k not in d, f'duplicate JSON key {k}'
   d[k]=v
  return d
 return json.loads(p.read_text(),object_pairs_hook=unique)
source=load(EN);context=load(CTX)
NOTES={
 'team_missing':'Team-name lookup failure retained; opaque role/name placeholder unchanged; no implication of member launch.',
 'registry_unreadable':'Registry readability failure retained without inventing corruption or exception details; agent definitions and retry both retained.',
 'config_unreadable':'Model configuration readability failure and inspect-configuration/retry remedy retained.',
 'tier_invalid_value':'Invalid value and both accepted alternatives retained: provider/model selector or configuration mapping object.',
 'tier_invalid_mapping':'Required model setting and optional effort setting remain distinct; supported reasoning-effort restriction retained. fr si nécessaire conveys optionality; Chinese effort可选 is explicit and the second 请 begins a separate directive after a semicolon.',
 'tier_unconfigured':'Absent configured model remains distinct from invalid model; literal tier path retained.',
 'tier_invalid_selector':'Invalid selector and exact provider/model format retained; no raw invalid value exposed.',
 'model_unresolvable':'Model resolution failure retained, not conflated with network resolution; selector and provider-configuration remedy both retained.',
 'unavailable':'Member availability fallback retained with model-configuration and agent-definition checks; no successful launch implication.'}
identity={'agent':'i18n-auditor','session':'89b8acddcc7c','model':'openai/gpt-6.1-sol','actual_model_source':'session-state model declaration: openai/gpt-6.1-sol','fallback_used':False}
for locale,path in TARGETS.items():
 target=load(path)
 hashes={label:{'path':str(p),'sha256':sha(p)} for label,p in [('en',EN),('context',CTX),(locale,path)]}
 assert all(v['sha256']==EXPECTED[k] for k,v in hashes.items())
 assert set(source)==set(context)==set(target) and len(target)==9
 # Run the repository's real catalogue validator, bounded to the assigned namespace
 # and in-memory scratch target. No catalogue root or generated asset is modified.
 with patch.object(check.catalogues,'namespaces',return_value=[NS]),patch.object(check.catalogues,'locales',return_value=['en',locale]),patch.object(check.catalogues,'load_catalogue',side_effect=lambda loc,ns:source if loc=='en' else target):
  problems=check.check_catalogues()
 assert not problems,problems
 per=[]
 for key,text in target.items():
  english=source[key]
  assert re.findall(r'\{[^{}]+\}',english)==re.findall(r'\{[^{}]+\}',text)
  assert not re.search(r'\b(plural|select|selectordinal)\s*,',text)
  assert re.findall(r'`[^`]*`',english)==re.findall(r'`[^`]*`',text)
  assert re.findall(r'</?[^>]+>',english)==re.findall(r'</?[^>]+>',text)
  for token in ['subagents.models.','provider/model']:
   assert english.count(token)==text.count(token),(key,token)
  if key.endswith('tier_invalid_mapping'):
   for token in ['model','effort']: assert re.search(r'(?<![A-Za-z])'+token+r'(?![A-Za-z])',text)
  assert not any(ord(c) in range(0x202a,0x202f) or ord(c) in range(0x2066,0x206a) for c in text)
  per.append({'key':key,'source':english,'current_text':text,'status':'passed','mechanical':'Exact placeholder names/types and technical syntax; no plural/select, tags, code spans, flags or commands to alter. No added/dropped keys.','linguistic_review':NOTES[key.rsplit('.',1)[-1]],'idiom_register_terminology':('Natural professional French, vous via imperative; équipe/agent/modèle; curly apostrophes; no French punctuation requiring a narrow no-break space.' if locale=='fr' else 'Natural Simplified Chinese diagnostic and directive clauses; 请 present in remedies, 智能体/团队/模型 consistent; no added spaces around technical tokens; full-width prose punctuation.'),'fit':{'chars':len(text),'cells':cell_len(text),'maxChars':context[key].get('maxChars'),'maxCells':context[key].get('maxCells'),'assessment':'Wrapping desktop warning; no fixed budget in sidecar. Width measured, not a rendered fit claim.'},'findings':[]})
 report={'locale':locale,'namespace':NS,'audit_date':'2026-11-19','audit_kind':'second_independent_linguistic_mechanical_pass','key_count':9,'auditor':dict(identity,independent_from_translator_session=('34a8703ccbff' if locale=='fr' else '530f154495de'),independent_from_first_auditor_session=('32077ea6da37' if locale=='fr' else '2b66cec83cdb')),'input_hashes':hashes,'audit_sequence':'Source/context/target reviewed and this per-key verdict recorded before opening first-auditor receipts.','current_report_verdict':{'status':'passed','unresolved_blocker_major_count':0,'finding_ids':[]},'findings':[],'per_key':per,'evidence_limits':['No translation/MT service used.','Scoped repository catalogue validator plus independent structural checks only.','No plural/select messages; CLDR branch completeness not applicable.','No pseudo-localization, fallback, bidi/mixed-direction rendered review, HTTP serving, or UI integration execution.','Neither shipped status nor runtime visual fit is certified.'],'audit_notes_not_execution_log':True}
 (OUT/f'report.{locale}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 log={'locale':locale,'namespace':NS,'auditor':identity,'input_hashes':hashes,'validator':'scripts.i18n.check.check_catalogues() with namespaces/locales/load_catalogue scoped in memory to frozen source and scratch contribution; real context sidecar remains loaded from frozen tree','validator_problems':problems,'independent_checks':'Strict duplicate-free JSON; nine-key source/context/target parity; exact placeholders; literal tokens; no ICU plural/select or bidi controls; width measurements','result':'PASS','key_count':9,'first_auditor_receipt_opened':False}
 (OUT/f'mechanical.{locale}.json').write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
 print(locale, 'PASS 9/9; source/context/target hashes verified; repository scoped validator 0 problems; independent linguistic verdict committed to scratch before first-verdict consultation')
