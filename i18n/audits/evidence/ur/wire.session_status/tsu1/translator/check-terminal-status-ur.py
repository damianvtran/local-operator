"""Scoped mechanical validation only; not an independent linguistic audit.
Uses the frozen repository parser and catalogue-validator helpers without
writing a repository catalogue, generating assets, or exercising runtime UI.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

repo = Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
scratch = Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0, str(repo))
from local_operator.i18n import runtime
from scripts.i18n.check import arguments_match, _check_plural_categories

namespace = 'wire.session_status'
paths = {
    'en': repo / 'local_operator/i18n/catalogues/en/wire.session_status.json',
    'context': repo / 'local_operator/i18n/catalogues/en/wire.session_status.context.json',
    'frozen_target': repo / 'local_operator/i18n/catalogues/ur/wire.session_status.json',
    'target': scratch / 'terminal-status-ur.json',
    'style': repo / 'i18n/style-guides/ur.md',
    'glossary': repo / 'i18n/glossary.md',
}
def digest(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f'duplicate key: {key}'
        result[key] = value
    return result

en, context, old, target = [json.loads(paths[k].read_text(), object_pairs_hook=unique_object) for k in ('en','context','frozen_target','target')]
assert digest(paths['en'])['sha256'] == '52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert digest(paths['context'])['sha256'] == '5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == '2930492e05a6968a2e1adfe34a091b94802bacd0'
assert len(target) == 17 and list(target) == list(en) == list(context)
assert target['wire.session_status.retired_label'] == old['wire.session_status.retired_label']
plural = json.loads((repo / 'local_operator/i18n/data/plural_rules.json').read_text())
categories = plural['locales']['ur']['categories']
assert set(categories) == {'one','other'}

def structure(nodes):
    result = []
    for node in nodes:
        if isinstance(node, runtime._Arg):
            result.append(('argument', node.name, node.kind, node.skeleton,
                           tuple((code, structure(children)) for code, children in node.parts)))
        elif isinstance(node, runtime._Hash):
            result.append(('hash',))
    return tuple(result)

problems = []
for key, value in target.items():
    assert isinstance(value, str) and '\n' not in value
    assert arguments_match(runtime.message_arguments(en[key]), runtime.message_arguments(value), translated_source=value, locale='ur', plural=plural), key
    assert structure(runtime.parse_message(en[key])) == structure(runtime.parse_message(value)), key
    _check_plural_categories(problems, 'ur', namespace, key, value, plural, 'ur')
    for selectors in runtime.plural_selectors(value):
        assert set(selectors) == set(categories), (key, selectors)
    assert not any(ord(c) in set(range(0x202A,0x202F)) | set(range(0x2066,0x206A)) | {0x200E,0x200F,0x061C} for c in value), key
    for token_re in (r'`[^`]*`', r'%[sdif]', r'https?://\S+', r'</?b>'):
        assert re.findall(token_re,en[key]) == re.findall(token_re,value), (key, token_re)
    assert en[key] != value, key
assert not problems, problems

results = {
    'scope': 'scratch catalogue mechanical validation only',
    'status': 'passed',
    'key_count': len(target),
    'new_keys_translated': len(target)-len(old),
    'preserved_existing_keys': list(old),
    'ur_cldr_categories_from_frozen_repo': categories,
    'checks': ['frozen HEAD/source/context hashes','JSON duplicate-key detection','17-key exact parity and source order','frozen retired_label exact value','repository ICU syntax and arguments_match','full recursive select-code/placeholder/hash-token structure equality','repository _check_plural_categories plus complete one/other arms','no manual bidi controls','single logical line','backtick/printf/URL/bold token equality','no unchanged English values'],
    'problems': problems,
    'not_claimed': ['independent linguistic/safety audit','integrated runtime','pseudo-locale','visual','native shaping','render-layer isolation','full-repository i18n check'],
}
(scratch / 'terminal-status-ur-checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
provenance = {
    'locale': 'ur', 'namespace': namespace, 'status': 'translated',
    'translator': {'agent':'i18n-translator','session':'7b97856c53df','model':'openai/gpt-6.1-sol','actual_model_source':'host session-state model declaration','fallback_used':False},
    'worktree':str(repo), 'frozen_head':'2930492e05a6968a2e1adfe34a091b94802bacd0',
    'input_hashes':{k:digest(v) for k,v in paths.items() if k != 'target'},
    'output_hashes':{'target':digest(paths['target']), 'checks':digest(scratch/'terminal-status-ur-checks.json'), 'checker':digest(Path(__file__))},
    'counts':{'total_keys':17,'new_keys_translated':16,'frozen_values_preserved':1},
    'audit_state':'NOT AUDITED; two independent safety audits required',
    'reasoning': [
      'Unseen is rendered as کیفیت ابھی نہیں پڑھی گئی (status not yet read), explicitly separating read-state from cause and avoiding an unseen=unknown outcome reading.',
      'Signal stop uses سگنل سے روکا گیا; forced kill uses زبردستی بند کیا گیا. command select none stays empty and other attributes the literal parameter as کمانڈ: {command}.',
      'Retirement explicitly names the runtime leaving for an update; frozen retired_label remains untouched even though the new whole-message phrasing differs.',
      'Dormant uses غیر فعال, not cancelled or scheduled; بیداری/بیداریاں follows glossary and one/other.',
      'Stall-bound self-termination and progress-bound update departure remain distinct; update failure explicitly retains the loaded build; runtime_killed explicitly says no stop was requested.',
      'The context provides no fixed cell/character caps; no truncation or width exception is claimed.'
    ],
    'glossary_proposals':{'runtime':'رن ٹائم','signal':'سگنل','forced kill':'زبردستی بند کرنا','dormant':'غیر فعال','turn':'باری','loaded build':'لوڈ شدہ بلڈ','unseen status':'کیفیت ابھی نہیں پڑھی گئی'},
    'programme_gap':'Unknown nonempty external diagnostic retains legacy English outside this namespace: unresolved programme gap, not accepted exception.',
    'restrictions':'Scratch outputs only; no repository/generated/history writes, commits, builds, installs, servers, PR operations, translation services or self-audit.'
}
(scratch/'terminal-status-ur-provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
print('TARGET_SHA256='+digest(paths['target'])['sha256'])
print('PROVENANCE='+str(scratch/'terminal-status-ur-provenance.json'))
