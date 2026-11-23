import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

repo = Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
scratch = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))
from scripts.i18n import check
from local_operator.i18n import runtime

ns = 'wire.teams.effective'
source = repo / f'local_operator/i18n/catalogues/en/{ns}.json'
context = source.with_name(ns + '.context.json')
target = scratch / f'catalogues/hi/{ns}.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source) == 'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501'
assert sha(context) == '01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417'
(scratch / 'catalogues/en').mkdir(exist_ok=True)
for path in (source, context):
    shutil.copyfile(path, scratch / 'catalogues/en' / path.name)
en = json.loads(source.read_text())
hi = json.loads(target.read_text())
assert list(en) == list(hi) and len(hi) == 9
for key in en:
    assert collections.Counter(re.findall(r'\{[^{}]+\}', en[key])) == collections.Counter(re.findall(r'\{[^{}]+\}', hi[key]))
    assert runtime.message_arguments(en[key]) == runtime.message_arguments(hi[key])
    assert not re.search(r',\s*(plural|select|selectordinal)\b', hi[key])
    for token in ('subagents.models.', 'provider/model'):
        assert en[key].count(token) == hi[key].count(token), (key, token)
    if key.endswith('.tier_invalid_mapping'):
        for token in ('model', 'effort'):
            # English also uses these as prose nouns; only the sidecar-designated
            # config keys must stay literal in Hindi, not each English occurrence.
            pattern = r'(?<![\w/])' + token + r'(?![\w/])'
            assert len(re.findall(pattern, hi[key])) == 1, (key, token)
    assert not re.search('[\u200e\u200f\u202a-\u202e\u2066-\u2069]', hi[key])

with patch.object(check.catalogues, 'catalogue_root', return_value=scratch / 'catalogues'):
    problems = check.check_catalogues()
    assert problems == [], problems
    # In-memory counterfactual: a missing parameter must fail the same checker.
    original_loader = check.catalogues.load_catalogue
    def bad_loader(locale, namespace):
        values = original_loader(locale, namespace)
        if locale == 'hi':
            values = dict(values)
            key = ns + '.team_missing'
            values[key] = values[key].replace('{role}', '')
        return values
    with patch.object(check.catalogues, 'load_catalogue', side_effect=bad_loader):
        negative = check.check_catalogues()
    assert negative and any('team_missing' in p for p in negative), negative

hashes = {
    'en': {'path': str(source.relative_to(repo)), 'sha256': sha(source)},
    'context': {'path': str(context.relative_to(repo)), 'sha256': sha(context)},
    'target': {'path': str(target), 'sha256': sha(target)},
}
receipt = {
    'locale': 'hi', 'namespace': ns, 'source_sha256': sha(source),
    'target_sha256': sha(target), 'context_sha256': sha(context),
    'translator': {
        'agent': 'i18n-translator', 'session': '47e37aebb87c',
        'model': 'openai/gpt-6.1-sol', 'date': '2026-11-19',
        'actual_model_source': 'Host session-state Model: openai/gpt-6.1-sol',
        'fallback': 'none observed',
    },
    'auditor': {'agent': None, 'model': None, 'result': 'not_audited'},
    'status': 'translated', 'keys_translated': 9, 'input_hashes': hashes,
    'source_commit': 'e70bc818b8553dca1fad007705257817be49ee37',
    'note': 'Scratch Hindi candidate; translated, not audited. Mechanical checks only. Two independent audits required for these high-visibility remedies; no pseudo/bidi/runtime/rendered proof or shipping approval.',
    'glossary_proposals': {
        'registry': 'रजिस्ट्री', 'agent definition': 'एजेंट की परिभाषा',
        'model mapping': 'मॉडल मैपिंग',
        'model selector': 'मॉडल चुनने के लिए दिया गया मान',
        'reasoning effort': 'तर्क-प्रयास (existing wire.settings usage)',
        'provider': 'प्रोवाइडर (existing wire.settings usage)',
    },
    'translation_decisions': [
        'No fixed width budget in the sidecar; wrap naturally, no truncation.',
        'Model selector uses an explanatory Hindi phrase rather than an English noun; provider/model stays literal.',
        'Mapping remains the developer loan मैपिंग, denoting a config object; model required, effort optional.',
        'Unresolvable uses पहचाना नहीं जा सका without asserting a missing provider or corrupt file.',
    ],
    'evidence_limits': {'independent_audits': 0, 'pseudo': 'not_run', 'bidi_runtime': 'not_run', 'rendered': 'not_run'},
}
(scratch / 'translator-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
report = {
    'keys': 9, 'key_order': 'PASS', 'ICU_argument_and_placeholder_parity': 'PASS',
    'technical_tokens': 'PASS', 'no_plural_select_or_manual_bidi_controls': 'PASS',
    'repo_validator': 'scripts.i18n.check.check_catalogues() on scratch en/hi fixture',
    'validator_problems': problems,
    'negative_control_missing_role': negative,
    'source_target_context_hashes': hashes,
    'scope': 'Mechanical only; not a linguistic self-audit or rendered/runtime proof',
}
(scratch / 'mechanical-checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))

env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
cli = [sys.executable, str(repo / 'scripts/i18n/ledger.py')]
get_sha = subprocess.run(cli + ['history-sha', '--locale', 'hi', '--namespace', ns], text=True, capture_output=True, env=env)
assert get_sha.returncode == 0, get_sha.stderr
expected = get_sha.stdout.strip()
args = cli + ['append', '--receipt', str(scratch / 'translator-receipt.json'), '--expected-history-sha', expected, '--dry-run']
result = subprocess.run(args, text=True, capture_output=True, env=env)
log = {'command': args, 'expected_history_sha': expected, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'apply': False}
(scratch / 'ledger-dry-run.json').write_text(json.dumps(log, indent=2) + '\n')
print('LEDGER_DRY_RUN', json.dumps(log))
assert result.returncode == 0
