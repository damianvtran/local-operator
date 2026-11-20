"""Scoped mechanical translation checks only; not a semantic audit or UI proof."""
import os
import sys
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from local_operator.i18n import runtime
from scripts.i18n.check import arguments_match, _check_plural_categories

NS = 'wire.session_status'
inputs = {
    'en': ROOT / f'local_operator/i18n/catalogues/en/{NS}.json',
    'context': ROOT / f'local_operator/i18n/catalogues/en/{NS}.context.json',
    'existing_target': ROOT / f'local_operator/i18n/catalogues/hi/{NS}.json',
    'style': ROOT / 'i18n/style-guides/hi.md',
    'glossary': ROOT / 'i18n/glossary.md',
    'target': OUT / f'{NS}.json',
}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
hashes = {key: {'path': str(path), 'sha256': digest(path)} for key, path in inputs.items()}
assert hashes['en']['sha256'] == '52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert hashes['context']['sha256'] == '5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
en = json.loads(inputs['en'].read_text())
target = json.loads(inputs['target'].read_text())
existing = json.loads(inputs['existing_target'].read_text())
context = json.loads(inputs['context'].read_text())
plural = json.loads((ROOT / 'local_operator/i18n/data/plural_rules.json').read_text())
assert list(target) == list(en) == list(context)
assert len(target) == 17
assert target[f'{NS}.retired_label'] == existing[f'{NS}.retired_label']

def selectors(nodes):
    found = []
    for node in nodes:
        if isinstance(node, runtime._Arg):
            if node.parts:
                found.append((node.name, node.kind, tuple(k for k, _ in node.parts)))
            for _, children in node.parts:
                found.extend(selectors(children))
    return found

def tokens(text):
    return re.findall(r'`[^`]*`|%[sdif]|</?[^>]+>', text)

results = []
for key, text in target.items():
    assert isinstance(text, str) and '\n' not in text
    assert not any('\u202a' <= c <= '\u202e' or '\u2066' <= c <= '\u2069' for c in text)
    expected_args = runtime.message_arguments(en[key])
    actual_args = runtime.message_arguments(text)
    assert arguments_match(expected_args, actual_args, translated_source=text, locale='hi', plural=plural), key
    assert selectors(runtime.parse_message(en[key])) == selectors(runtime.parse_message(text)), key
    problems = []
    _check_plural_categories(problems, 'hi', NS, key, text, plural, 'hi')
    assert not problems, problems
    assert tokens(en[key]) == tokens(text), key
    results.append({'key': key, 'icu_parse': 'PASS', 'argument_parity': 'PASS', 'select_arm_names_order': 'PASS', 'plural_categories': 'PASS', 'literal_tokens': 'PASS', 'single_line_no_bidi_controls': 'PASS'})

# These are formatter-only counterexamples, not integrated runtime/visual proof.
command_samples = []
opaque = '/stop --all'
for suffix in ['signal_stop', 'unseen_signal_stop', 'forced_kill', 'unseen_forced_kill']:
    for command in ['none', opaque]:
        value = runtime.render_message(target[f'{NS}.{suffix}'], {'command': command}, 'hi')
        assert value.count(opaque) == (command == opaque)
        assert 'none' not in value
        command_samples.append({'key': suffix, 'command': command, 'output': value})
plural_samples = []
for count in [0, 1, 2, 5, 21]:
    value = runtime.render_message(target[f'{NS}.dormant'], {'count': count}, 'hi')
    assert str(count) in value
    assert ('जागृतियाँ' in value) == (count not in [0, 1])
    plural_samples.append({'count': count, 'output': value})
negative_controls = {}
try:
    runtime.parse_message('{count, plural, one {bad}')
except runtime.MessageSyntaxError:
    negative_controls['malformed_icu_rejected'] = 'PASS'
else:
    raise AssertionError('syntax negative control accepted')
assert selectors(runtime.parse_message(en[f'{NS}.signal_stop'])) != selectors(runtime.parse_message(target[f'{NS}.signal_stop'].replace('none {', 'missing {')))
negative_controls['renamed_machine_arm_detected'] = 'PASS'
assert not arguments_match(runtime.message_arguments(en[f'{NS}.retired_label']), runtime.message_arguments('{renamed}'))
negative_controls['renamed_argument_detected'] = 'PASS'

report = {'status': 'mechanical_checks_passed', 'scope': 'Hindi scratch candidate only; repository parser and scoped checker helpers, not whole-corpus check.py CLI', 'key_count': 17, 'new_translations': 16, 'preserved_existing': 1, 'results': results, 'command_formatter_samples': command_samples, 'plural_formatter_samples': plural_samples, 'negative_controls': negative_controls, 'input_hashes': hashes}
(OUT / 'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('PASS: 17/17 keys and key order; frozen source/context hashes; retired_label unchanged')
print('PASS: repository ICU parser, argument parity, machine arm names/order, hi one/other, literal token parity')
print('PASS: 8 opaque command formatter samples; 5 plural formatter samples; 3 negative controls')
print('NOT PERFORMED: semantic audit, integrated runtime, pseudo, visual, native-client validation, full-corpus validator')
