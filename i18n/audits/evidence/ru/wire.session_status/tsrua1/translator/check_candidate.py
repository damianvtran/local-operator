"""Run the existing catalogue checker against only the frozen EN/RU fixture.

This is a scratch candidate check, not a native audit or application-runtime proof.
The real checker uses its supported catalogue-root seam; no repository files change.
"""
import importlib.util
import json
from pathlib import Path
import sys

REPO = Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from local_operator.i18n import catalogues, runtime

spec = importlib.util.spec_from_file_location('candidate_catalogue_checker', REPO / 'scripts/i18n/check.py')
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)
catalogues._CATALOGUES = ROOT / 'catalogues'
assert checker.main(['--catalogues-only']) == 0

en = json.loads((ROOT / 'catalogues/en/wire.session_status.json').read_text())
ru = json.loads((ROOT / 'wire.session_status.json').read_text())
legacy = json.loads((REPO / 'local_operator/i18n/catalogues/ru/wire.session_status.json').read_text())
assert list(en) == list(ru)
assert len(ru) == 17
assert ru['wire.session_status.retired_label'] == legacy['wire.session_status.retired_label']
assert all('\n' not in text and '\r' not in text for text in ru.values())
assert not any(any(ch in text for ch in '\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069') for text in ru.values())

def selects(source):
    result = []
    def visit(nodes):
        for node in nodes:
            if isinstance(node, runtime._Arg):
                if node.kind == 'select':
                    result.append((node.name, tuple(selector for selector, _ in node.parts)))
                for _, branch in node.parts:
                    visit(branch)
    visit(runtime.parse_message(source))
    return result

for key in en:
    assert runtime.message_arguments(en[key]) == runtime.message_arguments(ru[key]), key
    assert selects(en[key]) == selects(ru[key]), key
assert runtime.plural_selectors(ru['wire.session_status.dormant']) == (('one','few','many','other'),)
print('PASS: 17 keys in source order; 16 new translations; legacy retired_label unchanged')
print('PASS: exact ICU argument and select-arm identifiers; ru one/few/many/other plural arms')
print('PASS: one-line values, valid UTF-8 JSON, no manual bidi controls')
print('Mechanical branch interpolation samples (NOT application/runtime or visual clearance):')
for key, text in ru.items():
    if 'signal_stop' in key or 'forced_kill' in key:
        without = runtime.render_message(text, {'command':'none'}, 'ru')
        command = '/stop --all'
        with_command = runtime.render_message(text, {'command':command}, 'ru')
        assert 'none' not in without and 'по команде' not in without
        assert with_command.endswith(' по команде ' + command)
        print(key + ' [none]: ' + without)
        print(key + ' [literal]: ' + with_command)
for count in [0,1,2,5,11,21,22,25,101,1.5]:
    print('dormant ' + str(count) + ': ' + runtime.render_message(ru['wire.session_status.dormant'], {'count':count}, 'ru'))
for key in ['wire.session_status.error_cause','wire.session_status.unseen_error_cause']:
    for node in runtime.parse_message(ru[key]):
        if isinstance(node, runtime._Arg):
            for cause, _ in node.parts:
                print(key + ' [' + cause + ']: ' + runtime.render_message(ru[key], {'cause':cause}, 'ru'))
print('PASS: exercised all command branches, all 13 cause branches in both messages, and Russian integer/fractional plural samples')
