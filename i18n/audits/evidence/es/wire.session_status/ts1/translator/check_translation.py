"""Mechanical translation checks only; not an independent semantic audit.

Uses the repository's real catalogue validator on an isolated, namespace-only
fixture so concurrent/frozen catalogues remain untouched. No generated writes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

repo = Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
out = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))
from local_operator.i18n import catalogues, runtime
from scripts.i18n import check, ledger

ns = 'wire.session_status'
source = repo / f'local_operator/i18n/catalogues/en/{ns}.json'
context = source.with_name(f'{ns}.context.json')
previous = repo / f'local_operator/i18n/catalogues/es/{ns}.json'
target = out / f'{ns}.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source) == '52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert sha(context) == '5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
en, es, old = (json.loads(p.read_text()) for p in (source, target, previous))
assert list(es) == list(en) and len(es) == 17
assert es[f'{ns}.retired_label'] == old[f'{ns}.retired_label']
print('PASS: frozen source/context SHA-256, 17-key parity/order, unchanged legacy retired_label')
fixture = out / 'catalogue-fixture'
for locale in ('en', 'es'):
    (fixture / locale).mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, fixture / 'en' / source.name)
shutil.copyfile(context, fixture / 'en' / context.name)
shutil.copyfile(target, fixture / 'es' / target.name)
catalogues._CATALOGUES = fixture
problems = check.check_catalogues()
assert not problems, problems
print('PASS: repository scripts.i18n.check.check_catalogues(): 0 problems (isolated en/es namespace fixture)')

def selects(nodes):
    result = []
    for node in nodes:
        if isinstance(node, runtime._Arg):
            if node.kind == 'select':
                result.append((node.name, tuple(label for label, _ in node.parts)))
            for _, body in node.parts:
                result.extend(selects(body))
    return result

for key in en:
    assert runtime.message_arguments(en[key]) == runtime.message_arguments(es[key]), key
    assert selects(runtime.parse_message(en[key])) == selects(runtime.parse_message(es[key])), key
    assert '\n' not in es[key] and '\r' not in es[key], key
    assert not any(c in es[key] for c in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069'), key
assert runtime.plural_selectors(es[f'{ns}.dormant']) == (('one', 'many', 'other'),)
print('PASS: argument name/kind parity; all cause/command select-arm identifiers and order; Spanish one/many/other; one logical line; no bidi controls')

samples = []
for name in ('signal_stop', 'unseen_signal_stop', 'forced_kill', 'unseen_forced_kill'):
    for command in ('none', '/stop --all', 'lop sessions stop --session qa-synthetic'):
        result = runtime.render_message(es[f'{ns}.{name}'], {'command': command}, 'es')
        if command == 'none':
            assert 'none' not in result and 'mediante' not in result
        else:
            assert result.endswith(' mediante ' + command)
        samples.append({'key': name, 'params': {'command': command}, 'text': result})
for count in (0, 1, 2, 1000000, 1500000):
    samples.append({'key': 'dormant', 'params': {'count': count}, 'text': runtime.render_message(es[f'{ns}.dormant'], {'count': count}, 'es')})
for name in ('error_cause', 'unseen_error_cause'):
    arms = selects(runtime.parse_message(es[f'{ns}.{name}']))[0][1]
    for cause in arms:
        result = runtime.render_message(es[f'{ns}.{name}'], {'cause': cause}, 'es')
        assert result
        samples.append({'key': name, 'params': {'cause': cause}, 'text': result})
(out / 'format-samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2) + '\n')
print(f'PASS: {len(samples)} ICU formatting samples; absent command attribution omitted; technical command bytes preserved')

identity = {'agent': 'i18n-translator (terminal-status-es)', 'session': 'b39246e0d986', 'model': 'openai/gpt-6.1-sol', 'actual_model_source': 'session-state Model field', 'fallback': 'No fallback reported in this session', 'date': '2026-11-19'}
inputs = {'en': source, 'context': context, 'previous_es': previous, 'style_guide': repo / 'i18n/style-guides/es.md', 'glossary': repo / 'i18n/glossary.md', 'HOWTO': repo / 'docs/I18N.md', 'RFC': repo / 'docs/design/i18n.md', 'validator': repo / 'scripts/i18n/check.py', 'runtime': repo / 'local_operator/i18n/runtime.py'}
receipt = {'locale': 'es', 'namespace': ns, 'source_sha256': sha(source), 'context_sha256': sha(context), 'target_sha256': sha(target), 'translator': identity, 'auditor': {'agent': None, 'model': None, 'result': 'not audited; two independent safety-copy audits required'}, 'status': 'translated', 'note': '17 keys: 16 newly translated, legacy retired_label unchanged. Translator b39246e0d986; two independent safety-copy audits pending. Mechanical checks only; no visual/runtime clearance. Unknown external diagnostic English path remains an unresolved programme gap.'}
assert len(receipt['note']) <= 280
(out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
current = ledger.history_sha256(ledger._group(ledger.load_entries(repo))[('es', ns)])
path = ledger.append(repo, receipt, expected_history_sha=current, apply=False)
print(f'PASS: current append-compatible receipt dry-run: {path}; expected_history_sha256={current}; no history written')
record = {'kind': 'translation_provenance_not_audit', 'status': 'translated', 'locale': 'es', 'namespace': ns, 'date': '2026-11-19', 'key_count': 17, 'newly_translated_key_count': 16, 'preserved_key_count': 1, 'source_commit': '2930492e05a6968a2e1adfe34a091b94802bacd0', 'translator': identity, 'audited': False, 'independent_safety_copy_audits_required': 2, 'input_hashes': {label: {'path': str(p), 'sha256': sha(p)} for label, p in inputs.items()}, 'target': {'path': str(target), 'sha256': sha(target)}, 'expected_history_sha256': current, 'append_dry_run_destination': path, 'validation_scope': 'Mechanical catalogue/ICU checks only. Namespace fixture; not the full repository check. No rendered UI or integrated application/runtime clearance.', 'unknown_external_diagnostic': 'Legacy English branch intentionally unchanged; unresolved programme gap, not an allowed exception.', 'executed_at': datetime.now(timezone.utc).isoformat()}
(out / 'provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print('STATUS: translated; NOT independently audited; no repository/generated/history writes')
