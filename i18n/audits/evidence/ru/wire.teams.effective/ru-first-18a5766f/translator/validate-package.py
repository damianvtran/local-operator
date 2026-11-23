"""Mechanical packaging only: not an independent language audit or rendered proof."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT = Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
OUT = Path(__file__).resolve().parent
NS = 'wire.teams.effective'
for key in list(os.environ):
    if key.startswith('CMUX_'):
        del os.environ[key]
sys.path.insert(0, str(ROOT))
from scripts.i18n import check

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
en = ROOT / f'local_operator/i18n/catalogues/en/{NS}.json'
context = ROOT / f'local_operator/i18n/catalogues/en/{NS}.context.json'
target = OUT / f'{NS}.json'
assert sha(en) == 'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501'
assert sha(context) == '01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417'
source = json.loads(en.read_text())
translated = json.loads(target.read_text())
assert len(source) == 9
assert list(source) == list(translated), 'key/order parity'
for key in source:
    assert re.findall(r'\{[a-z_]+\}', source[key]) == re.findall(r'\{[a-z_]+\}', translated[key]), key
    technical_tokens = ['subagents.models.', 'provider/model']
    if key.endswith('.tier_invalid_mapping'):
        assert 'Set model to' in source[key] and 'effort to' in source[key]
        for identifier in ('model', 'effort'):
            assert len(re.findall(r'(?<![A-Za-z_/])' + identifier + r'(?![A-Za-z_/])', translated[key])) == 1, (key, identifier)
    for token in technical_tokens:
        # Only context-declared syntax is invariant; English prose 'model' translates.
        assert len(re.findall(r'(?<![A-Za-z_])' + re.escape(token) + r'(?![A-Za-z_])', source[key])) == len(re.findall(r'(?<![A-Za-z_])' + re.escape(token) + r'(?![A-Za-z_])', translated[key])), (key, token)
    assert not any(c in translated[key] for c in '\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069'), key
fixture = OUT / 'catalogues'
for locale in ('en', 'ru'):
    (fixture / locale).mkdir(parents=True, exist_ok=True)
(fixture / 'en' / en.name).write_bytes(en.read_bytes())
(fixture / 'en' / context.name).write_bytes(context.read_bytes())
(fixture / 'ru' / target.name).write_bytes(target.read_bytes())
check.catalogues.catalogue_root = lambda: fixture
output = io.StringIO()
with contextlib.redirect_stdout(output):
    result = check.main(['--catalogues-only'])
log = output.getvalue() + f'validator exit code: {result}\n9/9 keys; source order, placeholder parity, technical token counts, no bidi controls: PASS\n'
(OUT / 'validation.log').write_text(log, encoding='utf-8')
print(log, end='')
assert result == 0
assert sha(target) == sha(fixture / 'ru' / target.name)
def file_record(path):
    return {'path': str(path), 'sha256': sha(path)}
receipt = {
    'locale': 'ru', 'namespace': NS, 'key_count': len(translated),
    'status': 'translated', 'audited': False,
    'created_at': datetime.now(timezone.utc).isoformat(),
    'worktree': str(ROOT), 'revision': 'e70bc818b8553dca1fad007705257817be49ee37',
    'translator': {'agent': 'i18n-translator', 'task': 'effective-reasons-ru', 'session': '9588c120c894', 'model': 'openai/gpt-6.1-sol', 'actual_model_source': 'session-state model declaration', 'fallback_used': False},
    'input_hashes': {'en': file_record(en), 'context': file_record(context)},
    'target': file_record(target),
    'guidance': [file_record(ROOT / p) for p in ['docs/I18N.md', 'docs/design/i18n.md', 'i18n/glossary.md', 'i18n/style-guides/ru.md']],
    'artifacts': [file_record(OUT / 'validation.log'), file_record(Path(__file__)), file_record(fixture / 'en' / en.name), file_record(fixture / 'en' / context.name), file_record(fixture / 'ru' / target.name)],
    'mechanical_validation': {'command': 'PYTHONDONTWRITEBYTECODE=1 python3 ' + str(Path(__file__)), 'repo_entrypoint': 'scripts.i18n.check.main(["--catalogues-only"])', 'scope': 'isolated scratch catalogue root with only frozen en source, context and Russian target; all validator problems retained', 'exit_code': result, 'stdout': output.getvalue(), 'key_order_parity': True, 'placeholder_parity': True, 'technical_token_parity': True, 'target_package_byte_parity': True, 'bidi_controls_absent': True},
    'language_choices': {
        'register': 'Impersonal diagnostic statements and neutral plural/polite imperatives; no padding. Sentence case and Russian guillemets.',
        'agreement': 'Team name {role} follows с именем and is quoted without inflection. Literal tier path follows в; selector follows модель without changing its opaque value. Masculine участник governs его.',
        'glossary': 'team → команда; agent → агент; model → модель; member → участник (Russian wave-2 style guide). Technical subagents.models., provider/model, model, effort and placeholders are unchanged.',
        'ICU_CLDR': 'Only simple ICU placeholders occur. No plural or select branches are present or added. Russian one/few/many/other would apply to future cardinal messages.',
        'bidi': 'Russian LTR, no manual bidi controls. Technical Latin tokens remain in logical order.',
        'fit': 'All context notes allow natural wrapping and specify no fixed cell limit; no truncation or width exception.',
    },
    'ambiguities': [
        {'key': 'tier_invalid_mapping', 'choice': 'model mapping → объект настроек; effort → уровень рассуждений', 'reason': 'Context explicitly defines mapping as a configuration object, with required model and optional effort. Russian keeps that distinction rather than implying a model alias table.'},
        {'key': 'model_unresolvable', 'choice': 'could not be resolved → не удалось определить', 'reason': 'Avoids suggesting the model was launched or exposing exception details; remediation preserves checks of model identifier and provider configuration.'}
    ],
    'glossary_proposals': {'model selector': 'идентификатор модели', 'model mapping': 'объект с настройками модели / объект настроек (in the same immediate context)', 'reasoning effort': 'уровень рассуждений', 'agent registry': 'реестр агентов', 'agent definition': 'определение агента', 'provider': 'провайдер (consistent with existing ru wire.settings.account_policy)'},
    'limitations': ['Not independently audited; independent i18n-auditor still mandatory.', 'No rendered proof, live UI execution or full repository validator/generation claimed.', 'No repository, generated catalogue, ledger or audit-history writes; no build, install, server or PR.']
}
receipt_path = OUT / 'translator-receipt.json'
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for path in (target, receipt_path, OUT / 'validation.log'):
    print(sha(path), str(path))
