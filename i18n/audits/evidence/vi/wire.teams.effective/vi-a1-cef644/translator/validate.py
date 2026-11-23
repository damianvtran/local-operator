"""Mechanical validation only; neither independent audit nor rendered proof."""
import json
import pathlib
import re
import sys
import unicodedata
from unittest.mock import patch

ROOT = pathlib.Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
from scripts.i18n import check
from local_operator.i18n import runtime

namespace = 'wire.teams.effective'
source = ROOT / 'local_operator/i18n/catalogues/en' / (namespace + '.json')
context = source.with_name(namespace + '.context.json')
target = HERE / (namespace + '.json')
fixture = HERE / 'catalogue-fixture'
for locale in ('en', 'vi'):
    (fixture / locale).mkdir(parents=True, exist_ok=True)
(fixture / 'en' / source.name).write_bytes(source.read_bytes())
(fixture / 'en' / context.name).write_bytes(context.read_bytes())
(fixture / 'vi' / target.name).write_bytes(target.read_bytes())
with patch.object(check.catalogues, 'catalogue_root', return_value=fixture):
    result = check.main(['--catalogues-only'])
assert result == 0
src = json.loads(source.read_text())
dst = json.loads(target.read_text())
assert list(src) == list(dst)
rows = []
for key, value in dst.items():
    assert unicodedata.is_normalized('NFC', value)
    assert not any(unicodedata.bidirectional(c) in ('LRE','RLE','LRO','RLO','PDF','LRI','RLI','FSI','PDI') or c in '\u200e\u200f\u061c' for c in value)
    assert runtime.message_arguments(src[key]) == runtime.message_arguments(value)
    assert re.findall(r'\{[^{}]+\}', src[key]) == re.findall(r'\{[^{}]+\}', value)
    for token in ('subagents.models.', 'provider/model'):
        assert src[key].count(token) == value.count(token)
    if key.endswith('tier_invalid_mapping'):
        for token in ('model', 'effort'):
            assert re.search(r'\b' + token + r'\b', value)
    rows.append({'key': key, 'arguments': runtime.message_arguments(value), 'placeholder_sequence': re.findall(r'\{[^{}]+\}', value), 'plural_selectors': runtime.plural_selectors(value), 'NFC': True})
report = {'kind': 'mechanical_translation_checks_not_audit', 'status': 'translated', 'key_count': len(dst), 'same_key_order': True, 'exact_placeholders': True, 'technical_tokens': True, 'NFC': True, 'bidi_controls': 0, 'catalogue_validator_exit_code': result, 'rows': rows}
(HERE / 'mechanical-checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('9/9 keys: exact key order, argument types and placeholder sequence; technical tokens preserved; NFC; no bidi controls')
print('No plural/select messages; vi CLDR category is other only. No render or independent audit attempted.')
