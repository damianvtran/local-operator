"""Validate only this scratch translation through the checkout's real catalogue checker."""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

repo = Path('/Users/damian/local-operator/.worktrees/i18n-effective-reasons')
out = Path(__file__).parent
sys.path.insert(0, str(repo))
from scripts.i18n import check
from local_operator.i18n import catalogues

ns = 'wire.teams.effective'
en = repo / f'local_operator/i18n/catalogues/en/{ns}.json'
context = en.with_name(f'{ns}.context.json')
target = out / f'{ns}.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(en) == 'fa6fe7ce2807a00786dc71449a586ef4e2336ce11dbd199122f794dd82c0a501'
assert sha(context) == '01dbb15b434691a77d3417860fbf3e5cc91f243046906d4d77d7c9402b0fd417'
source = json.loads(en.read_text())
translated = json.loads(target.read_text())
assert list(source) == list(translated) and len(source) == 9
checks = []
controls = set('\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
for key, text in translated.items():
    source_tokens = re.findall(r'\{[^{}]+\}', source[key])
    target_tokens = re.findall(r'\{[^{}]+\}', text)
    assert source_tokens == target_tokens, key
    assert not any(c in controls for c in text), key
    literals = ['subagents.models.', 'provider/model']
    if key.endswith('.tier_invalid_mapping'):
        literals += ['model', 'effort']
    for literal in literals:
        # Isolated Latin configuration terms only; model inside provider/model is not a separate key.
        pattern = r'(?<![A-Za-z0-9_/])' + re.escape(literal) + r'(?![A-Za-z0-9_/])'
        expected_count = 1 if literal in ('model', 'effort') else len(re.findall(pattern, source[key]))
        assert expected_count == len(re.findall(pattern, text)), (key, literal)
    checks.append({'key': key, 'placeholders': target_tokens, 'placeholder_parity': True, 'bidi_controls': 0})
root = out / 'catalogue-fixture'
(root / 'en').mkdir(parents=True, exist_ok=True)
(root / 'ur').mkdir(parents=True, exist_ok=True)
shutil.copyfile(en, root / 'en' / en.name)
shutil.copyfile(context, root / 'en' / context.name)
shutil.copyfile(target, root / 'ur' / target.name)
catalogues._CATALOGUES = root
exit_code = check.main(['--catalogues-only'])
assert exit_code == 0
print('Scratch-only catalogue fixture: en + context + ur; no repository catalogue writes')
print('PASS: 9 ordered keys; placeholder and technical-token parity; no bidi controls; frozen input hashes match')
result = {'key_count': 9, 'key_order_matches': True, 'checks': checks, 'input_hashes': {'en': {'path': str(en), 'sha256': sha(en)}, 'context': {'path': str(context), 'sha256': sha(context)}, 'ur': {'path': str(target), 'sha256': sha(target)}}, 'validator': {'entrypoint': 'scripts.i18n.check.main(["--catalogues-only"])', 'catalogue_root': str(root), 'exit_code': exit_code, 'scope': 'scratch fixture, namespace only'}, 'independent_audit': 'not performed', 'pseudo_bidi_visual_runtime': 'not performed'}
(out / 'key-placeholder-hash-checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
