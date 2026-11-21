import os, sys, json, hashlib, shutil, runpy, re
from pathlib import Path
repo = Path('/Users/damian/local-operator/.worktrees/i18n-terminal-status')
scratch = Path(os.environ['LOCAL_OPERATOR_SCRATCHPAD'])
sys.path.insert(0, str(repo))
from local_operator.i18n import catalogues, runtime
ns = 'wire.session_status'
source = repo / 'local_operator/i18n/catalogues/en' / (ns + '.json')
context = source.with_name(ns + '.context.json')
previous = repo / 'local_operator/i18n/catalogues/fr' / (ns + '.json')
target = scratch / (ns + '.fr.json')
en = json.loads(source.read_text())
fr = json.loads(target.read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest() == '52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326'
assert hashlib.sha256(context.read_bytes()).hexdigest() == '5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525'
assert list(en) == list(fr) and len(fr) == 17
assert fr[ns + '.retired_label'] == json.loads(previous.read_text())[ns + '.retired_label']
print('PASS frozen source/context SHA-256; 17 keys in source order; historical retired_label value unchanged')
def structures(nodes, kind):
    out = []
    for node in nodes:
        if isinstance(node, runtime._Arg):
            if node.kind == kind:
                out.append((node.name, tuple(key for key, _ in node.parts)))
            for _, body in node.parts:
                out.extend(structures(body, kind))
    return out
for key in en:
    assert runtime.message_arguments(en[key]) == runtime.message_arguments(fr[key]), key
    assert structures(runtime.parse_message(en[key]), 'select') == structures(runtime.parse_message(fr[key]), 'select'), key
    assert re.findall(r'`[^`]*`|https?://\S+|%[sd]', en[key]) == re.findall(r'`[^`]*`|https?://\S+|%[sd]', fr[key]), key
    assert '\n' not in fr[key]
    assert not any(c in fr[key] for c in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
print('PASS all 17 ICU syntax/ordered argument signatures; select identifiers/order; literal token parity; single-line/no manual bidi controls')
plurals = structures(runtime.parse_message(fr[ns + '.dormant']), 'plural')
assert plurals == [('count', ('one', 'many', 'other'))]
print('PASS French count plural arms: one/many/other; # retained in all three arms')
for key in ['signal_stop','unseen_signal_stop','forced_kill','unseen_forced_kill']:
    ast = runtime.parse_message(fr[ns + '.' + key])
    arg = next(node for node in ast if isinstance(node, runtime._Arg))
    assert arg.parts[0] == ('none', ())
    assert any(isinstance(n, runtime._Arg) and n.name == 'command' and n.kind == 'simple' for n in arg.parts[1][1])
print('PASS command sentinel none is empty; other interpolates original command unchanged, with localized connector')
# Run the repository catalogue validator itself on this isolated namespace fixture.
fixture = scratch / 'catalogue-fixture'
for locale in ['en','fr']:
    (fixture / locale).mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, fixture / 'en' / source.name)
shutil.copyfile(context, fixture / 'en' / context.name)
shutil.copyfile(target, fixture / 'fr' / source.name)
catalogues._CATALOGUES = fixture
checker = runpy.run_path(str(repo / 'scripts/i18n/check.py'), run_name='fr_scoped_validator')
exit_code = checker['main'](['--catalogues-only'])
assert exit_code == 0
print('PASS repository scripts/i18n/check.py --catalogues-only; isolated scratch en/fr namespace fixture; exit 0')

def hashed(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
receipt = {
    'status': 'translated', 'audited': False,
    'locale': 'fr', 'namespace': ns, 'key_count': len(fr),
    'new_values_translated': 16, 'historical_values_preserved': 1,
    'frozen_commit': '2930492e05a6968a2e1adfe34a091b94802bacd0',
    'translator': {'agent':'i18n-translator / terminal-status-fr', 'session':'a14cc322d1da','actual_model':'openai/gpt-6.1-sol','actual_model_source':'session system model declaration','fallback_used':False},
    'input_hashes': {'en':hashed(source),'context':hashed(context),'previous_fr':hashed(previous),'style_guide':hashed(repo/'i18n/style-guides/fr.md'),'glossary':hashed(repo/'i18n/glossary.md'),'howto':hashed(repo/'docs/I18N.md'),'rfc':hashed(repo/'docs/design/i18n.md')},
    'artifacts': {'target':hashed(target),'check_script':hashed(Path(__file__))},
    'validation': {'command':'PYTHONDONTWRITEBYTECODE=1 /Users/damian/local-operator/.venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/check_fr.py"','repo_catalogue_validator':'scripts/i18n/check.py --catalogues-only via main() with scratch-only catalogue root','exit_code':exit_code,'scope':'17-key en/fr namespace fixture only; mechanical structural checks, not semantic audit or app/runtime/visual evidence'},
    'decisions': [
      'Unseen = non lue (not yet read), never invisible or failed; completion = Fin de session non lue.',
      'Signal stop = arrêtée par un signal; killed = arrêt forcé. Unknown command does not imply user attribution.',
      'Retired = retirée pour une mise à jour, not operator stop. Historical retired_label kept byte-identical as a VALUE, including its masculine retiré.',
      'runtime_stall_bound explicitly self-termination; runtime_overdue update-related departure after no-progress deadline. incidents.py _stall_bound_cause_sentence/_overdue_cause_sentence confirm deadline semantics.',
      'runtime_update_failed states loaded version retained, unlike update-related departure.',
      'disposed rendered session fermée, not supprimée: disposal does not assert historical data deletion.',
      'dormant = réveils en sommeil, not cancelled or scheduled to fire. one/many/other, with de after million multiples.'
    ],
    'glossary_proposals': {'unseen':'non lu / non lue','forced kill':'arrêt forcé','dormant wake':'réveil en sommeil','turn':'tour','stall/progress bound':'délai maximal sans progression'},
    'length': 'No fixed cap in source sidecar; all values one logical line; no truncation.',
    'limitations': ['Not independently audited. Two independent safety-copy audits pending under manager ownership.','Unknown nonempty external reason remains the legacy English path outside this namespace: UNRESOLVED programme gap. Stored prose not translated.','No application runtime or rendered/visual proof; no complete-status-copy localization claim.','No repo/generated/ledger writes, build, install, test suite, commit or PR. Scratch artifacts are handoff candidates, not tracked audit records.']
}
(scratch / 'translator-receipt.fr.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
print('Receipt written: ' + str(scratch / 'translator-receipt.fr.json'))
print('Target SHA-256: ' + hashed(target)['sha256'])
