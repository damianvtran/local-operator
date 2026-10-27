# Russian Notifications help — delta audit nt25a2

Verdict: **passed / clean**, one key re-read and audited. Prior finding **R5 fixed** by the original coder. No new findings. This does not erase the failed nt25a1 evidence or its provenance.

Auditor: translation-auditor, session/instance `ffd16b7ff158`, actual current model `openai/gpt-6-astra` from the resumed session-state header. Same independent auditing session; no authorship of translations, catalogue edits or ledger/git writes. Prior model remains as recorded in nt25a1; this round uses the current model, not a claimed unchanged model.

## Delta and semantic verdict

Key: `wire.slash.notifications_page.help`

Before: Вопросы и новости агентов; read — всё прочитано

After: Запросы, обновления; read помечает всё прочитанным

The after text matches the prior proposal exactly. «Запросы» and «обновления» now name the same concepts as the UI; «помечает всё прочитанным» explicitly describes the marking operation rather than a resulting state. The compressed enumeration is idiomatic command-palette/help copy. `read` remains a literal Latin command token. The measured Rich terminal cell width is **50/50**; there is no width margin and no rendered layout claim.

## Compatibility retained, not re-audited

Actual SHA-256 comparisons show English/context for this key unchanged; all three English/context/Russian inputs for `wire.errors.notifications` unchanged; and all three English/context/Russian UI `ui.notifications` inputs unchanged. The prior UI verdict (3 minor findings) and core-errors verdict (1 minor finding) remain clean-with-findings, not newly clean. The exact hashes appear in audit.log. No full UI or unchanged-namespace semantic audit repeated.

Current HEAD: `329b5c581784df1082dbd0630c0f814c7a3c0c8f`; repaired target is an uncommitted working-tree change. The input hashes, not HEAD alone, bind this result.

## Runtime evidence

From the core worktree, executed:

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/ru-a2-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/ru-a2-home/config" PATH="$PATH" PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TERM=xterm-256color .venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/audit-ru-round2.py"
```

Exit status: 0. Imported runtime path verified as this worktree's `local_operator/i18n/runtime.py`. Both English and Russian parsed and rendered verbatim. Four in-memory negative controls rejected malformed ICU, an absent runtime argument, a translated `read` token and an over-width label (55 cells). No audited file was mutated. Isolation uses a scratch HOME/config and env -i to strip inherited CMUX/XPC variables.

`audit.log` contains the actual output from this run. Reproduction script follows:

```python
import json, hashlib, re
from pathlib import Path
from rich.cells import cell_len
from local_operator.i18n import runtime
CORE=Path('/Users/damian/workspace/repos/lo-notifications-l1')
UI=Path('/Users/damian/local-operator-ui/.worktrees/notifications-ui')
NS='wire.slash.notifications_page';KEY=NS+'.help'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert Path(runtime.__file__).resolve()==CORE/'local_operator/i18n/runtime.py'
print('CORE_RUNTIME',runtime.__file__)
for ns in [NS,'wire.errors.notifications']:
 rec=json.loads((CORE/f'i18n/audits/evidence/ru/{ns}/nt25a1/record.json').read_text())
 for role,v in rec['input_hashes'].items():
  digest=sha(CORE/v['path']);same=digest==v['sha256']
  print('INPUT',ns,role,'UNCHANGED' if same else 'CHANGED',digest)
  if ns!=NS or role!='target':assert same
r=json.loads((UI/'i18n/receipts/nt-ru-ffd16b7ff158.json').read_text())
for role,p,h in [('en','en/ui.notifications.json',r['en_sha256']),('context','en/ui.notifications.context.json',r['context_sha256']),('target','ru/ui.notifications.json',r['target_sha256'])]:
 digest=sha(UI/'src/i18n/catalogues'/p);assert digest==h;print('RETAIN_UI',role,'UNCHANGED',digest)
base=CORE/'local_operator/i18n/catalogues'
en=json.loads((base/'en'/f'{NS}.json').read_text());ru=json.loads((base/'ru'/f'{NS}.json').read_text());ctx=json.loads((base/'en'/f'{NS}.context.json').read_text())
assert list(en)==list(ru)==[KEY]
for locale,cat in [('en',en),('ru',ru)]:
 s=cat[KEY];ast=runtime.parse_message(s);rendered=runtime.render_message(s,{},locale)
 assert rendered==s
 print('PARSE_RENDER',locale,repr(ast),json.dumps(rendered,ensure_ascii=False),'cells='+str(cell_len(rendered)))
s=ru[KEY];assert re.search(r'(?<!\w)read(?!\w)',s);assert cell_len(s)<=ctx[KEY]['maxCells']
print('LITERAL_READ PASS; MAXCELLS',cell_len(s),'<=',ctx[KEY]['maxCells'])
# Negative controls stay in memory: no mutation of audited catalogue bytes.
try:runtime.parse_message(s+' {broken')
except runtime.MessageSyntaxError as e:print('NEGATIVE malformed ICU REJECTED',type(e).__name__,str(e))
else:raise AssertionError('parser accepted malformed ICU')
try:runtime.render_message(s+' {missing}',{},'ru')
except runtime.MessageFormatError as e:print('NEGATIVE missing runtime argument REJECTED',type(e).__name__,str(e))
else:raise AssertionError('renderer accepted absent argument')
assert not re.search(r'(?<!\w)read(?!\w)',s.replace('read','прочитать'))
print('NEGATIVE translated command token REJECTED')
assert cell_len(s+' ABCD')>ctx[KEY]['maxCells']
print('NEGATIVE width overrun REJECTED',cell_len(s+' ABCD'))
print('RESULT PASS: one-key delta; parser/render, literal token, maxCells, four negative controls; source/context and previous clean scopes byte-identical')
```

## Limits

LLM textual audit, not native-speaker certification. No browser, pixel-fit, TUI frame or release/activation validation. Only the one repaired help key is newly approved.
