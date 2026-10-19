# Numeric graphic repair evidence

The source is the **unmodified** stored component in session `13c37091ac12`, digest
`a395a5c68b1c72d18274bc5ca3d77a55`. Its three price cells are `$0.15`, `$0.60`, `$0.015`.
The harness assembler produced standalone documents served over loopback; the harness
browser tool loaded and photographed them, without a separate browser engine.

- [Before](before.png): labels, but no bars or ticks. The regression test executes the
  shipped prelude and fails with `float('NaN') >= 0` false.
- [After](after.png): three bars and finite ticks, with values 0.15, 0.6 and 0.015.
- [Gap](gap.png): a synthetic copy replacing only `$0.60` with `N/A` has two bars,
  intact ticks and an `N/A` label beside Output. The source attachment was never edited.

Screenshots are standalone fallback-theme documents, not screenshots of the desktop
transcript. No stylesheet, frame size or label-color change is part of this fix.

## Validation order

`split_component()` and `data_json()` remain lossless: validation also calls both.
Only `assemble_document()` normalises a rendered copy of dataset rows. To support the
prompt's plain numbers **and** legacy currency-formatted components, the validator's
numeric reader uses the same conservative normaliser on component and evidence cells.
The 2×3 provenance matrix covers evidence `$0.15` / `0.15` and component `$0.15` /
`0.15` / numeric `0.15`; every case also rejects an invented `0.16`.

Only explicit decimals, valid thousands grouping, currency prefixes (`$`, `€`, `£`, `¥`)
and a percent suffix are recognised. Currency and percent cannot be combined. A percent
is not rescaled. Leading-zero identifiers, magnitude suffixes, guesses, malformed
commas, unsafe integers, and decimals whose meaning would round/underflow stay strings.
The optional rejection of all-gap charts is deliberately not added: stored documents
remain renderable, with zero-scale ticks and labelled bar gaps rather than fabricated data.

## Verification

All pytest commands used a fresh scratch HOME/config, stripped inherited environment,
`TERM=xterm-256color`, `--no-cov -n 0`, and an explicit scratch `--basetemp`.

- Targeted document/prelude/validator/fixture files: **99 passed in 3.42s**.
- Supplement directory: **690 passed, 2 failed** before the final gap-alignment rebuild;
  the frozen rebuild run was **689 passed, 3 failed** (the same two sweep failures plus a
  stale digest pin). After correcting only that pin, the 99-test affected set passed.
- The two sweep failures are unchanged permission tests. A baseline test-file control
  (only its relative conftest import made absolute) reproduced **2 failed, 14 deselected
  in 0.60s**. `sweep.py` is byte-identical to base `06cdb3300e`, SHA256
  `3d638aaf7e22fa1d2c435a41b295975dfa343ed073479679f160faa756097291`.
  This is a same-source control, not a claim of a clean whole-tree baseline run.
- Scoped flake8, Black 26.1.0 and isort 5.13.2 checks passed. Bounded scoped pyright:
  **0 errors, 0 warnings, 0 informations**, rc 0, wall 2.7s.
- `check-changed --run` widened all gates because JS/HTML are outside the import graph.
  It caught an overlong test line (fixed), then was stopped before whole-tree tests to
  obey the explicit no-full-local-suite budget. CI owns whole-tree verification.
- Prelude build: esbuild **0.28.2**. Pair **10,226 B raw / 4,602 B gzip -9**, deltas
  **+123 / +29 B**, caps **11,264 / 4,608 B**. No trim or cap increase was needed.

Actual implementation model: `openai/gpt-6.1-sol`, a quota fallback from
`anthropic/claude-sonnet-5-5`. Independent review/QA/design and CI remain merge gates.

## Coupled UI regeneration (not performed here)

The UI checkout's current HEAD predates these files; they exist on its `origin/main`.
The manager must use an up-to-date **UI worktree**, not edit its shared checkout.
Refresh exactly:

- `src/renderer/src/features/chat/canonical/supplement-document-fixtures.ts`
  (populated/error constants and provenance header, now PRELUDE_VERSION 4).
- `src/renderer/src/features/chat/canonical/supplement-closeout.generated.html`.

From this core worktree at the reviewed head, with `UI_ROOT` set to the manager's UI
worktree, the following command regenerates both using the real assembler. It preserves
the closeout fixture's data/body, replacing only its assembled envelope/prelude:

```sh
UI_ROOT=/path/to/ui-worktree .venv/bin/python - <<'PY'
import json, os, re, subprocess
from pathlib import Path
from local_operator.supplements.document import assemble_document, assemble_stored_component

ui = Path(os.environ['UI_ROOT']) / 'src/renderer/src/features/chat/canonical'
fixture = ui / 'supplement-document-fixtures.ts'
text = fixture.read_text()
for name in ('populated', 'error'):
    blob = Path(f'tests/fixtures/supplements/components/{name}.html').read_text()
    html = assemble_stored_component(blob)
    constant = f'SUPPLEMENT_STORY_{name.upper()}_DOCUMENT'
    pattern = r'(export const ' + constant + r'\s*=\s*)(?:\'(?:\\.|[^\'\\])*\'|"(?:\\.|[^"\\])*");'
    text, count = re.subn(pattern, lambda m: m[1] + json.dumps(html, ensure_ascii=False) + ';', text)
    assert count == 1, constant
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
text, count = re.subn(r' \* PROVENANCE:.*?(?= \* `populated`)',
    f' * PROVENANCE: core {head}, PRELUDE_VERSION 4.\n'
    ' * Regenerate from core tests/fixtures/supplements/components/{populated,error}.html\n'
    ' * using supplements.document.assemble_stored_component; see core numeric repair evidence.\n *\n',
    text, flags=re.S)
assert count == 1
fixture.write_text(text)
closeout = ui / 'supplement-closeout.generated.html'
old = closeout.read_text()
data = json.loads(old.split('id="lo-data">', 1)[1].split('</script>', 1)[0])
body = old.split('</script>', 2)[2].removesuffix('\n</body></html>').strip('\n')
closeout.write_text(assemble_document(body, data))
PY
```

No UI files were modified by this PR. Run the UI's own formatter/tests and capture its
rendered story states in the coupled PR; the standalone frames above do not replace that.
