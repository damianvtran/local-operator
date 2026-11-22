# Private source handoff: read-only composer footer

Status: **English-source-only; not ready to publish.** This is a bounded footer
patch, not agent-badge, role-metadata, native-copy or composer architecture work.
Allocation: hotfix/source patch, <=30 minutes. Base: latest cached `origin/main`
`bd7bf7520d5a87a438a04d760f556ab67199c678`; no fetch or transport was performed.
The delegated brief named an older cached base; the hardcoded footer was still
present at the actual base used here.

## Frozen translation input

- Namespace: `tui.editor_readonly`; exactly one immutable key:
  `tui.editor_readonly.hint`.
- Source: `local_operator/i18n/catalogues/en/tui.editor_readonly.json`.
- Exact ICU source: `Read-only · {key} back`.
- Exact English rendering with the production argument: `Read-only · esc back`
  (20 ASCII/Unicode terminal cells; the middle dot occupies one cell).
- Source file SHA-256:
  `a2a8bfb8a250b38e3a950bb2b02d885ec76cc726d77521086801099b6e99b5f2`.
- Context: adjacent `tui.editor_readonly.context.json`; SHA-256:
  `46c48a659223ad5e4861a1407c2c7a685183b9c7caecf6e19f53adb8131389ee`.
- Typed API: `tui_editor_readonly.hint(key="esc")`, emitted only by the normal
  generator. Runtime consumer: `editor.read_only_placeholder()`, called by
  `OperatorApp._composer_placeholder_for()` including live-language repaint.

Translate the ordinary prose **Read-only** and **back** idiomatically, retaining
its state/action meaning. `{key}` is a technical keyboard token; preserve the
placeholder and never translate its supplied value `esc`. The composer refuses
input while the read-only page owns focus; Esc returns to the chat. One short
line, within the existing 40-column split context; no role/agent badge, user
string, session name or prompt belongs in this message. No exact semantic key
existed in the base English catalogues, so this small namespace avoids staling
an already translated namespace for an unrelated one-key addition.

Required target locales from `i18n/languages.json`: **fr, es, zh-CN, ru, vi, ur,
hi**. None carries this namespace yet. Independent LLM authors and independent
A/B audits, including Urdu token/projection and real rendered width review,
remain outstanding. This implementer authored no target-language copy, audit
rows, allowance entries or eligibility overrides. Regeneration left
`data/shipped.json` byte-identical (en plus all seven existing target locales);
this is normal English fallback, **not localization completion**. The manifest
coverage gate will require the seven missing translations before public ship.

## Reproduction and visual evidence boundary

The delegator supplied the production-owned before-frame at
`~/.local-operator/sessions/ff0e39d33743/scratchpad/p1-valid-comparison-1.png`;
its file existence was checked here. The delegator, not this implementer,
identified its French/Spanish/Russian/Vietnamese English footer. Base source
independently confirms `READ_ONLY_PLACEHOLDER = "Read-only · esc back"`, imported
by the app and returned by the read-only mode ladder.

No new screenshot, native app, browser, packaged build or full locale matrix was
run under this source-only allocation. The English source/rendering above is
frozen translator input, **not an approved fresh rendered frame**. Request a
fresh authorized English frame and seven-locale rendered audits from the owner
before claiming visual or native approval. The old frame is evidence of the
original symptom only. Previously frozen native-candidate worktrees and their
evidence were not changed or re-certified by this patch.

## Verification performed

Primary checkout interpreter with `PYTHONPATH` pointing at this worktree verified
`local_operator.__file__` resolves to this worktree. No dependency install or
copied venv/node_modules was used.

- `scripts/i18n/generate.py`: exit 0, normal generation only.
- `scripts/i18n/generate.py --check`: exit 0; **201 artifacts byte-identical**.
  Only the new typed Python module and its generated package index differ from
  the base; existing bundled/eligibility artifacts remain byte-identical.
- `scripts/i18n/check.py --update local_operator/tui/widgets/editor.py`: exit 0;
  no baseline byte changes. The checker counts call-site literals, not all module
  constants: both affected modules scan as 0 before and after, so do not infer a
  global literal census from that number.
- Normal `scripts/i18n/check.py`: exit 0; 242 files scanned, 2,755 literals,
  **114 advisory findings**. Advisory green is not global i18n coverage proof.
- Exact tests, isolated fresh HOME/config, `env -i`, `TERM=xterm-256color`,
  `PYTHONPATH=<worktree>`, primary interpreter, **serial `-n 0`**:
  `tests/unit/i18n/test_editor_readonly_footer.py::test_readonly_hint_preserves_english_and_the_keyboard_token`,
  `::test_readonly_hint_re_resolves_on_live_repaint_without_enabling_input`, and
  `tests/unit/tui/test_app_pilot.py::test_the_read_only_composer_shows_neither_caret_nor_bright_chevron`:
  **3 passed, 2 warnings, 2.48 s**. New cases prove exact English/key/token,
  render-time resolver invocation through mounted `_apply_language("en")`, input
  refusal, and no overwrite of a different mode's credential-prompt sentinel.
  The English resolver marker is synthetic wiring evidence, not a translation.
  The neighbor verifies caret/chevron state and restoration. The prior same
  three-case invocation inherited automatic workers (3 passed); the reported
  final invocation was explicitly rerun serially.
- `git diff --check`: clean. Version, baseline, coverage allowance, immutable
  ledger/history and shipped eligibility remained unchanged.

Generator, checker and pytest commands used `scripts/run_bounded.py` with bounds
<=600 seconds. The first
cleanup-required generator invocation refused before spawning on this macOS
backend (125). Subsequent calls used its documented `--contract=command-status`:
actual command exits are verified, but **owned-group cleanup remains unknown**,
not certified. No dev server or browser was started.

Actual implementation model: `openai/gpt-6.1-sol`, temporary fallback after the
primary provider quota failure (initial guide reads used the primary model).
No review/audit approval is implied by authorship or these targeted tests.
