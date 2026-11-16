# hi · cli.failures · audit gbt2 (independent, fresh round)

Auditor: i18n-auditor (audit2-hi), session de3fcb4a492b. Model: started on
openai/gpt-6.1-sol; harness switched this run to anthropic/claude-opus-5-5
(provider quota HTTP 429) before any judgement or artifact was written.
Not the translator session. Worktree commit 80a55b208 (base f0689a27).

Scope: all 5 keys of `cli.failures` (new: `cli.failures.github_borrow_timeout`).
Inputs read: glossary, hi style guide, docs/I18N.md, en catalogue + context
sidecar, hi catalogue. No MT service, no earlier receipts for this key.

## Verdict: PASSED — BLOCKER 0 · MAJOR 0 · MINOR 1 · NIT 0

## Mechanical
- `scripts/i18n/check.py --catalogues-only`: rc 0, 0 problems (audit.log).
- Keys identical to en (5/5); placeholders identical per key
  (`{prog} {message} {names} {argument} {value} {choices} {group} {command} {owner} {seconds}`);
  no ICU plurals in this namespace; valid JSON; no ZWJ/ZWNJ/bidi marks.
- Tool output masks the Hindi credential word as `[redacted]`; on-disk bytes
  verified by codepoint dump: the word is U+0915 U+094D U+0930 U+0947 U+0921
  U+0947 U+0902 U+0936 U+093F U+092F U+0932 = क्रेडेंशियल, byte-identical to the
  style guide's `credentials →` rendering and the glossary hi cell.
  Full dump of the new value in audit.log.
- Width (rich cell_len): new key 59 cells / 79 code points; no maxCells.

## Key-by-key
- `error_line` `{prog}: त्रुटि: {message}` — shape kept. OK.
- `required` `इन आर्ग्युमेंट की ज़रूरत है: {names}` — terse, natural; majority
  spelling आर्ग्युमेंट (22 vs 3 in hi catalogues). OK.
- `invalid_choice` `आर्ग्युमेंट {argument}: अमान्य विकल्प: {value} ({choices} में से चुनें)` — OK.
- `invalid_command` `अमान्य {group} कमांड: {command}` — OK.
- `github_borrow_timeout` `{owner} से GitHub क्रेडेंशियल {seconds} सेकंड में नहीं मिला; कमांड फिर से चलाएँ`
  — meaning preserved: source device, deadline, not received, user re-runs.
  GitHub kept Latin; glossary term क्रेडेंशियल; masculine agreement (मिला)
  matches shipped `wire.errors.auth.credentials_missing`; सेकंड and
  `फिर से चलाएँ` match existing hi copy; polite imperative; semicolon and no
  final danda mirror the en line. No denial/revoke/login/auto-retry/lease
  extension implied. `seconds` is rendered with `:g` (e.g. 30, 1.5); सेकंड
  does not inflect, so any value reads correctly.

## Findings
- **H1 MINOR** `cli.failures.github_borrow_timeout` — `… सेकंड में नहीं मिला`.
  मिलना covers both "be received" and "be found"; hi copy already uses
  `… क्रेडेंशियल नहीं मिला` for "no stored credential found". The time bound
  makes "not received" the natural reading, but "did not arrive" is more
  exactly and unambiguously `नहीं पहुँचा`, and "within" is `के भीतर`.
  Suggested: `{owner} से GitHub क्रेडेंशियल {seconds} सेकंड के भीतर नहीं पहुँचा; कमांड फिर से चलाएँ`.
  Optional; does not block.

## Observation (not a translation finding)
- The new line in hi (and en) cli.failures.json is indented 1 space where
  siblings use 2; JSON is valid, cosmetic only.
- The commit regenerates `local_operator/i18n/data/shipped.json` to `["en"]`
  pending current-hash ledger rows; outside this audit's scope.

## Limitations
Independent LLM review, not native-speaker certification. Scalar text
evidence only; no terminal frame rendered. The ledger append was run with
`--dry-run` only (result reported to the requesting session, not stored here
because the receipt hashes this bundle).
