# French independent audit — cli.serve_readiness / sd1

- Audit date: 2026-11-16 (UTC)
- Auditor: Audit daemon French / i18n-auditor; session `19f657b90086`; actual model `openai/gpt-6.1-sol` (session runtime declaration; no fallback observed or reported).
- Translator: task/session `acf0a84cd992`; actual model `openai/gpt-6.1-sol`, as supplied in the task. Auditor is a different session and did not author or edit the translation. Same model does not mean same session.
- Source revision: `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca` plus uncommitted catalogues. This report binds the hashes below, not an assumed committed catalogue.
- English SHA-256: `6c6de62a68a037ac88ba7a63f75f8335845acc61549f78ca98599ac25d29821e`
- Context SHA-256: `e8c5895f98c71f39385a3defe974919ba776a9eda4bf195ac687a6db9b0db693`
- French SHA-256: `fa709445a56aaae44736e248854e28b4c87721a3d079be8710d7b1a92958ed44`
- Scope: all **11/11 keys**, all branches. This is one independent pass; it does not represent a second independent auditor or complete that separate safety/high-visibility gate.

## Verdict

**passed** for this independent catalogue pass: **0 BLOCKER, 0 MAJOR, 1 MINOR, 0 NIT**. No unresolved blocker/major finding. One optional terminology polish is recorded below; no translation edits made.

## Findings

### FR-SD1-01 — MINOR — `cli.serve_readiness.not_reporting`

- Current: `est enregistré mais ne répond pas (son heartbeat s’est arrêté)`
- Problem: `heartbeat` is understandable engineering jargon, so this does not obscure or reverse the diagnosis. However, it is unnecessary untranslated jargon in French product prose and is not among the French guide’s recorded kept-English terms. This is a polish issue, not a claimed glossary violation or safety defect.
- Suggested: `est enregistré mais ne répond pas (il n’envoie plus de signal de vie)`
- Meaning retained: the record exists, but reporting has stopped; this does not claim that the process is dead or that the PID identity is proven.

## Mechanical checks — executed, then independently inspected

`audit.log` is **actual command stdout/stderr and exit status**, not audit notes. `verify.py.txt` is the exact pure-parser verification code used with the checkout’s existing Python interpreter.

- `.venv/bin/python scripts/i18n/check.py --catalogues-only`: exit 0, `0 problem(s)`.
- Independent duplicate-rejecting JSON loads: English, context and French each contain 11 matching keys; no extras, omissions or duplicates.
- Parser verifies argument names, kinds and skeletons both globally and per branch; all 11 pass.
- All three plural messages (`unknown`, `stale`, `other_unit`) carry exactly `one`, `many`, `other`. Rendered all branches using 0, 1, 2 and 1,000,000. French zero correctly uses `one`; the million case uses `many`. No numerals are authored in copy, and the count is a selector, not a displayed number.
- Backtick-command literals are byte-identical per branch. No source Rich/HTML tags, flags, paths or identifiers are removed or modified. Source prose uses `pid` and target prose uses conventional `PID`; this is not a machine token.
- No bidi formatting controls in the French catalogue. RTL-specific requirements are not applicable to `fr`; opaque tokens are preserved.
- Manual punctuation inspection: curly apostrophes, U+202F before French colons and semicolons, no accidental ASCII-apostrophe ICU escape issue; literal command spaces are untouched.

## Every-key semantic, register and terminology assessment

| Key suffix | Assessment |
|---|---|
| `unproven` | Preserves inability to prove process identity and inability to compare a start-time stamp. Does not say the recorded PID identifies the daemon. |
| `not_reporting` | Preserves stopped reporting, not confirmed death; MINOR FR-SD1-01 only. |
| `not_this_daemon` | Correctly says a different process now has the PID; no instruction to act on that process. |
| `unknown_state` | Neutral unknown recorded state, no guessed diagnosis. |
| `install_unknown` | Preserves comparison failure, not a confirmed stale build. `Build` is ordinary French engineering usage here; no glossary-prescribed French counterpart exists. |
| `cgroup_unknown` | Preserves inability of this check to place/classify the cgroup. Does not assert that it is unsupervised. The compact `situer` is comprehensible in this diagnostic context. |
| `recorded_gap` | Explicitly a **recorded** build and unproved process identity; never asserts that the live process runs the recorded version. Both placeholders intact and in source order. |
| `unknown` | Singular/plural agreement correct in all categories, including capped diagnostic lists. Unknown install correspondence and unknown supervision remain unknown. `Impossible de savoir` is conservative wording of the check’s inability, not a positive proof. |
| `stale` | Build mismatch and effect on hosted remote sessions retained. Correct singular/plural agents and pronouns; no inference from an unproved identity introduced by the translation. |
| `other_unit` | Other service vs lop serve unit distinction and negative scope preserved. `Service`, `unité` and `lop serve` follow the glossary. |
| `identity_restart` | Inspect first, then restart to record process identities; per-daemon checks before action remain explicit. Commands byte-identical; `lancez` is correct vous-address in an instructional sentence. No claim of unconditional restart, deletion or killing a reused PID. |

`Supervisé`, `session`, `installation`, `service`, `unité` are consistent; product/command identifiers are unchanged. `Daemon` is intelligible established developer vocabulary, not a glossary conflict. Tone is factual and non-alarming; no tu-form or bureaucratic please added.

## Fit and terminal risk

There are **no maxCells/maxChars hints** for any of these keys. They are long diagnostic sentences/fragments, not labels for fixed-width buttons or tabs. Pure runtime renders with synthetic short bounded parameter values measure 35–207 cells, as recorded per key/branch in `audit.log`. Thus they need normal diagnostic wrapping; width alone without a budget is not a demonstrated overflow defect. Dynamic values can lengthen them. No wide CJK glyphs, combining-mark-dependent spelling or unexpected bidi controls were found. U+202F counts as one Rich cell; no fragile multi-cell symbol hint is introduced.

## Limits and provenance

- This report is `audit_notes_not_execution_log`; only `audit.log` proves the validator/parser executions. No tests, installations, live provider requests or translation/MT services were used.
- No live terminal layout, screenshots, native shaping or visual fit certification is claimed. The captured renders are the existing ICU runtime’s text output, not a live application screenshot.
- Ledger append is parent-owned. A schema-valid candidate receipt and append **dry-run** are supplied; no history was appended by this auditor, no commits were made, and these new repository evidence files are not claimed to be on main.
- Before use, parent must keep these input hashes current and arrange any required second independent safety pass.
