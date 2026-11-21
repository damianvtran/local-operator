# French remediation audit — cli.serve_readiness / sd1r

**Verdict: passed. FR-SD1-01 fixed; no open findings or unresolved BLOCKER/MAJOR.**

## Provenance and scope

- Audit date: 2026-11-16 (UTC).
- Auditor: Audit daemon French (i18n-auditor), session `19f657b90086`, actual model `openai/gpt-6.1-sol`, from the session runtime model declaration; no fallback observed or reported.
- Original translator and remediator: task `acf0a84cd992`, actual model `openai/gpt-6.1-sol`, per requesting brief. This auditor is a different session and has not edited translation files.
- This is the original auditor’s **remediation pass**, not the second independent audit. Initial 11/11-key semantic coverage is retained from `i18n/audits/evidence/fr/cli.serve_readiness/sd1/report.md` and its hash-bound record.
- Current source revision is `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca` plus uncommitted catalogues. No commit or on-main claim.

## Exact delta and finding disposition

### FR-SD1-01 — original severity MINOR — FIXED

Key: `cli.serve_readiness.not_reporting`

- Previous: `est enregistré mais ne répond pas (son heartbeat s’est arrêté)`
- Original suggestion: `est enregistré mais ne répond pas (il n’envoie plus de signal de vie)`
- Current: `est enregistré mais ne répond pas (il n’envoie plus de signal de vie)`

The suggested replacement was applied **exactly**. `Signal de vie` is natural French for the diagnostic. It states only that reporting has stopped, not that the daemon is proven dead, that the PID identity is established, or that a process can be killed without checks. No omission, added action or polarity change. No placeholder, markup, command, numeral or special glyph is introduced.

The new fragment integrates correctly in `unknown` for `one`, `many`, and `other` branches. Its `il` refers to the single process represented by each reason entry, even when the outer list concerns multiple daemons. French punctuation and curly apostrophes remain correct. No fixed width budget exists; the change does not introduce wide or combining-mark-dependent glyphs. This is French LTR copy, not an RTL audit.

## Retained coverage and direct semantic neighbors

The verification reconstructs the previous **entire French file byte sequence** by replacing the exact new value with the exact old value, then compares its SHA-256 with the sd1 target hash. It matches. Therefore the other 10 keys and all formatting bytes outside this replacement are unchanged. English and context hashes also match sd1. The style guide/glossary/process document hashes still match the recorded policy inputs.

Scoped semantic recheck: `not_reporting` plus direct neighbors `unknown`, `unproven`, `not_this_daemon`, `unknown_state`, `recorded_gap`, and `identity_restart`. The distinction between stopped heartbeat, unknown identity, a reused PID and a recorded-only build remains intact. `unknown` still states uncertainty; `identity_restart` still preserves inspect-first order and per-daemon checks before action. No unsafe reinterpretation was introduced. Original assessments of the unchanged `install_unknown`, `cgroup_unknown`, `stale`, and `other_unit` are retained, not reopened for new polish findings.

## Mechanical evidence

Actual outputs are in `audit.log`; this report is **audit_notes_not_execution_log**.

- Existing checkout interpreter, `.venv/bin/python scripts/i18n/check.py --catalogues-only`: exit 0, `0 problem(s)`.
- Hash-verified original pure-parser check rerun against current inputs: 11/11 keys; duplicate-free valid JSON; matching key and context sets; argument names/kinds and code/tag literals preserved per branch; all 3 plural messages contain French `one/many/other`; rendered 0, 1, 2 and 1,000,000.
- Direct `unknown` integration rendered using the actual new fragment for counts 1, 2 and 1,000,000.
- No installs, heavy/full tests, translation/MT services, live providers, translation edits or live process operations.
- No live terminal layout or screenshot certification. Runtime render output is not a rendered application frame.

## Current input SHA-256

| Input | SHA-256 |
|---|---|
| English | `6c6de62a68a037ac88ba7a63f75f8335845acc61549f78ca98599ac25d29821e` |
| Context | `e8c5895f98c71f39385a3defe974919ba776a9eda4bf195ac687a6db9b0db693` |
| French | `4f3fcbc1a716d020cc9735382e607ca315e5cdbf2d922750f8d333f2e4f9ae68` |

## Ledger and retention

`ledger-receipt.json` is the append-ready current-input receipt, with a record-linked note <=280 characters. `ledger-dry-run.log` contains the actual CLI help, history digest and append dry-run. Parent owns the append; this auditor does not append history or commit. Existing sd1/sd2 bundles and existing ledger rows are preserved. New sd1r files are repository evidence awaiting the parent’s tracking commit, not claimed to be on main.
