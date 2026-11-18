# Vietnamese remediation re-audit — cli.serve_readiness / sd1r

**PASSED: zero open findings; VI-SD1-01 FIXED.**

Date: 2026-11-16 UTC. Auditor: Audit daemon Vietnamese; session
`77442ce206e1`; actual model `openai/gpt-6.1-sol`, as declared by the host.
No fallback incident observed. This is the same independent auditor resumed after
the original translator's remediation, not another independent second-pass agent.
Translator task `acf0a84cd992`, actual model `openai/gpt-6.1-sol`, as supplied by
the requesting task. No translation/MT service used. No translation edits made.

## Scoped remediation and retained coverage

The original audit report and record remain unchanged at
`i18n/audits/evidence/vi/cli.serve_readiness/sd1/`.
English, context and all retained guidance hashes still match sd1. Replacing only
the current `cgroup_unknown` value with its old value and serializing in the
original format reproduces sd1's complete target SHA-256. Thus all ten other
values, key order, formatting and catalogue bytes are retained unchanged.
Initial full 11-key semantic/idiom/glossary/fit coverage is retained for unchanged
content; this re-audit focuses on the one changed key and direct semantic neighbors.
It does not re-open previously approved unrelated copy.

## Finding disposition

**VI-SD1-01 — MINOR — FIXED**

Key: `cli.serve_readiness.cgroup_unknown`

- Original: `chạy trong cgroup mà lần kiểm tra này không thể xác định vị trí`
- Requested wording: `chạy trong cgroup mà lần kiểm tra này chưa phân loại được`
- Current: `chạy trong cgroup mà lần kiểm tra này chưa phân loại được`
- Result: exact suggested replacement applied. “chưa phân loại được” naturally
  conveys inability to classify the cgroup, without implying a missing directory
  or claiming either successful identity proof or a known supervisor.

No finding remains open. No new out-of-scope minor/nit is introduced.

## Direct semantic neighbors

- `unknown`: composed with the new fragment, it still says current-build match
  and supervision are not known; no unknown-to-negative polarity conversion.
- `other_unit`: remains the distinct known-other-service diagnosis, not confused
  with a cgroup that cannot yet be classified. `dịch vụ` and kept `unit` remain
  glossary-conformant.
- `recorded_gap`: still qualifies the build as recorded and the identity as
  unproven; the cgroup change adds no assertion about the live PID's build.
- `unproven`: inability to compare start-time evidence remains distinct from
  inability to classify supervision. No unsafe PID identity inference is added.
- `identity_restart`: still orders inspection before the guarded restart and
  retains exact commands. No new kill, permission grant or unconditional success
  promise is introduced.

## Mechanical, fit and direction checks

The repository lightweight catalogue validator reported 0 problems (exit 0).
Additional parser checks covered 11/11 keys: strict JSON with duplicate-key
rejection; key/context parity; exact argument names, kinds and order; valid ICU;
all three Vietnamese plural messages have only the required `other` category;
backtick commands and markup preserved; NFC with no combining-mark sequences.
Composed neighbor samples exercised counts 0, 1, 2, 5 and 101.
The changed string has 57 characters / 57 terminal cells under Rich, down from
63; no width hint exists for it. It introduces no wide glyphs or bidi hazards.
Vietnamese is LTR; Urdu-specific RTL requirements do not apply. No live rendered
UI, screenshot, full test suite or provider execution is claimed.

## Evidence and ledger handoff

This report is **audit_notes_not_execution_log**. `audit.log` retains actual
commands, stdout, stderr and exit codes. `ledger-receipt.json` is a single
append-ready ledger row; its actual append dry-run passed. The auditor did not
append history or commit files. The parent must obtain a fresh history digest
before applying if history changes. `receipt.json` is the bundle-integrity
companion, explicitly referencing `record.json` for auditor provenance.

This is new repo-only evidence, not a replacement of lost originals or a claim
that this bundle is committed/merged/on main. sd1 and other bundles are preserved.
