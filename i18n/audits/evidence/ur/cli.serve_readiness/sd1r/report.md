# Urdu remediation audit — cli.serve_readiness — sd1r

**Verdict: passed.** Zero unresolved BLOCKER/MAJOR findings. **UR-SD1-M1: fixed.**

Audit date: 2026-11-16. Auditor: i18n-auditor / Audit daemon Urdu, session c736f176af6f, actual model openai/gpt-6.1-sol from session runtime declaration; no fallback reported. Translator/remediator remains the original task acf0a84cd992, actual openai/gpt-6.1-sol, independent of this auditor. No translation edits, history append, commit, installation, full test, live provider or MT service used.

## Scope and retained coverage

This is a focused remediation audit, not a replacement or rewriting of the original sd1 record. Exact byte comparison proves English/context unchanged and only the accepted Urdu cgroup key changed. All ten unchanged keys retain the full initial semantic/idiom/terminology/fit/RTL coverage in `../sd1/report.md`, bound to `../sd1/record.json`. Mechanical checks run on all eleven current keys. This continuation by the same auditor does not claim to be the manager's second separate independent audit.

## Finding disposition

### UR-SD1-M1 — MAJOR — FIXED

Key: `cli.serve_readiness.cgroup_unknown`

Prior: ایسے cgroup میں چلتا ہے جس کی جگہ یہ جانچ متعین نہیں کر سکتی

Current: ایسے cgroup میں چلتا ہے جس کی نوعیت یہ جانچ متعین نہیں کر سکتی

The original translator applied the exact sd1 suggestion. `نوعیت` describes the cgroup's type/nature, removing the location calque. It remains an inability of this check to classify, not proof of absent supervision, an unknown physical path, a dead daemon or a different running build. `cgroup` remains unchanged Latin. No new placeholder, punctuation, control character or unsafe assertion was introduced.

## Direct semantic neighbors independently checked

- `unknown`, both branches: a composed `pid 123: <corrected reason>` still leads to uncertainty about installed-build currency and supervision. Singular/plural agreement remains correct. It does not convert an unclassified context into a definite unsupervised verdict.
- `other_unit`, both branches: known other-service ownership remains a distinct diagnosis, not conflated with unknown cgroup type. `سروس` and `یونٹ` remain glossary-consistent.
- `install_unknown`: inability to compare builds remains separate from inability to classify supervision.
- `unproven` / `recorded_gap`: process-identity uncertainty and merely recorded build labels remain explicitly conservative; the changed cgroup wording introduces no implication that an unproven process identity has been established.
- `identity_restart`: status inspection first, guarded restart second, and checking each daemon before action remain unchanged and safe.

These semantic assessments are **audit_notes_not_execution_log**. Actual command execution is separately retained in `audit.log`.

## Mechanical, fit and RTL

Repository catalogue-only validator and the retained narrow parser reproduction run against current input. The delta script proves that replacing the current sentence with the exact previous sentence reproduces the original Urdu file SHA-256; consequently all other bytes are unchanged. The prior mechanical script is hash-checked before reuse. Current key parity, argument name/type/order parity, valid ICU, full Urdu one/other categories, command/backtick/tag parity and absence of inserted bidi controls/native digits pass. Plural inputs 0/1/2 are exercised, including the changed reason embedded in `unknown`.

The replacement adds only the short Urdu noun `نوعیت` in place of `جگہ`; there are no maxCells/maxChars hints, no compact button/tab, no mirrored arrows and no new combining-mark or token-order hazard. Existing diagnostic wrapping and Latin-token bidi integration caveats from sd1 remain. No actual terminal frame/shaping/mirroring proof is claimed, and no new out-of-scope minor is raised.

## Artifacts and history

`record.json` binds source/context/current target hashes and actual provenance. `ledger-receipt.json` is a single append-ready passed row, dry-run only; parent owns append. `receipt.json` hashes the companions and references the authoritative record. Existing sd1/sd2 evidence and the earlier changes_requested ledger row are preserved. This new canonical repo-only bundle must be staged/committed by the parent; it is not claimed to be on main.
