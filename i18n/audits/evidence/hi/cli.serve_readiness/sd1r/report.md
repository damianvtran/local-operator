# Independent Hindi remediation audit — cli.serve_readiness / sd1r

**Date:** 2026-11-16. **Verdict: passed.** Unresolved BLOCKER/MAJOR: **0**. No new findings.

Auditor: **Audit daemon Hindi**, session `1ee2c984b1e5`, actual model `openai/gpt-6.1-sol` according to host session-state; no fallback indicated. Independent of original translator session `acf0a84cd992`, also reported as `openai/gpt-6.1-sol`. The translator, not this auditor, applied remediation. No translation or MT service was used.

## Scope and retained coverage

This is a remediation-delta audit, not a second independently staffed pass. It retains the complete 11-key initial audit in `../sd1/record.json` and `../sd1/report.md`, and freshly reviews the changed key plus six direct semantic neighbors. It does not overwrite, reclassify or erase the historical `changes_requested` result.

Current English/context/glossary/Hindi style-guide hashes equal those recorded in sd1. The exact proposed replacement appears once. Replacing only that new phrase with the old phrase **in memory** restores the complete initial Hindi catalogue SHA-256 `44980d359b70259062ec915d495235869e62996247beaa7e851e05b8a65ab439`. Thus all other 10 keys and all surrounding bytes are unchanged. This is checked in the actual execution log, not assumed from the manager's summary.

## Finding resolution

### HI-SD1-01 — FIXED (previously MAJOR)

**Key:** `cli.serve_readiness.cgroup_unknown`

**Initial text:** ऐसे cgroup में चलता है जिसका स्थान यह जाँच तय नहीं कर सकती

**Current text / exact suggested replacement:** ऐसे cgroup में चलता है जिसकी श्रेणी इस जाँच से पता नहीं चल पा रही है

The literal location/placement calque has been removed. The new wording naturally states that the check cannot determine the cgroup's category. In context this is its supervision classification, not the location of the cgroup, process identity, or proof of a lack of supervision. It remains conservative: no claim that the process is unsupervised, that identity has been proved, or that a particular service holds it. The sentence reads naturally as a reason inside the surrounding unknown-status message. No further wording change is needed.

## Mechanical verification

- Fresh `.venv/bin/python scripts/i18n/check.py --catalogues-only`: **0 problems, exit 0**.
- Fresh duplicate-rejecting JSON parse and exact key-set check: **11 English = 11 context = 11 Hindi**.
- Fresh direct parsed-AST comparison for all 11 keys: names, types, skeletons, branch structure and exact backtick command spans match English.
- All plural messages still contain complete Hindi **one/other** CLDR categories. Fresh neighbor renders cover counts **0, 1, 2, 5, 21**; initial `stale` branch coverage is retained because it is byte-identical.
- The changed message introduces no placeholder, markup, flag, digit, bidi control or technical-token mutation. Literal `cgroup` is unchanged. Initial unchanged markup/command/digit checks remain applicable by whole-file byte-reconstruction proof.

## Fresh direct-neighbor review

| Key suffix | Remediation/regression assessment |
|---|---|
| `cgroup_unknown` | Exact accepted fix; diagnostic category uncertainty is accurate and idiomatic. |
| `unknown` | New reason composes coherently in both plural branches. Installed-build currency and supervision remain unknown; no unsafe factual conclusion appears. |
| `other_unit` | Still distinguishes supervision by another service from a lop serve unit; not confused with an unknown category. Both branches remain grammatical. |
| `install_unknown` | Still limits uncertainty to comparing installed and running builds; no invented mismatch. |
| `unproven` | Still says process identity cannot be proved, not that the daemon definitely died or PID identity was checked. |
| `recorded_gap` | Recorded-build evidence remains explicitly separate from proof of the current process's identity or running build. |
| `identity_restart` | Records inspection, then guarded restart; per-daemon checks remain explicit and commands exact. No unguarded kill/restart instruction was introduced. |

The remaining four keys (`not_reporting`, `not_this_daemon`, `unknown_state`, `stale`) retain their initial complete audit coverage; no fresh whole-namespace semantic re-audit is claimed. No new out-of-scope minors were introduced into this terminal remediation pass.

## Terminology, fit and directionality

The changed wording respects the Hindi style guide and shared glossary; settled technical `cgroup` remains Latin. No change to polite imperative or process-supervision terminology occurs. The new message measures **68 code points / 49 Rich cells** in the actual text-only check. There are no maxCells/maxChars hints. The longer composed diagnostic is expected to wrap, as before; no demonstrated fixed-width overflow is claimed. Rich cell measurements are not physical Devanagari shaping proof. Hindi is LTR; no Urdu RTL sign-off is applicable.

## Evidence and limits

`report.md` is **audit_notes_not_execution_log**. `audit.log` contains actual new validator, structural/render verification and ledger dry-run execution. `record.json` binds current input hashes and retains the sd1 record hash. `receipt.json` hashes companions, and `ledger-receipt.json` is the append-ready proposed passing row.

No full tests, installs, new tree, live provider, service action, rendered frame, translation edit, history append or commit was performed. Append is dry-run only; parent appends history and adds/commits the new repo-only evidence. Existing sd1/sd2 bundles and prior ledger rows are preserved. This pass does not claim completion of any separately required second independent safety audit.
