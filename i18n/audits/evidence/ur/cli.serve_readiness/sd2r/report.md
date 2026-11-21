# Independent second-pass remediation audit — ur / cli.serve_readiness / sd2r

**Current verdict: PASSED.** Changed values independently assessed on fresh target bytes; zero unresolved findings in this reassessment. This is not a claim that the earlier initial clean sd2 verdict overrode first-pass findings.

Auditor: Local Operator, session `85af124f4a9f`, actual model `openai/gpt-6.1-sol` (host declaration), no fallback observed/reported. Translator is `acf0a84cd992`; auditor authored/edited no translation. Not a native-speaker, terminal-shaping, design or release-eligibility approval.
Audit time: 2026-11-16T04:52:46.184602+00:00; HEAD `e4a6b6e1d01ef269604fe6d005e6bce1453b66ca`; working-tree bytes are authoritative.

## Fresh-byte inputs

- English SHA256: `6c6de62a68a037ac88ba7a63f75f8335845acc61549f78ca98599ac25d29821e`
- Context SHA256: `e8c5895f98c71f39385a3defe974919ba776a9eda4bf195ac687a6db9b0db693`
- Current target SHA256: `4180f78e34dd0ca0f9d0fa8b05a53de5abe8a103090d7f2bb9e0ae3446441f2a`
- Prior target SHA256 (sd2): `07c4be789363310c6f1863fd643974ca7741f18c99b7fd6c9404872ed84d9674`

English, context, glossary and locale style guide match the prior second-pass input hashes. Only the declared changed values differ: restoring their old sd2 rendered values and serializing the complete catalogue reproduces the exact prior whole-file SHA256. This independently verifies unchanged values and formatting, not merely the count of changed keys.

## Finding dispositions on current targets

### UR-SD1-M1 — MAJOR — FIXED
Key: `cli.serve_readiness.cgroup_unknown`
نوعیت ... متعین نہیں کر سکتی means this check cannot determine the nature/type. It removes spatial جگہ and preserves uncertainty. cgroup stays Latin in logical order; no manual bidi controls are introduced.

## Direct semantic neighbors and retained coverage

Rechecked the relationship of not_reporting, cgroup_unknown, unproven, recorded_gap, unknown and identity_restart. A non-answering recorded daemon is not asserted dead; unproven identity stays unproven. recorded_gap remains stored-build information, unlike the confirmed-live stale message. The unknown wrapper stays noncommittal about installed-build match and supervision. Guarded restart still checks each daemon before action; both literal commands and status-then-restart order survive.
The unchanged stale and other_unit wording continues to distinguish installed versus running build and an alternate supervisor versus no supervision. Unchanged key semantics retain the prior independent all-key sd2 coverage, rather than receiving a new unrelated full linguistic audit.

Changed keys (1): `cli.serve_readiness.cgroup_unknown`.
Unchanged keys carried forward: `cli.serve_readiness.unproven`, `cli.serve_readiness.not_reporting`, `cli.serve_readiness.not_this_daemon`, `cli.serve_readiness.unknown_state`, `cli.serve_readiness.install_unknown`, `cli.serve_readiness.recorded_gap`, `cli.serve_readiness.unknown`, `cli.serve_readiness.stale`, `cli.serve_readiness.other_unit`, `cli.serve_readiness.identity_restart`.

## Lightweight catalogue validation

Production ICU parser/runtime: 11 keys mechanically checked, 47 ordinary renders, 47 opaque/pseudo-parameter renders; counts [0, 1, 2, 3, 4, 5, 11, 21, 22, 25, 1.5, 1000000, 1500000]; every declared CLDR category exercised. Per-branch placeholders, argument order/kinds, literals and raw Unicode checked. `rendered.txt` includes the current changed values and composed uncertainty wrappers. Urdu stays logical order with no embedded bidi controls, and technical Latin bytes remain unchanged; no actual terminal bidi/shaping or screenshot proof is claimed. No arbitrary width cap is invented because the context declares none.

Actual shared command output: `i18n/audits/evidence/fr/cli.serve_readiness/sd2r/audit.log`. Reproducer: `i18n/audits/evidence/fr/cli.serve_readiness/sd2r/catalogue-check.py.txt`. No tests/fixtures/rigs, installs, generated files, translation edits, commits, or ledger/history appends. First auditors/parent own any fresh ledger receipts.

No remaining blocker, major, minor or nit is raised by this fresh-target reassessment. Approval is limited to the target SHA256 above and the unchanged source/context; further edits require a new binding.
