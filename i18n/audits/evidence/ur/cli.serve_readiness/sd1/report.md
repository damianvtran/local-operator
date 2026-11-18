# Independent Urdu audit — cli.serve_readiness — sd1

- Audit date: 2026-11-16; source revision: e4a6b6e1d01ef269604fe6d005e6bce1453b66ca plus the new uncommitted namespace inputs.
- Auditor: i18n-auditor / Audit daemon Urdu; session c736f176af6f; actual model openai/gpt-6.1-sol, from session runtime declaration; no fallback reported.
- Translator: task acf0a84cd992, actual openai/gpt-6.1-sol, as supplied by the delegation. Different session; this auditor did not translate or edit the copy.
- Scope: all 11 Urdu keys, including both branches of all three plural messages. No other locale audited. No MT/translation service, live provider, installs, full tests or daemon operations.
- Verdict: **changes_requested** — 0 BLOCKER, 1 MAJOR, 0 MINOR, 0 NIT.
- This is one independent audit, not a substitute for the second separate auditor required for safety/high-visibility copy. No earlier audit verdict was relied on.

## UR-SD1-M1 — MAJOR — literal calque obscures the diagnostic

**Key:** `cli.serve_readiness.cgroup_unknown`

**Current:** ایسے cgroup میں چلتا ہے جس کی جگہ یہ جانچ متعین نہیں کر سکتی

**Problem:** `جس کی جگہ ... متعین نہیں کر سکتی` says the check cannot determine the cgroup's location. English “cannot place” here means the check cannot classify its supervision context. `serve_supervision()` reads a supplied path and classifies a known serve unit, another service, a login scope, or unknown; an unknown classification does not necessarily mean the path/location is missing. This is a conspicuous literal calque in operational diagnostic copy, potentially directing the reader toward a location/path failure rather than an unclassified supervision context.

**Suggested wording:** ایسے cgroup میں چلتا ہے جس کی نوعیت یہ جانچ متعین نہیں کر سکتی

This expresses inability to determine the cgroup's type/nature, retains the technical token and uncertainty, and does not assert an absent service unit or a definite supervision failure. No translation edit made.

## Every-key accuracy, idiom, register and terminology pass

The following is **audit_notes_not_execution_log**, not proof of command execution.

| Key suffix | Assessment |
|---|---|
| unproven | Accept. Recorded versus proven identity remains distinct. `نہیں کی جا سکتی` is not a positive identity claim. Parenthesis preserves inability to compare a start-time stamp. |
| not_reporting | Accept. Recorded-but-not-answering and stopped heartbeat are retained; does not claim the process is proven dead. |
| not_this_daemon | Accept. Says another process now holds that PID; no assertion about that other process's build or supervision. `PID` is the normal acronym, not a renamed identifier. |
| unknown_state | Accept. Recorded state remains unknown; no stronger claim added. |
| install_unknown | Accept. Running build cannot be compared with the installed build; does not assert a mismatch. `بلڈ` is intelligible developer-register copy. |
| cgroup_unknown | Changes requested: UR-SD1-M1. |
| recorded_gap | Accept. Explicit `ریکارڈ میں درج بلڈ` and `عمل کی شناخت ثابت نہیں` correctly prevent a recorded/install gap becoming evidence about the current process. Placeholder labels/order retained. |
| unknown | Accept both branches. `معلوم نہیں ... یا نہیں` scopes uncertainty over installed build and supervision. Singular `رہا ہے` versus plural `رہے ہیں` agrees with count. Invariant loanword `ڈیمن` is intelligible with this agreement. `زیرِ نگرانی` follows the glossary. |
| stale | Accept both branches. Running versus installed builds remain distinct, and hosted remote sessions run the other build(s). `اس کی` versus `ان کی` and singular/plural verbs match. `ریموٹ سیشن` follows the guide/glossary. |
| other_unit | Accept both branches. Another service is distinguished from a lop serve unit; `سروس` and `یونٹ` follow the glossary. Singular/plural agreement is retained. Does not imply a known lop serve unit supervises the daemon. |
| identity_restart | Accept. Respectful `چلائیں`; inspect first, restart second, purpose is recording process identities. Guard before action remains explicit; no guarantee of unguarded restart or newly proven identity. Commands unchanged. |

## Mechanical pass

Actual executions, return codes and output are in `audit.log`; repeatable narrow parser input is `mechanical.py.txt`.

- Repository `scripts/i18n/check.py --catalogues-only`: passes with zero problems using the already-installed harness Python 3.14 interpreter. Initial `/usr/bin/python3` attempt failed because that interpreter is Python 3.9 and lacks `tomllib`; it was not treated as successful validation. No installation was needed.
- Independently verify duplicate-free valid JSON, exact source/Urdu/context key parity (11 each), parser argument name/kind/order parity, selectors, command/backtick text, tags, absence of manual bidi controls and native digits.
- All three plural messages have the full Urdu CLDR set `one` / `other`; no select messages or markup tags. Render every key with the repo parser; plural inputs 0/1/2 exercise both categories. The source intentionally uses count for agreement without printing it; translation preserves that contract.
- No changed flags, code or identifiers. `lop serve` outside backticks in `other_unit` is also retained, as in the source. `cgroup` remains Latin.

## Fit, punctuation and RTL

- No `maxCells` or `maxChars` hints exist for these 11 keys. These are diagnostic sentences, not buttons/tabs. Long composed reasons and opaque device/build/holder tokens may wrap; without a pinned width hint or actual terminal frame, no numerical fit certification is claimed.
- Urdu punctuation `،` and `؛` is appropriate. Diagnostic colons and parentheses remain sensible boundaries around Latin tokens. No arrows to mirror, no native digit substitution, no manual bidi controls.
- Urdu logical sentence order and postpositions are natural. Literal commands are intact LTR islands; do not add controls to the catalogue. Device names, PID lists, build strings and holder strings can carry mixed-direction runs, comma-separated IDs or paths. Their internal order and pairing need renderer isolation. Raw parser output is not evidence that a terminal reorders/shapes them correctly.
- No actual terminal/browser rendered frame was captured. Rendered RTL/shaping and fixed-width fit remain integration evidence outside this bounded catalogue-only audit; this report does not certify them or introduce a new catalogue defect to compensate for terminal limitations.

## Provenance and retention

`record.json` binds the exact input hashes, actual auditor identity, report, log and ledger receipt. `receipt.json` hashes companions and explicitly references that record; `ledger-receipt.json` is the single row accepted by `scripts/i18n/ledger.py append --receipt ...` and is dry-run validated. No append is applied: the parent owns history mutation.

Files were newly created only under the assigned canonical repo-only evidence directory. They are uncommitted until the parent stages/commits them; no claim is made that this bundle is on main. No original evidence was lost or recreated, and no existing bundle/addendum/manifest was changed. The report is a new original audit dated today, not a historical R2 recreation.
